extern "C" {

__device__ __forceinline__ float smx_warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

__global__ __launch_bounds__(256) void smx_cuda_givens_fwd(
    const float* __restrict__ x,
    const float* __restrict__ angles,
    float* __restrict__ y,
    int totalHalf,
    int halfDim)
{
    int stride = (int)gridDim.x * 256;
    int dim = halfDim << 1;
    for (int idx = (int)blockIdx.x * 256 + (int)threadIdx.x; idx < totalHalf; idx += stride) {
        int m = idx / halfDim;
        int i = idx - m * halfDim;
        long long rowOff = (long long)m * dim;
        float s, c;
        __sincosf(angles[i], &s, &c);
        float x0 = x[rowOff + i];
        float x1 = x[rowOff + halfDim + i];
        y[rowOff + i] = x0 * c - x1 * s;
        y[rowOff + halfDim + i] = x0 * s + x1 * c;
    }
}

__global__ __launch_bounds__(256) void smx_cuda_givens_bwd_dx(
    const float* __restrict__ dy,
    const float* __restrict__ angles,
    float* __restrict__ dx,
    int totalHalf,
    int halfDim)
{
    int stride = (int)gridDim.x * 256;
    int dim = halfDim << 1;
    for (int idx = (int)blockIdx.x * 256 + (int)threadIdx.x; idx < totalHalf; idx += stride) {
        int m = idx / halfDim;
        int i = idx - m * halfDim;
        long long rowOff = (long long)m * dim;
        float s, c;
        __sincosf(angles[i], &s, &c);
        float dy0 = dy[rowOff + i];
        float dy1 = dy[rowOff + halfDim + i];
        dx[rowOff + i] = dy0 * c + dy1 * s;
        dx[rowOff + halfDim + i] = -dy0 * s + dy1 * c;
    }
}

__global__ __launch_bounds__(128) void smx_cuda_givens_bwd_dphi(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ angles,
    float* __restrict__ dAngles,
    int rows,
    int halfDim)
{
    __shared__ float s_warp[4];
    int i = (int)blockIdx.x;
    if (i >= halfDim) return;
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;

    float s, c;
    __sincosf(angles[i], &s, &c);
    int dim = halfDim << 1;
    float acc = 0.0f;
    for (int m = tid; m < rows; m += 128) {
        long long rowOff = (long long)m * dim;
        float x0 = x[rowOff + i];
        float x1 = x[rowOff + halfDim + i];
        float dy0 = dy[rowOff + i];
        float dy1 = dy[rowOff + halfDim + i];
        acc += dy0 * (-x0 * s - x1 * c) + dy1 * (x0 * c - x1 * s);
    }
    acc = smx_warp_sum(acc);
    if (lane == 0) s_warp[warpId] = acc;
    __syncthreads();
    if (tid == 0) {
        dAngles[i] = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    }
}

__global__ __launch_bounds__(256) void smx_cuda_conv2_fwd(
    const float* __restrict__ x,
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    float* __restrict__ y,
    int total,
    int seqLen,
    int channels)
{
    int stride = (int)gridDim.x * 256;
    for (int idx = (int)blockIdx.x * 256 + (int)threadIdx.x; idx < total; idx += stride) {
        int c = idx % channels;
        int bt = idx / channels;
        int t = bt % seqLen;
        float xv = x[idx];
        float xp = t > 0 ? x[idx - channels] : 0.0f;
        y[idx] = w0[c] * xv + w1[c] * xp;
    }
}

__global__ __launch_bounds__(256) void smx_cuda_conv2_bwd_dx(
    const float* __restrict__ dy,
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    float* __restrict__ dx,
    int total,
    int seqLen,
    int channels)
{
    int stride = (int)gridDim.x * 256;
    for (int idx = (int)blockIdx.x * 256 + (int)threadIdx.x; idx < total; idx += stride) {
        int c = idx % channels;
        int bt = idx / channels;
        int t = bt % seqLen;
        float dyv = dy[idx];
        float dyn = (t + 1 < seqLen) ? dy[idx + channels] : 0.0f;
        dx[idx] = w0[c] * dyv + w1[c] * dyn;
    }
}

__global__ __launch_bounds__(128) void smx_cuda_conv2_bwd_dw(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    float* __restrict__ dw0,
    float* __restrict__ dw1,
    int batch,
    int seqLen,
    int channels)
{
    __shared__ float s_w0[4];
    __shared__ float s_w1[4];
    int c = (int)blockIdx.x;
    if (c >= channels) return;
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;

    int rows = batch * seqLen;
    float a0 = 0.0f;
    float a1 = 0.0f;
    for (int r = tid; r < rows; r += 128) {
        int t = r % seqLen;
        long long cur = (long long)r * channels + c;
        float dyv = dy[cur];
        a0 += dyv * x[cur];
        if (t > 0) a1 += dyv * x[cur - channels];
    }
    a0 = smx_warp_sum(a0);
    a1 = smx_warp_sum(a1);
    if (lane == 0) {
        s_w0[warpId] = a0;
        s_w1[warpId] = a1;
    }
    __syncthreads();
    if (tid == 0) {
        dw0[c] = s_w0[0] + s_w0[1] + s_w0[2] + s_w0[3];
        dw1[c] = s_w1[0] + s_w1[1] + s_w1[2] + s_w1[3];
    }
}

__global__ __launch_bounds__(128) void smx_cuda_ropel2_fwd(
    const float* __restrict__ x,
    const float* __restrict__ cosBuf,
    const float* __restrict__ sinBuf,
    float* __restrict__ y,
    int totalHeads,
    int seqLen,
    int numHeads,
    int headDim,
    int startPos,
    float eps,
    float postScale)
{
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    int half = headDim >> 1;
    int nPairs = (half + 31) >> 5;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;

    for (int r = globalWarp; r < totalHeads; r += totalWarps) {
        int bt = r / numHeads;
        int t = bt % seqLen;
        long long cOff = (long long)(startPos + t) * half;
        long long xOff = (long long)r * headDim;
        float r0[8];
        float r1[8];
        float localSq = 0.0f;
        for (int k = 0; k < nPairs; k++) {
            int p = (k << 5) + lane;
            if (p < half) {
                float x0 = x[xOff + 2 * p];
                float x1 = x[xOff + 2 * p + 1];
                float c = cosBuf[cOff + p];
                float s = sinBuf[cOff + p];
                float v0 = x0 * c - x1 * s;
                float v1 = x0 * s + x1 * c;
                r0[k] = v0;
                r1[k] = v1;
                localSq += v0 * v0 + v1 * v1;
            }
        }
        float normSq = smx_warp_sum(localSq);
        float sc = postScale * rsqrtf(normSq + eps);
        for (int k = 0; k < nPairs; k++) {
            int p = (k << 5) + lane;
            if (p < half) {
                y[xOff + 2 * p] = r0[k] * sc;
                y[xOff + 2 * p + 1] = r1[k] * sc;
            }
        }
    }
}

__global__ __launch_bounds__(128) void smx_cuda_ropel2_bwd(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ cosBuf,
    const float* __restrict__ sinBuf,
    float* __restrict__ dx,
    int totalHeads,
    int seqLen,
    int numHeads,
    int headDim,
    int startPos,
    float eps,
    float postScale)
{
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    int half = headDim >> 1;
    int nPairs = (half + 31) >> 5;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;

    for (int r = globalWarp; r < totalHeads; r += totalWarps) {
        int bt = r / numHeads;
        int t = bt % seqLen;
        long long cOff = (long long)(startPos + t) * half;
        long long xOff = (long long)r * headDim;
        float r0[8];
        float r1[8];
        float dy0[8];
        float dy1[8];
        float cp[8];
        float sp[8];
        float localSq = 0.0f;
        for (int k = 0; k < nPairs; k++) {
            int p = (k << 5) + lane;
            if (p < half) {
                float x0 = x[xOff + 2 * p];
                float x1 = x[xOff + 2 * p + 1];
                float c = cosBuf[cOff + p];
                float s = sinBuf[cOff + p];
                float v0 = x0 * c - x1 * s;
                float v1 = x0 * s + x1 * c;
                r0[k] = v0;
                r1[k] = v1;
                cp[k] = c;
                sp[k] = s;
                dy0[k] = dy[xOff + 2 * p];
                dy1[k] = dy[xOff + 2 * p + 1];
                localSq += v0 * v0 + v1 * v1;
            }
        }
        float invNorm = rsqrtf(smx_warp_sum(localSq) + eps);
        float localDot = 0.0f;
        for (int k = 0; k < nPairs; k++) {
            int p = (k << 5) + lane;
            if (p < half) {
                r0[k] *= invNorm;
                r1[k] *= invNorm;
                localDot += dy0[k] * r0[k] + dy1[k] * r1[k];
            }
        }
        float dot = smx_warp_sum(localDot);
        float sc = postScale * invNorm;
        for (int k = 0; k < nPairs; k++) {
            int p = (k << 5) + lane;
            if (p < half) {
                float dr0 = sc * (dy0[k] - r0[k] * dot);
                float dr1 = sc * (dy1[k] - r1[k] * dot);
                float c = cp[k];
                float s = sp[k];
                dx[xOff + 2 * p] = dr0 * c + dr1 * s;
                dx[xOff + 2 * p + 1] = -dr0 * s + dr1 * c;
            }
        }
    }
}

__global__ __launch_bounds__(128) void smx_cuda_ropequad_fwd(
    const float* __restrict__ x,
    const float* __restrict__ cosBuf,
    const float* __restrict__ sinBuf,
    float* __restrict__ y,
    int totalHeads,
    int seqLen,
    int numHeads,
    int mb,
    int outDim,
    int startPos)
{
    __shared__ float s_r[4][64];
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    int half = mb >> 1;
    int mb2 = mb * mb;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;

    for (int r = globalWarp; r < totalHeads; r += totalWarps) {
        int bt = r / numHeads;
        int t = bt % seqLen;
        long long cOff = (long long)(startPos + t) * half;
        long long xOff = (long long)r * mb;
        long long yOff = (long long)r * outDim;
        for (int p = lane; p < half; p += 32) {
            float x0 = x[xOff + 2 * p];
            float x1 = x[xOff + 2 * p + 1];
            float c = cosBuf[cOff + p];
            float s = sinBuf[cOff + p];
            s_r[warpId][2 * p] = x0 * c - x1 * s;
            s_r[warpId][2 * p + 1] = x0 * s + x1 * c;
        }
        __syncwarp();
        for (int idx = lane; idx < outDim; idx += 32) {
            if (idx < mb2) {
                int u = idx / mb;
                int v = idx - u * mb;
                y[yOff + idx] = s_r[warpId][u] * s_r[warpId][v];
            } else {
                y[yOff + idx] = 0.0f;
            }
        }
        __syncwarp();
    }
}

__global__ __launch_bounds__(128) void smx_cuda_ropequad_bwd(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ cosBuf,
    const float* __restrict__ sinBuf,
    float* __restrict__ dx,
    int totalHeads,
    int seqLen,
    int numHeads,
    int mb,
    int outDim,
    int startPos)
{
    __shared__ float s_r[4][64];
    __shared__ float s_dr[4][64];
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    int half = mb >> 1;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;

    for (int r = globalWarp; r < totalHeads; r += totalWarps) {
        int bt = r / numHeads;
        int t = bt % seqLen;
        long long cOff = (long long)(startPos + t) * half;
        long long xOff = (long long)r * mb;
        long long yOff = (long long)r * outDim;
        for (int p = lane; p < half; p += 32) {
            float x0 = x[xOff + 2 * p];
            float x1 = x[xOff + 2 * p + 1];
            float c = cosBuf[cOff + p];
            float s = sinBuf[cOff + p];
            s_r[warpId][2 * p] = x0 * c - x1 * s;
            s_r[warpId][2 * p + 1] = x0 * s + x1 * c;
        }
        __syncwarp();
        for (int u = lane; u < mb; u += 32) {
            float acc = 0.0f;
            for (int v = 0; v < mb; v++) {
                acc += (dy[yOff + u * mb + v] + dy[yOff + v * mb + u]) * s_r[warpId][v];
            }
            s_dr[warpId][u] = acc;
        }
        __syncwarp();
        for (int p = lane; p < half; p += 32) {
            float dr0 = s_dr[warpId][2 * p];
            float dr1 = s_dr[warpId][2 * p + 1];
            float c = cosBuf[cOff + p];
            float s = sinBuf[cOff + p];
            dx[xOff + 2 * p] = dr0 * c + dr1 * s;
            dx[xOff + 2 * p + 1] = -dr0 * s + dr1 * c;
        }
        __syncwarp();
    }
}

__global__ __launch_bounds__(128) void smx_cuda_projres_fwd(
    const float* __restrict__ h,
    const float* __restrict__ u,
    const float* __restrict__ gateLogit,
    float* __restrict__ y,
    int rows,
    int dim,
    float eps)
{
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    int nVec = (dim + 31) >> 5;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;

    float g = 1.0f / (1.0f + expf(-gateLogit[0]));
    float beta = 1.0f - g;

    for (int m = globalWarp; m < rows; m += totalWarps) {
        long long off = (long long)m * dim;
        float rh[32];
        float ru[32];
        float localDot = 0.0f;
        float localNormSq = 0.0f;
        for (int v = 0; v < nVec; v++) {
            int j = (v << 5) + lane;
            if (j < dim) {
                float hj = h[off + j];
                float uj = u[off + j];
                rh[v] = hj;
                ru[v] = uj;
                localDot += uj * hj;
                localNormSq += hj * hj;
            }
        }
        float dot = smx_warp_sum(localDot);
        float normSq = smx_warp_sum(localNormSq);
        float scaleH = 1.0f - beta * (dot / (normSq + eps));
        for (int v = 0; v < nVec; v++) {
            int j = (v << 5) + lane;
            if (j < dim) {
                y[off + j] = ru[v] + scaleH * rh[v];
            }
        }
    }
}

__global__ __launch_bounds__(128) void smx_cuda_projres_bwd_rows(
    const float* __restrict__ dy,
    const float* __restrict__ h,
    const float* __restrict__ u,
    const float* __restrict__ gateLogit,
    float* __restrict__ dh,
    float* __restrict__ du,
    float* __restrict__ blockPartials,
    int rows,
    int dim,
    float eps)
{
    __shared__ float s_warp[4];
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    int nVec = (dim + 31) >> 5;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;

    float g = 1.0f / (1.0f + expf(-gateLogit[0]));
    float beta = 1.0f - g;
    float warpGateSum = 0.0f;

    for (int m = globalWarp; m < rows; m += totalWarps) {
        long long off = (long long)m * dim;
        float rh[32];
        float ru[32];
        float rdy[32];
        float localDot = 0.0f;
        float localNormSq = 0.0f;
        float localPm = 0.0f;
        for (int v = 0; v < nVec; v++) {
            int j = (v << 5) + lane;
            if (j < dim) {
                float hj = h[off + j];
                float uj = u[off + j];
                float dyj = dy[off + j];
                rh[v] = hj;
                ru[v] = uj;
                rdy[v] = dyj;
                localDot += uj * hj;
                localNormSq += hj * hj;
                localPm += dyj * hj;
            }
        }
        float dot = smx_warp_sum(localDot);
        float denom = smx_warp_sum(localNormSq) + eps;
        float pm = smx_warp_sum(localPm);
        float coeff = dot / denom;
        float scaleH = 1.0f - beta * coeff;
        float kFactor = (beta * pm) / denom;
        if (lane == 0) warpGateSum += coeff * pm;
        for (int v = 0; v < nVec; v++) {
            int j = (v << 5) + lane;
            if (j < dim) {
                float hj = rh[v];
                float uj = ru[v];
                float dyj = rdy[v];
                du[off + j] = dyj - kFactor * hj;
                dh[off + j] = scaleH * dyj - kFactor * (uj - 2.0f * coeff * hj);
            }
        }
    }

    if (lane == 0) s_warp[warpId] = warpGateSum;
    __syncthreads();
    if (tid == 0) {
        blockPartials[blockIdx.x] = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    }
}

__global__ __launch_bounds__(256) void smx_cuda_projres_bwd_dgate(
    const float* __restrict__ blockPartials,
    const float* __restrict__ gateLogit,
    float* __restrict__ dGateLogit,
    int numBlocks)
{
    __shared__ float s_warp[8];
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    float acc = 0.0f;
    for (int b = tid; b < numBlocks; b += 256) {
        acc += blockPartials[b];
    }
    acc = smx_warp_sum(acc);
    if (lane == 0) s_warp[warpId] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3] + s_warp[4] + s_warp[5] + s_warp[6] + s_warp[7];
        float g = 1.0f / (1.0f + expf(-gateLogit[0]));
        dGateLogit[0] = g * (1.0f - g) * sum;
    }
}

__global__ __launch_bounds__(256) void smx_cuda_speccirc_fwd(
    const float* __restrict__ diagScale,
    const float* __restrict__ specRe,
    const float* __restrict__ specIm,
    float* __restrict__ outMat,
    int batch,
    int dim)
{
    __shared__ float s_h[1024];
    int b = (int)blockIdx.x;
    if (b >= batch) return;
    int tid = (int)threadIdx.x;
    int half = dim >> 1;
    int nf = half + 1;
    int specOff = b * nf;
    float dimF = (float)dim;
    float invD = 1.0f / dimF;
    float twoPiOverD = 6.2831853071795864769f * invD;

    for (int m = tid; m < dim; m += 256) {
        float signNyq = (m & 1) == 0 ? 1.0f : -1.0f;
        float sum = specRe[specOff] + signNyq * specRe[specOff + half];
        for (int f = 1; f < half; f++) {
            int fm = (f * m) % dim;
            float sn, cs;
            __sincosf(twoPiOverD * (float)fm, &sn, &cs);
            sum += 2.0f * (specRe[specOff + f] * cs - specIm[specOff + f] * sn);
        }
        s_h[m] = sum * invD;
    }
    __syncthreads();

    int dd = dim * dim;
    int matBase = b * dd;
    int diagBase = b * dim;
    for (int idx = tid; idx < dd; idx += 256) {
        int r = idx / dim;
        int c = idx - r * dim;
        int shift = c - r;
        if (shift < 0) shift += dim;
        outMat[matBase + idx] = diagScale[diagBase + r] * s_h[shift];
    }
}

__global__ __launch_bounds__(256) void smx_cuda_speccirc_bwd(
    const float* __restrict__ dOut,
    const float* __restrict__ diagScale,
    const float* __restrict__ specRe,
    const float* __restrict__ specIm,
    float* __restrict__ dDiag,
    float* __restrict__ dRe,
    float* __restrict__ dIm,
    int batch,
    int dim)
{
    __shared__ float s_h[1024];
    __shared__ float s_dh[1024];
    int b = (int)blockIdx.x;
    if (b >= batch) return;
    int tid = (int)threadIdx.x;
    int half = dim >> 1;
    int nf = half + 1;
    int specOff = b * nf;
    int matBase = b * dim * dim;
    int vecBase = b * dim;
    float dimF = (float)dim;
    float invD = 1.0f / dimF;
    float twoPiOverD = 6.2831853071795864769f * invD;

    for (int m = tid; m < dim; m += 256) {
        float signNyq = (m & 1) == 0 ? 1.0f : -1.0f;
        float sum = specRe[specOff] + signNyq * specRe[specOff + half];
        for (int f = 1; f < half; f++) {
            int fm = (f * m) % dim;
            float sn, cs;
            __sincosf(twoPiOverD * (float)fm, &sn, &cs);
            sum += 2.0f * (specRe[specOff + f] * cs - specIm[specOff + f] * sn);
        }
        s_h[m] = sum * invD;
    }
    __syncthreads();

    for (int idx = tid; idx < dim; idx += 256) {
        float accDiag = 0.0f;
        float accH = 0.0f;
        for (int k = 0; k < dim; k++) {
            int shift = k - idx;
            if (shift < 0) shift += dim;
            accDiag += dOut[matBase + idx * dim + k] * s_h[shift];
            int col = k + idx;
            if (col >= dim) col -= dim;
            accH += dOut[matBase + k * dim + col] * diagScale[vecBase + k];
        }
        dDiag[vecBase + idx] = accDiag;
        s_dh[idx] = accH;
    }
    __syncthreads();

    for (int f = tid; f < nf; f += 256) {
        if (f == 0) {
            float s0 = 0.0f;
            for (int m = 0; m < dim; m++) s0 += s_dh[m];
            dRe[specOff] = s0 * invD;
            dIm[specOff] = 0.0f;
        } else if (f == half) {
            float sn = 0.0f;
            for (int m = 0; m < dim; m++) sn += ((m & 1) == 0 ? 1.0f : -1.0f) * s_dh[m];
            dRe[specOff + half] = sn * invD;
            dIm[specOff + half] = 0.0f;
        } else {
            float accRe = 0.0f;
            float accIm = 0.0f;
            for (int m = 0; m < dim; m++) {
                int fm = (f * m) % dim;
                float sn, cs;
                __sincosf(twoPiOverD * (float)fm, &sn, &cs);
                float dhv = s_dh[m];
                accRe += dhv * cs;
                accIm -= dhv * sn;
            }
            float sc = 2.0f * invD;
            dRe[specOff + f] = accRe * sc;
            dIm[specOff + f] = accIm * sc;
        }
    }
}

__global__ __launch_bounds__(128) void smx_cuda_sigtopk_mix_fwd(
    const float* __restrict__ logits,
    const float* __restrict__ expertTable,
    const float* __restrict__ routingBias,
    float* __restrict__ y,
    float* __restrict__ savedInfo,
    float* __restrict__ loadCounts,
    int rows,
    int numExperts,
    int topK,
    int channels,
    float eps)
{
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;
    int nSlots = (numExperts + 31) >> 5;
    int strideInfo = 2 * topK + 1;

    for (int m = globalWarp; m < rows; m += totalWarps) {
        long long logOff = (long long)m * numExperts;
        long long infoOff = (long long)m * strideInfo;
        float regS[8];
        float regB[8];
        for (int v = 0; v < nSlots; v++) {
            int j = (v << 5) + lane;
            if (j < numExperts) {
                float sj = 1.0f / (1.0f + expf(-logits[logOff + j]));
                regS[v] = sj;
                regB[v] = sj + routingBias[j];
            } else {
                regS[v] = 0.0f;
                regB[v] = -3.4e38f;
            }
        }

        int chosenIdx[16];
        float chosenSig[16];
        float sumSel = 0.0f;
        for (int r = 0; r < topK; r++) {
            float localVal = -3.4e38f;
            int localIdx = 2147483647;
            float localSig = 0.0f;
            for (int v = 0; v < nSlots; v++) {
                int j = (v << 5) + lane;
                if (j < numExperts) {
                    float bv = regB[v];
                    if (localIdx == 2147483647 || bv > localVal || (bv == localVal && j < localIdx)) {
                        localVal = bv;
                        localIdx = j;
                        localSig = regS[v];
                    }
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                float oVal = __shfl_xor_sync(0xffffffffu, localVal, offset);
                int oIdx = __shfl_xor_sync(0xffffffffu, localIdx, offset);
                float oSig = __shfl_xor_sync(0xffffffffu, localSig, offset);
                if (oIdx < 2147483647 && (localIdx == 2147483647 || oVal > localVal || (oVal == localVal && oIdx < localIdx))) {
                    localVal = oVal;
                    localIdx = oIdx;
                    localSig = oSig;
                }
            }
            chosenIdx[r] = localIdx;
            chosenSig[r] = localSig;
            sumSel += localSig;
            int winLane = localIdx & 31;
            int winSlot = localIdx >> 5;
            if (lane == winLane) {
                regB[winSlot] = -3.4e38f;
            }
        }

        float denom = sumSel >= eps ? sumSel : eps;
        float invDenom = 1.0f / denom;
        if (lane < topK) {
            int eIdx = chosenIdx[lane];
            savedInfo[infoOff + lane] = (float)eIdx;
            savedInfo[infoOff + topK + lane] = chosenSig[lane] * invDenom;
            if (loadCounts != 0) {
                atomicAdd(&loadCounts[eIdx], 1.0f);
            }
        }
        if (lane == 0) {
            savedInfo[infoOff + 2 * topK] = sumSel;
        }

        long long yOff = (long long)m * channels;
        for (int c = lane; c < channels; c += 32) {
            float acc = 0.0f;
            for (int r = 0; r < topK; r++) {
                acc += (chosenSig[r] * invDenom) * expertTable[(long long)chosenIdx[r] * channels + c];
            }
            y[yOff + c] = acc;
        }
    }
}

__global__ __launch_bounds__(256) void smx_cuda_sigtopk_bias_upd(
    const float* __restrict__ loadCounts,
    float* __restrict__ routingBias,
    int rows,
    int numExperts,
    int topK,
    float rate)
{
    int e = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (e >= numExperts) return;
    float mean = (float)((long long)rows * topK) / (float)numExperts;
    float diff = mean - loadCounts[e];
    float sgn = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
    routingBias[e] += rate * sgn;
}

__global__ __launch_bounds__(128) void smx_cuda_sigtopk_mix_bwd_dlog(
    const float* __restrict__ dy,
    const float* __restrict__ logits,
    const float* __restrict__ expertTable,
    const float* __restrict__ savedInfo,
    float* __restrict__ dLogits,
    int rows,
    int numExperts,
    int topK,
    int channels,
    float eps)
{
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;
    int strideInfo = 2 * topK + 1;

    for (int m = globalWarp; m < rows; m += totalWarps) {
        long long logOff = (long long)m * numExperts;
        for (int j = lane; j < numExperts; j += 32) {
            dLogits[logOff + j] = 0.0f;
        }
        long long infoOff = (long long)m * strideInfo;
        long long yOff = (long long)m * channels;
        float sumSel = savedInfo[infoOff + 2 * topK];

        int idxArr[16];
        float dGateArr[16];
        float dotDg = 0.0f;
        for (int r = 0; r < topK; r++) {
            int eIdx = (int)savedInfo[infoOff + r];
            float gateR = savedInfo[infoOff + topK + r];
            float part = 0.0f;
            for (int c = lane; c < channels; c += 32) {
                part += dy[yOff + c] * expertTable[(long long)eIdx * channels + c];
            }
            float dGate = smx_warp_sum(part);
            idxArr[r] = eIdx;
            dGateArr[r] = dGate;
            dotDg += dGate * gateR;
        }
        if (lane < topK) {
            int eIdx = idxArr[lane];
            float dGate = dGateArr[lane];
            float dSel = sumSel >= eps ? ((dGate - dotDg) / sumSel) : (dGate / eps);
            float sj = 1.0f / (1.0f + expf(-logits[logOff + eIdx]));
            dLogits[logOff + eIdx] = dSel * sj * (1.0f - sj);
        }
    }
}

__global__ __launch_bounds__(128) void smx_cuda_sigtopk_mix_bwd_dtbl(
    const float* __restrict__ dy,
    const float* __restrict__ savedInfo,
    float* __restrict__ dTable,
    int rows,
    int numExperts,
    int topK,
    int channels)
{
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;
    int totalCells = numExperts * channels;
    int strideInfo = 2 * topK + 1;

    for (int cell = globalWarp; cell < totalCells; cell += totalWarps) {
        int e = cell / channels;
        int c = cell - e * channels;
        float acc = 0.0f;
        for (int m = lane; m < rows; m += 32) {
            long long infoOff = (long long)m * strideInfo;
            float dyc = dy[(long long)m * channels + c];
            for (int r = 0; r < topK; r++) {
                if ((int)savedInfo[infoOff + r] == e) {
                    acc += savedInfo[infoOff + topK + r] * dyc;
                }
            }
        }
        float sum = smx_warp_sum(acc);
        if (lane == 0) dTable[cell] = sum;
    }
}

__global__ __launch_bounds__(256) void smx_cuda_dualswiglu_fwd(
    const float* __restrict__ guProj,
    const float* __restrict__ guGain,
    const float* __restrict__ upBase,
    float* __restrict__ y,
    int total,
    int hiddenDim)
{
    int stride = (int)gridDim.x * 256;
    for (int gid = (int)blockIdx.x * 256 + (int)threadIdx.x; gid < total; gid += stride) {
        int m = gid / hiddenDim;
        int j = gid - m * hiddenDim;
        long long off2 = (long long)m * (2 * hiddenDim);
        float gRaw = guProj[off2 + j];
        float uRaw = guProj[off2 + hiddenDim + j];
        float gGain = 1.0f + guGain[off2 + j];
        float uGain = upBase[j] + guGain[off2 + hiddenDim + j];
        float g = gRaw * gGain;
        float sig = 1.0f / (1.0f + expf(-g));
        y[gid] = (g * sig * uRaw) * uGain;
    }
}

__global__ __launch_bounds__(256) void smx_cuda_dualswiglu_bwd_elem(
    const float* __restrict__ dy,
    const float* __restrict__ guProj,
    const float* __restrict__ guGain,
    const float* __restrict__ upBase,
    float* __restrict__ dGuProj,
    float* __restrict__ dGuGain,
    int total,
    int hiddenDim)
{
    int stride = (int)gridDim.x * 256;
    for (int gid = (int)blockIdx.x * 256 + (int)threadIdx.x; gid < total; gid += stride) {
        int m = gid / hiddenDim;
        int j = gid - m * hiddenDim;
        long long off2 = (long long)m * (2 * hiddenDim);
        float dyv = dy[gid];
        float gRaw = guProj[off2 + j];
        float uRaw = guProj[off2 + hiddenDim + j];
        float gGain = 1.0f + guGain[off2 + j];
        float uGain = upBase[j] + guGain[off2 + hiddenDim + j];
        float g = gRaw * gGain;
        float sig = 1.0f / (1.0f + expf(-g));
        float silu = g * sig;
        float h = silu * uRaw;
        dGuGain[off2 + hiddenDim + j] = dyv * h;
        float dh = dyv * uGain;
        dGuProj[off2 + hiddenDim + j] = dh * silu;
        float dg = (dh * uRaw) * sig * (1.0f + g * (1.0f - sig));
        dGuProj[off2 + j] = dg * gGain;
        dGuGain[off2 + j] = dg * gRaw;
    }
}

__global__ __launch_bounds__(128) void smx_cuda_dualswiglu_bwd_dbase(
    const float* __restrict__ dy,
    const float* __restrict__ guProj,
    const float* __restrict__ guGain,
    float* __restrict__ dUpBase,
    int rows,
    int hiddenDim)
{
    __shared__ float s_warp[4];
    int j = (int)blockIdx.x;
    if (j >= hiddenDim) return;
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    long long stride2 = 2LL * hiddenDim;
    float acc = 0.0f;
    for (int m = tid; m < rows; m += 128) {
        long long off2 = (long long)m * stride2;
        float g = guProj[off2 + j] * (1.0f + guGain[off2 + j]);
        float sig = 1.0f / (1.0f + expf(-g));
        acc += dy[(long long)m * hiddenDim + j] * (g * sig * guProj[off2 + hiddenDim + j]);
    }
    acc = smx_warp_sum(acc);
    if (lane == 0) s_warp[warpId] = acc;
    __syncthreads();
    if (tid == 0) {
        dUpBase[j] = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    }
}

__global__ __launch_bounds__(256) void smx_cuda_bornratio_fwd(
    const float* __restrict__ scanOut,
    const float* __restrict__ sinkLogAlpha,
    float* __restrict__ y,
    int totalOut,
    int numHeads,
    int headDim,
    int scanDim,
    int hasSink,
    float eps)
{
    int stride = (int)gridDim.x * 256;
    for (int gid = (int)blockIdx.x * 256 + (int)threadIdx.x; gid < totalOut; gid += stride) {
        int row = gid / headDim;
        int j = gid - row * headDim;
        int h = row % numHeads;
        float sinkTerm = hasSink != 0 ? expf(sinkLogAlpha[h]) : 0.0f;
        long long sOff = (long long)row * scanDim;
        float den = scanOut[sOff + headDim] + eps + sinkTerm;
        y[gid] = scanOut[sOff + j] / den;
    }
}

__global__ __launch_bounds__(128) void smx_cuda_bornratio_bwd_dscan(
    const float* __restrict__ dy,
    const float* __restrict__ scanOut,
    const float* __restrict__ sinkLogAlpha,
    float* __restrict__ dScan,
    int totalRows,
    int numHeads,
    int headDim,
    int scanDim,
    int hasSink,
    float eps)
{
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;

    for (int row = globalWarp; row < totalRows; row += totalWarps) {
        int h = row % numHeads;
        float sinkTerm = hasSink != 0 ? expf(sinkLogAlpha[h]) : 0.0f;
        long long sOff = (long long)row * scanDim;
        long long yOff = (long long)row * headDim;
        float invDen = 1.0f / (scanOut[sOff + headDim] + eps + sinkTerm);
        float localDot = 0.0f;
        for (int j = lane; j < headDim; j += 32) {
            float dyj = dy[yOff + j];
            localDot += dyj * scanOut[sOff + j];
            dScan[sOff + j] = dyj * invDen;
        }
        float dot = smx_warp_sum(localDot);
        if (lane == 0) {
            dScan[sOff + headDim] = -dot * invDen * invDen;
        }
        for (int j = headDim + 1 + lane; j < scanDim; j += 32) {
            dScan[sOff + j] = 0.0f;
        }
    }
}

__global__ __launch_bounds__(128) void smx_cuda_bornratio_bwd_dsink(
    const float* __restrict__ dy,
    const float* __restrict__ scanOut,
    const float* __restrict__ sinkLogAlpha,
    float* __restrict__ dSink,
    int totalRows,
    int numHeads,
    int headDim,
    int scanDim,
    float eps)
{
    __shared__ float s_warp[4];
    int h = (int)blockIdx.x;
    if (h >= numHeads) return;
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    float sinkTerm = expf(sinkLogAlpha[h]);
    int nPositions = totalRows / numHeads;
    float acc = 0.0f;
    for (int pos = tid; pos < nPositions; pos += 128) {
        int row = pos * numHeads + h;
        long long sOff = (long long)row * scanDim;
        long long yOff = (long long)row * headDim;
        float invDen = 1.0f / (scanOut[sOff + headDim] + eps + sinkTerm);
        float dot = 0.0f;
        for (int j = 0; j < headDim; j++) {
            dot += dy[yOff + j] * scanOut[sOff + j];
        }
        acc += -dot * invDen * invDen;
    }
    acc = smx_warp_sum(acc);
    if (lane == 0) s_warp[warpId] = acc;
    __syncthreads();
    if (tid == 0) {
        dSink[h] = (s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3]) * sinkTerm;
    }
}

__global__ __launch_bounds__(256) void smx_cuda_boundsig_fwd(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int total,
    int cols,
    float flr)
{
    int stride = (int)gridDim.x * 256;
    float scale = 1.0f - flr;
    for (int gid = (int)blockIdx.x * 256 + (int)threadIdx.x; gid < total; gid += stride) {
        int c = gid % cols;
        float sig = 1.0f / (1.0f + expf(-(logits[gid] + bias[c])));
        y[gid] = flr + scale * sig;
    }
}

__global__ __launch_bounds__(256) void smx_cuda_boundsig_bwd_dlog(
    const float* __restrict__ dy,
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    float* __restrict__ dLogits,
    int total,
    int cols,
    float flr)
{
    int stride = (int)gridDim.x * 256;
    float scale = 1.0f - flr;
    for (int gid = (int)blockIdx.x * 256 + (int)threadIdx.x; gid < total; gid += stride) {
        int c = gid % cols;
        float sig = 1.0f / (1.0f + expf(-(logits[gid] + bias[c])));
        dLogits[gid] = dy[gid] * scale * sig * (1.0f - sig);
    }
}

__global__ __launch_bounds__(128) void smx_cuda_boundsig_bwd_dbias(
    const float* __restrict__ dy,
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    float* __restrict__ dBias,
    int rows,
    int cols,
    float flr)
{
    __shared__ float s_warp[4];
    int c = (int)blockIdx.x;
    if (c >= cols) return;
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    float scale = 1.0f - flr;
    float bc = bias[c];
    float acc = 0.0f;
    for (int r = tid; r < rows; r += 128) {
        long long idx = (long long)r * cols + c;
        float sig = 1.0f / (1.0f + expf(-(logits[idx] + bc)));
        acc += dy[idx] * scale * sig * (1.0f - sig);
    }
    acc = smx_warp_sum(acc);
    if (lane == 0) s_warp[warpId] = acc;
    __syncthreads();
    if (tid == 0) {
        dBias[c] = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    }
}

} // extern "C"
