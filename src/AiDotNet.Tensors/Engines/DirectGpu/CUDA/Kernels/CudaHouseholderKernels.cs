// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Single-launch shared-SRAM warp-shuffle forward and 2-launch zero-atomic deterministic backward CUDA kernels
/// for chained Householder reflections <c>y = x H_0 H_1 ... H_{K-1}</c> (and reverse order when <c>trans != 0</c>).
/// </summary>
internal static class CudaHouseholderKernels
{
    public const int MaxReflectors = 16;
    public const int MaxDim = 1024;
    public const int MaxForwardReflectorElements = 4096;
    public const int MaxBackwardReflectorElements = 2048;
    public const int BlockThreads = 128;
    public const int WarpsPerBlock = BlockThreads / 32;
    public const int ReduceBlockThreads = 256;

    public static string[] GetKernelNames() => new[]
    {
        "householder_chain_fwd",
        "householder_chain_bwd_rows",
        "householder_chain_bwd_reduce",
    };

    public static string GetSource() => @"
extern ""C"" {

__device__ __forceinline__ float hh_warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}

__device__ __forceinline__ float hh_warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// Shared memory: [K * D] floats for s_uHat + [4] floats for s_warp.
__global__ __launch_bounds__(128) void householder_chain_fwd(
    const float* __restrict__ x,
    const float* __restrict__ u,
    float* __restrict__ y,
    int rows,
    int K,
    int D,
    int trans,
    float sqrtFloor)
{
    extern __shared__ float smem[];
    float* s_uHat = smem;
    float* s_warp = smem + K * D;

    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;

    for (int j = 0; j < K; j++) {
        const float* uj = u + j * D;
        float localMax = 0.0f;
        for (int i = tid; i < D; i += 128) {
            localMax = fmaxf(localMax, fabsf(uj[i]));
        }
        localMax = hh_warp_max(localMax);
        if (lane == 0) s_warp[warpId] = localMax;
        __syncthreads();
        float maxAbs = fmaxf(fmaxf(s_warp[0], s_warp[1]), fmaxf(s_warp[2], s_warp[3]));
        __syncthreads();
        float rowScale = fmaxf(maxAbs, sqrtFloor * 0.0625f);

        float localSq = 0.0f;
        for (int i = tid; i < D; i += 128) {
            float sc = uj[i] / rowScale;
            localSq += sc * sc;
        }
        localSq = hh_warp_sum(localSq);
        if (lane == 0) s_warp[warpId] = localSq;
        __syncthreads();
        float scaledSq = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
        float sfr = sqrtFloor / rowScale;
        float denom = fmaxf(scaledSq, sfr * sfr);
        float r = 1.0f / sqrtf(denom);
        float* uHatJ = s_uHat + j * D;
        for (int i = tid; i < D; i += 128) {
            uHatJ[i] = (uj[i] / rowScale) * r;
        }
        __syncthreads();
    }

    int nVec = (D + 31) >> 5;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;

    for (int m = globalWarp; m < rows; m += totalWarps) {
        const float* xm = x + (long long)m * D;
        float* ym = y + (long long)m * D;
        float rx[32];
        for (int v = 0; v < nVec; v++) {
            int col = (v << 5) + lane;
            rx[v] = col < D ? xm[col] : 0.0f;
        }
        for (int s = 0; s < K; s++) {
            int j = trans != 0 ? (K - 1 - s) : s;
            const float* uHatJ = s_uHat + j * D;
            float dot = 0.0f;
            for (int v = 0; v < nVec; v++) {
                int col = (v << 5) + lane;
                if (col < D) dot += rx[v] * uHatJ[col];
            }
            dot = hh_warp_sum(dot);
            for (int v = 0; v < nVec; v++) {
                int col = (v << 5) + lane;
                if (col < D) {
                    float step = dot * uHatJ[col];
                    rx[v] = (rx[v] - step) - step;
                }
            }
        }
        for (int v = 0; v < nVec; v++) {
            int col = (v << 5) + lane;
            if (col < D) ym[col] = rx[v];
        }
    }
}

// Shared memory: [K * D] s_uHat + [4 * K * D] s_duWarp + [16] s_invNorm + [16] s_active + [4] s_warp.
__global__ __launch_bounds__(128) void householder_chain_bwd_rows(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ u,
    float* __restrict__ dx,
    float* __restrict__ dUPartials,
    float* __restrict__ uHatOut,
    float* __restrict__ metaOut,
    int rows,
    int K,
    int D,
    int trans,
    float sqrtFloor)
{
    extern __shared__ float smem[];
    int kd = K * D;
    float* s_uHat = smem;
    float* s_duWarp = s_uHat + kd;
    float* s_invNorm = s_duWarp + 4 * kd;
    float* s_active = s_invNorm + 16;
    float* s_warp = s_active + 16;

    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;

    for (int idx = tid; idx < 4 * kd; idx += 128) {
        s_duWarp[idx] = 0.0f;
    }

    for (int j = 0; j < K; j++) {
        const float* uj = u + j * D;
        float localMax = 0.0f;
        for (int i = tid; i < D; i += 128) {
            localMax = fmaxf(localMax, fabsf(uj[i]));
        }
        localMax = hh_warp_max(localMax);
        if (lane == 0) s_warp[warpId] = localMax;
        __syncthreads();
        float maxAbs = fmaxf(fmaxf(s_warp[0], s_warp[1]), fmaxf(s_warp[2], s_warp[3]));
        __syncthreads();
        float rowScale = fmaxf(maxAbs, sqrtFloor * 0.0625f);

        float localSq = 0.0f;
        for (int i = tid; i < D; i += 128) {
            float sc = uj[i] / rowScale;
            localSq += sc * sc;
        }
        localSq = hh_warp_sum(localSq);
        if (lane == 0) s_warp[warpId] = localSq;
        __syncthreads();
        float scaledSq = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
        float sfr = sqrtFloor / rowScale;
        float scaledFloor = sfr * sfr;
        float denom = fmaxf(scaledSq, scaledFloor);
        float r = 1.0f / sqrtf(denom);
        if (tid == 0) {
            s_invNorm[j] = r / rowScale;
            s_active[j] = scaledSq > scaledFloor ? 1.0f : (scaledSq == scaledFloor ? 0.5f : 0.0f);
        }
        float* uHatJ = s_uHat + j * D;
        for (int i = tid; i < D; i += 128) {
            uHatJ[i] = (uj[i] / rowScale) * r;
        }
        __syncthreads();
    }

    int nVec = (D + 31) >> 5;
    int globalWarp = (int)blockIdx.x * 4 + warpId;
    int totalWarps = (int)gridDim.x * 4;
    float* myDu = s_duWarp + warpId * kd;

    for (int m = globalWarp; m < rows; m += totalWarps) {
        const float* xm = x + (long long)m * D;
        const float* dym = dy + (long long)m * D;
        float* dxm = dx + (long long)m * D;

        float rx[32];
        float rg[32];
        float p[16];
        for (int v = 0; v < nVec; v++) {
            int col = (v << 5) + lane;
            rx[v] = col < D ? xm[col] : 0.0f;
            rg[v] = col < D ? dym[col] : 0.0f;
        }

        for (int s = 0; s < K; s++) {
            int j = trans != 0 ? (K - 1 - s) : s;
            const float* uHatJ = s_uHat + j * D;
            float dot = 0.0f;
            for (int v = 0; v < nVec; v++) {
                int col = (v << 5) + lane;
                if (col < D) dot += rx[v] * uHatJ[col];
            }
            dot = hh_warp_sum(dot);
            p[s] = dot;
            for (int v = 0; v < nVec; v++) {
                int col = (v << 5) + lane;
                if (col < D) {
                    float step = dot * uHatJ[col];
                    rx[v] = (rx[v] - step) - step;
                }
            }
        }

        for (int s = K - 1; s >= 0; s--) {
            int j = trans != 0 ? (K - 1 - s) : s;
            const float* uHatJ = s_uHat + j * D;
            float* duJ = myDu + j * D;
            float ps = p[s];

            float q = 0.0f;
            for (int v = 0; v < nVec; v++) {
                int col = (v << 5) + lane;
                if (col < D) q += rg[v] * uHatJ[col];
            }
            q = hh_warp_sum(q);

            for (int v = 0; v < nVec; v++) {
                int col = (v << 5) + lane;
                if (col < D) {
                    float xStage = xm[col];
                    for (int t = 0; t < s; t++) {
                        int jt = trans != 0 ? (K - 1 - t) : t;
                        float stepT = p[t] * s_uHat[jt * D + col];
                        xStage = (xStage - stepT) - stepT;
                    }
                    float duTerm = q * xStage + ps * rg[v];
                    duJ[col] = (duJ[col] - duTerm) - duTerm;
                }
            }

            for (int v = 0; v < nVec; v++) {
                int col = (v << 5) + lane;
                if (col < D) {
                    float stepQ = q * uHatJ[col];
                    rg[v] = (rg[v] - stepQ) - stepQ;
                }
            }
        }

        for (int v = 0; v < nVec; v++) {
            int col = (v << 5) + lane;
            if (col < D) dxm[col] = rg[v];
        }
    }

    __syncthreads();

    float* blockOut = dUPartials + (long long)blockIdx.x * kd;
    for (int idx = tid; idx < kd; idx += 128) {
        blockOut[idx] = s_duWarp[idx] + s_duWarp[kd + idx] + s_duWarp[2 * kd + idx] + s_duWarp[3 * kd + idx];
    }
    if (blockIdx.x == 0) {
        for (int idx = tid; idx < kd; idx += 128) {
            uHatOut[idx] = s_uHat[idx];
        }
        if (tid < K) {
            metaOut[2 * tid] = s_invNorm[tid];
            metaOut[2 * tid + 1] = s_active[tid];
        }
    }
}

__global__ __launch_bounds__(256) void householder_chain_bwd_reduce(
    const float* __restrict__ dUPartials,
    const float* __restrict__ uHat,
    const float* __restrict__ meta,
    float* __restrict__ dU,
    int numBlocks,
    int K,
    int D)
{
    __shared__ float s_red[8];
    int j = (int)blockIdx.x;
    if (j >= K) return;
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    int kd = K * D;
    int rowOff = j * D;

    float invNorm = meta[2 * j];
    float active = meta[2 * j + 1];

    float localDot = 0.0f;
    for (int i = tid; i < D; i += 256) {
        float sum = 0.0f;
        for (int b = 0; b < numBlocks; b++) {
            sum += dUPartials[(long long)b * kd + rowOff + i];
        }
        dU[rowOff + i] = sum;
        localDot += sum * uHat[rowOff + i];
    }
    localDot = hh_warp_sum(localDot);
    if (lane == 0) s_red[warpId] = localDot;
    __syncthreads();

    float totalDot = 0.0f;
    if (warpId == 0) {
        float v = lane < 8 ? s_red[lane] : 0.0f;
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    totalDot = s_red[0];

    for (int i = tid; i < D; i += 256) {
        float duVal = dU[rowOff + i];
        dU[rowOff + i] = invNorm * (duVal - active * totalDot * uHat[rowOff + i]);
    }
}

} // extern ""C""
";
}
