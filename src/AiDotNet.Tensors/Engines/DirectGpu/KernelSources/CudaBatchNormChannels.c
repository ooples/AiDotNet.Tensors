// BatchNorm -> LeakyReLU(slope > 0) over the split variant; each entry takes the slope last.
extern "C" __global__ __launch_bounds__(256) void batchnorm_fwd_leaky_split(const float* input, const float* residual,
    float* output, const float* gamma, const float* beta, float* saveMean, float* saveVar, const float* partials,
    int batch, int channels, int spatial, float epsilon, int splits, float slope)
{ bn_fwd_split_apply<true, false, true>(input, residual, output, gamma, beta, saveMean, saveVar, partials, batch,
    channels, spatial, epsilon, splits, slope); }
#define BN_SPLIT_BWD_LEAKY_ENTRY(MODE, SUFFIX) \
extern "C" __global__ __launch_bounds__(256) void batchnorm_bwd##SUFFIX##_split_reduce(const float* gradOutput, \
    const float* maskSource, const float* gradOutput2, const float* input, const float* gamma, const float* saveMean, \
    const float* saveVar, float* partials, int batch, int channels, int spatial, float epsilon, int splits, \
    float slope) \
{ bn_bwd_split_reduce<MODE, true>(gradOutput, maskSource, gradOutput2, input, gamma, saveMean, saveVar, partials, \
    batch, channels, spatial, epsilon, splits, slope); } \
extern "C" __global__ __launch_bounds__(256) void batchnorm_bwd##SUFFIX##_split_apply(const float* gradOutput, \
    const float* maskSource, const float* gradOutput2, float* gradResidual, const float* input, const float* gamma, \
    const float* saveMean, const float* saveVar, float* gradInput, float* gradGamma, float* gradBeta, \
    const float* partials, int batch, int channels, int spatial, float epsilon, int splits, int accumulate, \
    float slope) \
{ bn_bwd_split_apply<MODE, false, true>(gradOutput, maskSource, gradOutput2, gradResidual, input, gamma, saveMean, \
    saveVar, gradInput, gradGamma, gradBeta, partials, batch, channels, spatial, epsilon, splits, accumulate, slope); }
BN_SPLIT_BWD_LEAKY_ENTRY(1, _leaky)
BN_SPLIT_BWD_LEAKY_ENTRY(2, _leakymask)

// ---- Channels-last (NHWC) training BatchNorm: the activation as [rows = N*H*W, C], C contiguous ----
// The split variant's arithmetic over the other layout: per (row chunk, channel) shifted sums (the shift the channel's
// first value), merged in double per channel; bn_affine for the normalized value; the same masked backward. Blocks are
// 32 x 8 threads, each thread four consecutive channels (C a multiple of four), so a warp reads 512 contiguous bytes.
__device__ __forceinline__ void bn_nhwc_range(int rows, int splits, int& begin, int& end)
{
    const int chunk = (rows + splits - 1) / splits;
    begin = blockIdx.y * chunk;
    end = min(rows, begin + chunk);
}

// Sums the block's eight row lanes of a float4 per channel quad into row lane 0.
__device__ __forceinline__ float4 bn_nhwc_lane_sum(float4 v, float4 (*red)[32])
{
    red[threadIdx.y][threadIdx.x] = v;
    __syncthreads();
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (threadIdx.y == 0)
        for (int i = 0; i < 8; i++)
        {
            const float4 r = red[i][threadIdx.x];
            acc.x += r.x; acc.y += r.y; acc.z += r.z; acc.w += r.w;
        }
    __syncthreads();
    return acc;
}

extern "C" __global__ __launch_bounds__(256) void batchnorm_nhwc_stats(const float* __restrict__ input,
    float* __restrict__ partials, int rows, int channels, int splits)
{
    __shared__ float4 red[8][32];
    const int c = (blockIdx.x * 32 + threadIdx.x) * 4;
    const bool live = c < channels;
    int begin, end;
    bn_nhwc_range(rows, splits, begin, end);
    float4 s1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), s2 = s1;
    if (live)
    {
        const float4 shift = *reinterpret_cast<const float4*>(input + c);
        for (int r = begin + threadIdx.y; r < end; r += 8)
        {
            const float4 v = *reinterpret_cast<const float4*>(input + (long long)r * channels + c);
            const float a = v.x - shift.x, bb = v.y - shift.y, cc = v.z - shift.z, d = v.w - shift.w;
            s1.x += a; s1.y += bb; s1.z += cc; s1.w += d;
            s2.x += a * a; s2.y += bb * bb; s2.z += cc * cc; s2.w += d * d;
        }
    }
    s1 = bn_nhwc_lane_sum(s1, red);
    s2 = bn_nhwc_lane_sum(s2, red);
    if (live && threadIdx.y == 0)
    {
        const float count = (float)max(end - begin, 0);
        const float a1[4] = { s1.x, s1.y, s1.z, s1.w }, a2[4] = { s2.x, s2.y, s2.z, s2.w };
        for (int j = 0; j < 4; j++)
        {
            float* p = partials + ((long long)(c + j) * splits + blockIdx.y) * 3;
            p[0] = count; p[1] = a1[j]; p[2] = a2[j];
        }
    }
}

// Sums a double across the block (blockDim.x a multiple of 32, at most 1024) into thread 0.
__device__ __forceinline__ double bn_nhwc_block_sum(double v, double* shared)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    v = threadIdx.x < warps ? shared[threadIdx.x] : 0.0;
    if (warp == 0)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// One block per channel: its splits' partials merged in double, derived as bn_fwd_split_apply derives them.
extern "C" __global__ void batchnorm_nhwc_finalize(const float* __restrict__ input,
    const float* __restrict__ partials, float* __restrict__ saveMean, float* __restrict__ saveVar, int channels,
    int splits)
{
    __shared__ double red[32];
    const int c = blockIdx.x;
    if (c >= channels) return;
    double n = 0.0, s1 = 0.0, s2 = 0.0;
    for (int i = threadIdx.x; i < splits; i += blockDim.x)
    {
        const float* p = partials + ((long long)c * splits + i) * 3;
        n += (double)p[0]; s1 += (double)p[1]; s2 += (double)p[2];
    }
    n = bn_nhwc_block_sum(n, red);
    s1 = bn_nhwc_block_sum(s1, red);
    s2 = bn_nhwc_block_sum(s2, red);
    if (threadIdx.x != 0) return;
    const double m = s1 / n, varD = s2 / n - m * m;
    saveMean[c] = (float)((double)input[c] + m);
    saveVar[c] = (float)(varD > 0.0 ? varD : 0.0);
}

// act: 0 none, 1 ReLU, 2 LeakyReLU of `slope`.
__device__ __forceinline__ float bn_nhwc_act(float z, int act, float slope)
{
    return act == 2 ? bn_act<true>(z, slope) : act == 1 ? bn_act<false>(z, slope) : z;
}

extern "C" __global__ void batchnorm_nhwc_apply(const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ gamma, const float* __restrict__ beta, const float* __restrict__ saveMean,
    const float* __restrict__ saveVar, long long quads, int channels, float epsilon, int act, float slope)
{
    const int cq = channels >> 2;
    for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < quads; q += (long long)gridDim.x * blockDim.x)
    {
        const int c = (int)(q % cq) * 4;
        const float4 v = reinterpret_cast<const float4*>(input)[q];
        const float4 m = *reinterpret_cast<const float4*>(saveMean + c), var = *reinterpret_cast<const float4*>(saveVar + c);
        const float4 g = *reinterpret_cast<const float4*>(gamma + c), bt = *reinterpret_cast<const float4*>(beta + c);
        float4 y;
        y.x = bn_nhwc_act(bn_affine(v.x, m.x, rsqrtf(var.x + epsilon), g.x, bt.x), act, slope);
        y.y = bn_nhwc_act(bn_affine(v.y, m.y, rsqrtf(var.y + epsilon), g.y, bt.y), act, slope);
        y.z = bn_nhwc_act(bn_affine(v.z, m.z, rsqrtf(var.z + epsilon), g.z, bt.z), act, slope);
        y.w = bn_nhwc_act(bn_affine(v.w, m.w, rsqrtf(var.w + epsilon), g.w, bt.w), act, slope);
        reinterpret_cast<float4*>(output)[q] = y;
    }
}

// The masked output gradient of one value. mode 0: none; 1: from the activation output in maskSource; 2: from the
// rebuilt pre-activation bn_affine(x) (maskSource is beta, by channel). act 2 LeakyReLU, else ReLU.
__device__ __forceinline__ float bn_nhwc_dy(float dy, float dy2, bool hasDy2, float o, float x, float mean, float inv,
    float g, float bt, int mode, int act, float slope)
{
    if (hasDy2) dy = dy + dy2;
    if (mode == 0) return dy;
    const bool positive = mode == 1 ? o > 0.0f : bn_affine(x, mean, inv, g, bt) > 0.0f;
    return act == 2 ? bn_act_grad<true>(positive, dy, slope) : bn_act_grad<false>(positive, dy, slope);
}

__device__ __forceinline__ float4 bn_nhwc_dy4(const float* __restrict__ gradOutput, const float* __restrict__ gradOutput2,
    const float* __restrict__ maskSource, long long at, int c, float4 x, float4 mean, float4 inv, float4 g, int mode,
    int act, float slope)
{
    const float4 dy = *reinterpret_cast<const float4*>(gradOutput + at);
    const bool has2 = gradOutput2 != nullptr;
    const float4 d2 = has2 ? *reinterpret_cast<const float4*>(gradOutput2 + at) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f), bt = o;
    if (mode == 1) o = *reinterpret_cast<const float4*>(maskSource + at);
    else if (mode == 2) bt = *reinterpret_cast<const float4*>(maskSource + c);
    float4 r;
    r.x = bn_nhwc_dy(dy.x, d2.x, has2, o.x, x.x, mean.x, inv.x, g.x, bt.x, mode, act, slope);
    r.y = bn_nhwc_dy(dy.y, d2.y, has2, o.y, x.y, mean.y, inv.y, g.y, bt.y, mode, act, slope);
    r.z = bn_nhwc_dy(dy.z, d2.z, has2, o.z, x.z, mean.z, inv.z, g.z, bt.z, mode, act, slope);
    r.w = bn_nhwc_dy(dy.w, d2.w, has2, o.w, x.w, mean.w, inv.w, g.w, bt.w, mode, act, slope);
    return r;
}

__device__ __forceinline__ float4 bn_nhwc_inv4(const float* __restrict__ saveVar, int c, float epsilon)
{
    const float4 v = *reinterpret_cast<const float4*>(saveVar + c);
    return make_float4(rsqrtf(v.x + epsilon), rsqrtf(v.y + epsilon), rsqrtf(v.z + epsilon), rsqrtf(v.w + epsilon));
}

extern "C" __global__ __launch_bounds__(256) void batchnorm_nhwc_bwd_reduce(const float* __restrict__ gradOutput,
    const float* __restrict__ maskSource, const float* __restrict__ gradOutput2, const float* __restrict__ input,
    const float* __restrict__ gamma, const float* __restrict__ saveMean, const float* __restrict__ saveVar,
    float* __restrict__ partials, int rows, int channels, float epsilon, int splits, int mode, int act, float slope)
{
    __shared__ float4 red[8][32];
    const int c = (blockIdx.x * 32 + threadIdx.x) * 4;
    const bool live = c < channels;
    int begin, end;
    bn_nhwc_range(rows, splits, begin, end);
    float4 sdy = make_float4(0.0f, 0.0f, 0.0f, 0.0f), sdyx = sdy;
    if (live)
    {
        const float4 mean = *reinterpret_cast<const float4*>(saveMean + c), inv = bn_nhwc_inv4(saveVar, c, epsilon);
        const float4 g = *reinterpret_cast<const float4*>(gamma + c);
        for (int r = begin + threadIdx.y; r < end; r += 8)
        {
            const long long at = (long long)r * channels + c;
            const float4 x = *reinterpret_cast<const float4*>(input + at);
            const float4 dy = bn_nhwc_dy4(gradOutput, gradOutput2, maskSource, at, c, x, mean, inv, g, mode, act, slope);
            sdy.x += dy.x; sdy.y += dy.y; sdy.z += dy.z; sdy.w += dy.w;
            sdyx.x += dy.x * ((x.x - mean.x) * inv.x); sdyx.y += dy.y * ((x.y - mean.y) * inv.y);
            sdyx.z += dy.z * ((x.z - mean.z) * inv.z); sdyx.w += dy.w * ((x.w - mean.w) * inv.w);
        }
    }
    sdy = bn_nhwc_lane_sum(sdy, red);
    sdyx = bn_nhwc_lane_sum(sdyx, red);
    if (live && threadIdx.y == 0)
    {
        const float a1[4] = { sdy.x, sdy.y, sdy.z, sdy.w }, a2[4] = { sdyx.x, sdyx.y, sdyx.z, sdyx.w };
        for (int j = 0; j < 4; j++)
        {
            float* p = partials + ((long long)(c + j) * splits + blockIdx.y) * 2;
            p[0] = a1[j]; p[1] = a2[j];
        }
    }
}

// One block per channel, as batchnorm_nhwc_finalize.
extern "C" __global__ void batchnorm_nhwc_bwd_finalize(const float* __restrict__ partials,
    float* __restrict__ gradGamma, float* __restrict__ gradBeta, int channels, int splits)
{
    __shared__ double red[32];
    const int c = blockIdx.x;
    if (c >= channels) return;
    double sdy = 0.0, sdyx = 0.0;
    for (int i = threadIdx.x; i < splits; i += blockDim.x)
    {
        const float* p = partials + ((long long)c * splits + i) * 2;
        sdy += (double)p[0]; sdyx += (double)p[1];
    }
    sdy = bn_nhwc_block_sum(sdy, red);
    sdyx = bn_nhwc_block_sum(sdyx, red);
    if (threadIdx.x != 0) return;
    gradGamma[c] = (float)sdyx;
    gradBeta[c] = (float)sdy;
}

extern "C" __global__ void batchnorm_nhwc_bwd_apply(const float* __restrict__ gradOutput,
    const float* __restrict__ maskSource, const float* __restrict__ gradOutput2, const float* __restrict__ input,
    const float* __restrict__ gamma, const float* __restrict__ saveMean, const float* __restrict__ saveVar,
    const float* __restrict__ gradGamma, const float* __restrict__ gradBeta, float* __restrict__ gradInput,
    long long quads, int channels, float epsilon, int mode, int act, float slope, int accumulate, float invN)
{
    const int cq = channels >> 2;
    for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < quads; q += (long long)gridDim.x * blockDim.x)
    {
        const int c = (int)(q % cq) * 4;
        const long long at = q * 4;
        const float4 mean = *reinterpret_cast<const float4*>(saveMean + c), inv = bn_nhwc_inv4(saveVar, c, epsilon);
        const float4 g = *reinterpret_cast<const float4*>(gamma + c);
        const float4 sDy = *reinterpret_cast<const float4*>(gradBeta + c), sDyX = *reinterpret_cast<const float4*>(gradGamma + c);
        const float4 x = reinterpret_cast<const float4*>(input)[q];
        const float4 dy = bn_nhwc_dy4(gradOutput, gradOutput2, maskSource, at, c, x, mean, inv, g, mode, act, slope);
        float4 r;
        r.x = inv.x * (dy.x * g.x - invN * (g.x * sDy.x + ((x.x - mean.x) * inv.x) * (g.x * sDyX.x)));
        r.y = inv.y * (dy.y * g.y - invN * (g.y * sDy.y + ((x.y - mean.y) * inv.y) * (g.y * sDyX.y)));
        r.z = inv.z * (dy.z * g.z - invN * (g.z * sDy.z + ((x.z - mean.z) * inv.z) * (g.z * sDyX.z)));
        r.w = inv.w * (dy.w * g.w - invN * (g.w * sDy.w + ((x.w - mean.w) * inv.w) * (g.w * sDyX.w)));
        float4* dst = reinterpret_cast<float4*>(gradInput) + q;
        if (accumulate) { const float4 o = *dst; r.x += o.x; r.y += o.y; r.z += o.z; r.w += o.w; }
        *dst = r;
    }
}
