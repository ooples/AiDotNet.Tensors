namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Sparse multi-head graph attention (CpuEngine.GraphAttentionCsr's device form): one thread per (graph, node, head)
/// row walks the node's CSR neighbourhood; the backward gathers into each node through the transposed (CSC) pattern,
/// so nothing uses atomics and the sums are deterministic. The dropout draw is CpuEngine.GraphAttentionUniform.
/// </summary>
internal static class CudaGraphAttentionKernels
{
    public static string[] GetKernelNames() =>
        ["gat_csr_scores", "gat_csr_forward", "gat_csr_backward_rows", "gat_csr_backward_cols", "gat_csr_attention_grad"];

    public static string GetSource() => @"
__device__ __forceinline__ float gat_uniform(unsigned int seed, unsigned int index)
{
    unsigned int x = index * 0x9E3779B1u + seed;
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return (float)(x >> 8) * (1.0f / 16777216.0f);
}

// selfScore[r, h] = Wh[r, h, :] . a[h, :F]; neighScore[r, h] = Wh[r, h, :] . a[h, F:]  (r over batch * nodes)
extern ""C"" __global__ __launch_bounds__(256) void gat_csr_scores(
    const float* __restrict__ wh, const float* __restrict__ attn,
    float* __restrict__ selfScore, float* __restrict__ neighScore, int rows, int heads, int features)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * heads) return;
    int h = t % heads;
    const float* w = wh + (long long)t * features;
    const float* as = attn + h * 2 * features;
    float s = 0.0f, q = 0.0f;
    for (int f = 0; f < features; f++) { float v = w[f]; s += v * as[f]; q += v * as[features + f]; }
    selfScore[t] = s;
    neighScore[t] = q;
}

extern ""C"" __global__ __launch_bounds__(256) void gat_csr_forward(
    const float* __restrict__ wh, const float* __restrict__ selfScore, const float* __restrict__ neighScore,
    const int* __restrict__ rowPtr, const int* __restrict__ col, float* __restrict__ out,
    int batch, int nodes, int heads, int features, int edges, float slope, float rate, float keep, unsigned int seed)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= batch * nodes * heads) return;
    int h = t % heads;
    int bi = t / heads;
    int b = bi / nodes, i = bi % nodes;
    int r0 = rowPtr[i], r1 = rowPtr[i + 1];
    float self = selfScore[t];
    int nb = b * nodes;
    float mx = -3.402823466e38f;
    for (int e = r0; e < r1; e++)
    {
        float z = self + neighScore[(nb + col[e]) * heads + h];
        float ev = z > 0.0f ? z : slope * z;
        mx = fmaxf(mx, ev);
    }
    float sum = 0.0f;
    for (int e = r0; e < r1; e++)
    {
        float z = self + neighScore[(nb + col[e]) * heads + h];
        float ev = z > 0.0f ? z : slope * z;
        sum += expf(ev - mx);
    }
    float inv = 1.0f / sum;
    float acc[32];
    float* o = out + (long long)t * features;
    for (int f0 = 0; f0 < features; f0 += 32)
    {
        int fc = min(32, features - f0);
        for (int f = 0; f < fc; f++) acc[f] = 0.0f;
        for (int e = r0; e < r1; e++)
        {
            float z = self + neighScore[(nb + col[e]) * heads + h];
            float ev = z > 0.0f ? z : slope * z;
            float c = expf(ev - mx) * inv;
            if (rate > 0.0f)
                c = gat_uniform(seed, (unsigned int)(((long long)b * edges + e) * heads + h)) < rate ? 0.0f : c * keep;
            if (c == 0.0f) continue;
            const float* w = wh + ((long long)(nb + col[e]) * heads + h) * features + f0;
            for (int f = 0; f < fc; f++) acc[f] += c * w[f];
        }
        for (int f = 0; f < fc; f++) o[f0 + f] = acc[f];
    }
}

// Per row: recompute the coefficients; gz[e] = c (g - sum_k c_k g_k) LeakyReLU'(z), g = m (gout_i . Wh_j);
// cp[e] = c m; gSelf[row] = sum_e gz[e].
extern ""C"" __global__ __launch_bounds__(256) void gat_csr_backward_rows(
    const float* __restrict__ wh, const float* __restrict__ gout,
    const float* __restrict__ selfScore, const float* __restrict__ neighScore,
    const int* __restrict__ rowPtr, const int* __restrict__ col,
    float* __restrict__ gz, float* __restrict__ cp, float* __restrict__ gSelf,
    int batch, int nodes, int heads, int features, int edges, float slope, float rate, float keep, unsigned int seed)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= batch * nodes * heads) return;
    int h = t % heads;
    int bi = t / heads;
    int b = bi / nodes, i = bi % nodes;
    int r0 = rowPtr[i], r1 = rowPtr[i + 1];
    float self = selfScore[t];
    int nb = b * nodes;
    float mx = -3.402823466e38f;
    for (int e = r0; e < r1; e++)
    {
        float z = self + neighScore[(nb + col[e]) * heads + h];
        mx = fmaxf(mx, z > 0.0f ? z : slope * z);
    }
    float sum = 0.0f;
    for (int e = r0; e < r1; e++)
    {
        float z = self + neighScore[(nb + col[e]) * heads + h];
        sum += expf((z > 0.0f ? z : slope * z) - mx);
    }
    float inv = 1.0f / sum;
    const float* g = gout + (long long)t * features;
    float dot = 0.0f;
    for (int e = r0; e < r1; e++)
    {
        float z = self + neighScore[(nb + col[e]) * heads + h];
        float c = expf((z > 0.0f ? z : slope * z) - mx) * inv;
        float m = 1.0f;
        long long slot = ((long long)b * edges + e) * heads + h;
        if (rate > 0.0f) m = gat_uniform(seed, (unsigned int)slot) < rate ? 0.0f : keep;
        float gv = 0.0f;
        if (m != 0.0f)
        {
            const float* w = wh + ((long long)(nb + col[e]) * heads + h) * features;
            for (int f = 0; f < features; f++) gv += g[f] * w[f];
            gv *= m;
        }
        cp[slot] = c * m;
        gz[slot] = gv;
        dot += c * gv;
    }
    float gs = 0.0f;
    for (int e = r0; e < r1; e++)
    {
        float z = self + neighScore[(nb + col[e]) * heads + h];
        float c = expf((z > 0.0f ? z : slope * z) - mx) * inv;
        long long slot = ((long long)b * edges + e) * heads + h;
        float ge = c * (gz[slot] - dot) * (z > 0.0f ? 1.0f : slope);
        gz[slot] = ge;
        gs += ge;
    }
    gSelf[t] = gs;
}

// Per column j: gWh[j] = sum over edges (i -> j) cp * gout_i + gSelf_j a_self + gNeigh_j a_neigh.
extern ""C"" __global__ __launch_bounds__(256) void gat_csr_backward_cols(
    const float* __restrict__ gout, const float* __restrict__ attn,
    const float* __restrict__ gz, const float* __restrict__ cp, const float* __restrict__ gSelf,
    const int* __restrict__ colPtr, const int* __restrict__ tRow, const int* __restrict__ tEdge,
    float* __restrict__ gNeigh, float* __restrict__ gWh,
    int batch, int nodes, int heads, int features, int edges)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= batch * nodes * heads) return;
    int h = t % heads;
    int bj = t / heads;
    int b = bj / nodes, j = bj % nodes;
    int k0 = colPtr[j], k1 = colPtr[j + 1];
    int nb = b * nodes;
    float gn = 0.0f;
    for (int k = k0; k < k1; k++) gn += gz[((long long)b * edges + tEdge[k]) * heads + h];
    gNeigh[t] = gn;
    float gs = gSelf[t];
    const float* as = attn + h * 2 * features;
    float* o = gWh + (long long)t * features;
    float acc[32];
    for (int f0 = 0; f0 < features; f0 += 32)
    {
        int fc = min(32, features - f0);
        for (int f = 0; f < fc; f++) acc[f] = gs * as[f0 + f] + gn * as[features + f0 + f];
        for (int k = k0; k < k1; k++)
        {
            float c = cp[((long long)b * edges + tEdge[k]) * heads + h];
            if (c == 0.0f) continue;
            const float* g = gout + ((long long)(nb + tRow[k]) * heads + h) * features + f0;
            for (int f = 0; f < fc; f++) acc[f] += c * g[f];
        }
        for (int f = 0; f < fc; f++) o[f0 + f] = acc[f];
    }
}

// One block per attention entry (h, side, f): sum over rows of score-grad * Wh.
extern ""C"" __global__ __launch_bounds__(256) void gat_csr_attention_grad(
    const float* __restrict__ wh, const float* __restrict__ gSelf, const float* __restrict__ gNeigh,
    float* __restrict__ gAttn, int rows, int heads, int features)
{
    int entry = blockIdx.x;                 // h * 2F + side * F + f
    int h = entry / (2 * features);
    int rem = entry % (2 * features);
    int side = rem / features, f = rem % features;
    const float* gsrc = side == 0 ? gSelf : gNeigh;
    float acc = 0.0f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x)
        acc += gsrc[r * heads + h] * wh[((long long)r * heads + h) * features + f];
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) gAttn[entry] = red[0];
}
";
}
