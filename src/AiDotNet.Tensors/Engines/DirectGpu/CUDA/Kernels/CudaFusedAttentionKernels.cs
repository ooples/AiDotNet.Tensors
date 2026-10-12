namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// #1141: tiled FlashAttention-2 for float32 on CUDA -- the device counterpart of the CPU fused attention (#889) and
/// of <c>MultiHeadAttentionCore</c>. One kernel family serves both:
/// <list type="bullet">
/// <item>operands are strided [B, H, S, D] views with a unit-stride last axis, so the head-interleaved [B, S, H * D]
/// layout of <c>MultiHeadAttentionCore</c> is read and written in place (no permutes);</item>
/// <item>an optional boolean mask (int32 0 / 1 on the device) broadcast through its own strides, causal masking with a
/// diagonal offset (0: PyTorch's top-left <c>is_causal</c>; seqK - seqQ: the bottom-right alignment of the attention
/// core), and a Gemma-2 soft-cap <c>cap * tanh(s / cap)</c>;</item>
/// <item>a row with no allowed key outputs zeros and passes no gradient (log-sum-exp -inf), as the CPU kernels.</item>
/// </list>
/// The forward runs 4 threads per query row, each over every 4th key of a 64-key tile staged in padded shared memory,
/// rescaling its running max / sum once per tile; the four partials are merged with warp shuffles in a fixed order.
/// The backward is FlashAttention-2's two passes without atomics: a delta = rowsum(dO * O) pass, a key-block pass
/// (2 threads per key row, half the head dimension each) that owns dK / dV, and a query-block pass (2 threads per query
/// row) that owns dQ. Every reduction has a fixed order, so results do not depend on scheduling.
/// Head dimension D (= value dimension) is a template parameter: 16, 32 and 64 are compiled.
/// <para>
/// The default backward (<c>fa_backward_fused_d*</c>) is a CUDA-only implementation of
/// <see cref="IFusedAttentionBackend.FusedAttentionBackward"/>, not a new operation: it computes the same dQ, dK and dV
/// as the two-pass kernels, which <c>DeterministicReductions</c> still selects. It needs a float atomic add on device
/// memory for dQ. WGSL has none, and Vulkan and OpenCL offer one only through optional extensions, so it cannot be
/// written once for every backend. The other five backends get the same interface method from per-element kernels
/// that have no cross-block sums and need no atomics (#1179, implemented in #1181).
/// </para>
/// </summary>
internal static class CudaFusedAttentionKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "fa_forward_d16", "fa_forward_d32", "fa_forward_d64",
        "fa_backward_kv_d16", "fa_backward_kv_d32", "fa_backward_kv_d64",
        "fa_backward_q_d16", "fa_backward_q_d32", "fa_backward_q_d64",
        "fa_backward_fused_d16", "fa_backward_fused_d32", "fa_backward_fused_d64",
        "fa_backward_delta",
    };

    public static string GetSource() => @"
#define FA_BR 32        // query rows per forward block (4 threads each: 128 threads)
#define FA_BC 64        // keys per shared-memory tile (backward-q pass)
#define FA_FBC 64       // keys per shared-memory tile (forward)
#define FA_SPLIT 4      // threads per query row in the forward
#define FA_KB 128       // key rows per backward-kv block (2 threads each: 256 threads)
#define FA_QB 128       // query rows per backward-q block (2 threads each: 256 threads)
#define FA_QT 32        // query rows per shared-memory tile in the backward-kv pass
#define FA_FKB 64       // key rows per fused-backward block (2 threads each: 128 threads)
// The two threads of a backward row own alternating 4-element chunks: element FA_EL(dd, half) for dd in [0, D / 2).
// Their float4 shared-memory reads of one row then hit 8 distinct banks (no conflict) and each feeds 4 FMAs.
#define FA_EL(dd, half) (8 * ((dd) >> 2) + 4 * (half) + ((dd) & 3))

struct FaParams
{
    int B, H, Sq, Sk;
    // element strides (the last axis is unit-stride): batch, head, row
    long long qb, qh, qr, kb, kh, kr, vb, vh, vr, ob, oh, orow;
    float scale, softcap;
    int causal, causalOffset;      // allowed iff j <= i + causalOffset
    int hasMask;
    long long mb, mh, mi, mj;      // mask strides (0 on a broadcast axis)
};

__device__ __forceinline__ bool fa_allowed(const FaParams& p, const int* __restrict__ mask, int b, int h, int i, int j)
{
    if (p.causal && j > i + p.causalOffset) return false;
    if (p.hasMask && mask[b * p.mb + h * p.mh + i * p.mi + j * p.mj] == 0) return false;
    return true;
}

// The capped score and d(capped)/d(raw).
__device__ __forceinline__ float fa_cap(float raw, float cap, float* slope)
{
    if (cap > 0.0f) { float t = tanhf(raw / cap); *slope = 1.0f - t * t; return cap * t; }
    *slope = 1.0f;
    return raw;
}

// ---------------------------------------------------------------------------------------------------------------
// Forward: grid (ceil(Sq / FA_BR), B * H), block FA_BR * FA_SPLIT threads. Thread (r, c) handles query row r of the
// block and keys c, c + 4, ... of each tile; the 4 lanes of a row are adjacent in the warp and merged by shuffles.
// ---------------------------------------------------------------------------------------------------------------
template <int D>
__device__ void fa_forward(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ o, float* __restrict__ lse, const int* __restrict__ mask, FaParams p)
{
    __shared__ __align__(16) float Ks[FA_FBC][D + 4];
    __shared__ __align__(16) float Vs[FA_FBC][D + 4];
    int bh = blockIdx.y, b = bh / p.H, h = bh % p.H;
    int r = threadIdx.x / FA_SPLIT, c = threadIdx.x % FA_SPLIT;
    // Causal: the last query blocks see the most keys, so they are launched first (shorter tail).
    int qBlock = p.causal ? (int)(gridDim.x - 1 - blockIdx.x) : (int)blockIdx.x;
    int i = qBlock * FA_BR + r;
    bool active = i < p.Sq;

    float qr[D], acc[D];
    const float* qrow = q + b * p.qb + h * p.qh + (long long)(active ? i : 0) * p.qr;
    #pragma unroll
    for (int d = 0; d < D; d++) { qr[d] = active ? qrow[d] * p.scale : 0.0f; acc[d] = 0.0f; }
    float m = -INFINITY, l = 0.0f;

    // Keys this block can need: up to the last row's causal limit.
    int lastRow = min(p.Sq - 1, qBlock * FA_BR + FA_BR - 1);
    int keyEnd = p.causal ? min(p.Sk, lastRow + p.causalOffset + 1) : p.Sk;
    const float* kbase = k + b * p.kb + h * p.kh;
    const float* vbase = v + b * p.vb + h * p.vh;
    for (int j0 = 0; j0 < keyEnd; j0 += FA_FBC)
    {
        int cols = min(FA_FBC, keyEnd - j0);
        for (int e = threadIdx.x; e < cols * (D / 4); e += blockDim.x)
        {
            int jj = e / (D / 4), d4 = e % (D / 4);
            *reinterpret_cast<float4*>(&Ks[jj][4 * d4]) = *reinterpret_cast<const float4*>(kbase + (long long)(j0 + jj) * p.kr + 4 * d4);
            *reinterpret_cast<float4*>(&Vs[jj][4 * d4]) = *reinterpret_cast<const float4*>(vbase + (long long)(j0 + jj) * p.vr + 4 * d4);
        }
        __syncthreads();
        if (active)
        {
            float s[FA_FBC / FA_SPLIT];
            float tileMax = -INFINITY;
            #pragma unroll
            for (int t = 0; t < FA_FBC / FA_SPLIT; t++)
            {
                int jj = c + t * FA_SPLIT, j = j0 + jj;
                float sv = -INFINITY;
                if (jj < cols && fa_allowed(p, mask, b, h, i, j))
                {
                    float dot = 0.0f;
                    const float4* k4 = reinterpret_cast<const float4*>(&Ks[jj][0]);
                    #pragma unroll
                    for (int d4 = 0; d4 < D / 4; d4++)
                    {
                        float4 kv = k4[d4];
                        dot += qr[4 * d4] * kv.x + qr[4 * d4 + 1] * kv.y + qr[4 * d4 + 2] * kv.z + qr[4 * d4 + 3] * kv.w;
                    }
                    float slope;
                    sv = fa_cap(dot, p.softcap, &slope);
                }
                s[t] = sv;
                tileMax = fmaxf(tileMax, sv);
            }
            if (tileMax != -INFINITY)
            {
                float mNew = fmaxf(m, tileMax);
                float rescale = m == -INFINITY ? 0.0f : __expf(m - mNew);
                l *= rescale;
                #pragma unroll
                for (int d = 0; d < D; d++) acc[d] *= rescale;
                #pragma unroll
                for (int t = 0; t < FA_FBC / FA_SPLIT; t++)
                {
                    if (s[t] == -INFINITY) continue;
                    float pr = __expf(s[t] - mNew);
                    l += pr;
                    int jj = c + t * FA_SPLIT;
                    const float4* v4 = reinterpret_cast<const float4*>(&Vs[jj][0]);
                    #pragma unroll
                    for (int d4 = 0; d4 < D / 4; d4++)
                    {
                        float4 vv = v4[d4];
                        acc[4 * d4] += pr * vv.x; acc[4 * d4 + 1] += pr * vv.y;
                        acc[4 * d4 + 2] += pr * vv.z; acc[4 * d4 + 3] += pr * vv.w;
                    }
                }
                m = mNew;
            }
        }
        __syncthreads();
    }

    // Merge the FA_SPLIT partials of the row (lanes r*4 .. r*4+3 of the warp), xor 1 then xor 2.
    #pragma unroll
    for (int off = 1; off < FA_SPLIT; off <<= 1)
    {
        float mo = __shfl_xor_sync(0xffffffffu, m, off);
        float lo = __shfl_xor_sync(0xffffffffu, l, off);
        float mNew = fmaxf(m, mo);
        float a = m == -INFINITY ? 0.0f : __expf(m - mNew);
        float bo = mo == -INFINITY ? 0.0f : __expf(mo - mNew);
        #pragma unroll
        for (int d = 0; d < D; d++)
        {
            float other = __shfl_xor_sync(0xffffffffu, acc[d], off);
            acc[d] = acc[d] * a + other * bo;
        }
        l = l * a + lo * bo;
        m = mNew;
    }
    if (active && c == 0)
    {
        float* orow = o + b * p.ob + h * p.oh + (long long)i * p.orow;
        float inv = l > 0.0f ? 1.0f / l : 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) orow[d] = acc[d] * inv;
        lse[(long long)bh * p.Sq + i] = l > 0.0f ? m + __logf(l) : -INFINITY;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// delta[bh, i] = sum_d dO[i, d] * O[i, d]   (one thread per query row)
// ---------------------------------------------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void fa_backward_delta(const float* __restrict__ dO,
    const float* __restrict__ o, float* __restrict__ delta, FaParams p, int D)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rows = (long long)p.B * p.H * p.Sq;
    if (idx >= rows) return;
    int i = (int)(idx % p.Sq); int bh = (int)(idx / p.Sq); int b = bh / p.H, h = bh % p.H;
    long long base = b * p.ob + h * p.oh + (long long)i * p.orow;   // dO shares O's layout
    float sum = 0.0f;
    for (int d = 0; d < D; d++) sum += dO[base + d] * o[base + d];
    delta[idx] = sum;
}

// ---------------------------------------------------------------------------------------------------------------
// Backward, key pass: grid (ceil(Sk / FA_KB), B * H), block FA_KB * 2 threads. Thread (j, half) owns key row j and
// half of the head dimension of dK[j] and dV[j]; query rows stream through shared memory in tiles of FA_QB.
// ---------------------------------------------------------------------------------------------------------------
template <int D>
__device__ void fa_backward_kv(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ dO, const float* __restrict__ lse, const float* __restrict__ delta,
    float* __restrict__ dK, float* __restrict__ dV, const int* __restrict__ mask, FaParams p)
{
    const int HALF = D / 2;
    __shared__ __align__(16) float Qs[FA_QT][D + 4];
    __shared__ __align__(16) float Gs[FA_QT][D + 4];
    __shared__ float Ls[FA_QT], Ds[FA_QT];
    int bh = blockIdx.y, b = bh / p.H, h = bh % p.H;
    int jr = threadIdx.x / 2, half = threadIdx.x % 2;
    int j = blockIdx.x * FA_KB + jr;
    bool active = j < p.Sk;
    // The two threads of a row own interleaved elements (2 * dd + half), so their shared-memory reads of one row hit
    // adjacent banks rather than the same bank D / 2 apart.

    float kr[HALF], vr[HALF], dk[HALF], dv[HALF];
    const float* krow = k + b * p.kb + h * p.kh + (long long)(active ? j : 0) * p.kr;
    const float* vrow = v + b * p.vb + h * p.vh + (long long)(active ? j : 0) * p.vr;
    #pragma unroll
    for (int d = 0; d < HALF; d++) { kr[d] = active ? krow[FA_EL(d, half)] : 0.0f; vr[d] = active ? vrow[FA_EL(d, half)] : 0.0f; dk[d] = 0.0f; dv[d] = 0.0f; }

    // Causal: query rows before the first key of this block minus the offset see none of its keys.
    int firstKey = blockIdx.x * FA_KB;
    int iStart = p.causal ? max(0, firstKey - p.causalOffset) : 0;
    iStart = iStart / FA_QT * FA_QT;
    const float* qbase = q + b * p.qb + h * p.qh;
    const float* gbase = dO + b * p.ob + h * p.oh;
    for (int i0 = iStart; i0 < p.Sq; i0 += FA_QT)
    {
        int rows = min(FA_QT, p.Sq - i0);
        for (int e = threadIdx.x; e < rows * D; e += blockDim.x)
        {
            int ii = e / D, d = e % D;
            Qs[ii][d] = qbase[(long long)(i0 + ii) * p.qr + d] * p.scale;
            Gs[ii][d] = gbase[(long long)(i0 + ii) * p.orow + d];
        }
        for (int e = threadIdx.x; e < rows; e += blockDim.x)
        {
            Ls[e] = lse[(long long)bh * p.Sq + i0 + e];
            Ds[e] = delta[(long long)bh * p.Sq + i0 + e];
        }
        __syncthreads();
        // Unrolled so neighbouring rows' independent dot products overlap (the pass runs at low occupancy).
        #pragma unroll 4
        for (int ii = 0; ii < rows; ii++)
        {
            int i = i0 + ii;
            float sPart = 0.0f, dpPart = 0.0f;
            #pragma unroll
            for (int c4 = 0; c4 < HALF / 4; c4++)
            {
                float4 qv = *reinterpret_cast<const float4*>(&Qs[ii][8 * c4 + 4 * half]);
                float4 gv = *reinterpret_cast<const float4*>(&Gs[ii][8 * c4 + 4 * half]);
                sPart += qv.x * kr[4 * c4] + qv.y * kr[4 * c4 + 1] + qv.z * kr[4 * c4 + 2] + qv.w * kr[4 * c4 + 3];
                dpPart += gv.x * vr[4 * c4] + gv.y * vr[4 * c4 + 1] + gv.z * vr[4 * c4 + 2] + gv.w * vr[4 * c4 + 3];
            }
            float raw = sPart + __shfl_xor_sync(0xffffffffu, sPart, 1);
            float dp = dpPart + __shfl_xor_sync(0xffffffffu, dpPart, 1);
            float rowLse = Ls[ii];
            if (!active || rowLse == -INFINITY || !fa_allowed(p, mask, b, h, i, j)) continue;
            float slope;
            float s = fa_cap(raw, p.softcap, &slope);
            float pr = __expf(s - rowLse);
            float ds = pr * (dp - Ds[ii]) * slope;
            #pragma unroll
            for (int c4 = 0; c4 < HALF / 4; c4++)
            {
                float4 qv = *reinterpret_cast<const float4*>(&Qs[ii][8 * c4 + 4 * half]);
                float4 gv = *reinterpret_cast<const float4*>(&Gs[ii][8 * c4 + 4 * half]);
                dv[4 * c4] += pr * gv.x; dv[4 * c4 + 1] += pr * gv.y; dv[4 * c4 + 2] += pr * gv.z; dv[4 * c4 + 3] += pr * gv.w;
                dk[4 * c4] += ds * qv.x; dk[4 * c4 + 1] += ds * qv.y; dk[4 * c4 + 2] += ds * qv.z; dk[4 * c4 + 3] += ds * qv.w;
            }
        }
        __syncthreads();
    }
    if (active)
    {
        float* dkrow = dK + b * p.kb + h * p.kh + (long long)j * p.kr;
        float* dvrow = dV + b * p.vb + h * p.vh + (long long)j * p.vr;
        #pragma unroll
        for (int d = 0; d < HALF; d++) { dkrow[FA_EL(d, half)] = dk[d]; dvrow[FA_EL(d, half)] = dv[d]; }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Backward, query pass: grid (ceil(Sq / FA_QB), B * H), block FA_QB * 2 threads. Thread (i, half) owns query row i
// and half of dQ[i]; key / value rows stream through shared memory in tiles of FA_BC.
// ---------------------------------------------------------------------------------------------------------------
template <int D>
__device__ void fa_backward_q(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ dO, const float* __restrict__ lse, const float* __restrict__ delta,
    float* __restrict__ dQ, const int* __restrict__ mask, FaParams p)
{
    const int HALF = D / 2;
    __shared__ __align__(16) float Ks[FA_BC][D + 4];
    __shared__ __align__(16) float Vs[FA_BC][D + 4];
    int bh = blockIdx.y, b = bh / p.H, h = bh % p.H;
    int ir = threadIdx.x / 2, half = threadIdx.x % 2;
    int qBlock = p.causal ? (int)(gridDim.x - 1 - blockIdx.x) : (int)blockIdx.x;   // heaviest first when causal
    int i = qBlock * FA_QB + ir;
    bool active = i < p.Sq;

    float qr[HALF], gr[HALF], dq[HALF];
    const float* qrow = q + b * p.qb + h * p.qh + (long long)(active ? i : 0) * p.qr;
    const float* grow = dO + b * p.ob + h * p.oh + (long long)(active ? i : 0) * p.orow;
    #pragma unroll
    for (int d = 0; d < HALF; d++) { qr[d] = active ? qrow[FA_EL(d, half)] * p.scale : 0.0f; gr[d] = active ? grow[FA_EL(d, half)] : 0.0f; dq[d] = 0.0f; }
    float rowLse = active ? lse[(long long)bh * p.Sq + i] : -INFINITY;
    float rowDelta = active ? delta[(long long)bh * p.Sq + i] : 0.0f;

    int lastRow = min(p.Sq - 1, qBlock * FA_QB + FA_QB - 1);
    int keyEnd = p.causal ? max(0, min(p.Sk, lastRow + p.causalOffset + 1)) : p.Sk;
    const float* kbase = k + b * p.kb + h * p.kh;
    const float* vbase = v + b * p.vb + h * p.vh;
    for (int j0 = 0; j0 < keyEnd; j0 += FA_BC)
    {
        int cols = min(FA_BC, keyEnd - j0);
        for (int e = threadIdx.x; e < cols * D; e += blockDim.x)
        {
            int jj = e / D, d = e % D;
            Ks[jj][d] = kbase[(long long)(j0 + jj) * p.kr + d];
            Vs[jj][d] = vbase[(long long)(j0 + jj) * p.vr + d];
        }
        __syncthreads();
        #pragma unroll 4
        for (int jj = 0; jj < cols; jj++)
        {
            int j = j0 + jj;
            float sPart = 0.0f, dpPart = 0.0f;
            #pragma unroll
            for (int c4 = 0; c4 < HALF / 4; c4++)
            {
                float4 kv = *reinterpret_cast<const float4*>(&Ks[jj][8 * c4 + 4 * half]);
                float4 vv = *reinterpret_cast<const float4*>(&Vs[jj][8 * c4 + 4 * half]);
                sPart += qr[4 * c4] * kv.x + qr[4 * c4 + 1] * kv.y + qr[4 * c4 + 2] * kv.z + qr[4 * c4 + 3] * kv.w;
                dpPart += gr[4 * c4] * vv.x + gr[4 * c4 + 1] * vv.y + gr[4 * c4 + 2] * vv.z + gr[4 * c4 + 3] * vv.w;
            }
            float raw = sPart + __shfl_xor_sync(0xffffffffu, sPart, 1);
            float dp = dpPart + __shfl_xor_sync(0xffffffffu, dpPart, 1);
            if (!active || rowLse == -INFINITY || !fa_allowed(p, mask, b, h, i, j)) continue;
            float slope;
            float s = fa_cap(raw, p.softcap, &slope);
            float pr = __expf(s - rowLse);
            float ds = pr * (dp - rowDelta) * slope;
            #pragma unroll
            for (int c4 = 0; c4 < HALF / 4; c4++)
            {
                float4 kv = *reinterpret_cast<const float4*>(&Ks[jj][8 * c4 + 4 * half]);
                dq[4 * c4] += ds * kv.x; dq[4 * c4 + 1] += ds * kv.y; dq[4 * c4 + 2] += ds * kv.z; dq[4 * c4 + 3] += ds * kv.w;
            }
        }
        __syncthreads();
    }
    if (active)
    {
        float* dqrow = dQ + b * p.qb + h * p.qh + (long long)i * p.qr;
        #pragma unroll
        for (int d = 0; d < HALF; d++) dqrow[FA_EL(d, half)] = dq[d] * p.scale;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Fused backward (the default; the two-pass kv / q kernels serve DeterministicReductions): grid (ceil(Sk / FA_FKB),
// B * H), block FA_FKB * 2 threads. Each block owns dK / dV of its FA_FKB keys exactly as the kv pass, and per query
// tile also stages dS in shared memory, multiplies it by the block's keys (dQ_tile = dS K, 4 x D/16 outputs per
// thread) and adds that into dQ with atomics -- one per (row, column) per key block -- instead of a second pass that
// recomputes the scores. dQ must be zero on entry. Summation order of dQ across key blocks is not fixed.
// ---------------------------------------------------------------------------------------------------------------
template <int D>
__device__ void fa_backward_fused(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ dO, const float* __restrict__ lse, const float* __restrict__ delta,
    float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dV, const int* __restrict__ mask, FaParams p)
{
    const int HALF = D / 2;
    constexpr int OC = D / 16;
    __shared__ __align__(16) float Qs[FA_QT][D + 4];
    __shared__ __align__(16) float Gs[FA_QT][D + 4];
    __shared__ __align__(16) float Ks[FA_FKB][D + 4];
    __shared__ __align__(16) float dSs[FA_QT][FA_FKB + 4];
    __shared__ float Ls[FA_QT], Ds[FA_QT];
    int bh = blockIdx.y, b = bh / p.H, h = bh % p.H;
    int jr = threadIdx.x / 2, half = threadIdx.x % 2;
    int j = blockIdx.x * FA_FKB + jr;
    bool active = j < p.Sk;

    float kr[HALF], vr[HALF], dk[HALF], dv[HALF];
    const float* krow = k + b * p.kb + h * p.kh + (long long)(active ? j : 0) * p.kr;
    const float* vrow = v + b * p.vb + h * p.vh + (long long)(active ? j : 0) * p.vr;
    #pragma unroll
    for (int d = 0; d < HALF; d++)
    {
        kr[d] = active ? krow[FA_EL(d, half)] : 0.0f;
        vr[d] = active ? vrow[FA_EL(d, half)] : 0.0f;
        dk[d] = 0.0f; dv[d] = 0.0f;
        Ks[jr][FA_EL(d, half)] = kr[d];   // inactive keys stay zero rows for the dQ product
    }
    int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;   // the dQ tile: rows ty*4 .. +4, columns tx*OC .. +OC

    int firstKey = blockIdx.x * FA_FKB;
    int iStart = p.causal ? max(0, firstKey - p.causalOffset) : 0;
    iStart = iStart / FA_QT * FA_QT;
    const float* qbase = q + b * p.qb + h * p.qh;
    const float* gbase = dO + b * p.ob + h * p.oh;
    float* dqbase = dQ + b * p.qb + h * p.qh;
    for (int i0 = iStart; i0 < p.Sq; i0 += FA_QT)
    {
        int rows = min(FA_QT, p.Sq - i0);
        for (int e = threadIdx.x; e < FA_QT * D; e += blockDim.x)
        {
            int ii = e / D, d = e % D;
            bool in = ii < rows;
            Qs[ii][d] = in ? qbase[(long long)(i0 + ii) * p.qr + d] * p.scale : 0.0f;
            Gs[ii][d] = in ? gbase[(long long)(i0 + ii) * p.orow + d] : 0.0f;
        }
        for (int e = threadIdx.x; e < FA_QT; e += blockDim.x)
        {
            Ls[e] = e < rows ? lse[(long long)bh * p.Sq + i0 + e] : -INFINITY;
            Ds[e] = e < rows ? delta[(long long)bh * p.Sq + i0 + e] : 0.0f;
        }
        __syncthreads();
        #pragma unroll 4
        for (int ii = 0; ii < FA_QT; ii++)
        {
            int i = i0 + ii;
            float sPart = 0.0f, dpPart = 0.0f;
            #pragma unroll
            for (int c4 = 0; c4 < HALF / 4; c4++)
            {
                float4 qv = *reinterpret_cast<const float4*>(&Qs[ii][8 * c4 + 4 * half]);
                float4 gv = *reinterpret_cast<const float4*>(&Gs[ii][8 * c4 + 4 * half]);
                sPart += qv.x * kr[4 * c4] + qv.y * kr[4 * c4 + 1] + qv.z * kr[4 * c4 + 2] + qv.w * kr[4 * c4 + 3];
                dpPart += gv.x * vr[4 * c4] + gv.y * vr[4 * c4 + 1] + gv.z * vr[4 * c4 + 2] + gv.w * vr[4 * c4 + 3];
            }
            float raw = sPart + __shfl_xor_sync(0xffffffffu, sPart, 1);
            float dp = dpPart + __shfl_xor_sync(0xffffffffu, dpPart, 1);
            float rowLse = Ls[ii];
            float ds = 0.0f;
            if (active && ii < rows && rowLse != -INFINITY && fa_allowed(p, mask, b, h, i, j))
            {
                float slope;
                float s = fa_cap(raw, p.softcap, &slope);
                float pr = __expf(s - rowLse);
                ds = pr * (dp - Ds[ii]) * slope;
                #pragma unroll
                for (int c4 = 0; c4 < HALF / 4; c4++)
                {
                    float4 qv = *reinterpret_cast<const float4*>(&Qs[ii][8 * c4 + 4 * half]);
                    float4 gv = *reinterpret_cast<const float4*>(&Gs[ii][8 * c4 + 4 * half]);
                    dv[4 * c4] += pr * gv.x; dv[4 * c4 + 1] += pr * gv.y; dv[4 * c4 + 2] += pr * gv.z; dv[4 * c4 + 3] += pr * gv.w;
                    dk[4 * c4] += ds * qv.x; dk[4 * c4 + 1] += ds * qv.y; dk[4 * c4 + 2] += ds * qv.z; dk[4 * c4 + 3] += ds * qv.w;
                }
            }
            if (half == 0) dSs[ii][jr] = ds;
        }
        __syncthreads();

        // dQ_tile = scale * dS K over this block's keys, added into dQ.
        float acc[4][OC];
        #pragma unroll
        for (int r = 0; r < 4; r++)
            #pragma unroll
            for (int u = 0; u < OC; u++) acc[r][u] = 0.0f;
        #pragma unroll 4
        for (int c = 0; c < FA_FKB; c++)
        {
            float kv[OC];
            #pragma unroll
            for (int u = 0; u < OC; u++) kv[u] = Ks[c][tx * OC + u];
            #pragma unroll
            for (int r = 0; r < 4; r++)
            {
                float sv = dSs[ty * 4 + r][c];
                #pragma unroll
                for (int u = 0; u < OC; u++) acc[r][u] += sv * kv[u];
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; r++)
        {
            int ii = ty * 4 + r;
            if (ii >= rows) continue;
            float* dqrow = dqbase + (long long)(i0 + ii) * p.qr;
            #pragma unroll
            for (int u = 0; u < OC; u++)
                if (acc[r][u] != 0.0f) atomicAdd(dqrow + tx * OC + u, acc[r][u] * p.scale);
        }
        __syncthreads();
    }
    if (active)
    {
        float* dkrow = dK + b * p.kb + h * p.kh + (long long)j * p.kr;
        float* dvrow = dV + b * p.vb + h * p.vh + (long long)j * p.vr;
        #pragma unroll
        for (int d = 0; d < HALF; d++) { dkrow[FA_EL(d, half)] = dk[d]; dvrow[FA_EL(d, half)] = dv[d]; }
    }
}

#define FA_INSTANTIATE(D) \
extern ""C"" __global__ __launch_bounds__(FA_BR * FA_SPLIT) void fa_forward_d##D(const float* __restrict__ q, \
    const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ o, float* __restrict__ lse, \
    const int* __restrict__ mask, FaParams p) { fa_forward<D>(q, k, v, o, lse, mask, p); } \
extern ""C"" __global__ __launch_bounds__(FA_KB * 2) void fa_backward_kv_d##D(const float* __restrict__ q, \
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ dO, \
    const float* __restrict__ lse, const float* __restrict__ delta, float* __restrict__ dK, float* __restrict__ dV, \
    const int* __restrict__ mask, FaParams p) { fa_backward_kv<D>(q, k, v, dO, lse, delta, dK, dV, mask, p); } \
extern ""C"" __global__ __launch_bounds__(FA_QB * 2) void fa_backward_q_d##D(const float* __restrict__ q, \
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ dO, \
    const float* __restrict__ lse, const float* __restrict__ delta, float* __restrict__ dQ, \
    const int* __restrict__ mask, FaParams p) { fa_backward_q<D>(q, k, v, dO, lse, delta, dQ, mask, p); } \
extern ""C"" __global__ __launch_bounds__(FA_FKB * 2) void fa_backward_fused_d##D(const float* __restrict__ q, \
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ dO, \
    const float* __restrict__ lse, const float* __restrict__ delta, float* __restrict__ dQ, float* __restrict__ dK, \
    float* __restrict__ dV, const int* __restrict__ mask, FaParams p) \
    { fa_backward_fused<D>(q, k, v, dO, lse, delta, dQ, dK, dV, mask, p); }

FA_INSTANTIATE(16)
FA_INSTANTIATE(32)
FA_INSTANTIATE(64)
";
}
