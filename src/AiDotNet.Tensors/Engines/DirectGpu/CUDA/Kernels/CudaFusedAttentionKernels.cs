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
/// </summary>
internal static class CudaFusedAttentionKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "fa_forward_d16", "fa_forward_d32", "fa_forward_d64",
        "fa_backward_kv_d16", "fa_backward_kv_d32", "fa_backward_kv_d64",
        "fa_backward_q_d16", "fa_backward_q_d32", "fa_backward_q_d64",
        "fa_backward_delta",
    };

    public static string GetSource() => @"
#define FA_BR 32        // query rows per forward block (4 threads each: 128 threads)
#define FA_BC 64        // keys per shared-memory tile (backward-q pass)
#define FA_FBC 64       // keys per shared-memory tile (forward)
#define FA_FWD_TILED 1  // forward as register-tiled GEMMs (S = Q K^T and O += P V per 4x4 thread tile)
#define FA_SPLIT 4      // threads per query row in the forward
#define FA_KB 128       // key rows per backward-kv block (2 threads each: 256 threads)
#define FA_QB 128       // query rows per backward-q block (2 threads each: 256 threads)
#define FA_QT 32        // query rows per shared-memory tile in the backward-kv pass
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
// Forward, register-tiled: grid (ceil(Sq / FA_BR), B * H), block 128 threads as 8 row groups (ty) x 16 lanes (tx).
// Per 64-key tile, thread (ty, tx) computes the 4 x 4 scores S[ty*4 .. +4][tx*4 .. +4] as outer products over the head
// dimension from Q^T (pre-scaled) and K^T in shared memory -- 16 FMAs per two float4 reads -- then updates the online
// softmax of its 4 rows (row max / sum over the 16 lanes of the row group by xor butterflies, identical in every lane)
// and accumulates O[ty*4 .. +4][tx*(D/16) .. +D/16] += P V with P staged in shared memory. K^T and V share one
// buffer in turn.
// ---------------------------------------------------------------------------------------------------------------
#define FA_KT(d, j) KV[(d) * (FA_FBC + 4) + (j)]
#define FA_VS(j, d) KV[(j) * (D + 4) + (d)]
template <int D>
__device__ void fa_forward_tiled(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ o, float* __restrict__ lse, const int* __restrict__ mask, FaParams p)
{
    constexpr int OC = D / 16;   // output columns per thread
    constexpr int KVN = (D * (FA_FBC + 4) > FA_FBC * (D + 4)) ? D * (FA_FBC + 4) : FA_FBC * (D + 4);
    __shared__ __align__(16) float Qt[D][FA_BR + 4];
    __shared__ __align__(16) float KV[KVN];
    __shared__ __align__(16) float Ps[FA_BR][FA_FBC + 4];
    int bh = blockIdx.y, b = bh / p.H, h = bh % p.H;
    int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    // Causal: the last query blocks see the most keys, so they are launched first (shorter tail).
    int qBlock = p.causal ? (int)(gridDim.x - 1 - blockIdx.x) : (int)blockIdx.x;
    int row0 = qBlock * FA_BR;

    const float* qbase = q + b * p.qb + h * p.qh;
    for (int e = threadIdx.x; e < FA_BR * (D / 4); e += blockDim.x)
    {
        int r = e % FA_BR, d4 = e / FA_BR;
        float4 qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (row0 + r < p.Sq) qv = *reinterpret_cast<const float4*>(qbase + (long long)(row0 + r) * p.qr + 4 * d4);
        Qt[4 * d4][r] = qv.x * p.scale; Qt[4 * d4 + 1][r] = qv.y * p.scale;
        Qt[4 * d4 + 2][r] = qv.z * p.scale; Qt[4 * d4 + 3][r] = qv.w * p.scale;
    }

    float acc[4][OC], m[4], l[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        m[i] = -INFINITY; l[i] = 0.0f;
        #pragma unroll
        for (int u = 0; u < OC; u++) acc[i][u] = 0.0f;
    }

    int lastRow = min(p.Sq - 1, row0 + FA_BR - 1);
    int keyEnd = p.causal ? min(p.Sk, lastRow + p.causalOffset + 1) : p.Sk;
    const float* kbase = k + b * p.kb + h * p.kh;
    const float* vbase = v + b * p.vb + h * p.vh;
    for (int j0 = 0; j0 < keyEnd; j0 += FA_FBC)
    {
        int cols = min(FA_FBC, keyEnd - j0);
        // K^T: consecutive threads take consecutive keys of one 4-wide chunk (conflict-free transposed stores).
        for (int e = threadIdx.x; e < FA_FBC * (D / 4); e += blockDim.x)
        {
            int jj = e % FA_FBC, d4 = e / FA_FBC;
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (jj < cols) kv = *reinterpret_cast<const float4*>(kbase + (long long)(j0 + jj) * p.kr + 4 * d4);
            FA_KT(4 * d4, jj) = kv.x; FA_KT(4 * d4 + 1, jj) = kv.y;
            FA_KT(4 * d4 + 2, jj) = kv.z; FA_KT(4 * d4 + 3, jj) = kv.w;
        }
        __syncthreads();

        float s[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int jx = 0; jx < 4; jx++) s[i][jx] = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < D; d++)
        {
            float4 a = *reinterpret_cast<const float4*>(&Qt[d][ty * 4]);
            float4 kk = *reinterpret_cast<const float4*>(&FA_KT(d, tx * 4));
            float av[4] = { a.x, a.y, a.z, a.w }, kvv[4] = { kk.x, kk.y, kk.z, kk.w };
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int jx = 0; jx < 4; jx++) s[i][jx] += av[i] * kvv[jx];
        }

        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            int row = row0 + ty * 4 + i;
            bool rowActive = row < p.Sq;
            float rmax = -INFINITY;
            #pragma unroll
            for (int jx = 0; jx < 4; jx++)
            {
                int jj = tx * 4 + jx;
                float sv = -INFINITY;
                if (rowActive && jj < cols && fa_allowed(p, mask, b, h, row, j0 + jj))
                {
                    float slope;
                    sv = fa_cap(s[i][jx], p.softcap, &slope);
                }
                s[i][jx] = sv;
                rmax = fmaxf(rmax, sv);
            }
            #pragma unroll
            for (int off = 1; off < 16; off <<= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, off));
            float mNew = fmaxf(m[i], rmax);
            float pr[4], rsum = 0.0f;
            #pragma unroll
            for (int jx = 0; jx < 4; jx++)
            {
                pr[jx] = s[i][jx] == -INFINITY ? 0.0f : __expf(s[i][jx] - mNew);
                rsum += pr[jx];
            }
            #pragma unroll
            for (int off = 1; off < 16; off <<= 1) rsum += __shfl_xor_sync(0xffffffffu, rsum, off);
            if (rmax != -INFINITY)
            {
                float alpha = m[i] == -INFINITY ? 0.0f : __expf(m[i] - mNew);
                l[i] = l[i] * alpha + rsum;
                #pragma unroll
                for (int u = 0; u < OC; u++) acc[i][u] *= alpha;
                m[i] = mNew;
            }
            *reinterpret_cast<float4*>(&Ps[ty * 4 + i][tx * 4]) = make_float4(pr[0], pr[1], pr[2], pr[3]);
        }
        __syncthreads();   // P staged; every thread is done with K^T

        for (int e = threadIdx.x; e < FA_FBC * (D / 4); e += blockDim.x)
        {
            int jj = e / (D / 4), d4 = e % (D / 4);
            float4 vv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);   // zero past the tile: 0 * garbage could be NaN
            if (jj < cols) vv = *reinterpret_cast<const float4*>(vbase + (long long)(j0 + jj) * p.vr + 4 * d4);
            *reinterpret_cast<float4*>(&FA_VS(jj, 4 * d4)) = vv;
        }
        __syncthreads();

        int cEnd = (cols + 3) & ~3;
        for (int c4 = 0; c4 < cEnd; c4 += 4)
        {
            float pv[4][4];
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float4 t = *reinterpret_cast<const float4*>(&Ps[ty * 4 + i][c4]);
                pv[i][0] = t.x; pv[i][1] = t.y; pv[i][2] = t.z; pv[i][3] = t.w;
            }
            #pragma unroll
            for (int cc = 0; cc < 4; cc++)
            {
                float vr[OC];
                #pragma unroll
                for (int u = 0; u < OC; u++) vr[u] = FA_VS(c4 + cc, tx * OC + u);
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    #pragma unroll
                    for (int u = 0; u < OC; u++) acc[i][u] += pv[i][cc] * vr[u];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int row = row0 + ty * 4 + i;
        if (row >= p.Sq) continue;
        float* orow = o + b * p.ob + h * p.oh + (long long)row * p.orow;
        float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
        #pragma unroll
        for (int u = 0; u < OC; u++) orow[tx * OC + u] = acc[i][u] * inv;
        if (tx == 0) lse[(long long)bh * p.Sq + row] = l[i] > 0.0f ? m[i] + __logf(l[i]) : -INFINITY;
    }
}
#undef FA_KT
#undef FA_VS

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

#if FA_FWD_TILED
#define FA_FORWARD_IMPL fa_forward_tiled
#else
#define FA_FORWARD_IMPL fa_forward
#endif
#define FA_INSTANTIATE(D) \
extern ""C"" __global__ __launch_bounds__(FA_BR * FA_SPLIT) void fa_forward_d##D(const float* __restrict__ q, \
    const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ o, float* __restrict__ lse, \
    const int* __restrict__ mask, FaParams p) { FA_FORWARD_IMPL<D>(q, k, v, o, lse, mask, p); } \
extern ""C"" __global__ __launch_bounds__(FA_KB * 2) void fa_backward_kv_d##D(const float* __restrict__ q, \
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ dO, \
    const float* __restrict__ lse, const float* __restrict__ delta, float* __restrict__ dK, float* __restrict__ dV, \
    const int* __restrict__ mask, FaParams p) { fa_backward_kv<D>(q, k, v, dO, lse, delta, dK, dV, mask, p); } \
extern ""C"" __global__ __launch_bounds__(FA_QB * 2) void fa_backward_q_d##D(const float* __restrict__ q, \
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ dO, \
    const float* __restrict__ lse, const float* __restrict__ delta, float* __restrict__ dQ, \
    const int* __restrict__ mask, FaParams p) { fa_backward_q<D>(q, k, v, dO, lse, delta, dQ, mask, p); }

FA_INSTANTIATE(16)
FA_INSTANTIATE(32)
FA_INSTANTIATE(64)
";
}
