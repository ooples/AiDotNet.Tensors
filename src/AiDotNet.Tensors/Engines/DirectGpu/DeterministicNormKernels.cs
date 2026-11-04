// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The deterministic global-norm kernels (<see cref="Gpu.DeterministicGradientNorm"/> is the reference they match bit
/// for bit) in one C source for CUDA, HIP, OpenCL and Metal; a prelude per dialect maps the qualifiers. Everything that
/// decides a result bit runs in integers: no float add, divide or square root, so fast-math, flush-to-zero and
/// reduction order cannot change it.
/// </summary>
internal static class DeterministicNormKernels
{
    internal enum Dialect { Cuda, Hip, OpenCl, Metal }

    /// <summary>Threads per chunk reduction and per finish; part of the kernels (each thread takes 16 elements).</summary>
    internal const int GroupSize = 256;

    internal static readonly string[] CudaKernelNames =
        { "dgn_chunk_partials", "dgn_finish", "dgn_clip_scale", "dgn_multi_tensor_scale" };

    internal static readonly string[] PerTensorKernelNames =
        { "dgn_chunk_partials_tensor", "dgn_finish", "dgn_clip_scale", "dgn_scale_tensor" };

    internal static string Source(Dialect dialect) => Prelude(dialect) + Common + Entries(dialect);

    private static string Prelude(Dialect d) => d switch
    {
        Dialect.Cuda or Dialect.Hip => @"
#define U32 unsigned int
#define U64 unsigned long long
#define FN __device__ static __forceinline__
#define GPTR(T) T*
#define CGPTR(T) const T*
#define LPTR(T) T*
#define PPTR(T) T*
#define BARRIER __syncthreads();
#define CLZ32(x) __clz((int)(x))
#define AS_FLOAT(u) __uint_as_float(u)
#define AS_UINT(f) __float_as_uint(f)
#define IMAX(a, b) max((a), (b))
#define UMAX(a, b) max((a), (b))
#define UMIN(a, b) min((a), (b))
",
        Dialect.OpenCl => @"
#define U32 uint
#define U64 ulong
#define FN static inline
#define GPTR(T) __global T*
#define CGPTR(T) __global const T*
#define LPTR(T) __local T*
#define PPTR(T) T*
#define BARRIER barrier(CLK_LOCAL_MEM_FENCE);
#define CLZ32(x) clz((uint)(x))
#define AS_FLOAT(u) as_float(u)
#define AS_UINT(f) as_uint(f)
#define IMAX(a, b) max((int)(a), (int)(b))
#define UMAX(a, b) max((uint)(a), (uint)(b))
#define UMIN(a, b) min((uint)(a), (uint)(b))
",
        _ => @"
#include <metal_stdlib>
using namespace metal;
#define U32 uint
#define U64 ulong
#define FN static inline
#define GPTR(T) device T*
#define CGPTR(T) device const T*
#define LPTR(T) threadgroup T*
#define PPTR(T) thread T*
#define BARRIER threadgroup_barrier(mem_flags::mem_threadgroup);
#define CLZ32(x) clz((uint)(x))
#define AS_FLOAT(u) as_type<float>((uint)(u))
#define AS_UINT(f) as_type<uint>((float)(f))
#define IMAX(a, b) max((int)(a), (int)(b))
#define UMAX(a, b) max((uint)(a), (uint)(b))
#define UMIN(a, b) min((uint)(a), (uint)(b))
",
    };

    // A port of Gpu.DeterministicGradientNorm; keep the two in step.
    private const string Common = @"
#define DGN_CHUNK 4096
#define DGN_ONE ((U64)1)
#define DGN_MASK52 ((((U64)0xFFFFFu) << 32) | (U64)0xFFFFFFFFu)

FN int dgn_bitlen64(U64 x)
{
    U32 hi = (U32)(x >> 32), lo = (U32)x;
    return hi != 0u ? 64 - (int)CLZ32(hi) : (lo != 0u ? 32 - (int)CLZ32(lo) : 0);
}

FN int dgn_bitlen128(U64 lo, U64 hi) { return hi != 0 ? 64 + dgn_bitlen64(hi) : dgn_bitlen64(lo); }

FN void dgn_add128(PPTR(U64) lo, PPTR(U64) hi, U64 blo, U64 bhi)
{
    U64 s = *lo + blo;
    *hi = *hi + bhi + (s < *lo ? DGN_ONE : (U64)0);
    *lo = s;
}

FN void dgn_shr128(U64 lo, U64 hi, int sh, PPTR(U64) rlo, PPTR(U64) rhi)
{
    if (sh <= 0) { *rlo = lo; *rhi = hi; }
    else if (sh >= 128) { *rlo = 0; *rhi = 0; }
    else if (sh >= 64) { *rlo = hi >> (sh - 64); *rhi = 0; }
    else { *rlo = (lo >> sh) | (hi << (64 - sh)); *rhi = hi >> sh; }
}

FN int dgn_lowbits(U64 lo, U64 hi, int bits)
{
    if (bits <= 0) return 0;
    if (bits >= 128) return (lo | hi) != 0;
    if (bits >= 64) return lo != 0 || (bits > 64 && (hi & ((DGN_ONE << (bits - 64)) - DGN_ONE)) != 0);
    return (lo & ((DGN_ONE << bits) - DGN_ONE)) != 0;
}

FN int dgn_bitat(U64 lo, U64 hi, int bit) { return bit < 64 ? (int)((lo >> bit) & DGN_ONE) : (int)((hi >> (bit - 64)) & DGN_ONE); }

// m^2 * 2^(68 - 2(emax - e)) for the float with bits b, truncated (0 for NaN and infinity).
FN void dgn_square(U32 b, U32 emax, PPTR(U64) lo, PPTR(U64) hi)
{
    U32 e = (b >> 23) & 0xFFu;
    *lo = 0; *hi = 0;
    if (e == 0xFFu) return;
    U64 m = (U64)((b & 0x7FFFFFu) | (e != 0u ? 0x800000u : 0u));
    if (e == 0u) e = 1u;
    U64 sq = m * m;
    int sh = 68 - 2 * (int)(emax - e);
    if (sh >= 64) *hi = sq << (sh - 64);
    else if (sh > 0) { *lo = sq << sh; *hi = sq >> (64 - sh); }
    else if (sh > -48) *lo = sq >> (-sh);
}

FN U64 dgn_isqrt(U64 a)
{
    U64 r = 0, bit = DGN_ONE << 62;
    while (bit > a) bit >>= 2;
    while (bit != 0)
    {
        if (a >= r + bit) { a -= r + bit; r = (r >> 1) + bit; }
        else r >>= 1;
        bit >>= 2;
    }
    return r;
}

FN U64 dgn_roundshift(U64 m, int sh, int sticky)
{
    if (sh > 64) return 0;
    U64 q = sh == 64 ? (U64)0 : (m >> sh);
    int roundBit = (int)((m >> (sh - 1)) & DGN_ONE);
    int rest = sticky || (sh > 1 && (m & (~(U64)0 >> (65 - sh))) != 0);
    if (roundBit && (rest || (q & DGN_ONE) != 0)) q++;
    return q;
}

FN U32 dgn_round_to_float(U64 m, int e, int sticky)
{
    if (m == 0) return 0u;
    int lead = e + dgn_bitlen64(m) - 1;
    int ulp = IMAX(lead - 23, -149);
    int sh = ulp - e;
    U64 q = sh > 0 ? dgn_roundshift(m, sh, sticky) : (m << (-sh));
    if (q >= (DGN_ONE << 24)) { q >>= 1; ulp++; }
    if (q < (DGN_ONE << 23)) return (U32)q;
    int biased = ulp + 23 + 127;
    if (biased >= 255) return 0x7F800000u;
    return ((U32)biased << 23) | (U32)(q & (U64)0x7FFFFFu);
}

FN void dgn_round24(U64 m, int e, int sticky, PPTR(U64) rm, PPTR(int) re)
{
    if (m == 0) { *rm = 0; *re = 0; return; }
    int len = dgn_bitlen64(m);
    if (len <= 24) { *rm = m << (24 - len); *re = e - (24 - len); return; }
    int sh = len - 24;
    U64 q = dgn_roundshift(m, sh, sticky);
    int ee = e + sh;
    if (q >= (DGN_ONE << 24)) { q >>= 1; ee++; }
    *rm = q; *re = ee;
}

FN void dgn_addround24(U64 am, int ae, U64 bm, int be, PPTR(U64) rm, PPTR(int) re)
{
    if (am == 0) { *rm = bm; *re = be; return; }
    if (bm == 0) { *rm = am; *re = ae; return; }
    if (ae < be) { U64 tm = am; am = bm; bm = tm; int te = ae; ae = be; be = te; }
    int d = ae - be;
    if (d > 30) { dgn_round24((am << 3) | DGN_ONE, ae - 3, 0, rm, re); return; }
    dgn_round24((am << d) + bm, be, 0, rm, re);
}

FN void dgn_sqrt(U64 alo, U64 ahi, int v, PPTR(U64) m, PPTR(int) e, PPTR(int) sticky)
{
    int len = dgn_bitlen128(alo, ahi);
    if (len == 0) { *m = 0; *e = 0; *sticky = 0; return; }
    int k = len - 54;
    if (((v + k) & 1) != 0) k++;
    U64 a, h;
    int dropped = 0;
    if (k >= 0) { dgn_shr128(alo, ahi, k, &a, &h); dropped = k > 0 && dgn_lowbits(alo, ahi, k); }
    else a = alo << (-k);
    U64 r = dgn_isqrt(a);
    *m = r; *e = (v + k) / 2; *sticky = dropped || r * r != a;
}

// min(1, fp32(maxNorm / fp32(fp32(norm) + 1e-6f))) as fp32 bits, unbounded exponent.
FN U32 dgn_clip_coef(U64 nm, int ne, int ns, U32 mb)
{
    if (mb == 0x7F800000u) return 0x3F800000u;
    U64 nM, dM; int nE, dE;
    dgn_round24(nm, ne, ns, &nM, &nE);
    dgn_addround24(nM, nE, (U64)0x8637BDu, 107 - 150, &dM, &dE);
    U32 me = (mb >> 23) & 0xFFu;
    U64 mM = (U64)((mb & 0x7FFFFFu) | (me != 0u ? 0x800000u : 0u));
    int mE = (me != 0u ? (int)me : 1) - 150;
    if (mM == 0) return 0u;
    while (mM < (U64)0x800000u) { mM <<= 1; mE--; }
    if (mE > dE || (mE == dE && mM >= dM)) return 0x3F800000u;
    U64 num = mM << 40;
    U64 q = num / dM, rem = num - q * dM;
    return dgn_round_to_float(q, mE - 40 - dE, rem != 0);
}

FN void dgn_to_double(U64 alo, U64 ahi, int v, PPTR(U32) wlo, PPTR(U32) whi)
{
    int len = dgn_bitlen128(alo, ahi);
    U64 bits = 0;
    if (len != 0)
    {
        U64 m; int e;
        if (len > 53)
        {
            int sh = len - 53;
            U64 top, h;
            dgn_shr128(alo, ahi, sh, &top, &h);
            int roundBit = dgn_bitat(alo, ahi, sh - 1);
            int rest = sh > 1 && dgn_lowbits(alo, ahi, sh - 1);
            m = top;
            if (roundBit && (rest || (m & DGN_ONE) != 0)) m++;
            e = v + sh;
            if (m == (DGN_ONE << 53)) { m >>= 1; e++; }
        }
        else { m = alo << (53 - len); e = v - (53 - len); }
        int lead = e + 52;
        bits = lead >= -1022 ? ((((U64)(lead + 1023)) << 52) | (m & DGN_MASK52)) : (m >> (-1022 - lead));
    }
    *wlo = (U32)bits; *whi = (U32)(bits >> 32);
}

// g * s, one IEEE multiply rounded to nearest even, with subnormal operands and results kept (no flush to zero).
FN U32 dgn_mul(U32 g, U32 s)
{
    if (s == 0x3F800000u) return g;   // unscaled: every element, NaN payloads included, keeps its bits
    U32 eg = (g >> 23) & 0xFFu, es = (s >> 23) & 0xFFu;
    if (eg - 1u < 254u && es - 1u < 254u && eg + es >= 128u) return AS_UINT(AS_FLOAT(g) * AS_FLOAT(s));
    U32 sign = (g ^ s) & 0x80000000u;
    U32 fg = g & 0x7FFFFFu, fs = s & 0x7FFFFFu;
    if ((eg == 0xFFu && fg != 0u) || (es == 0xFFu && fs != 0u)) return 0x7FC00000u;
    int gzero = eg == 0u && fg == 0u, szero = es == 0u && fs == 0u;
    if (eg == 0xFFu || es == 0xFFu) return (gzero || szero) ? 0x7FC00000u : (sign | 0x7F800000u);
    if (gzero || szero) return sign;
    U64 prod = (U64)(fg | (eg != 0u ? 0x800000u : 0u)) * (U64)(fs | (es != 0u ? 0x800000u : 0u));
    int e = (eg != 0u ? (int)eg : 1) + (es != 0u ? (int)es : 1) - 300;
    return dgn_round_to_float(prod, e, 0) | sign;
}

// One chunk's partial (8 words) from elements [start, end) of x; every thread of the 256-thread group calls it.
FN void dgn_chunk(CGPTR(U32) x, int start, int end, U32 tensor, GPTR(U32) outp, int tid,
                  LPTR(U32) su, LPTR(U64) slo, LPTR(U64) shi)
{
    U32 b[16];
    U32 emax = 1u, nan = 0u, inf = 0u;
    for (int j = 0; j < 16; j++)
    {
        int i = start + j * 256 + tid;
        U32 v = i < end ? x[i] : 0u;
        b[j] = v;
        U32 e = (v >> 23) & 0xFFu;
        if (e == 0xFFu) { if ((v & 0x7FFFFFu) != 0u) nan++; else inf++; }
        else if (e > emax) emax = e;
    }
    su[tid] = emax;
    BARRIER
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) su[tid] = UMAX(su[tid], su[tid + s]); BARRIER }
    emax = su[0];
    BARRIER
    su[tid] = (nan << 16) | inf;
    BARRIER
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) su[tid] = su[tid] + su[tid + s]; BARRIER }
    U32 counts = su[0];
    U64 lo = 0, hi = 0;
    for (int j = 0; j < 16; j++)
    {
        U64 a, h;
        dgn_square(b[j], emax, &a, &h);
        dgn_add128(&lo, &hi, a, h);
    }
    slo[tid] = lo; shi[tid] = hi;
    BARRIER
    for (int s = 128; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            U64 l = slo[tid], h = shi[tid];
            dgn_add128(&l, &h, slo[tid + s], shi[tid + s]);
            slo[tid] = l; shi[tid] = h;
        }
        BARRIER
    }
    if (tid == 0)
    {
        outp[0] = emax; outp[1] = counts >> 16; outp[2] = counts & 0xFFFFu; outp[3] = tensor;
        outp[4] = (U32)slo[0]; outp[5] = (U32)(slo[0] >> 32); outp[6] = (U32)shi[0]; outp[7] = (U32)(shi[0] >> 32);
    }
    BARRIER
}

// Combines `count` partials into the record's sum, counts and norm (one 256-thread group).
FN void dgn_finish_body(CGPTR(U32) p, int count, int headroom, GPTR(U32) rec, int tid,
                        LPTR(U32) su, LPTR(U32) su2, LPTR(U64) slo, LPTR(U64) shi)
{
    U32 g = 0u, first = 0xFFFFFFFFu;
    U64 nan = 0, inf = 0;
    for (int i = tid; i < count; i += 256)
    {
        CGPTR(U32) q = p + (U64)i * 8u;
        if ((q[4] | q[5] | q[6] | q[7]) != 0u && q[0] > g) g = q[0];
        nan += q[1]; inf += q[2];
        if ((q[1] | q[2]) != 0u && q[3] < first) first = q[3];
    }
    su[tid] = g; su2[tid] = first; slo[tid] = nan; shi[tid] = inf;
    BARRIER
    for (int s = 128; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            su[tid] = UMAX(su[tid], su[tid + s]); su2[tid] = UMIN(su2[tid], su2[tid + s]);
            slo[tid] = slo[tid] + slo[tid + s]; shi[tid] = shi[tid] + shi[tid + s];
        }
        BARRIER
    }
    g = su[0]; first = su2[0]; nan = slo[0]; inf = shi[0];
    BARRIER
    U64 lo = 0, hi = 0;
    if (g != 0u)
        for (int i = tid; i < count; i += 256)
        {
            CGPTR(U32) q = p + (U64)i * 8u;
            if ((q[4] | q[5] | q[6] | q[7]) == 0u) continue;
            U64 a, h;
            dgn_shr128((U64)q[4] | ((U64)q[5] << 32), (U64)q[6] | ((U64)q[7] << 32), 2 * (int)(g - q[0]) + headroom, &a, &h);
            dgn_add128(&lo, &hi, a, h);
        }
    slo[tid] = lo; shi[tid] = hi;
    BARRIER
    for (int s = 128; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            U64 l = slo[tid], h = shi[tid];
            dgn_add128(&l, &h, slo[tid + s], shi[tid + s]);
            slo[tid] = l; shi[tid] = h;
        }
        BARRIER
    }
    if (tid == 0)
    {
        lo = slo[0]; hi = shi[0];
        int v = g != 0u ? 2 * (int)g - 368 + headroom : 0;
        U32 nanS = nan > (U64)0xFFFFFFFFu ? 0xFFFFFFFFu : (U32)nan;
        U32 infS = inf > (U64)0xFFFFFFFFu ? 0xFFFFFFFFu : (U32)inf;
        rec[2] = (U32)lo; rec[3] = (U32)(lo >> 32); rec[4] = (U32)hi; rec[5] = (U32)(hi >> 32);
        rec[6] = (U32)v; rec[7] = nanS; rec[8] = infS; rec[9] = first;
        U32 wlo = 0u, whi = 0u, norm;
        if (nanS != 0u) { whi = 0x7FF80000u; norm = 0x7FC00000u; }
        else if (infS != 0u) { whi = 0x7FF00000u; norm = 0x7F800000u; }
        else
        {
            dgn_to_double(lo, hi, v, &wlo, &whi);
            U64 nm; int ne, ns;
            dgn_sqrt(lo, hi, v, &nm, &ne, &ns);
            norm = dgn_round_to_float(nm, ne, ns);
        }
        rec[0] = wlo; rec[1] = whi; rec[10] = norm; rec[11] = 0x3F800000u;
        rec[12] = (nanS | infS) != 0u ? 1u : 0u; rec[13] = 0u; rec[14] = 0u; rec[15] = 0u;
    }
}

// The clip's element update: NonFiniteGradientPolicy.ZeroNonFinite (zeroNonFinite != 0) writes +0 for a NaN or
// infinite element; every other element is dgn_mul(g, s).
FN U32 dgn_scale_elem(U32 g, U32 s, U32 zeroNonFinite)
{
    return (zeroNonFinite != 0u && ((g >> 23) & 0xFFu) == 0xFFu) ? 0u : dgn_mul(g, s);
}

// policy (NonFiniteGradientPolicy): 0 leaves the gradients unscaled on a non-finite element, 1 propagates as PyTorch
// (NaN, or 0 for +Inf), 2 clips by the norm of the finite elements (the sum already excludes the non-finite ones; the
// scale pass zeroes those). The host passes 0 for 3 (Error), which it raises after reading the record.
FN void dgn_clip_body(GPTR(U32) rec, U32 maxNormBits, U32 policy, GPTR(U32) scale)
{
    U32 s;
    if (rec[7] != 0u && policy != 2u) s = policy == 1u ? 0x7FC00000u : 0x3F800000u;
    else if (rec[8] != 0u && policy != 2u) s = policy == 1u ? 0u : 0x3F800000u;
    else
    {
        U64 nm; int ne, ns;
        dgn_sqrt((U64)rec[2] | ((U64)rec[3] << 32), (U64)rec[4] | ((U64)rec[5] << 32), (int)rec[6], &nm, &ne, &ns);
        s = dgn_clip_coef(nm, ne, ns, maxNormBits);
    }
    rec[11] = s; rec[13] = maxNormBits;
    scale[0] = s;
}
";

    private static string Entries(Dialect d) => d switch
    {
        Dialect.Cuda or Dialect.Hip => @"
extern ""C"" __global__ __launch_bounds__(256) void dgn_chunk_partials(
    const unsigned long long* __restrict__ ptrs, const int* __restrict__ sizes, const int* __restrict__ chunkTensor,
    const int* __restrict__ chunkStart, int totalChunks, unsigned int* __restrict__ partials)
{
    __shared__ U32 su[256];
    __shared__ U64 slo[256], shi[256];
    for (int c = blockIdx.x; c < totalChunks; c += gridDim.x)
    {
        int t = chunkTensor[c], start = chunkStart[c];
        int end = min(start + DGN_CHUNK, sizes[t]);
        dgn_chunk((const U32*)ptrs[t], start, end, (U32)t, partials + (size_t)c * 8u, (int)threadIdx.x, su, slo, shi);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void dgn_finish(
    const unsigned int* __restrict__ partials, int count, int headroom, unsigned int* __restrict__ record)
{
    __shared__ U32 su[256], su2[256];
    __shared__ U64 slo[256], shi[256];
    dgn_finish_body(partials, count, headroom, record, (int)threadIdx.x, su, su2, slo, shi);
}

extern ""C"" __global__ void dgn_clip_scale(unsigned int* __restrict__ record, unsigned int maxNormBits, unsigned int policy,
    unsigned int* __restrict__ scale)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) dgn_clip_body(record, maxNormBits, policy, scale);
}

extern ""C"" __global__ __launch_bounds__(256) void dgn_multi_tensor_scale(
    const unsigned long long* __restrict__ ptrs, const int* __restrict__ sizes, const int* __restrict__ chunkTensor,
    const int* __restrict__ chunkStart, int totalChunks, const unsigned int* __restrict__ scale, unsigned int zeroNonFinite)
{
    U32 s = scale[0];
    for (int c = blockIdx.x; c < totalChunks; c += gridDim.x)
    {
        int t = chunkTensor[c], start = chunkStart[c];
        int end = min(start + DGN_CHUNK, sizes[t]);
        U32* p = (U32*)ptrs[t];
        for (int i = start + (int)threadIdx.x; i < end; i += blockDim.x) p[i] = dgn_scale_elem(p[i], s, zeroNonFinite);
    }
}
",
        Dialect.OpenCl => @"
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void dgn_chunk_partials_tensor(
    __global const uint* x, const int n, const uint tensor, const int partialBase, __global uint* partials)
{
    __local uint su[256];
    __local ulong slo[256], shi[256];
    int c = (int)get_group_id(0);
    int start = c * DGN_CHUNK;
    dgn_chunk(x, start, min(start + DGN_CHUNK, n), tensor, partials + (ulong)(partialBase + c) * 8u, (int)get_local_id(0), su, slo, shi);
}

__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void dgn_finish(
    __global const uint* partials, const int count, const int headroom, __global uint* record)
{
    __local uint su[256], su2[256];
    __local ulong slo[256], shi[256];
    dgn_finish_body(partials, count, headroom, record, (int)get_local_id(0), su, su2, slo, shi);
}

__kernel void dgn_clip_scale(__global uint* record, const uint maxNormBits, const uint policy, __global uint* scale)
{
    if (get_global_id(0) == 0) dgn_clip_body(record, maxNormBits, policy, scale);
}

__kernel void dgn_scale_tensor(__global uint* x, const int n, __global const uint* scale, const uint zeroNonFinite)
{
    int i = (int)get_global_id(0);
    if (i < n) x[i] = dgn_scale_elem(x[i], scale[0], zeroNonFinite);
}
",
        _ => @"
kernel void dgn_chunk_partials_tensor(
    device const uint* x [[buffer(0)]], device uint* partials [[buffer(1)]], constant uint& n [[buffer(2)]],
    constant uint& tensor [[buffer(3)]], constant uint& partialBase [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint c [[threadgroup_position_in_grid]])
{
    threadgroup uint su[256];
    threadgroup ulong slo[256], shi[256];
    int start = (int)c * DGN_CHUNK;
    dgn_chunk(x, start, min(start + DGN_CHUNK, (int)n), tensor, partials + (ulong)(partialBase + c) * 8u, (int)tid, su, slo, shi);
}

kernel void dgn_finish(
    device const uint* partials [[buffer(0)]], device uint* record [[buffer(1)]], constant uint& count [[buffer(2)]],
    constant uint& headroom [[buffer(3)]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup uint su[256], su2[256];
    threadgroup ulong slo[256], shi[256];
    dgn_finish_body(partials, (int)count, (int)headroom, record, (int)tid, su, su2, slo, shi);
}

kernel void dgn_clip_scale(device uint* record [[buffer(0)]], device uint* scale [[buffer(1)]],
    constant uint& maxNormBits [[buffer(2)]], constant uint& policy [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid == 0) dgn_clip_body(record, maxNormBits, policy, scale);
}

kernel void dgn_scale_tensor(device uint* x [[buffer(0)]], device const uint* scale [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant uint& zeroNonFinite [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = dgn_scale_elem(x[gid], scale[0], zeroNonFinite);
}
",
    };
}
