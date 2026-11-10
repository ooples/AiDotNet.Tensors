// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// The global gradient-norm clip's arithmetic, defined in integers so that every backend (CPU, CUDA, HIP, OpenCL,
/// Metal, Vulkan, WebGPU) produces the same bits: the sum of squares, the norm, and the clip coefficient do not
/// depend on the device, the launch shape, scheduling, fast-math, or flush-to-zero. This class is the reference
/// every kernel is tested against.
/// </summary>
/// <remarks>
/// <para><b>Sum of squares.</b> Every tensor is cut into chunks of <see cref="Chunk"/> elements (a chunk never spans
/// two tensors). In a chunk, <c>emax</c> is the largest biased exponent of a finite element, with subnormals and
/// zeros counted as exponent 1. A finite element with biased exponent <c>E</c> (<c>e = max(E, 1)</c>) and integer
/// significand <c>m &lt; 2^24</c> has the exact square <c>m^2 * 2^(2e - 300)</c>. The chunk accumulates
/// <c>m^2 * 2^(68 - 2(emax - e))</c> in a 128-bit integer, truncating bits below the chunk's unit
/// <c>2^(2 emax - 368)</c>: the largest square lands below bit 116, so 4096 of them cannot overflow, and each
/// element is truncated on its own, which no summation order changes. The chunk partials then combine the same way:
/// with <c>G</c> the largest <c>emax</c> of a nonzero partial and <c>H = ceil(log2(partials))</c>, each partial is
/// shifted right by <c>2(G - emax) + H</c> and the shifted partials are summed exactly in 128 bits, giving
/// <c>S = A * 2^V</c>, <c>V = 2G - 368 + H</c>. Integer addition is associative, so any reduction tree gives the
/// same <c>A</c>. The truncation error is below 2^(2H - 115) of the largest square, far below an fp32 rounding,
/// and no finite input can overflow.</para>
/// <para><b>Norm and coefficient.</b> Computed from <c>(A, V)</c> in integers with round-to-nearest-even and an
/// unbounded exponent, following PyTorch's <c>clip_grad_norm_</c> sequence: <c>n = fp32(sqrt(S))</c>,
/// <c>d = fp32(n + 1e-6f)</c>, <c>coef = fp32(maxNorm / d)</c>, <c>scale = min(1, coef)</c>, each rounding as an
/// IEEE fp32 operation would, but with no overflow: a norm past <c>float.MaxValue</c> still yields its exact
/// coefficient (which may be subnormal). A non-finite element makes the norm NaN (any NaN) or +Inf, and the
/// coefficient NaN or 0 as PyTorch's would be under <see cref="NonFiniteGradientPolicy.Propagate"/> (1 under
/// <see cref="NonFiniteGradientPolicy.LeaveUnscaled"/>); the record counts them either way.</para>
/// <para><b>Scaling.</b> <c>g * scale</c> is one IEEE multiply. Kernels compute it in integers whenever an operand
/// or the product is subnormal, so devices that flush subnormals still match.</para>
/// </remarks>
internal static class DeterministicGradientNorm
{
    /// <summary>Elements per chunk. Part of the definition: changing it changes the truncation grid.</summary>
    public const int Chunk = 4096;

    /// <summary>uint words per chunk partial: emax, nan count, inf count, tensor index, A (4 words, low first).</summary>
    public const int PartialWords = 8;

    /// <summary>uint words in a norm record.</summary>
    public const int RecordWords = 16;

    // Record layout, in uint words. Words 0-1 keep the double sum of squares where every existing reader expects it.
    public const int WordSumSquaresLo = 0, WordSumSquaresHi = 1;
    public const int WordA0 = 2;                 // A, words 2..5, low first
    public const int WordV = 6;                  // int
    public const int WordNanCount = 7, WordInfCount = 8;
    public const int WordFirstNonFiniteTensor = 9; // uint.MaxValue when every element is finite
    public const int WordNorm = 10;              // fp32 bits of the norm (+Inf past float range, NaN, ...)
    public const int WordScale = 11;             // fp32 bits of min(1, coef), the factor the gradients are scaled by
    public const int WordNonFinite = 12;         // 1 when some element is NaN or infinite
    public const int WordMaxNorm = 13;           // fp32 bits of the bound the scale was computed for

    /// <summary>One chunk's partial.</summary>
    public struct Partial
    {
        public uint Emax, NanCount, InfCount, Tensor;
        public ulong Lo, Hi;
    }

    /// <summary>The partials of every chunk of every tensor, in tensor order.</summary>
    public static Partial[] Partials(IReadOnlyList<float[]> tensors)
    {
        if (tensors is null) throw new ArgumentNullException(nameof(tensors));
        var list = new List<Partial>();
        for (int t = 0; t < tensors.Count; t++)
        {
            var data = tensors[t];
            for (int start = 0; start < data.Length; start += Chunk)
                list.Add(ChunkPartial(new ReadOnlySpan<float>(data, start, Math.Min(Chunk, data.Length - start)), (uint)t));
        }
        return list.ToArray();
    }

    /// <summary>Number of chunks of a tensor of <paramref name="length"/> elements.</summary>
    public static int ChunkCount(int length) => (int)(((long)length + Chunk - 1) / Chunk);

    /// <summary>The combine's headroom bits, ceil(log2(partials)).</summary>
    public static int Headroom(int partials)
    {
        int h = 0;
        while ((1L << h) < partials) h++;
        return h;
    }

    /// <summary>The partials of one host tensor's chunks into <paramref name="dest"/> from <paramref name="offset"/>,
    /// over the thread pool when the tensor is large (chunks are independent; their order changes no bit).</summary>
    public static unsafe void ChunkPartials(float* data, int length, uint tensor, Partial[] dest, int offset)
    {
        int chunks = ChunkCount(length);
        if (chunks <= 8)
        {
            for (int c = 0; c < chunks; c++)
                dest[offset + c] = ChunkPartial(new ReadOnlySpan<float>(data + (long)c * Chunk, Math.Min(Chunk, length - c * Chunk)), tensor);
            return;
        }
        var p = (IntPtr)data;
        AiDotNet.Tensors.Helpers.CpuParallelSettings.LightweightParallel(chunks, c =>
        {
            var basePtr = (float*)p;
            dest[offset + c] = ChunkPartial(new ReadOnlySpan<float>(basePtr + (long)c * Chunk, Math.Min(Chunk, length - c * Chunk)), tensor);
        });
    }

    /// <summary>Multiplies <paramref name="data"/> in place by the record's scale (<see cref="Mul"/>); a scale of
    /// exactly 1 leaves it untouched.</summary>
    public static unsafe void Scale(float* data, int length, uint scale, bool zeroNonFinite = false)
    {
        if (scale == 0x3F800000u && !zeroNonFinite) return;
        var u = (uint*)data;
        if (length < 65536)
        {
            for (int i = 0; i < length; i++) u[i] = ScaleElement(u[i], scale, zeroNonFinite);
            return;
        }
        var p = (IntPtr)data;
        int blocks = (length + 65535) / 65536;
        AiDotNet.Tensors.Helpers.CpuParallelSettings.LightweightParallel(blocks, b =>
        {
            var q = (uint*)p;
            int end = Math.Min(length, (b + 1) * 65536);
            for (int i = b * 65536; i < end; i++) q[i] = ScaleElement(q[i], scale, zeroNonFinite);
        });
    }

    public static Partial ChunkPartial(ReadOnlySpan<float> chunk, uint tensor)
    {
        var p = new Partial { Tensor = tensor, Emax = 1 };
        for (int i = 0; i < chunk.Length; i++)
        {
            uint b = SingleToUInt32Bits(chunk[i]);
            uint e = (b >> 23) & 0xFF;
            if (e == 0xFF)
            {
                if ((b & 0x7FFFFF) != 0) p.NanCount = SatAdd(p.NanCount, 1);
                else p.InfCount = SatAdd(p.InfCount, 1);
                continue;
            }
            if (e == 0) e = 1;
            if (e > p.Emax) p.Emax = e;
        }
        for (int i = 0; i < chunk.Length; i++)
        {
            uint b = SingleToUInt32Bits(chunk[i]);
            uint e = (b >> 23) & 0xFF;
            if (e == 0xFF) continue;
            ulong m = (b & 0x7FFFFF) | (e != 0 ? 0x800000u : 0u);
            if (e == 0) e = 1;
            Square(m, 68 - 2 * (int)(p.Emax - e), out ulong lo, out ulong hi);
            Add128(ref p.Lo, ref p.Hi, lo, hi);
        }
        return p;
    }

    /// <summary>m^2 * 2^shift as a 128-bit integer, truncated; m &lt; 2^24, shift &lt;= 68.</summary>
    public static void Square(ulong m, int shift, out ulong lo, out ulong hi)
    {
        ulong sq = m * m;
        lo = 0; hi = 0;
        if (shift >= 64) { hi = sq << (shift - 64); }
        else if (shift > 0) { lo = sq << shift; hi = sq >> (64 - shift); }
        else if (shift > -48) { lo = sq >> -shift; }
    }

    /// <summary>Combines chunk partials into a norm record for <paramref name="maxNorm"/>.</summary>
    public static uint[] Combine(IReadOnlyList<Partial> partials, float maxNorm,
        NonFiniteGradientPolicy policy = NonFiniteGradientPolicy.LeaveUnscaled)
    {
        if (partials is null) throw new ArgumentNullException(nameof(partials));
        int h = Headroom(partials.Count);
        uint g = 0, nan = 0, inf = 0, first = uint.MaxValue;
        for (int i = 0; i < partials.Count; i++)
        {
            var p = partials[i];
            if ((p.Lo | p.Hi) != 0 && p.Emax > g) g = p.Emax;
            nan = SatAdd(nan, p.NanCount);
            inf = SatAdd(inf, p.InfCount);
            if (p.NanCount + (ulong)p.InfCount != 0 && p.Tensor < first) first = p.Tensor;
        }
        ulong aLo = 0, aHi = 0;
        if (g != 0)
            for (int i = 0; i < partials.Count; i++)
            {
                var p = partials[i];
                if ((p.Lo | p.Hi) == 0) continue;
                ShiftRight128(p.Lo, p.Hi, 2 * (int)(g - p.Emax) + h, out ulong lo, out ulong hi);
                Add128(ref aLo, ref aHi, lo, hi);
            }
        int v = g == 0 ? 0 : 2 * (int)g - 368 + h;
        return Finish(aLo, aHi, v, nan, inf, first, maxNorm, policy);
    }

    /// <summary>The record from the exact sum <c>A * 2^V</c> and the non-finite counts (the kernels' last step).</summary>
    public static uint[] Finish(ulong aLo, ulong aHi, int v, uint nan, uint inf, uint first, float maxNorm,
        NonFiniteGradientPolicy policy = NonFiniteGradientPolicy.LeaveUnscaled)
    {
        var r = new uint[RecordWords];
        r[WordA0] = (uint)aLo; r[WordA0 + 1] = (uint)(aLo >> 32); r[WordA0 + 2] = (uint)aHi; r[WordA0 + 3] = (uint)(aHi >> 32);
        r[WordV] = (uint)v;
        r[WordNanCount] = nan; r[WordInfCount] = inf; r[WordFirstNonFiniteTensor] = first;
        r[WordMaxNorm] = SingleToUInt32Bits(maxNorm);
        ulong dbl;
        uint norm, scale;
        bool propagate = policy == NonFiniteGradientPolicy.Propagate;
        bool finiteOnly = policy == NonFiniteGradientPolicy.ZeroNonFinite;
        if (nan != 0) { dbl = 0x7FF8000000000000UL; norm = 0x7FC00000; scale = propagate ? 0x7FC00000u : 0x3F800000u; }
        else if (inf != 0) { dbl = 0x7FF0000000000000UL; norm = 0x7F800000; scale = propagate ? 0u : 0x3F800000u; }
        else
        {
            dbl = ToDoubleBits(aLo, aHi, v);
            Sqrt(aLo, aHi, v, out ulong nm, out int ne, out bool ns);
            norm = RoundToFloat(nm, ne, ns);
            scale = ClipScale(nm, ne, ns, maxNorm);
        }
        if (finiteOnly && (nan != 0 || inf != 0))
        {
            // The sum excludes the non-finite elements: clip the rest by the norm of the finite ones.
            Sqrt(aLo, aHi, v, out ulong fm, out int fe, out bool fs);
            scale = ClipScale(fm, fe, fs, maxNorm);
        }
        r[WordSumSquaresLo] = (uint)dbl; r[WordSumSquaresHi] = (uint)(dbl >> 32);
        r[WordNorm] = norm; r[WordScale] = scale;
        r[WordNonFinite] = nan != 0 || inf != 0 ? 1u : 0u;
        return r;
    }

    /// <summary>The record for host tensors: the CPU engine's clip uses this so its bits match every device.</summary>
    public static uint[] Compute(IReadOnlyList<float[]> tensors, float maxNorm,
        NonFiniteGradientPolicy policy = NonFiniteGradientPolicy.LeaveUnscaled) => Combine(Partials(tensors), maxNorm, policy);

    /// <summary>sqrt(A * 2^V) as m * 2^e (m has 27 bits, or 0) plus a sticky bit for the inexact remainder.</summary>
    public static void Sqrt(ulong aLo, ulong aHi, int v, out ulong m, out int e, out bool sticky)
    {
        int len = BitLength(aLo, aHi);
        if (len == 0) { m = 0; e = 0; sticky = false; return; }
        int k = len - 54;
        if (((v + k) & 1) != 0) k++;
        ulong a; bool dropped;
        if (k >= 0)
        {
            ShiftRight128(aLo, aHi, k, out a, out _);
            dropped = k > 0 && LowBitsNonZero(aLo, aHi, k);
        }
        else { a = aLo << -k; dropped = false; }
        ulong r = ISqrt(a);
        m = r; e = (v + k) / 2; sticky = dropped || r * r != a;
    }

    /// <summary>The clip coefficient min(1, fp32(maxNorm / fp32(fp32(norm) + 1e-6f))) as fp32 bits.</summary>
    public static uint ClipScale(ulong nm, int ne, bool ns, float maxNorm)
    {
        if (float.IsPositiveInfinity(maxNorm)) return 0x3F800000;
        // n = fp32(norm) with an unbounded exponent, as 24-bit significand nM * 2^nE.
        Round24(nm, ne, ns, out ulong nM, out int nE);
        // d = fp32(n + 1e-6f). 1e-6f = 0x358637BD = 0x8637BD * 2^(107 - 150).
        AddRound24(nM, nE, 0x8637BDUL, 107 - 150, out ulong dM, out int dE);
        uint mb = SingleToUInt32Bits(maxNorm);
        uint me = (mb >> 23) & 0xFF;
        ulong mM = (mb & 0x7FFFFF) | (me != 0 ? 0x800000u : 0u);
        int mE = (me == 0 ? 1 : (int)me) - 150;
        if (mM == 0) return 0;
        while (mM < 0x800000) { mM <<= 1; mE--; }
        // coef >= 1 exactly when maxNorm >= d: compare significands at a common exponent (both are 24-bit).
        if (mE > dE || (mE == dE && mM >= dM)) return 0x3F800000;
        ulong q = (mM << 40) / dM, rem = (mM << 40) % dM;
        return RoundToFloat(q, mE - 40 - dE, rem != 0);
    }

    /// <summary>Rounds (m + sticky) * 2^e to 24 significant bits, ties to even, with an unbounded exponent.</summary>
    public static void Round24(ulong m, int e, bool sticky, out ulong rm, out int re)
    {
        if (m == 0) { rm = 0; re = 0; return; }
        int len = BitLength(m, 0);
        if (len <= 24) { rm = m << (24 - len); re = e - (24 - len); return; }
        int sh = len - 24;
        rm = RoundShift(m, sh, sticky);
        re = e + sh;
        if (rm >= 1UL << 24) { rm >>= 1; re++; }
    }

    /// <summary>fp32(a + b) for positive 24-bit significands with an unbounded exponent.</summary>
    public static void AddRound24(ulong am, int ae, ulong bm, int be, out ulong rm, out int re)
    {
        if (am == 0) { rm = bm; re = be; return; }
        if (bm == 0) { rm = am; re = ae; return; }
        if (ae < be) { (am, bm) = (bm, am); (ae, be) = (be, ae); }
        int d = ae - be;
        if (d > 30) { Round24((am << 3) | 1, ae - 3, false, out rm, out re); return; }
        Round24((am << d) + bm, be, false, out rm, out re);
    }

    /// <summary>fp32 bits of (m + sticky) * 2^e, ties to even, with subnormals and overflow to +Inf. Callers pass
    /// at least two bits beyond the float's precision whenever <paramref name="sticky"/> is set.</summary>
    public static uint RoundToFloat(ulong m, int e, bool sticky)
    {
        if (m == 0) return 0;
        int lead = e + BitLength(m, 0) - 1;
        int ulp = Math.Max(lead - 23, -149);   // exponent of the float's last significand bit here
        int sh = ulp - e;
        ulong q = sh > 0 ? RoundShift(m, sh, sticky) : m << -sh;
        if (q >= 1UL << 24) { q >>= 1; ulp++; }  // the rounding carried into a new bit (q was 2^24 exactly)
        if (q < 1UL << 23) return (uint)q;      // subnormal (ulp == -149), or zero
        int biased = ulp + 23 + 127;
        if (biased >= 255) return 0x7F800000;
        return ((uint)biased << 23) | (uint)(q & 0x7FFFFF);
    }

    /// <summary>The clip's element update: +0 for a NaN or infinite element under
    /// <see cref="NonFiniteGradientPolicy.ZeroNonFinite"/>, otherwise <see cref="Mul"/>.</summary>
    public static uint ScaleElement(uint g, uint s, bool zeroNonFinite) =>
        zeroNonFinite && ((g >> 23) & 0xFF) == 0xFF ? 0u : Mul(g, s);

    /// <summary>g * s as one IEEE fp32 multiply (ties to even, subnormals kept), with NaN results canonical
    /// (0x7FC00000) so they too have the same bits everywhere, and g itself when s is exactly 1; the kernels'
    /// dgn_mul.</summary>
    public static uint Mul(uint g, uint s)
    {
        if (s == 0x3F800000u) return g;   // unscaled: every element, NaN payloads included, keeps its bits
        uint eg = (g >> 23) & 0xFF, es = (s >> 23) & 0xFF;
        uint fg = g & 0x7FFFFF, fs = s & 0x7FFFFF, sign = (g ^ s) & 0x80000000u;
        if ((eg == 0xFF && fg != 0) || (es == 0xFF && fs != 0)) return 0x7FC00000;
        bool gzero = eg == 0 && fg == 0, szero = es == 0 && fs == 0;
        if (eg == 0xFF || es == 0xFF) return gzero || szero ? 0x7FC00000u : sign | 0x7F800000u;
        if (gzero || szero) return sign;
        ulong prod = (ulong)(fg | (eg != 0 ? 0x800000u : 0u)) * (fs | (es != 0 ? 0x800000u : 0u));
        int e = (eg != 0 ? (int)eg : 1) + (es != 0 ? (int)es : 1) - 300;
        return RoundToFloat(prod, e, false) | sign;
    }

    /// <summary>The exact sum of squares as a double (ties to even); A * 2^V never leaves the double range.</summary>
    public static ulong ToDoubleBits(ulong aLo, ulong aHi, int v)
    {
        int len = BitLength(aLo, aHi);
        if (len == 0) return 0;
        ulong m;
        int e;
        if (len > 53)
        {
            int sh = len - 53;
            ShiftRight128(aLo, aHi, sh, out ulong top, out _);
            bool half = BitAt(aLo, aHi, sh - 1);
            bool rest = sh > 1 && LowBitsNonZero(aLo, aHi, sh - 1);
            m = top;
            if (half && (rest || (m & 1) != 0)) m++;
            e = v + sh;
            if (m == 1UL << 53) { m >>= 1; e++; }
        }
        else { m = aLo << (53 - len); e = v - (53 - len); }
        int lead = e + 52;
        if (lead >= -1022) return ((ulong)(lead + 1023) << 52) | (m & 0xFFFFFFFFFFFFFUL);
        // Not reachable for squares of floats (the smallest is 2^-298); kept exact for completeness.
        return m >> (-1022 - lead);
    }

    public static ulong ISqrt(ulong a)
    {
        ulong r = 0, bit = 1UL << 62;
        while (bit > a) bit >>= 2;
        while (bit != 0)
        {
            if (a >= r + bit) { a -= r + bit; r = (r >> 1) + bit; }
            else r >>= 1;
            bit >>= 2;
        }
        return r;
    }

    /// <summary>(m + sticky) / 2^sh rounded to an integer, ties to even; sh &gt;= 1.</summary>
    private static ulong RoundShift(ulong m, int sh, bool sticky)
    {
        if (sh > 64) return 0;   // below half of one
        ulong q = sh == 64 ? 0 : m >> sh;
        bool half = ((m >> (sh - 1)) & 1) != 0;
        bool rest = sticky || (sh > 1 && (m & (ulong.MaxValue >> (65 - sh))) != 0);
        if (half && (rest || (q & 1) != 0)) q++;
        return q;
    }

    public static void Add128(ref ulong lo, ref ulong hi, ulong bLo, ulong bHi)
    {
        ulong s = lo + bLo;
        hi += bHi + (s < lo ? 1UL : 0UL);
        lo = s;
    }

    public static void ShiftRight128(ulong lo, ulong hi, int sh, out ulong rLo, out ulong rHi)
    {
        if (sh <= 0) { rLo = lo; rHi = hi; return; }
        if (sh >= 128) { rLo = 0; rHi = 0; return; }
        if (sh >= 64) { rLo = hi >> (sh - 64); rHi = 0; return; }
        rLo = (lo >> sh) | (hi << (64 - sh));
        rHi = hi >> sh;
    }

    private static bool LowBitsNonZero(ulong lo, ulong hi, int bits)
    {
        if (bits <= 0) return false;
        if (bits >= 128) return (lo | hi) != 0;
        if (bits >= 64) return lo != 0 || (bits > 64 && (hi & ((1UL << (bits - 64)) - 1)) != 0);
        return (lo & ((1UL << bits) - 1)) != 0;
    }

    private static bool BitAt(ulong lo, ulong hi, int bit) => bit < 64 ? ((lo >> bit) & 1) != 0 : ((hi >> (bit - 64)) & 1) != 0;

    public static int BitLength(ulong lo, ulong hi)
    {
        if (hi != 0) return 64 + BitLength64(hi);
        return BitLength64(lo);
    }

    private static int BitLength64(ulong x)
    {
        int n = 0;
        while (x != 0) { x >>= 1; n++; }
        return n;
    }

    private static uint SatAdd(uint a, uint b) => a > uint.MaxValue - b ? uint.MaxValue : a + b;

    public static unsafe uint SingleToUInt32Bits(float f) => *(uint*)&f;

    public static unsafe float UInt32BitsToSingle(uint u) => *(float*)&u;
}
