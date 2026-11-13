// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The raw-data kernels of IDataDecodeBackend (torch.frombuffer + .to(float32); numeric text parsing), written once in
// a C subset and emitted as OpenCL C, CUDA/HIP C++, Vulkan GLSL and Metal Shading Language. WGSL keeps its own copy
// (WebGpu/WebGpuDataDecodeKernels.cs).
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Five kernels over 256-wide groups, every buffer a 32-bit integer array (the text and raw bytes packed four per word,
/// little-endian; float results stored as their bits), so no value ever passes through a float register: a load that
/// flushes subnormals or quiets a NaN cannot change a decoded bit.
/// <list type="bullet">
/// <item><see cref="Decode"/>: element i of the packed buffer (any <see cref="NumericStorageType"/>, either byte order)
/// to float32, rounded to nearest even as the host's conversion rounds.</item>
/// <item><see cref="CountSeparators"/>: separators (the delimiter or a line feed) per 64-byte segment of the text.</item>
/// <item><see cref="Scan"/>: one group's exclusive prefix sum over an int array, in place, and the total.</item>
/// <item><see cref="ScatterBounds"/>: from each segment's first field index, every field's [start, end) byte range.</item>
/// <item><see cref="ParseFields"/>: every field parsed to float32, exactly as <c>float.Parse</c> rounds. A field outside
/// the exactly-rounded fast path (more than nineteen significant digits whose rounding the dropped digits could change,
/// a value that may fall outside the normal float range [1e-37, 1e38), NaN / infinity symbols, a negative zero, any
/// character the fast grammar does not accept) is flagged for the host instead of guessed.</item>
/// </list>
/// Per-element kernels take <c>firstItem</c>, so a grid larger than one dispatch's 65535 groups runs as several launches.
/// </summary>
internal static class DataDecodeKernelTemplate
{
    /// <summary>Threads per group.</summary>
    internal const int GroupSize = 256;

    /// <summary>Text bytes one separator-scan thread covers.</summary>
    internal const int SegmentBytes = 64;

    internal const string Decode = "dd_decode_bytes";
    internal const string CountSeparators = "dd_count_separators";
    internal const string Scan = "dd_exclusive_scan";
    internal const string ScatterBounds = "dd_scatter_bounds";
    internal const string ParseFields = "dd_parse_fields";

    /// <summary>Style bits <see cref="ParseFields"/> understands (its <c>styleMask</c>).</summary>
    internal const int StyleLeadingWhite = 1, StyleTrailingWhite = 2, StyleLeadingSign = 4, StyleDecimalPoint = 8,
        StyleExponent = 16;

    /// <summary>One buffer parameter: its name, element type (uint or int) and whether the kernel writes it.</summary>
    internal readonly struct Buffer
    {
        internal Buffer(string name, bool isUnsigned, bool written)
        {
            Name = name; IsUnsigned = isUnsigned; Written = written;
        }

        internal string Name { get; }
        internal bool IsUnsigned { get; }
        internal bool Written { get; }
    }

    /// <summary>One kernel: buffers in binding order, then int parameters, then the body.</summary>
    internal sealed class Spec
    {
        internal Spec(string name, Buffer[] buffers, string[] ints, string body, int sharedInts = 0)
        {
            Name = name; Buffers = buffers; Ints = ints; Body = body; SharedInts = sharedInts;
        }

        internal string Name { get; }
        internal Buffer[] Buffers { get; }
        internal string[] Ints { get; }
        internal string Body { get; }
        internal int SharedInts { get; }
    }

    // A text byte: words packs four per uint, byte k at bits 8 * (k % 4) of word k / 4.
    private const string ByteAt = "((words[({0}) >> 2] >> ((({0}) & 3) * 8)) & 255u)";

    private static string B(string index) => string.Format(System.Globalization.CultureInfo.InvariantCulture, ByteAt, index);

    // Scalar helpers (no buffer parameters: GLSL cannot pass a buffer to a function). Every one is pure integer code.
    private const string Helpers = @"
// Index of the highest set bit of x (x > 0).
FN int dd_msb(uint x)
{
    int r = 0;
    uint v = x;
    if (v >= 65536u) { v = v >> 16; r = r + 16; }
    if (v >= 256u) { v = v >> 8; r = r + 8; }
    if (v >= 16u) { v = v >> 4; r = r + 4; }
    if (v >= 4u) { v = v >> 2; r = r + 2; }
    if (v >= 2u) { r = r + 1; }
    return r;
}

// The low word of hi:lo >> s, for 0 <= s.
FN uint dd_shr64_lo(uint hi, uint lo, int s)
{
    if (s <= 0) return lo;
    if (s >= 64) return 0u;
    if (s >= 32) return hi >> (s - 32);
    return (lo >> s) | (hi << (32 - s));
}

// Bit s (0 <= s <= 63) of hi:lo.
FN uint dd_bit64(uint hi, uint lo, int s)
{
    if (s >= 32) return (hi >> (s - 32)) & 1u;
    return (lo >> s) & 1u;
}

// 1 when any of the low s bits of hi:lo is set.
FN int dd_low_bits_set(uint hi, uint lo, int s)
{
    if (s <= 0) return 0;
    if (s < 32) return (lo & ((1u << s) - 1u)) != 0u ? 1 : 0;
    if (lo != 0u) return 1;
    if (s == 32) return 0;
    if (s >= 64) return hi != 0u ? 1 : 0;
    return (hi & ((1u << (s - 32)) - 1u)) != 0u ? 1 : 0;
}

// hi:lo >> s (s >= 1) rounded to nearest, ties to even.
FN uint dd_round_shift(uint hi, uint lo, int s)
{
    uint m = dd_shr64_lo(hi, lo, s);
    uint rb = 0u;
    if (s <= 64) rb = dd_bit64(hi, lo, s - 1);
    int sticky = dd_low_bits_set(hi, lo, s - 1);
    if (rb != 0u && (sticky != 0 || (m & 1u) != 0u)) m = m + 1u;
    return m;
}

// The float32 bits of the unsigned 64-bit hi:lo, rounded to nearest even.
FN uint dd_u64_to_f32(uint hi, uint lo)
{
    if (hi == 0u && lo == 0u) return 0u;
    int p = 0;
    if (hi != 0u) p = 32 + dd_msb(hi); else p = dd_msb(lo);
    uint m = 0u;
    if (p <= 23) m = lo << (23 - p);
    else
    {
        m = dd_round_shift(hi, lo, p - 23);
        if (m == 16777216u) { m = 8388608u; p = p + 1; }
    }
    return (ITOU(p + 127) << 23) | (m & 8388607u);
}

// The float32 bits of the two's-complement 32-bit v.
FN uint dd_i32_to_f32(uint v)
{
    if ((v & 2147483648u) != 0u) return 2147483648u | dd_u64_to_f32(0u, (~v) + 1u);
    return dd_u64_to_f32(0u, v);
}

// The float32 bits of the two's-complement 64-bit hi:lo.
FN uint dd_i64_to_f32(uint hi, uint lo)
{
    if ((hi & 2147483648u) != 0u)
    {
        uint nlo = (~lo) + 1u;
        uint nhi = ~hi;
        if (nlo == 0u) nhi = nhi + 1u;
        return 2147483648u | dd_u64_to_f32(nhi, nlo);
    }
    return dd_u64_to_f32(hi, lo);
}

// The float32 bits of the binary16 h (exact).
FN uint dd_f16_to_f32(uint h)
{
    uint sgn = (h & 32768u) << 16;
    uint e = (h >> 10) & 31u;
    uint m = h & 1023u;
    if (e == 31u) return sgn | 2139095040u | (m << 13);
    if (e == 0u)
    {
        if (m == 0u) return sgn;
        int p = dd_msb(m);
        return sgn | (ITOU(p + 103) << 23) | ((m << (23 - p)) & 8388607u);
    }
    return sgn | ((e + 112u) << 23) | (m << 13);
}

// The float32 bits of the binary64 hi:lo, rounded to nearest even (subnormal results included); a NaN keeps its sign
// and its payload's top bits, quieted, as the x64 conversion does.
FN uint dd_f64_to_f32(uint hi, uint lo)
{
    uint sgn = hi & 2147483648u;
    int e = UTOI((hi >> 20) & 2047u);
    uint mhi = hi & 1048575u;
    if (e == 2047)
    {
        if (mhi != 0u || lo != 0u) return sgn | 2143289344u | (((mhi << 3) | (lo >> 29)) & 8388607u);
        return sgn | 2139095040u;
    }
    if (e == 0) return sgn;
    int ue = e - 1023;
    if (ue > 127) return sgn | 2139095040u;
    uint shi = mhi | 1048576u;
    if (ue >= -126)
    {
        uint m = dd_round_shift(shi, lo, 29);
        if (m == 16777216u) { m = 8388608u; ue = ue + 1; }
        if (ue > 127) return sgn | 2139095040u;
        return sgn | (ITOU(ue + 127) << 23) | (m & 8388607u);
    }
    // A subnormal (or zero) result in units of 2^-149; a carry to 2^23 is exactly the smallest normal's bits.
    return sgn | dd_round_shift(shi, lo, -97 - ue);
}

// The low word of lo * 10 + d, and the carry out of it, computed in 16-bit halves (d <= 9).
FN uint dd_mul10_lo(uint lo, uint d)
{
    uint l0 = (lo & 65535u) * 10u + d;
    uint l1 = (lo >> 16) * 10u + (l0 >> 16);
    return (l1 << 16) | (l0 & 65535u);
}

FN uint dd_mul10_carry(uint lo, uint d)
{
    uint l0 = (lo & 65535u) * 10u + d;
    uint l1 = (lo >> 16) * 10u + (l0 >> 16);
    return l1 >> 16;
}

// x = x * 10 over eight little-endian words (the callers' bounds keep the product below 2^256).
FN void dd_big_mul10(LIMBS x LIMBS_END)
{
    uint carry = 0u;
    for (int i = 0; i < 8; i++)
    {
        uint v = x[i];
        uint l0 = (v & 65535u) * 10u + carry;
        uint l1 = (v >> 16) * 10u + (l0 >> 16);
        x[i] = (l1 << 16) | (l0 & 65535u);
        carry = l1 >> 16;
    }
}

// The bit length of x (0 for zero).
FN int dd_big_bitlen(CLIMBS x LIMBS_END)
{
    int n = 0;
    for (int i = 7; i >= 0; i--)
    {
        if (n == 0 && x[i] != 0u) n = i * 32 + dd_msb(x[i]) + 1;
    }
    return n;
}

// x = x << s (s >= 0); bits past the top word are dropped (the callers' bounds keep them zero).
FN void dd_big_shl(LIMBS x LIMBS_END, int s)
{
    int ls = s >> 5;
    int bs = s & 31;
    for (int i = 7; i >= 0; i--)
    {
        int j = i - ls;
        uint v = 0u;
        if (j >= 0)
        {
            v = x[j] << ITOU(bs);
            if (bs != 0 && j >= 1) v = v | (x[j - 1] >> ITOU(32 - bs));
        }
        x[i] = v;
    }
}

// 1 when x >= y.
FN int dd_big_ge(CLIMBS x LIMBS_END, CLIMBS y LIMBS_END)
{
    int r = 1;
    int decided = 0;
    for (int i = 7; i >= 0; i--)
    {
        if (decided == 0 && x[i] != y[i])
        {
            decided = 1;
            if (x[i] < y[i]) r = 0;
        }
    }
    return r;
}

// x = x - y (x >= y).
FN void dd_big_sub(LIMBS x LIMBS_END, CLIMBS y LIMBS_END)
{
    uint borrow = 0u;
    for (int i = 0; i < 8; i++)
    {
        uint u = x[i];
        uint w = y[i];
        uint nb = 0u;
        if (u < w || (u == w && borrow != 0u)) nb = 1u;
        x[i] = u - w - borrow;
        borrow = nb;
    }
}

// The float32 bits of sHi:sLo * 10^e10 (the value > 0, bounded by the caller to the normal range) rounded to nearest
// even, by exact arithmetic: a = s * 10^max(e10, 0) and b = 10^max(-e10, 0) aligned to the same bit length, 25
// quotient bits by restoring division, the remainder as the sticky bit. stickyIn marks a value strictly above the
// quotient. 0 when the result is outside the normal range. The host twin is NumericTextFastPath.ExactToSingleBits.
FN uint dd_exact_to_f32(uint sHi, uint sLo, int e10, int stickyIn)
{
    uint a[8];
    uint b[8];
    for (int i = 0; i < 8; i++) { a[i] = 0u; b[i] = 0u; }
    a[0] = sLo;
    a[1] = sHi;
    b[0] = 1u;
    for (int t = 0; t < e10; t++) dd_big_mul10(a);
    for (int t = 0; t < -e10; t++) dd_big_mul10(b);
    int la = dd_big_bitlen(a);
    int lb = dd_big_bitlen(b);
    int ex2 = la - lb;
    if (la > lb) dd_big_shl(b, la - lb);
    else if (lb > la) dd_big_shl(a, lb - la);
    if (dd_big_ge(a, b) == 0) { dd_big_shl(a, 1); ex2 = ex2 - 1; }
    uint m = 0u;
    for (int k = 0; k < 25; k++)
    {
        uint bit = 0u;
        if (dd_big_ge(a, b) != 0) { dd_big_sub(a, b); bit = 1u; }
        m = (m << 1) | bit;
        dd_big_shl(a, 1);
    }
    int sticky = stickyIn;
    if (dd_big_bitlen(a) != 0) sticky = 1;
    uint rb = m & 1u;
    m = m >> 1;
    if (rb != 0u && (sticky != 0 || (m & 1u) != 0u)) m = m + 1u;
    if (m == 16777216u) { m = 8388608u; ex2 = ex2 + 1; }
    if (ex2 < -126 || ex2 > 127) return 0u;
    return (ITOU(ex2 + 127) << 23) | (m & 8388607u);
}
";

    private static readonly string DecodeBody = @"
    int i = gid + firstItem;
    if (i >= nItems) return;
    int at = i * elemSize;
    uint lo = 0u;
    uint hi = 0u;
    for (int j = 0; j < elemSize; j++)
    {
        int k = at + j;
        if (littleEndian == 0) k = at + elemSize - 1 - j;
        uint b = " + B("k") + @";
        if (j < 4) lo = lo | (b << (j * 8)); else hi = hi | (b << ((j - 4) * 8));
    }
    uint bits = 0u;
    if (typeCode == 0) bits = dd_f16_to_f32(lo);
    else if (typeCode == 1) bits = lo;
    else if (typeCode == 2) bits = dd_f64_to_f32(hi, lo);
    else if (typeCode == 3) { if ((lo & 128u) != 0u) lo = lo | 4294967040u; bits = dd_i32_to_f32(lo); }
    else if (typeCode == 5) { if ((lo & 32768u) != 0u) lo = lo | 4294901760u; bits = dd_i32_to_f32(lo); }
    else if (typeCode == 7) bits = dd_i32_to_f32(lo);
    else if (typeCode == 9) bits = dd_i64_to_f32(hi, lo);
    else bits = dd_u64_to_f32(0u, lo);
    outBits[i] = bits;
";

    private static readonly string CountBody = @"
    int seg = gid + firstItem;
    if (seg >= nSegs) return;
    int start = seg * 64;
    int stop = textLen;
    if (start + 64 < textLen) stop = start + 64;
    uint dl = ITOU(delim);
    int c = 0;
    for (int k = start; k < stop; k++)
    {
        uint b = " + B("k") + @";
        if (b == 10u || b == dl) c = c + 1;
    }
    counts[seg] = c;
";

    // One group: thread t sums a contiguous range, thread 0 scans the 256 partial sums, then every thread rewrites its
    // range as the running prefix. The barriers sit outside every thread-dependent branch.
    private const string ScanBody = @"
    int per = (nItems + 255) / 256;
    int lo0 = lid * per;
    int hi0 = nItems;
    if (lo0 + per < nItems) hi0 = lo0 + per;
    int s = 0;
    for (int k = lo0; k < hi0; k++) s = s + vals[k];
    sm[lid] = s;
    BARRIER();
    if (lid == 0)
    {
        int run = 0;
        for (int t = 0; t < 256; t++) { int v = sm[t]; sm[t] = run; run = run + v; }
        total[0] = run;
    }
    BARRIER();
    int acc = sm[lid];
    for (int k = lo0; k < hi0; k++) { int v = vals[k]; vals[k] = acc; acc = acc + v; }
";

    // A field ends at its separator; a carriage return before a line feed (or before the newline the host trimmed off
    // the end of the text) is not part of the field, unless the delimiter is itself a carriage return.
    private static readonly string ScatterBody = @"
    int seg = gid + firstItem;
    if (seg >= nSegs) return;
    int start = seg * 64;
    int stop = textLen;
    if (start + 64 < textLen) stop = start + 64;
    uint dl = ITOU(delim);
    int f = offs[seg];
    if (seg == 0) bounds[0] = 0;
    for (int k = start; k < stop; k++)
    {
        uint b = " + B("k") + @";
        if (b == 10u || b == dl)
        {
            int e = k;
            if (b == 10u && delim != 13 && k > 0)
            {
                uint pb = " + B("k - 1") + @";
                if (pb == 13u) e = k - 1;
            }
            bounds[2 * f + 1] = e;
            bounds[2 * f + 2] = k + 1;
            f = f + 1;
        }
    }
    if (seg == nSegs - 1)
    {
        int e = textLen;
        if (trailingNewline != 0 && delim != 13 && textLen > 0)
        {
            uint pb = " + B("textLen - 1") + @";
            if (pb == 13u) e = textLen - 1;
        }
        bounds[2 * f + 1] = e;
    }
";

    // The fast grammar: [white] [sign] digits [. digits] [(e|E) [sign] digits] [white], each part only where the style
    // allows it, at least one mantissa digit. Up to nineteen significant digits are kept (sHi:sLo); later ones only mark
    // the value as strictly above the kept prefix. A result certainly in the normal range ([1e-37, 1e38)) is then
    // rounded by exact multi-word division (dd_exact_to_f32), and when digits were dropped the prefix and the prefix
    // plus one unit must round to the same float, so the dropped digits cannot matter.
    private static readonly string ParseBody = @"
    int f = gid + firstItem;
    if (f >= nItems) return;
    int p = bounds[2 * f];
    int e = bounds[2 * f + 1];
    int state = 0;
    uint resultBits = ITOU(missingBits);
    if (p < e)
    {
        int q = p;
        while (q < e)
        {
            uint b = " + B("q") + @";
            if (b == 32u || (b >= 9u && b <= 13u)) q = q + 1; else break;
        }
        if (q > p && (styleMask & 1) == 0) state = 1;
        int r = e;
        while (r > q)
        {
            uint b = " + B("r - 1") + @";
            if (b == 32u || (b >= 9u && b <= 13u)) r = r - 1; else break;
        }
        if (r < e && (styleMask & 2) == 0) state = 1;
        if (q == r) state = 1;
        int neg = 0;
        if (state == 0)
        {
            uint b = " + B("q") + @";
            if (b == 45u || b == 43u)
            {
                if ((styleMask & 4) == 0) state = 1;
                if (b == 45u) neg = 1;
                q = q + 1;
            }
        }
        uint sLo = 0u;
        uint sHi = 0u;
        int nsig = 0;
        int dropped = 0;
        int scale = 0;
        int digits = 0;
        while (state == 0 && q < r)
        {
            uint b = " + B("q") + @";
            if (b < 48u || b > 57u) break;
            uint d = b - 48u;
            digits = digits + 1;
            if (nsig == 0 && d == 0u) { }
            else if (nsig < 19) { sHi = sHi * 10u + dd_mul10_carry(sLo, d); sLo = dd_mul10_lo(sLo, d); nsig = nsig + 1; }
            else { scale = scale + 1; if (d != 0u) dropped = 1; }
            q = q + 1;
        }
        if (state == 0 && q < r)
        {
            uint b = " + B("q") + @";
            if (b == 46u)
            {
                if ((styleMask & 8) == 0) state = 1;
                q = q + 1;
                while (state == 0 && q < r)
                {
                    uint c = " + B("q") + @";
                    if (c < 48u || c > 57u) break;
                    uint d = c - 48u;
                    digits = digits + 1;
                    if (nsig == 0 && d == 0u) scale = scale - 1;
                    else if (nsig < 19)
                    {
                        sHi = sHi * 10u + dd_mul10_carry(sLo, d);
                        sLo = dd_mul10_lo(sLo, d);
                        nsig = nsig + 1;
                        scale = scale - 1;
                    }
                    else if (d != 0u) dropped = 1;
                    q = q + 1;
                }
            }
        }
        if (digits == 0) state = 1;
        int ex = 0;
        if (state == 0 && q < r)
        {
            uint b = " + B("q") + @";
            if (b == 101u || b == 69u)
            {
                if ((styleMask & 16) == 0) state = 1;
                q = q + 1;
                int eneg = 0;
                if (q < r)
                {
                    uint c = " + B("q") + @";
                    if (c == 45u || c == 43u) { if (c == 45u) eneg = 1; q = q + 1; }
                }
                int ed = 0;
                while (q < r)
                {
                    uint c = " + B("q") + @";
                    if (c < 48u || c > 57u) break;
                    if (ex < 100000) ex = ex * 10 + UTOI(c - 48u);
                    ed = ed + 1;
                    q = q + 1;
                }
                if (ed == 0) state = 1;
                if (eneg != 0) ex = -ex;
            }
        }
        if (q != r) state = 1;
        if (state == 0)
        {
            if (nsig == 0)
            {
                if (neg != 0) state = 1; else resultBits = 0u;
            }
            else
            {
                int e10 = scale + ex;
                if (nsig - 1 + e10 < -37 || nsig + e10 > 38) state = 1;
                else
                {
                    uint lob = dd_exact_to_f32(sHi, sLo, e10, dropped);
                    if (lob == 0u) state = 1;
                    else if (dropped != 0)
                    {
                        uint hLo = sLo + 1u;
                        uint hHi = sHi;
                        if (hLo == 0u) hHi = hHi + 1u;
                        if (dd_exact_to_f32(hHi, hLo, e10, 1) != lob) state = 1;
                    }
                    if (state == 0)
                    {
                        resultBits = lob;
                        if (neg != 0) resultBits = lob | 2147483648u;
                    }
                }
            }
        }
    }
    if (state != 0) resultBits = ITOU(missingBits);
    outBits[f] = resultBits;
    status[f] = state;
";

    private static Buffer U(string name, bool written = false) => new(name, true, written);
    private static Buffer I(string name, bool written = false) => new(name, false, written);

    /// <summary>
    /// Decode: words, outBits (written) | nItems, typeCode, elemSize, littleEndian, firstItem.
    /// CountSeparators: words, counts (written) | textLen, delim, nSegs, firstItem.
    /// Scan: vals (in place), total (written) | nItems.
    /// ScatterBounds: words, offs, bounds (written) | textLen, delim, nSegs, trailingNewline, firstItem.
    /// ParseFields: words, bounds, outBits (written), status (written) | nItems, styleMask, missingBits, firstItem.
    /// </summary>
    internal static readonly Spec[] Kernels =
    {
        new(Decode, new[] { U("words"), U("outBits", true) },
            new[] { "nItems", "typeCode", "elemSize", "littleEndian", "firstItem" }, DecodeBody),
        new(CountSeparators, new[] { U("words"), I("counts", true) },
            new[] { "textLen", "delim", "nSegs", "firstItem" }, CountBody),
        new(Scan, new[] { I("vals", true), I("total", true) }, new[] { "nItems" }, ScanBody, sharedInts: 256),
        new(ScatterBounds, new[] { U("words"), I("offs"), I("bounds", true) },
            new[] { "textLen", "delim", "nSegs", "trailingNewline", "firstItem" }, ScatterBody),
        new(ParseFields, new[] { U("words"), I("bounds"), U("outBits", true), I("status", true) },
            new[] { "nItems", "styleMask", "missingBits", "firstItem" }, ParseBody),
    };

    /// <summary>Every kernel name, in <see cref="Kernels"/> order.</summary>
    internal static string[] GetKernelNames()
    {
        var names = new string[Kernels.Length];
        for (int i = 0; i < Kernels.Length; i++) names[i] = Kernels[i].Name;
        return names;
    }

    internal static Spec GetSpec(string name)
    {
        foreach (var spec in Kernels)
            if (spec.Name == name) return spec;
        throw new ArgumentException($"No data-decode kernel named '{name}'.", nameof(name));
    }

    /// <summary>The whole module (every kernel) in one dialect; GLSL has one kernel per shader (<see cref="GetGlslSource"/>).</summary>
    internal static string GetModuleSource(WinogradKernelDialect dialect)
    {
        if (dialect == WinogradKernelDialect.Glsl)
            throw new ArgumentException("GLSL has one kernel per shader; use GetGlslSource.", nameof(dialect));
        var sb = new StringBuilder();
        if (dialect == WinogradKernelDialect.Metal) sb.Append("#include <metal_stdlib>\nusing namespace metal;\n");
        if (dialect == WinogradKernelDialect.Hip) sb.Append("typedef unsigned int uint;\n");
        sb.Append(Spell(Helpers, dialect));
        foreach (var spec in Kernels)
        {
            sb.Append('\n').Append(dialect switch
            {
                WinogradKernelDialect.OpenCl => "__kernel void ",
                WinogradKernelDialect.Metal => "kernel void ",
                _ => "extern \"C\" __global__ void ",
            });
            sb.Append(spec.Name).Append('(');
            var parameters = new List<string>();
            for (int i = 0; i < spec.Buffers.Length; i++)
            {
                var b = spec.Buffers[i];
                string type = b.IsUnsigned ? "uint" : "int";
                parameters.Add(dialect switch
                {
                    WinogradKernelDialect.OpenCl => (b.Written ? "__global " : "__global const ") + type + "* restrict " + b.Name,
                    WinogradKernelDialect.Metal => (b.Written ? "device " : "device const ") + type + "* " + b.Name + " [[buffer(" + i + ")]]",
                    _ => (b.Written ? "" : "const ") + type + "* __restrict__ " + b.Name,
                });
            }
            for (int i = 0; i < spec.Ints.Length; i++)
                parameters.Add(dialect switch
                {
                    WinogradKernelDialect.OpenCl => "const int " + spec.Ints[i],
                    WinogradKernelDialect.Metal => "constant int& " + spec.Ints[i] + " [[buffer(" + (spec.Buffers.Length + i) + ")]]",
                    _ => "int " + spec.Ints[i],
                });
            if (dialect == WinogradKernelDialect.Metal)
            {
                parameters.Add("uint gidU [[thread_position_in_grid]]");
                parameters.Add("uint lidU [[thread_index_in_threadgroup]]");
            }
            sb.Append(string.Join(", ", parameters)).Append(")\n{\n");
            sb.Append(dialect switch
            {
                WinogradKernelDialect.OpenCl => "    int gid = (int)get_global_id(0);\n    int lid = (int)get_local_id(0);\n",
                WinogradKernelDialect.Metal => "    int gid = (int)gidU;\n    int lid = (int)lidU;\n",
                _ => "    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n    int lid = (int)threadIdx.x;\n",
            });
            if (spec.SharedInts > 0)
                sb.Append(dialect switch
                {
                    WinogradKernelDialect.OpenCl => $"    __local int sm[{spec.SharedInts}];\n",
                    WinogradKernelDialect.Metal => $"    threadgroup int sm[{spec.SharedInts}];\n",
                    _ => $"    __shared__ int sm[{spec.SharedInts}];\n",
                });
            sb.Append(Spell(spec.Body, dialect)).Append("}\n");
        }
        return sb.ToString();
    }

    /// <summary>One kernel as a Vulkan compute shader: the buffers at bindings 0.. and the ints as push constants, in
    /// <see cref="Spec.Ints"/> order, over 256-wide groups.</summary>
    internal static string GetGlslSource(string name)
    {
        var spec = GetSpec(name);
        var sb = new StringBuilder("#version 450\nlayout(local_size_x = 256) in;\n");
        for (int i = 0; i < spec.Buffers.Length; i++)
            sb.Append("layout(set=0, binding=").Append(i).Append(") buffer B").Append(i).Append(" { ")
                .Append(spec.Buffers[i].IsUnsigned ? "uint " : "int ").Append(spec.Buffers[i].Name).Append("[]; };\n");
        sb.Append("layout(push_constant) uniform PC { ");
        foreach (var p in spec.Ints) sb.Append("int ").Append(p).Append("; ");
        sb.Append("};\n");
        if (spec.SharedInts > 0) sb.Append("shared int sm[").Append(spec.SharedInts).Append("];\n");
        sb.Append(Spell(Helpers, WinogradKernelDialect.Glsl));
        sb.Append("void main()\n{\n    int gid = int(gl_GlobalInvocationID.x);\n    int lid = int(gl_LocalInvocationID.x);\n");
        sb.Append(Spell(spec.Body, WinogradKernelDialect.Glsl)).Append("}\n");
        return sb.ToString();
    }

    // The dialect-neutral spellings: helper-function qualifier, the work-group barrier, the int <-> uint conversions
    // (GLSL has no C cast) and the eight-word integer parameters (LIMBS name LIMBS_END, CLIMBS when read-only: an
    // array copied in / out in GLSL, a pointer to the caller's private array elsewhere).
    private static string Spell(string body, WinogradKernelDialect dialect)
        => body.Replace("FN ", dialect == WinogradKernelDialect.Hip ? "__device__ inline " : "")
            .Replace("CLIMBS ", dialect switch
            {
                WinogradKernelDialect.Glsl => "in uint ",
                WinogradKernelDialect.Metal => "thread const uint* ",
                _ => "const uint* ",
            })
            .Replace("LIMBS ", dialect switch
            {
                WinogradKernelDialect.Glsl => "inout uint ",
                WinogradKernelDialect.Metal => "thread uint* ",
                _ => "uint* ",
            })
            .Replace(" LIMBS_END", dialect == WinogradKernelDialect.Glsl ? "[8]" : "")
            .Replace("BARRIER();", dialect switch
            {
                WinogradKernelDialect.OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
                WinogradKernelDialect.Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
                WinogradKernelDialect.Glsl => "memoryBarrierShared(); barrier();",
                _ => "__syncthreads();",
            })
            .Replace("UTOI(", dialect == WinogradKernelDialect.Glsl ? "int(" : "(int)(")
            .Replace("ITOU(", dialect == WinogradKernelDialect.Glsl ? "uint(" : "(uint)(");
}
