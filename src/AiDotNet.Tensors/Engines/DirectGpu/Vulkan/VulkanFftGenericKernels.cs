// Copyright (c) AiDotNet. All rights reserved.
// GLSL kernels for the Vulkan generic-precision, arbitrary-length FFT (#964), the peers of
// CUDA/Kernels/CudaFFTGenericKernels.
//
// Storage is float32, float16 or bfloat16; every arithmetic expression is float32. Vulkan storage buffers here are
// arrays of 32-bit words, so a 16-bit type packs two values per word, and a kernel that writes typed storage gives
// each thread whole words to write (two values at a time): two threads never write halves of the same word. The
// narrowing conversions are written out bit by bit (round to nearest even), so they do not depend on a device's
// 16-bit float support or on how its packHalf2x16 rounds.
using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>One generic-FFT shader with the binding count and push-constant size its pipeline is created with.</summary>
internal readonly struct VulkanFftGenericKernel
{
    internal VulkanFftGenericKernel(string name, string source, int bindings, int pushWords)
    {
        Name = name;
        Source = source;
        Bindings = bindings;
        PushWords = pushWords;
    }

    internal string Name { get; }
    internal string Source { get; }
    internal int Bindings { get; }
    internal int PushWords { get; }
}

/// <summary>GLSL source for the Vulkan generic FFT path, specialized per <see cref="FftElementType"/>.</summary>
internal static class VulkanFftGenericKernels
{
    public const int LocalSize = 256;

    private static readonly ConcurrentDictionary<string, VulkanFftGenericKernel> Kernels =
        new ConcurrentDictionary<string, VulkanFftGenericKernel>(StringComparer.Ordinal);

    /// <summary>Values of <paramref name="type"/> stored in one 32-bit word.</summary>
    public static int ElementsPerWord(FftElementType type) => type == FftElementType.Float32 ? 1 : 2;

    /// <summary>Words that hold <paramref name="count"/> values of <paramref name="type"/>.</summary>
    public static int WordsFor(int count, FftElementType type) => type == FftElementType.Float32 ? count : (int)(((long)count + 1) / 2);

    private const string Header = "#version 450\nlayout(local_size_x = 256) in;\n";

    // Load / store helpers: ld(word, slot) widens one stored value, pack(v0, v1) narrows the values of one word.
    private static string Storage(FftElementType type) => type switch
    {
        FftElementType.Float32 => @"
#define EPW 1u
float ld(uint word, uint slot) { return uintBitsToFloat(word); }
uint pack(float v0, float v1) { return floatBitsToUint(v0); }
",
        FftElementType.Float16 => @"
#define EPW 2u
// float32 -> binary16, round to nearest even (overflow gives infinity, NaN stays NaN).
uint enc(float v)
{
    uint b = floatBitsToUint(v);
    uint sgn = (b >> 16) & 0x8000u;
    uint a = b & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return sgn | 0x7E00u;
    if (a >= 0x47800000u) return sgn | 0x7C00u;
    if (a < 0x33000000u) return sgn;
    int e = int(a >> 23) - 127;
    if (e < -14)
    {
        // Subnormal result: the 24-bit significand shifted down to units of 2^-24.
        uint m = (a & 0x7FFFFFu) | 0x800000u;
        uint shift = uint(-1 - e);
        uint r = m >> shift;
        uint rem = m & ((1u << shift) - 1u);
        uint halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (r & 1u) != 0u)) r++;
        return sgn | r;
    }
    // A rounding carry out of the significand increments the exponent, up to infinity.
    uint r = (uint(e + 15) << 10) | ((a >> 13) & 0x3FFu);
    uint rem = a & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u) != 0u)) r++;
    return sgn | r;
}
float ld(uint word, uint slot) { return unpackHalf2x16((word >> (slot * 16u)) & 0xFFFFu).x; }
uint pack(float v0, float v1) { return enc(v0) | (enc(v1) << 16); }
",
        FftElementType.BFloat16 => @"
#define EPW 2u
// float32 -> bfloat16: the top 16 bits, round to nearest even (NaN stays NaN).
uint enc(float v)
{
    uint b = floatBitsToUint(v);
    if ((b & 0x7FFFFFFFu) > 0x7F800000u) return ((b >> 16) & 0x8000u) | 0x7FC0u;
    return (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;
}
float ld(uint word, uint slot) { return uintBitsToFloat(((word >> (slot * 16u)) & 0xFFFFu) << 16); }
uint pack(float v0, float v1) { return enc(v0) | (enc(v1) << 16); }
",
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown FFT element type."),
    };

    // (cos, sin) of pi * num / den for 0 <= num < den <= 2^29. The angle is reduced to an octant in integers before
    // any float is formed, so the phase keeps full float precision however long the transform is.
    private const string Cis = @"
vec2 cis_fraction(uint num, uint den)
{
    uint quadrant = (4u * num) / den;
    uint rem = 4u * num - quadrant * den;
    const float quarterPi = 0.78539816339744830962;
    float phi = quarterPi * (float(rem) / float(den));
    float psi = quarterPi * (float(den - rem) / float(den));
    if (quadrant == 0u) return vec2(cos(phi), sin(phi));
    if (quadrant == 1u) return vec2(sin(psi), cos(psi));
    if (quadrant == 2u) return vec2(-sin(phi), cos(phi));
    return vec2(-cos(psi), sin(psi));
}

// (hi * 2^32 + lo) mod d, for d <= 2^30.
uint mod64(uint hi, uint lo, uint d)
{
    uint rem = hi % d;
    for (int bit = 31; bit >= 0; bit--)
    {
        rem = (rem << 1) | ((lo >> uint(bit)) & 1u);
        if (rem >= d) rem -= d;
    }
    return rem;
}

// exp(i pi k^2 / n): k^2 is reduced mod 2n in integers, because the phase lives in its low bits.
vec2 chirp(uint k, uint n)
{
    uint hi, lo;
    umulExtended(k, k, hi, lo);
    uint r = mod64(hi, lo, 2u * n);
    if (r >= n) return -cis_fraction(r - n, n);
    return cis_fraction(r, n);
}
";

    private static VulkanFftGenericKernel Get(string name, int bindings, int pushWords, Func<string> source)
        => Kernels.GetOrAdd(name, key => new VulkanFftGenericKernel(key, source(), bindings, pushWords));

    private static string Suffix(FftElementType type) => type.KernelSuffix();

    /// <summary>Every kernel a transform with <paramref name="type"/> storage can launch.</summary>
    public static VulkanFftGenericKernel[] All(FftElementType type)
    {
        var kernels = new System.Collections.Generic.List<VulkanFftGenericKernel>
        {
            BuildTwiddles, BuildChirp, BuildChirpPadded, BluesteinPointwise,
            BitReverse(type), Butterfly(type), Scale(type), BluesteinPremultiply(type), BluesteinPostmultiply(type),
        };
        if (type != FftElementType.Float32)
        {
            kernels.Add(FromFloat(type));
            kernels.Add(ToFloat(type));
            // The Bluestein workspace is float32 whatever the storage type.
            kernels.Add(BitReverse(FftElementType.Float32));
            kernels.Add(Butterfly(FftElementType.Float32));
            kernels.Add(Scale(FftElementType.Float32));
        }
        return kernels.ToArray();
    }

    /// <summary>Twiddle table of a length-n radix-2 transform: w[j] = exp(-+2 pi i j / n) for j &lt; n / 2.</summary>
    public static VulkanFftGenericKernel BuildTwiddles => Get("fftg_build_twiddles", 2, 3, () => Header + @"
layout(set = 0, binding = 0) writeonly buffer TR { float twRe[]; };
layout(set = 0, binding = 1) writeonly buffer TI { float twIm[]; };
layout(push_constant) uniform P { uint halfCount; uint n; uint inv; };
" + Cis + @"
void main()
{
    uint j = gl_GlobalInvocationID.x;
    if (j >= halfCount) return;
    vec2 w = n > 1u ? cis_fraction(2u * j, n) : vec2(1.0, 0.0);
    twRe[j] = w.x;
    twIm[j] = inv != 0u ? w.y : -w.y;
}
");

    /// <summary>Bluestein chirp c[k] = exp(+i pi k^2 / n) (conjugated when inverse), length n.</summary>
    public static VulkanFftGenericKernel BuildChirp => Get("fftg_build_chirp", 2, 2, () => Header + @"
layout(set = 0, binding = 0) writeonly buffer CR { float chRe[]; };
layout(set = 0, binding = 1) writeonly buffer CI { float chIm[]; };
layout(push_constant) uniform P { uint n; uint inv; };
" + Cis + @"
void main()
{
    uint k = gl_GlobalInvocationID.x;
    if (k >= n) return;
    vec2 c = chirp(k, n);
    chRe[k] = c.x;
    chIm[k] = inv != 0u ? -c.y : c.y;
}
");

    /// <summary>
    /// Bluestein convolution kernel, length m, symmetrically extended: b[j] = c[j] for j &lt; n, b[m - j] = c[j] for
    /// 1 &lt;= j &lt; n, zero elsewhere. The mirror is what makes the m-point cyclic convolution equal the linear one.
    /// </summary>
    public static VulkanFftGenericKernel BuildChirpPadded => Get("fftg_build_chirp_padded", 2, 3, () => Header + @"
layout(set = 0, binding = 0) writeonly buffer BR { float bRe[]; };
layout(set = 0, binding = 1) writeonly buffer BI { float bIm[]; };
layout(push_constant) uniform P { uint n; uint m; uint inv; };
" + Cis + @"
void main()
{
    uint j = gl_GlobalInvocationID.x;
    if (j >= m) return;
    bool mirrored = j >= n && j + n >= m + 1u;
    if (j >= n && !mirrored) { bRe[j] = 0.0; bIm[j] = 0.0; return; }
    vec2 c = chirp(mirrored ? m - j : j, n);
    bRe[j] = c.x;
    bIm[j] = inv != 0u ? -c.y : c.y;
}
");

    /// <summary>Pointwise product of the padded signal spectrum with the transformed chirp spectrum (float32).</summary>
    public static VulkanFftGenericKernel BluesteinPointwise => Get("fftg_bluestein_pointwise", 4, 2, () => Header + @"
layout(set = 0, binding = 0) buffer R { float re[]; };
layout(set = 0, binding = 1) buffer I { float im[]; };
layout(set = 0, binding = 2) readonly buffer KR { float kRe[]; };
layout(set = 0, binding = 3) readonly buffer KI { float kIm[]; };
layout(push_constant) uniform P { uint total; uint m; };
void main()
{
    uint o = gl_GlobalInvocationID.x;
    if (o >= total) return;
    uint idx = o % m;
    float aR = re[o], aI = im[o];
    float bR = kRe[idx], bI = kIm[idx];
    re[o] = aR * bR - aI * bI;
    im[o] = aR * bI + aI * bR;
}
");

    /// <summary>Narrows float32 into typed storage, one word (two values) per thread.</summary>
    public static VulkanFftGenericKernel FromFloat(FftElementType type) => Get("fftg_from_f32" + Suffix(type), 2, 1, () => Header + @"
layout(set = 0, binding = 0) readonly buffer S { float src[]; };
layout(set = 0, binding = 1) writeonly buffer D { uint dst[]; };
layout(push_constant) uniform P { uint count; };
" + Storage(type) + @"
void main()
{
    uint w = gl_GlobalInvocationID.x;
    uint e = w * EPW;
    if (e >= count) return;
    float v0 = src[e];
    float v1 = e + 1u < count ? src[e + 1u] : 0.0;
    dst[w] = pack(v0, v1);
}
");

    /// <summary>Widens typed storage to float32.</summary>
    public static VulkanFftGenericKernel ToFloat(FftElementType type) => Get("fftg_to_f32" + Suffix(type), 2, 1, () => Header + @"
layout(set = 0, binding = 0) readonly buffer S { uint src[]; };
layout(set = 0, binding = 1) writeonly buffer D { float dst[]; };
layout(push_constant) uniform P { uint count; };
" + Storage(type) + @"
void main()
{
    uint e = gl_GlobalInvocationID.x;
    if (e >= count) return;
    dst[e] = ld(src[e / EPW], e % EPW);
}
");

    /// <summary>Bit-reversal permutation of every row into a scratch pair, one output word per thread.</summary>
    public static VulkanFftGenericKernel BitReverse(FftElementType type) => Get("fftg_batched_bit_reverse" + Suffix(type), 4, 3, () => Header + @"
layout(set = 0, binding = 0) readonly buffer SR { uint srcRe[]; };
layout(set = 0, binding = 1) readonly buffer SI { uint srcIm[]; };
layout(set = 0, binding = 2) writeonly buffer DR { uint dstRe[]; };
layout(set = 0, binding = 3) writeonly buffer DI { uint dstIm[]; };
layout(push_constant) uniform P { uint count; uint n; uint log2n; };
" + Storage(type) + @"
void main()
{
    uint w = gl_GlobalInvocationID.x;
    if (w * EPW >= count) return;
    float vr[2] = float[2](0.0, 0.0);
    float vi[2] = float[2](0.0, 0.0);
    for (uint q = 0u; q < EPW; q++)
    {
        uint e = w * EPW + q;
        if (e >= count) break;
        uint row = e / n, idx = e % n;
        uint rev = 0u, t = idx;
        for (uint i = 0u; i < log2n; i++) { rev = (rev << 1) | (t & 1u); t >>= 1; }
        uint s = row * n + rev;
        vr[q] = ld(srcRe[s / EPW], s % EPW);
        vi[q] = ld(srcIm[s / EPW], s % EPW);
    }
    dstRe[w] = pack(vr[0], vr[1]);
    dstIm[w] = pack(vi[0], vi[1]);
}
");

    /// <summary>
    /// One radix-2 stage, in place, over every row (rows are contiguous and a multiple of the stride long, so the
    /// butterflies are indexed over the whole buffer). A thread owns EPW adjacent wings: one word of tops and one word
    /// of bottoms. With two values per word the first stage (stride 2) has its top and bottom in the same word, so
    /// there a thread owns one word.
    /// </summary>
    public static VulkanFftGenericKernel Butterfly(FftElementType type) => Get("fftg_batched_butterfly" + Suffix(type), 4, 3, () => Header + @"
layout(set = 0, binding = 0) buffer R { uint re[]; };
layout(set = 0, binding = 1) buffer I { uint im[]; };
layout(set = 0, binding = 2) readonly buffer TR { float twRe[]; };
layout(set = 0, binding = 3) readonly buffer TI { float twIm[]; };
layout(push_constant) uniform P { uint count; uint n; uint stride; };
" + Storage(type) + @"
void main()
{
    uint t = gl_GlobalInvocationID.x;
    uint halfStride = stride >> 1;
    if (halfStride < EPW)
    {
        // stride 2 with two values per word: word t is (top, bottom), and the only twiddle is w[0].
        if (t * 2u >= count) return;
        uint wr = re[t], wi = im[t];
        float tR = ld(wr, 0u), bR = ld(wr, 1u), tI = ld(wi, 0u), bI = ld(wi, 1u);
        float twR = twRe[0], twI = twIm[0];
        float xR = bR * twR - bI * twI, xI = bR * twI + bI * twR;
        re[t] = pack(tR + xR, tR - xR);
        im[t] = pack(tI + xI, tI - xI);
        return;
    }
    uint groups = halfStride / EPW;
    uint block = t / groups, group = t % groups;
    uint top = block * stride + group * EPW;
    if (top >= count) return;
    uint bottom = top + halfStride;
    uint twStep = n / stride;
    uint topR = re[top / EPW], topI = im[top / EPW], bottomR = re[bottom / EPW], bottomI = im[bottom / EPW];
    float outTopR[2] = float[2](0.0, 0.0), outTopI[2] = float[2](0.0, 0.0);
    float outBottomR[2] = float[2](0.0, 0.0), outBottomI[2] = float[2](0.0, 0.0);
    for (uint q = 0u; q < EPW; q++)
    {
        uint wing = group * EPW + q;
        float twR = twRe[wing * twStep], twI = twIm[wing * twStep];
        float tR = ld(topR, q), tI = ld(topI, q), bR = ld(bottomR, q), bI = ld(bottomI, q);
        float xR = bR * twR - bI * twI, xI = bR * twI + bI * twR;
        outTopR[q] = tR + xR; outTopI[q] = tI + xI;
        outBottomR[q] = tR - xR; outBottomI[q] = tI - xI;
    }
    re[top / EPW] = pack(outTopR[0], outTopR[1]);
    im[top / EPW] = pack(outTopI[0], outTopI[1]);
    re[bottom / EPW] = pack(outBottomR[0], outBottomR[1]);
    im[bottom / EPW] = pack(outBottomI[0], outBottomI[1]);
}
");

    /// <summary>Scales every value (the inverse transform's 1 / n), one word per thread.</summary>
    public static VulkanFftGenericKernel Scale(FftElementType type) => Get("fftg_scale" + Suffix(type), 2, 2, () => Header + @"
layout(set = 0, binding = 0) buffer R { uint re[]; };
layout(set = 0, binding = 1) buffer I { uint im[]; };
layout(push_constant) uniform P { uint count; float s; };
" + Storage(type) + @"
void main()
{
    uint w = gl_GlobalInvocationID.x;
    uint e = w * EPW;
    if (e >= count) return;
    uint wr = re[w], wi = im[w];
    bool second = EPW > 1u && e + 1u < count;
    re[w] = pack(ld(wr, 0u) * s, second ? ld(wr, 1u) * s : 0.0);
    im[w] = pack(ld(wi, 0u) * s, second ? ld(wi, 1u) * s : 0.0);
}
");

    /// <summary>Bluestein step 1: x[k] * conj(chirp[k]) into the zero-padded float32 workspace of length m per row.</summary>
    public static VulkanFftGenericKernel BluesteinPremultiply(FftElementType type) => Get("fftg_bluestein_premul" + Suffix(type), 6, 3, () => Header + @"
layout(set = 0, binding = 0) readonly buffer XR { uint xRe[]; };
layout(set = 0, binding = 1) readonly buffer XI { uint xIm[]; };
layout(set = 0, binding = 2) readonly buffer CR { float chRe[]; };
layout(set = 0, binding = 3) readonly buffer CI { float chIm[]; };
layout(set = 0, binding = 4) writeonly buffer WR { float wRe[]; };
layout(set = 0, binding = 5) writeonly buffer WI { float wIm[]; };
layout(push_constant) uniform P { uint total; uint n; uint m; };
" + Storage(type) + @"
void main()
{
    uint o = gl_GlobalInvocationID.x;
    if (o >= total) return;
    uint row = o / m, idx = o % m;
    if (idx >= n) { wRe[o] = 0.0; wIm[o] = 0.0; return; }
    uint s = row * n + idx;
    float aR = ld(xRe[s / EPW], s % EPW), aI = ld(xIm[s / EPW], s % EPW);
    float cR = chRe[idx], cI = -chIm[idx];
    wRe[o] = aR * cR - aI * cI;
    wIm[o] = aR * cI + aI * cR;
}
");

    /// <summary>
    /// Bluestein step 3: the first n workspace values of each row times conj(chirp) and the scale, written back in the
    /// caller's storage type, one output word per thread (rows of odd length do not align to words).
    /// </summary>
    public static VulkanFftGenericKernel BluesteinPostmultiply(FftElementType type) => Get("fftg_bluestein_postmul" + Suffix(type), 6, 4, () => Header + @"
layout(set = 0, binding = 0) readonly buffer WR { float wRe[]; };
layout(set = 0, binding = 1) readonly buffer WI { float wIm[]; };
layout(set = 0, binding = 2) readonly buffer CR { float chRe[]; };
layout(set = 0, binding = 3) readonly buffer CI { float chIm[]; };
layout(set = 0, binding = 4) writeonly buffer YR { uint yRe[]; };
layout(set = 0, binding = 5) writeonly buffer YI { uint yIm[]; };
layout(push_constant) uniform P { uint count; uint n; uint m; float scale; };
" + Storage(type) + @"
void main()
{
    uint w = gl_GlobalInvocationID.x;
    if (w * EPW >= count) return;
    float vr[2] = float[2](0.0, 0.0);
    float vi[2] = float[2](0.0, 0.0);
    for (uint q = 0u; q < EPW; q++)
    {
        uint e = w * EPW + q;
        if (e >= count) break;
        uint row = e / n, idx = e % n;
        uint o = row * m + idx;
        float aR = wRe[o] * scale, aI = wIm[o] * scale;
        float cR = chRe[idx], cI = -chIm[idx];
        vr[q] = aR * cR - aI * cI;
        vi[q] = aR * cI + aI * cR;
    }
    yRe[w] = pack(vr[0], vr[1]);
    yIm[w] = pack(vi[0], vi[1]);
}
");
}
