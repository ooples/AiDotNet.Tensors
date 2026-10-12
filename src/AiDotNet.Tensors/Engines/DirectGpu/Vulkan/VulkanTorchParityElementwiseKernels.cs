// Copyright (c) AiDotNet. All rights reserved.
// Vulkan device kernels for the PyTorch-parity elementwise ops (#1127), the peers of
// CUDA/Kernels/CudaTorchParityElementwiseKernels.
//
// Each kernel mirrors the CPU reference lambda (CpuEngine.TorchMisc / TorchSpecial): the same branches, evaluated in
// double through VulkanTorchParityMath and rounded once to float. NaN is tested by bit pattern, and Math.Max / Math.Min
// clamps propagate a NaN from the input or either bound, while fmax / fmin ignore it.
//
// The bodies below are written with bare floating-point literals and passed through VulkanTorchParityMath.Doubles,
// which marks them all as doubles: do not mix float arithmetic into a double kernel.
using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>GLSL compute shaders for the Vulkan PyTorch-parity elementwise kernels.</summary>
internal static class VulkanTorchParityElementwiseKernels
{
    /// <summary>Elements per work group of every kernel here except the scan.</summary>
    public const int LocalSize = 256;

    /// <summary>Elements per work group (and per shared-memory block) of the inclusive scan.</summary>
    public const int ScanBlock = 256;

    private static readonly ConcurrentDictionary<string, string> Sources = new ConcurrentDictionary<string, string>(StringComparer.Ordinal);

    private static string Cached(string key, Func<string> build) => Sources.GetOrAdd(key, _ => build());

    /// <summary>Every shader of this kernel set, by name: what a test compiles to prove the sources are valid GLSL.</summary>
    public static System.Collections.Generic.IEnumerable<(string Name, string Source)> All()
    {
        foreach (TorchParityUnaryOp op in Enum.GetValues(typeof(TorchParityUnaryOp))) yield return ("unary." + op, Unary(op));
        foreach (TorchParityBinaryOp op in Enum.GetValues(typeof(TorchParityBinaryOp))) yield return ("binary." + op, Binary(op));
        yield return ("polynomial", Polynomial);
        yield return ("range", Range);
        yield return ("logspace", Logspace);
        yield return ("window", Window);
        yield return ("tri_indices", TriIndices);
        yield return ("diagflat", Diagflat);
        yield return ("diagonal_scatter", DiagonalScatter);
        yield return ("quantile_rows", QuantileRows);
        yield return ("flag_reset", FlagReset);
        yield return ("flag_merge", FlagMerge);
        yield return ("sample", Sample);
        yield return ("bernoulli", Bernoulli);
        yield return ("poisson", Poisson);
        yield return ("binomial", Binomial);
        yield return ("multinomial.replacement", Multinomial(true));
        yield return ("multinomial", Multinomial(false));
        yield return ("dropout_affine.alpha", DropoutAffine(true));
        yield return ("dropout_affine", DropoutAffine(false));
        yield return ("rrelu", Rrelu);
        yield return ("random_keys", RandomKeys);
        yield return ("nonzero_static_scatter", NonzeroStaticScatter);
        yield return ("unravel_axis", UnravelAxis);
        yield return ("slice_bitonic_step", SliceBitonicStep);
        yield return ("slice_first_flags", SliceFirstFlags);
        yield return ("slice_gather_unique", SliceGatherUnique);
        yield return ("scan_block", ScanBlockSource);
        yield return ("scan_add", ScanAdd);
        yield return ("int_binary", IntBinary);
        yield return ("int_not", IntNot);
        yield return ("int64_binary", Int64Binary);
        yield return ("int64_not", Int64Not);
        yield return ("grid_sample_3d", GridSample3D);
        yield return ("grid_sample_3d_backward", GridSample3DBackward);
    }

    private const string Int64Extension = "#extension GL_ARB_gpu_shader_int64 : require\n";

    private static string Head(bool int64, int localSize = LocalSize)
        => "#version 450\n" + (int64 ? Int64Extension : "") + "layout(local_size_x = " + localSize + ") in;\n";

    // A kernel that computes in double: header, declarations, the double library, then the body.
    private static string DoubleShader(string declarations, string body, bool special, bool int64 = false)
        => Head(int64) + declarations + VulkanTorchParityMath.Tables + VulkanTorchParityMath.ElementaryFunctions
            + Conversions + (special ? VulkanTorchParityMath.SpecialFunctionsSource : "") + VulkanTorchParityMath.Doubles(body);

    // Buffer values cross between float and double by bit pattern: D widens exactly and F rounds to nearest even,
    // both including the subnormal floats a GPU's own conversions flush to zero (logit of 6e-39 must not be -inf).
    private const string Conversions = @"
double D(float f) { return d_from_float_bits(floatBitsToUint(f)); }
float F(double d) { return uintBitsToFloat(d_to_float_bits(d)); }
float F(int v) { return float(v); }
// A positive weight, subnormal included (NaN and infinity are rejected before this is asked).
bool f_positive(float f) { uint u = floatBitsToUint(f); return u != 0u && u <= 0x7F800000u; }
";

    // A kernel in float / integer arithmetic only: runs on devices without shaderFloat64.
    private static string FloatShader(string declarations, string body, bool int64 = false, int localSize = LocalSize)
        => Head(int64, localSize) + declarations + FloatHelpers + body;

    private const string FloatHelpers = @"
bool f_isnan(float x) { return (floatBitsToUint(x) & 0x7FFFFFFFu) > 0x7F800000u; }
bool f_isinf(float x) { return (floatBitsToUint(x) & 0x7FFFFFFFu) == 0x7F800000u; }
float f_nan() { return uintBitsToFloat(0x7FC00000u); }
";

    /// <summary>
    /// y = <paramref name="expression"/> of the double x, over doubles stored as two words each: how the tests measure
    /// the double library against .NET, one function at a time.
    /// </summary>
    public static string DoubleProbe(string expression, bool special) => Cached("probe:" + special + ":" + expression, () => DoubleShader(@"
layout(set = 0, binding = 0) readonly buffer A { uint a[]; };
layout(set = 0, binding = 1) writeonly buffer O { uint o[]; };
layout(push_constant) uniform P { uint n; };
", @"
void main()
{
" + Index + @"
    double x = packDouble2x32(uvec2(a[2u * i], a[2u * i + 1u]));
    double y = " + expression + @";
    uvec2 bits = unpackDouble2x32(y);
    o[2u * i] = bits.x;
    o[2u * i + 1u] = bits.y;
}
", special));

    private const string UnaryBuffers = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { uint n; float p0; float p1; };
";

    private const string BinaryBuffers = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float b[]; };
layout(set = 0, binding = 2) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { uint n; };
";

    private const string Index = "    uint i = gl_GlobalInvocationID.x;\n    if (i >= n) return;\n";

    // ------------------------------------------------------------------------------------------------------------
    // Unary and binary elementwise ops
    // ------------------------------------------------------------------------------------------------------------

    /// <summary>True when <paramref name="op"/>'s kernel computes in double (and so needs shaderFloat64).</summary>
    public static bool NeedsDoubles(TorchParityUnaryOp op) => op switch
    {
        TorchParityUnaryOp.Hardtanh or TorchParityUnaryOp.Softsign or TorchParityUnaryOp.Signbit or TorchParityUnaryOp.IsPosInf
            or TorchParityUnaryOp.IsNegInf or TorchParityUnaryOp.Angle or TorchParityUnaryOp.NonZero => false,
        _ => true,
    };

    /// <summary>True when <paramref name="op"/>'s kernel computes in double (and so needs shaderFloat64).</summary>
    public static bool NeedsDoubles(TorchParityBinaryOp op)
        => op is TorchParityBinaryOp.Igamma or TorchParityBinaryOp.Igammac or TorchParityBinaryOp.FloorDivide;

    public static string Unary(TorchParityUnaryOp op) => Cached("unary:" + op, () =>
    {
        if (!NeedsDoubles(op))
        {
            string f = op switch
            {
                // Math.Min(Math.Max(x, lo), hi), a NaN from x or either bound propagating.
                TorchParityUnaryOp.Hardtanh => "(f_isnan(v) || f_isnan(p0) || f_isnan(p1)) ? f_nan() : ((v < p0 ? p0 : v) > p1 ? p1 : (v < p0 ? p0 : v))",
                TorchParityUnaryOp.Softsign => "v / (1.0 + abs(v))",
                TorchParityUnaryOp.Signbit => "(floatBitsToUint(v) >> 31) != 0u ? 1.0 : 0.0",
                TorchParityUnaryOp.IsPosInf => "floatBitsToUint(v) == 0x7F800000u ? 1.0 : 0.0",
                TorchParityUnaryOp.IsNegInf => "floatBitsToUint(v) == 0xFF800000u ? 1.0 : 0.0",
                // angle of a real: NaN stays NaN, negatives (including -inf) give pi, everything else (including -0) 0.
                TorchParityUnaryOp.Angle => "f_isnan(v) ? v : (v < 0.0 ? 3.14159274101257324 : 0.0)",
                // all / any indicator: 1 where non-zero (NaN counts), else 0.
                _ => "(floatBitsToUint(v) & 0x7FFFFFFFu) != 0u ? 1.0 : 0.0",
            };
            return FloatShader(UnaryBuffers, "void main()\n{\n" + Index + "    float v = a[i];\n    o[i] = " + f + ";\n}\n");
        }

        string d = op switch
        {
            TorchParityUnaryOp.Celu => "v > 0.0 ? v : D(p0) * (d_exp(v / D(p0)) - 1.0)",
            // min(x, 0) - log(1 + exp(-|x|)), as the CPU writes it.
            TorchParityUnaryOp.LogSigmoid => "d_isnan(v) ? v : (v < 0.0 ? v : 0.0) - d_log(1.0 + d_exp(-abs(v)))",
            TorchParityUnaryOp.Logit => "parity_logit(v, D(p0), D(p1))",
            TorchParityUnaryOp.Sinc => "v == 0.0 ? 1.0 : (d_isinf(v) ? 0.0 : sf_sin_pi(v) / (SF_PI * v))",
            TorchParityUnaryOp.Erf => "sf_erf(v)",
            TorchParityUnaryOp.BesselJ0 => "sf_bessel_j(0, v)",
            TorchParityUnaryOp.BesselJ1 => "sf_bessel_j(1, v)",
            TorchParityUnaryOp.BesselY0 => "sf_bessel_y(0, v)",
            TorchParityUnaryOp.BesselY1 => "sf_bessel_y(1, v)",
            TorchParityUnaryOp.ModifiedBesselI0 => "sf_bessel_i(0, v)",
            TorchParityUnaryOp.ModifiedBesselI1 => "sf_bessel_i(1, v)",
            TorchParityUnaryOp.Erfcx => "sf_erfcx(v)",
            TorchParityUnaryOp.Ndtr => "sf_ndtr(v)",
            TorchParityUnaryOp.Ndtri => "sf_ndtri(v)",
            TorchParityUnaryOp.Entr => "d_isnan(v) ? v : (v > 0.0 ? -v * d_log(v) : (v == 0.0 ? 0.0 : -d_inf()))",
            TorchParityUnaryOp.LogNdtr => "sf_log_ndtr(v)",
            TorchParityUnaryOp.ModifiedBesselK0 => "sf_bessel_k(0.0, v)",
            TorchParityUnaryOp.ModifiedBesselK1 => "sf_bessel_k(1.0, v)",
            TorchParityUnaryOp.ScaledModifiedBesselK0 => "sf_bessel_k_scaled(0.0, v)",
            TorchParityUnaryOp.ScaledModifiedBesselK1 => "sf_bessel_k_scaled(1.0, v)",
            TorchParityUnaryOp.SphericalBesselJ0 => "sf_spherical_bessel_j0(v)",
            TorchParityUnaryOp.AiryAi => "sf_airy_ai(v)",
            TorchParityUnaryOp.Mvlgamma => "parity_mvlgamma(v, int(p0))",
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        const string helpers = @"
// c = Math.Min(Math.Max(x, lo), hi) (lo / hi are -inf / +inf without eps), a NaN from x or a bound propagating;
// then log(c / (1 - c)).
double parity_logit(double v, double lo, double hi)
{
    if (d_isnan(v) || d_isnan(lo) || d_isnan(hi)) return d_nan();
    double c = v < lo ? lo : v;
    c = c > hi ? hi : c;
    return d_log(c / (1.0 - c));
}

// mvlgamma (p >= 1): NaN at or below (p - 1) / 2, else p (p - 1) / 4 log(pi) + sum_j lgamma(x - j / 2).
double parity_mvlgamma(double v, int p)
{
    if (v <= double(p - 1) / 2.0) return d_nan();
    return sf_mvlgamma(v, p);
}
";
        return DoubleShader(UnaryBuffers, helpers + "void main()\n{\n" + Index + "    double v = D(a[i]);\n    o[i] = F(" + d + ");\n}\n",
            special: true);
    });

    public static string Binary(TorchParityBinaryOp op) => Cached("binary:" + op, () =>
    {
        if (!NeedsDoubles(op))
        {
            string f = op switch
            {
                // NaN-ignoring max / min.
                TorchParityBinaryOp.Fmax => "f_isnan(x) ? y : (f_isnan(y) ? x : max(x, y))",
                TorchParityBinaryOp.Fmin => "f_isnan(x) ? y : (f_isnan(y) ? x : min(x, y))",
                TorchParityBinaryOp.Heaviside => "f_isnan(x) ? x : (x < 0.0 ? 0.0 : (x > 0.0 ? 1.0 : y))",
                // A comparison with NaN is false, as in .NET.
                TorchParityBinaryOp.GreaterEqual => "(!f_isnan(x) && !f_isnan(y) && x >= y) ? 1.0 : 0.0",
                _ => "(!f_isnan(x) && !f_isnan(y) && x <= y) ? 1.0 : 0.0",
            };
            return FloatShader(BinaryBuffers, "void main()\n{\n" + Index + "    float x = a[i], y = b[i];\n    o[i] = " + f + ";\n}\n");
        }

        string d = op switch
        {
            TorchParityBinaryOp.Igamma => "sf_igamma(x, y)",
            TorchParityBinaryOp.Igammac => "sf_igammac(x, y)",
            // floor(a / b) in double, as the CPU's Math.Floor(x / y).
            _ => "floor(x / y)",
        };
        return DoubleShader(BinaryBuffers,
            "void main()\n{\n" + Index + "    double x = D(a[i]), y = D(b[i]);\n    o[i] = F(" + d + ");\n}\n",
            special: op != TorchParityBinaryOp.FloorDivide);
    });

    /// <summary>
    /// Orthogonal polynomials: the CPU recurrence in double. Degree: NaN gives NaN; truncated toward zero; negative gives
    /// 0 (checked before a NaN x, as on the CPU). The host keeps infinite / &gt; 2^20 degrees on the CPU path.
    /// </summary>
    public static string Polynomial => Cached("polynomial", () => DoubleShader(@"
layout(set = 0, binding = 0) readonly buffer X { float x[]; };
layout(set = 0, binding = 1) readonly buffer Deg { float degree[]; };
layout(set = 0, binding = 2) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { uint n; int kind; int shifted; };
", @"
void main()
{
" + Index + @"
    double d = D(degree[i]);
    if (d_isnan(d)) { o[i] = F(d_nan()); return; }
    double t = d_trunc(d);
    if (t < 0.0) { o[i] = 0.0f; return; }
    if (t > 1048576.0) { o[i] = F(d_nan()); return; }
    double v = D(x[i]);
    if (d_isnan(v)) { o[i] = F(d_nan()); return; }
    if (shifted != 0) v = 2.0 * v - 1.0;
    o[i] = F(sf_polynomial(kind, v, int(t)));
}
", special: true));

    // ------------------------------------------------------------------------------------------------------------
    // Generators: the CPU's FromDoubles lambdas in double, same operation order, rounded once to float.
    // A double parameter arrives as two push-constant words (low, high).
    // ------------------------------------------------------------------------------------------------------------

    private const string Unpack = "double U(uint lo, uint hi) { return packDouble2x32(uvec2(lo, hi)); }\n";

    /// <summary>arange / range / linspace-style points: start + i * span / denominator.</summary>
    public static string Range => Cached("range", () => DoubleShader(@"
layout(set = 0, binding = 0) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { uint n; uint pad0; uint startLo; uint startHi; uint spanLo; uint spanHi; uint denLo; uint denHi; };
", Unpack + @"
void main()
{
" + Index + @"
    precise double scaled = double(i) * U(spanLo, spanHi);
    precise double v = U(startLo, startHi) + scaled / U(denLo, denHi);
    o[i] = F(v);
}
", special: false));

    /// <summary>logspace: base ^ (single ? start : start + i * span / denominator).</summary>
    public static string Logspace => Cached("logspace", () => DoubleShader(@"
layout(set = 0, binding = 0) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { uint n; uint pad0; uint startLo; uint startHi; uint spanLo; uint spanHi; uint denLo; uint denHi; uint baseLo; uint baseHi; uint single; };
", Unpack + @"
void main()
{
" + Index + @"
    precise double scaled = double(i) * U(spanLo, spanHi);
    precise double e = single != 0u ? U(startLo, startHi) : U(startLo, startHi) + scaled / U(denLo, denHi);
    o[i] = F(d_pow(U(baseLo, baseHi), e));
}
", special: false));

    /// <summary>Windows over t = i / (m - 1): 0 Hann, 1 Hamming(alpha, beta), 2 Blackman, 3 Bartlett, 4 Kaiser(beta = alpha).</summary>
    public static string Window => Cached("window", () => DoubleShader(@"
layout(set = 0, binding = 0) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { uint n; int kind; int m; uint pad0; uint alphaLo; uint alphaHi; uint betaLo; uint betaHi; };
", Unpack + @"
void main()
{
" + Index + @"
    if (n == 1u) { o[i] = 1.0f; return; }
    double alpha = U(alphaLo, alphaHi), beta = U(betaLo, betaHi);
    double t = double(i) / double(m - 1), v;
    if (kind == 0) v = 0.5 - 0.5 * d_cos(2.0 * SF_PI * t);
    else if (kind == 1) v = alpha - beta * d_cos(2.0 * SF_PI * t);
    else if (kind == 2) v = 0.42 - 0.5 * d_cos(2.0 * SF_PI * t) + 0.08 * d_cos(4.0 * SF_PI * t);
    else if (kind == 3) v = 1.0 - abs(2.0 * t - 1.0);
    else
    {
        double r = 2.0 * t - 1.0;
        v = sf_bessel_i(0, alpha * sqrt(max(0.0, 1.0 - r * r))) / sf_bessel_i(0, alpha);
    }
    o[i] = F(v);
}
", special: true));

    /// <summary>
    /// tril / triu indices as a [2, total] tensor (row indices, then column indices), one thread per row; each row's
    /// output offset is the kept-count of the rows before it.
    /// </summary>
    public static string TriIndices => Cached("triIndices", () => FloatShader(@"
layout(set = 0, binding = 0) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { int row; int col; int offset; int lower; int total; };
", @"
// Columns kept in row r: lower keeps j - r <= offset, upper keeps j - r >= offset.
void tri_columns(int r, out int lo, out int hi)
{
    int edge = r + offset;
    if (lower != 0) { lo = 0; hi = min(edge, col - 1); }
    else { lo = max(edge, 0); hi = col - 1; }
}

void main()
{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= row) return;
    int lo, hi, at = 0;
    for (int r = 0; r < i; r++)
    {
        tri_columns(r, lo, hi);
        if (hi >= lo) at += hi - lo + 1;
    }
    tri_columns(i, lo, hi);
    for (int j = lo; j <= hi; j++, at++)
    {
        o[at] = float(i);
        o[total + at] = float(j);
    }
}
"));

    // ------------------------------------------------------------------------------------------------------------
    // Index, reduction-feeding and order-statistic ops
    // ------------------------------------------------------------------------------------------------------------

    /// <summary>diagflat: a size x size matrix with the n input values on diagonal `offset`, zeros elsewhere.</summary>
    public static string Diagflat => Cached("diagflat", () => FloatShader(@"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { int total; int size; int offset; int n; };
", @"
void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= total) return;
    int r = idx / size, c = idx % size;
    int k = offset >= 0 ? (c - r == offset ? r : -1) : (r - c == -offset ? c : -1);
    o[idx] = (k >= 0 && k < n) ? a[k] : 0.0;
}
"));

    /// <summary>
    /// diagonal_scatter: writes src (torch.diagonal's layout: the other axes in order, then the diagonal) into the copy of
    /// the input already in the output.
    /// </summary>
    public static string DiagonalScatter => Cached("diagonalScatter", () => FloatShader(@"
layout(set = 0, binding = 0) readonly buffer S { float src[]; };
layout(set = 0, binding = 1) buffer O { float o[]; };
layout(push_constant) uniform P { int count; int otherRank; int otherExtent[8]; int otherStride[8]; int stride1; int stride2; int start1; int start2; int diagonal; };
", @"
void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= count) return;
    int outer = idx / diagonal, t = idx % diagonal;
    int base = 0, rest = outer;
    for (int j = otherRank - 1; j >= 0; j--) { base += rest % otherExtent[j] * otherStride[j]; rest /= otherExtent[j]; }
    o[base + (t + start1) * stride1 + (t + start2) * stride2] = src[idx];
}
"));

    /// <summary>
    /// quantile over sorted rows (ascending, NaN last). mode: 0 linear, 1 lower, 2 higher, 3 nearest (ties to even),
    /// 4 midpoint. Without ignoreNan a row holding a NaN gives NaN; a row with no values gives NaN.
    /// </summary>
    public static string QuantileRows => Cached("quantileRows", () => DoubleShader(@"
layout(set = 0, binding = 0) readonly buffer S { float sorted[]; };
layout(set = 0, binding = 1) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { int rows; int length; uint qLo; uint qHi; int mode; int ignoreNan; };
", Unpack + @"
bool f_isnan(float x) { return (floatBitsToUint(x) & 0x7FFFFFFFu) > 0x7F800000u; }

void main()
{
    int r = int(gl_GlobalInvocationID.x);
    if (r >= rows) return;
    int start = r * length;
    int lo = 0, hi = length;   // the first NaN (NaN sorts last)
    while (lo < hi) { int mid = (lo + hi) >> 1; if (f_isnan(sorted[start + mid])) hi = mid; else lo = mid + 1; }
    int count = lo;
    if (count == 0 || (count < length && ignoreNan == 0)) { o[r] = F(d_nan()); return; }
    double position = U(qLo, qHi) * double(count - 1);
    int below = int(floor(position)), above = int(ceil(position));
    double w = position - double(below);
    if (mode == 1) { above = below; w = 0.0; }
    else if (mode == 2) { below = above; w = 0.0; }
    else if (mode == 3) { below = int(d_rint(position)); above = below; w = 0.0; }
    else if (mode == 4) { w = below == above ? 0.0 : 0.5; }
    precise double v = (1.0 - w) * D(sorted[start + below]) + w * D(sorted[start + above]);
    o[r] = F(v);
}
", special: false));

    // ------------------------------------------------------------------------------------------------------------
    // Random sampling: the counter-based Philox4x32-10 generator of the CUDA kernels (same streams). Element i draws
    // its k-th uniform from counter (i, k), so a seed gives the same tensor on every run and launch shape.
    //
    // Invalid per-element parameters do not produce values. CUDA lowers a 64-bit flag with a 64-bit atomicMin; GLSL has
    // 32-bit atomics only, so a kernel lowers the offending element's index into a slot per (rank, code) of a small
    // scratch buffer, and FlagMerge then folds the slots into the 64-bit flag ((rank << 58 | index) << 3 | code).
    // ------------------------------------------------------------------------------------------------------------

    private static string FlagSlots(int binding) => @"
layout(set = 0, binding = " + binding + @") buffer FlagSlots { uint flagSlots[]; };
void parity_flag(uint rank, uint element, uint code) { atomicMin(flagSlots[rank * 4u + code], element); }
";

    private const string Rng = @"
struct ParityRng { uint key0; uint key1; uint element; uint draw; };

ParityRng parity_rng(uint seedLo, uint seedHi, uint element)
{
    ParityRng r;
    r.key0 = seedLo; r.key1 = seedHi; r.element = element; r.draw = 0u;
    return r;
}

// A uniform double in [0, 1) with 53 random bits.
double parity_uniform(inout ParityRng r)
{
    uvec4 c = uvec4(r.element, 0u, r.draw, 0x5EEDu);
    r.draw += 1u;
    uint k0 = r.key0, k1 = r.key1;
    for (int round = 0; round < 10; round++)
    {
        uint lo0, hi0, lo1, hi1;
        umulExtended(0xD2511F53u, c.x, hi0, lo0);
        umulExtended(0xCD9E8D57u, c.z, hi1, lo1);
        c = uvec4(hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0);
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return (double(c.x >> 5) * 67108864.0 + double(c.y >> 6)) * (1.0 / 9007199254740992.0);
}

// Box-Muller, as the CPU's StandardNormal: sqrt(-2 log(1 - U1)) cos(2 pi U2).
double parity_standard_normal(inout ParityRng r)
{
    double u1 = parity_uniform(r), u2 = parity_uniform(r);
    return sqrt(-2.0 * d_log(1.0 - u1)) * d_cos(2.0 * 3.141592653589793 * u2);
}
";

    /// <summary>Sets a 64-bit validation flag to "no error".</summary>
    public static string FlagReset => Cached("flagReset", () => FloatShader(@"
layout(set = 0, binding = 0) writeonly buffer F { uint flag[]; };
layout(push_constant) uniform P { uint unused; };
", @"
void main()
{
    if (gl_GlobalInvocationID.x != 0u) return;
    flag[0] = 0xFFFFFFFFu;
    flag[1] = 0xFFFFFFFFu;
}
"));

    /// <summary>Folds the scratch slots into the 64-bit flag (kept at its minimum) and clears them.</summary>
    public static string FlagMerge => Cached("flagMerge", () => FloatShader(@"
layout(set = 0, binding = 0) buffer S { uint slots[]; };
layout(set = 0, binding = 1) buffer F { uint flag[]; };
layout(push_constant) uniform P { uint unused; };
", @"
void main()
{
    if (gl_GlobalInvocationID.x != 0u) return;
    uint bestLo = flag[0], bestHi = flag[1];
    for (uint rank = 0u; rank < 2u; rank++)
        for (uint code = 1u; code < 4u; code++)
        {
            uint element = slots[rank * 4u + code];
            if (element == 0xFFFFFFFFu) continue;
            slots[rank * 4u + code] = 0xFFFFFFFFu;
            uint lo = (element << 3) | code, hi = (element >> 29) | (rank << 29);
            if (hi < bestHi || (hi == bestHi && lo < bestLo)) { bestHi = hi; bestLo = lo; }
        }
    flag[0] = bestLo;
    flag[1] = bestHi;
}
"));

    /// <summary>
    /// Shape-only distributions: 0 uniform(a = low, b = high), 1 cauchy(a = median, b = sigma), 2 exponential(a = rate),
    /// 3 geometric(a = p), 4 log-normal(a = mean, b = std), 5 standard normal, 6 randint(a = low, b = span).
    /// </summary>
    public static string Sample => Cached("sample", () => DoubleShader(@"
layout(set = 0, binding = 0) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { uint n; int kind; uint aLo; uint aHi; uint bLo; uint bHi; uint seedLo; uint seedHi; };
", Unpack + Rng + @"
void main()
{
" + Index + @"
    double a = U(aLo, aHi), b = U(bLo, bHi);
    ParityRng r = parity_rng(seedLo, seedHi, i);
    double v;
    if (kind == 0) v = a + (b - a) * parity_uniform(r);
    else if (kind == 1) v = a + b * d_tan(3.141592653589793 * (parity_uniform(r) - 0.5));
    else if (kind == 2) v = -d_log(1.0 - parity_uniform(r)) / a;
    else if (kind == 3) v = a == 1.0 ? 1.0 : max(1.0, ceil(d_log(1.0 - parity_uniform(r)) / d_log1p(-a)));
    else if (kind == 4) v = d_exp(a + b * parity_standard_normal(r));
    else if (kind == 5) v = parity_standard_normal(r);
    else v = a + min(floor(parity_uniform(r) * b), b - 1.0);
    o[i] = F(v);
}
", special: false));

    /// <summary>Bernoulli: 1 with probability p; code 1 for p outside [0, 1] or NaN.</summary>
    public static string Bernoulli => Cached("bernoulli", () => DoubleShader(@"
layout(set = 0, binding = 0) readonly buffer Pr { float p[]; };
layout(set = 0, binding = 1) writeonly buffer O { float o[]; };
" + FlagSlots(2) + @"
layout(push_constant) uniform P { uint n; uint pad0; uint seedLo; uint seedHi; };
", Rng + @"
void main()
{
" + Index + @"
    double q = D(p[i]);
    if (d_isnan(q) || q < 0.0 || q > 1.0) { parity_flag(0u, i, 1u); return; }
    ParityRng r = parity_rng(seedLo, seedHi, i);
    o[i] = parity_uniform(r) < q ? 1.0f : 0.0f;
}
", special: false));

    /// <summary>Poisson: Knuth's product method below rate 10, Hormann's PTRS above; code 1 for a negative or NaN rate.</summary>
    public static string Poisson => Cached("poisson", () => DoubleShader(@"
layout(set = 0, binding = 0) readonly buffer R { float rates[]; };
layout(set = 0, binding = 1) writeonly buffer O { float o[]; };
" + FlagSlots(2) + @"
layout(push_constant) uniform P { uint n; uint pad0; uint seedLo; uint seedHi; };
", Rng + @"
void main()
{
" + Index + @"
    double rate = D(rates[i]);
    if (d_isnan(rate) || rate < 0.0) { parity_flag(0u, i, 1u); return; }
    if (rate == 0.0) { o[i] = 0.0f; return; }
    ParityRng r = parity_rng(seedLo, seedHi, i);
    if (rate < 10.0)
    {
        double limit = d_exp(-rate), product = parity_uniform(r);
        int k = 0;
        while (product > limit) { k++; product *= parity_uniform(r); }
        o[i] = F(k);
        return;
    }
    double slam = sqrt(rate), logLam = d_log(rate);
    double bb = 0.931 + 2.53 * slam, aa = -0.059 + 0.02483 * bb, invAlpha = 1.1239 + 1.1328 / (bb - 3.4), vr = 0.9277 - 3.6224 / (bb - 2.0);
    while (true)
    {
        double u = parity_uniform(r) - 0.5, v = parity_uniform(r), us = 0.5 - abs(u);
        double k = floor((2.0 * aa / us + bb) * u + rate + 0.43);
        if (us >= 0.07 && v <= vr) { o[i] = F(k); return; }
        if (k < 0.0 || (us < 0.013 && v > us)) continue;
        if (d_log(v) + d_log(invAlpha) - d_log(aa / (us * us) + bb) <= -rate + k * logLam - sf_lgamma(k + 1.0)) { o[i] = F(k); return; }
    }
}
", special: true));

    /// <summary>
    /// Binomial: geometric waiting times while n min(p, 1-p) &lt; 10, Hormann's BTRS above (as the CPU and PyTorch).
    /// Code 1: count not a whole number of trials within the long range; code 2: count &lt; 0, or p outside [0, 1] / NaN.
    /// </summary>
    public static string Binomial => Cached("binomial", () => DoubleShader(@"
layout(set = 0, binding = 0) readonly buffer C { float counts[]; };
layout(set = 0, binding = 1) readonly buffer Pr { float probs[]; };
layout(set = 0, binding = 2) writeonly buffer O { float o[]; };
" + FlagSlots(3) + @"
layout(push_constant) uniform P { uint n; uint pad0; uint seedLo; uint seedHi; };
", Rng + @"
double parity_stirling_tail(double k)
{
    const double table[10] = double[10](
        0.08106146679532726, 0.04134069595540929, 0.02767792568499834, 0.02079067210376509, 0.01664469118982119,
        0.01387612882307075, 0.01189670994589177, 0.01041126526197209, 0.009255462182712733, 0.008330563433362871);
    if (k <= 9.0) return table[int(k)];
    double kp1sq = (k + 1.0) * (k + 1.0);
    return (1.0 / 12.0 - (1.0 / 360.0 - 1.0 / 1260.0 / kp1sq) / kp1sq) / (k + 1.0);
}

void main()
{
" + Index + @"
    double count = D(counts[i]), p = D(probs[i]);
    if (d_isnan(count) || d_isinf(count) || count != floor(count) || count >= 9.2233720368547758E18) { parity_flag(0u, i, 1u); return; }
    if (count < 0.0 || d_isnan(p) || p < 0.0 || p > 1.0) { parity_flag(0u, i, 2u); return; }
    if (count == 0.0 || p == 0.0) { o[i] = 0.0f; return; }
    if (p == 1.0) { o[i] = F(count); return; }
    int64_t trials = int64_t(count);
    ParityRng r = parity_rng(seedLo, seedHi, i);
    bool flip = p > 0.5;
    double q = flip ? 1.0 - p : p, draw;
    if (count * q >= 10.0)
    {
        double stddev = sqrt(count * q * (1.0 - q)), bb = 1.15 + 2.53 * stddev, aa = -0.0873 + 0.0248 * bb + 0.01 * q;
        double c = count * q + 0.5, vR = 0.92 - 4.2 / bb, ratio = q / (1.0 - q), alpha = (2.83 + 5.1 / bb) * stddev;
        double m = floor(double(trials + 1L) * q);
        while (true)
        {
            double u = parity_uniform(r) - 0.5, v = parity_uniform(r), us = 0.5 - abs(u);
            double k = floor((2.0 * aa / us + bb) * u + c);
            if (k < 0.0 || k > count) continue;
            if (us >= 0.07 && v <= vR) { draw = k; break; }
            v = d_log(v * alpha / (aa / (us * us) + bb));
            double upper = (m + 0.5) * d_log((m + 1.0) / (ratio * (count - m + 1.0)))
                + (count + 1.0) * d_log((count - m + 1.0) / (count - k + 1.0))
                + (k + 0.5) * d_log(ratio * (count - k + 1.0) / (k + 1.0))
                + parity_stirling_tail(m) + parity_stirling_tail(count - m) - parity_stirling_tail(k) - parity_stirling_tail(count - k);
            if (v <= upper) { draw = k; break; }
        }
    }
    else
    {
        // Each step is checked against the remaining trials before it is added, so an enormous (or infinite) step ends
        // the walk instead of overflowing the position.
        double logQ = d_log1p(-q);
        int64_t successes = 0L, position = 0L;
        while (true)
        {
            double advance = floor(d_log(1.0 - parity_uniform(r)) / logQ) + 1.0;
            if (d_isnan(advance) || !(advance <= double(trials - position))) break;
            position += int64_t(advance);
            successes++;
        }
        draw = double(successes);
    }
    o[i] = F(flip ? count - draw : draw);
}
", special: false, int64: true));

    /// <summary>
    /// Multinomial: one thread per row walks numSamples draws over its categories, as the CPU. Code 1: a negative, NaN
    /// or infinite weight (flagged at its element); code 2: a row with no positive weight left (flagged at the row,
    /// ranked after every weight error). With replacement the scratch holds double prefix sums (2 words per category).
    /// </summary>
    public static string Multinomial(bool replacement) => Cached("multinomial:" + replacement, () => DoubleShader(@"
layout(set = 0, binding = 0) readonly buffer W { float w[]; };
layout(set = 0, binding = 1) buffer S { " + (replacement ? "double prefix[];" : "float s[];") + @" };
layout(set = 0, binding = 2) writeonly buffer O { float o[]; };
" + FlagSlots(3) + @"
layout(push_constant) uniform P { uint rows; int categories; int numSamples; uint pad0; uint seedLo; uint seedHi; };
", Rng + @"
void main()
{
    uint row = gl_GlobalInvocationID.x;
    if (row >= rows) return;
    int first = int(row) * categories;
    for (int c = 0; c < categories; c++)
    {
        double v = D(w[first + c]);
        if (d_isnan(v) || v < 0.0 || d_isinf(v)) { parity_flag(0u, uint(first + c), 1u); return; }
    }
    ParityRng r = parity_rng(seedLo, seedHi, row);
    int outFirst = int(row) * numSamples;
" + (replacement ? @"
    // The weights never change, so the CPU's walk is a search of the double prefix sums (accumulated in the CPU's
    // order): the first category whose running total exceeds u, which is always a positive-weight one.
    double total = 0.0;
    int last = -1;
    for (int c = 0; c < categories; c++) { total += D(w[first + c]); prefix[first + c] = total; if (f_positive(w[first + c])) last = c; }
    if (!(total > 0.0)) { parity_flag(1u, uint(first), 2u); return; }
    for (int k = 0; k < numSamples; k++)
    {
        double u = parity_uniform(r) * total;
        int lo = 0, hi = categories;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (u < prefix[first + mid]) hi = mid; else lo = mid + 1; }
        o[outFirst + k] = F(lo < categories ? lo : last);
    }
" : @"
    for (int c = 0; c < categories; c++) s[first + c] = w[first + c];
    for (int k = 0; k < numSamples; k++)
    {
        double total = 0.0;
        for (int c = 0; c < categories; c++) total += D(s[first + c]);
        if (!(total > 0.0)) { parity_flag(1u, uint(first), 2u); return; }
        double u = parity_uniform(r) * total, running = 0.0;
        int chosen = -1;
        for (int c = categories - 1; c >= 0; c--) if (f_positive(s[first + c])) { chosen = c; break; }
        for (int c = 0; c < categories; c++)
        {
            running += D(s[first + c]);
            if (u < running && f_positive(s[first + c])) { chosen = c; break; }
        }
        o[outFirst + k] = F(chosen);
        s[first + chosen] = 0.0f;
    }
") + @"
}
", special: false));

    /// <summary>
    /// Dropout keep mask, one draw per block of `block` elements: keep = (U &gt;= p). Alpha dropout writes the per-element
    /// scale and offset (a * keep, a * sat * (1 - keep) + b); channel dropout writes the scale only (keep * a).
    /// </summary>
    public static string DropoutAffine(bool alpha) => Cached("dropoutAffine:" + alpha, () => DoubleShader(@"
layout(set = 0, binding = 0) writeonly buffer Sc { float scale[]; };
" + (alpha ? "layout(set = 0, binding = 1) writeonly buffer Of { float offset[]; };\n" : "") + @"
layout(push_constant) uniform P { uint n; uint block; uint pLo; uint pHi; uint aLo; uint aHi; uint satLo; uint satHi; uint bLo; uint bHi; uint seedLo; uint seedHi; };
", Unpack + Rng + @"
void main()
{
" + Index + @"
    ParityRng r = parity_rng(seedLo, seedHi, i / block);
    double keep = parity_uniform(r) >= U(pLo, pHi) ? 1.0 : 0.0;
    double a = U(aLo, aHi);
" + (alpha
    ? "    scale[i] = F(a * keep);\n    offset[i] = F(a * U(satLo, satHi) * (1.0 - keep) + U(bLo, bHi));\n"
    : "    scale[i] = F(keep * a);\n") + @"
}
", special: false));

    /// <summary>rrelu: slope 1 above zero; below, the eval slope (training = 0) or a uniform draw in [lower, upper].</summary>
    public static string Rrelu => Cached("rrelu", () => DoubleShader(@"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { uint n; uint pad0; uint lowerLo; uint lowerHi; uint upperLo; uint upperHi; int training; uint pad1; uint seedLo; uint seedHi; };
", Unpack + Rng + @"
void main()
{
" + Index + @"
    double v = D(a[i]), lower = U(lowerLo, lowerHi), upper = U(upperLo, upperHi), slope;
    if (v > 0.0) slope = 1.0;
    else if (training == 0) slope = (lower + upper) / 2.0;
    else { ParityRng r = parity_rng(seedLo, seedHi, i); slope = lower + (upper - lower) * parity_uniform(r); }
    o[i] = F(slope * v);
}
", special: false));

    /// <summary>randperm keys: one uniform per position.</summary>
    public static string RandomKeys => Cached("randomKeys", () => DoubleShader(@"
layout(set = 0, binding = 0) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { uint n; uint pad0; uint seedLo; uint seedHi; };
", Rng + @"
void main()
{
" + Index + @"
    ParityRng r = parity_rng(seedLo, seedHi, i);
    o[i] = F(parity_uniform(r));
}
", special: false));

    // ------------------------------------------------------------------------------------------------------------
    // nonzero_static / unravel_index / unique_dim / scan / integer ops / grid_sample 3-D
    // ------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// nonzero_static: an element that is non-zero (NaN counts; -0 does not) and whose inclusive-scan position is within
    /// `size` writes its coordinates to row (position - 1) of the [size, rank] output.
    /// </summary>
    public static string NonzeroStaticScatter => Cached("nonzeroStaticScatter", () => FloatShader(@"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer Ps { float positions[]; };
layout(set = 0, binding = 2) buffer O { float o[]; };
layout(push_constant) uniform P { uint n; int size; int rank; int extent[8]; };
", @"
void main()
{
" + Index + @"
    if ((floatBitsToUint(a[i]) & 0x7FFFFFFFu) == 0u) return;
    int row = int(positions[i]) - 1;
    if (row >= size) return;
    int rest = int(i);
    for (int k = rank - 1; k >= 0; k--) { o[row * rank + k] = float(rest % extent[k]); rest /= extent[k]; }
}
"));

    /// <summary>
    /// unravel_index, one launch per output axis: ((flat + total) % total) / stride % extent. A validating launch flags
    /// (code 1) an index whose truncation lies outside [-total, total), or a NaN.
    /// </summary>
    public static string UnravelAxis => Cached("unravelAxis", () => DoubleShader(@"
layout(set = 0, binding = 0) readonly buffer A { float indices[]; };
layout(set = 0, binding = 1) writeonly buffer O { float o[]; };
" + FlagSlots(2) + @"
layout(push_constant) uniform P { uint n; uint pad0; uint totalLo; uint totalHi; uint strideLo; uint strideHi; uint extentLo; uint extentHi; int validate; };
", @"
int64_t L(uint lo, uint hi) { return int64_t((uint64_t(hi) << 32) | uint64_t(lo)); }

void main()
{
" + Index + @"
    int64_t total = L(totalLo, totalHi), stride = L(strideLo, strideHi), extent = L(extentLo, extentHi);
    double t = d_trunc(D(indices[i]));
    if (d_isnan(t) || t < -double(total) || t >= double(total))
    {
        if (validate != 0) parity_flag(0u, i, 1u);
        return;
    }
    int64_t f = int64_t(t);
    o[i] = F(double(((f + total) % total) / stride % extent));
}
", special: false, int64: true));

    // unique_dim's slice order, as the CPU's comparer: element by element, NaN below every number and equal to NaN,
    // -0 equal to 0.
    private const string SliceCompare = @"
int slice_compare(int a, int b, int length)
{
    for (int k = 0; k < length; k++)
    {
        float x = data[a * length + k], y = data[b * length + k];
        bool nx = f_isnan(x), ny = f_isnan(y);
        if (nx || ny) { if (nx && ny) continue; return nx ? -1 : 1; }
        if (x < y) return -1;
        if (x > y) return 1;
    }
    return 0;
}
";

    /// <summary>One bitonic step over slice indices (padding = -1 sorts last; equal slices keep index order).</summary>
    public static string SliceBitonicStep => Cached("sliceBitonicStep", () => FloatShader(@"
layout(set = 0, binding = 0) readonly buffer D { float data[]; };
layout(set = 0, binding = 1) buffer Or { float order[]; };
layout(push_constant) uniform P { int rowLen; int k; int j; int length; };
", SliceCompare + @"
void main()
{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= rowLen) return;
    int ixj = i ^ j;
    if (ixj <= i) return;
    float fa = order[i], fb = order[ixj];
    int a = int(fa), b = int(fb);
    bool bBeforeA, aBeforeB;
    if (a < 0 || b < 0) { bBeforeA = a < 0 && b >= 0; aBeforeB = b < 0 && a >= 0; }
    else
    {
        int c = slice_compare(a, b, length);
        bBeforeA = c > 0 || (c == 0 && b < a);
        aBeforeB = c < 0 || (c == 0 && a < b);
    }
    bool up = (i & k) == 0;
    if (up ? bBeforeA : aBeforeB) { order[i] = fb; order[ixj] = fa; }
}
"));

    /// <summary>1 where a sorted slice differs from the one before it (the first slice always), else 0.</summary>
    public static string SliceFirstFlags => Cached("sliceFirstFlags", () => FloatShader(@"
layout(set = 0, binding = 0) readonly buffer D { float data[]; };
layout(set = 0, binding = 1) readonly buffer Or { float order[]; };
layout(set = 0, binding = 2) writeonly buffer Fl { float flags[]; };
layout(push_constant) uniform P { uint n; int length; };
", SliceCompare + @"
void main()
{
" + Index + @"
    flags[i] = (i == 0u || slice_compare(int(order[i]), int(order[i - 1u]), length) != 0) ? 1.0 : 0.0;
}
"));

    /// <summary>Copies each first-of-a-run slice (in sorted order) to row (position - 1) of the [unique, length] output.</summary>
    public static string SliceGatherUnique => Cached("sliceGatherUnique", () => FloatShader(@"
layout(set = 0, binding = 0) readonly buffer D { float data[]; };
layout(set = 0, binding = 1) readonly buffer Or { float order[]; };
layout(set = 0, binding = 2) readonly buffer Fl { float flags[]; };
layout(set = 0, binding = 3) readonly buffer Ps { float positions[]; };
layout(set = 0, binding = 4) buffer O { float o[]; };
layout(push_constant) uniform P { int total; int length; };
", @"
void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= total) return;
    int s = idx / length, k = idx % length;
    if (flags[s] == 0.0) return;
    int row = int(positions[s]) - 1;
    o[row * length + k] = data[int(order[s]) * length + k];
}
"));

    /// <summary>
    /// Inclusive prefix sum of one block per work group (Hillis-Steele in shared memory); each block's total goes to
    /// `sums`, which the host scans the same way and adds back with <see cref="ScanAdd"/>.
    /// </summary>
    public static string ScanBlockSource => Cached("scanBlock", () => FloatShader(@"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer O { float o[]; };
layout(set = 0, binding = 2) writeonly buffer Sm { float sums[]; };
layout(push_constant) uniform P { uint n; };
shared float s[" + ScanBlock + @"];
", @"
void main()
{
    uint t = gl_LocalInvocationID.x, i = gl_GlobalInvocationID.x;
    s[t] = i < n ? a[i] : 0.0;
    barrier();
    for (uint offset = 1u; offset < " + ScanBlock + @"u; offset <<= 1)
    {
        float v = t >= offset ? s[t - offset] : 0.0;
        barrier();
        s[t] += v;
        barrier();
    }
    if (i < n) o[i] = s[t];
    if (t == " + (ScanBlock - 1) + @"u) sums[gl_WorkGroupID.x] = s[t];
}
", localSize: ScanBlock));

    public static string ScanAdd => Cached("scanAdd", () => FloatShader(@"
layout(set = 0, binding = 0) buffer O { float o[]; };
layout(set = 0, binding = 1) readonly buffer C { float carries[]; };
layout(push_constant) uniform P { uint n; };
", @"
void main()
{
" + Index + @"
    uint block = i / " + ScanBlock + @"u;
    if (block > 0u) o[i] += carries[block - 1u];
}
"));

    /// <summary>
    /// Integer ops on raw int32 storage, as the CPU's IntegerBinary: 64-bit arithmetic on the sign-extended operands,
    /// wrapped to int32. op: 0 and, 1 or, 2 xor, 3 left shift, 4 right shift, 5 gcd, 6 lcm, 7 floor divide (code 1 for
    /// a zero divisor). Shifts follow C#'s long shifts: a count of 64 or more gives 0 (left) or the sign fill (right);
    /// otherwise the count is masked to its low 6 bits.
    /// </summary>
    public static string IntBinary => Cached("intBinary", () => FloatShader(@"
layout(set = 0, binding = 0) readonly buffer A { int a[]; };
layout(set = 0, binding = 1) readonly buffer B { int b[]; };
layout(set = 0, binding = 2) writeonly buffer O { int o[]; };
" + FlagSlots(3) + @"
layout(push_constant) uniform P { uint n; int op; };
", @"
// Binary (Stein's) gcd on the magnitudes: shifts and subtraction only. The AMD Vulkan driver's 64-bit % gave wrong
// remainders (gcd(12, int.MinValue) came back 12, gcd(-9223372036854775807, 3) came back 0x100000001).
int64_t parity_gcd(int64_t x, int64_t y)
{
    uint64_t a = x < 0L ? uint64_t(-x) : uint64_t(x), b = y < 0L ? uint64_t(-y) : uint64_t(y);
    if (a == 0UL) return int64_t(b);
    if (b == 0UL) return int64_t(a);
    int shift = 0;
    while (((a | b) & 1UL) == 0UL) { a >>= 1; b >>= 1; shift++; }
    while ((a & 1UL) == 0UL) a >>= 1;
    while (b != 0UL)
    {
        while ((b & 1UL) == 0UL) b >>= 1;
        if (a > b) { uint64_t s = a; a = b; b = s; }
        b -= a;
    }
    return int64_t(a << shift);
}
void main()
{
" + Index + @"
    int64_t x = int64_t(a[i]), v = int64_t(b[i]), r;
    if (op == 0) r = x & v;
    else if (op == 1) r = x | v;
    else if (op == 2) r = x ^ v;
    else if (op == 3) r = v >= 64L ? 0L : int64_t(uint64_t(x) << int(v & 63L));
    else if (op == 4) r = v >= 64L ? (x < 0L ? -1L : 0L) : x >> int(v & 63L);
    else if (op == 5) r = parity_gcd(x, v);
    else if (op == 6)
    {
        int64_t g = parity_gcd(x, v);
        int64_t m = g == 0L ? 0L : x / g * v;
        r = m < 0L ? -m : m;
    }
    else
    {
        if (v == 0L) { parity_flag(0u, i, 1u); return; }
        // Floor division from the magnitudes (GLSL leaves % of a negative operand undefined).
        int64_t ax = x < 0L ? -x : x, av = v < 0L ? -v : v;
        int64_t q = ax / av;
        r = ((x < 0L) == (v < 0L)) ? q : -(q + (ax % av != 0L ? 1L : 0L));
    }
    o[i] = int(r);
}
", int64: true));

    public static string IntNot => Cached("intNot", () => FloatShader(@"
layout(set = 0, binding = 0) readonly buffer A { int a[]; };
layout(set = 0, binding = 1) writeonly buffer O { int o[]; };
layout(push_constant) uniform P { uint n; };
", @"
void main()
{
" + Index + @"
    o[i] = ~a[i];
}
"));

    // #1135: the same ops on the raw int64 layout (two words per element) that long, ulong, short, ushort, sbyte, byte
    // and uint tensors use on the device. Values are the element's int64 widening (ulong keeps its bits), computed as
    // the CPU's IntegerBinary does -- in signed 64 bits -- and narrowed to the element type t (0 long, 1 ulong,
    // 2 short, 3 ushort, 4 sbyte, 5 byte, 6 uint), as its FromLong. Flag codes: 1 a zero divisor
    // (DivideByZeroException), 2 the CPU's OverflowException (Math.Abs of long.MinValue in gcd / lcm,
    // long.MinValue / -1).
    private const string Int64Narrow = @"
int64_t parity_int_narrow(int64_t r, int t)
{
    if (t == 2) return (r << 48) >> 48;
    if (t == 3) return r & 0xFFFFL;
    if (t == 4) return (r << 56) >> 56;
    if (t == 5) return r & 0xFFL;
    if (t == 6) return r & 0xFFFFFFFFL;
    return r;
}
";

    public static string Int64Binary => Cached("int64Binary", () => FloatShader(@"
layout(set = 0, binding = 0) readonly buffer A { int64_t a[]; };
layout(set = 0, binding = 1) readonly buffer B { int64_t b[]; };
layout(set = 0, binding = 2) writeonly buffer O { int64_t o[]; };
" + FlagSlots(3) + @"
layout(push_constant) uniform P { uint n; int op; int t; };
", Int64Narrow + @"
// Both operands are non-negative here: the caller rejects long.MinValue before taking magnitudes.
// Binary (Stein's) gcd on the magnitudes: shifts and subtraction only. The AMD Vulkan driver's 64-bit % gave wrong
// remainders (gcd(12, int.MinValue) came back 12, gcd(-9223372036854775807, 3) came back 0x100000001).
int64_t parity_gcd(int64_t x, int64_t y)
{
    uint64_t a = x < 0L ? uint64_t(-x) : uint64_t(x), b = y < 0L ? uint64_t(-y) : uint64_t(y);
    if (a == 0UL) return int64_t(b);
    if (b == 0UL) return int64_t(a);
    int shift = 0;
    while (((a | b) & 1UL) == 0UL) { a >>= 1; b >>= 1; shift++; }
    while ((a & 1UL) == 0UL) a >>= 1;
    while (b != 0UL)
    {
        while ((b & 1UL) == 0UL) b >>= 1;
        if (a > b) { uint64_t s = a; a = b; b = s; }
        b -= a;
    }
    return int64_t(a << shift);
}
void main()
{
" + Index + @"
    const int64_t minValue = -9223372036854775807L - 1L;
    int64_t x = a[i], v = b[i], r;
    if (op == 0) r = x & v;
    else if (op == 1) r = x | v;
    else if (op == 2) r = x ^ v;
    else if (op == 3) r = v >= 64L ? 0L : int64_t(uint64_t(x) << int(v & 63L));
    else if (op == 4) r = v >= 64L ? (x < 0L ? -1L : 0L) : x >> int(v & 63L);
    else if (op == 5)
    {
        if (x == minValue || v == minValue) { parity_flag(0u, i, 2u); return; }
        r = parity_gcd(x, v);
    }
    else if (op == 6)
    {
        if (x == minValue || v == minValue) { parity_flag(0u, i, 2u); return; }
        int64_t g = parity_gcd(x, v);
        // x / g from the magnitudes (g divides x exactly), then a wrapping multiply, as C#'s unchecked long.
        int64_t ax = x < 0L ? -x : x;
        int64_t xg = g == 0L ? 0L : ax / g;
        if (x < 0L) xg = -xg;
        int64_t m = g == 0L ? 0L : int64_t(uint64_t(xg) * uint64_t(v));
        if (m == minValue) { parity_flag(0u, i, 2u); return; }
        r = m < 0L ? -m : m;
    }
    else
    {
        if (v == 0L) { parity_flag(0u, i, 1u); return; }
        if (x == minValue && v == -1L) { parity_flag(0u, i, 2u); return; }
        // Floor division from unsigned magnitudes (GLSL leaves % of a negative operand undefined, and the magnitude
        // of long.MinValue only fits unsigned).
        uint64_t ax = x < 0L ? uint64_t(0) - uint64_t(x) : uint64_t(x);
        uint64_t av = v < 0L ? uint64_t(0) - uint64_t(v) : uint64_t(v);
        uint64_t q = ax / av;
        if ((x < 0L) == (v < 0L)) r = int64_t(q);
        else r = int64_t(uint64_t(0) - (q + (ax % av != uint64_t(0) ? uint64_t(1) : uint64_t(0))));
    }
    o[i] = parity_int_narrow(r, t);
}
", int64: true));

    public static string Int64Not => Cached("int64Not", () => FloatShader(@"
layout(set = 0, binding = 0) readonly buffer A { int64_t a[]; };
layout(set = 0, binding = 1) writeonly buffer O { int64_t o[]; };
layout(push_constant) uniform P { uint n; int t; };
", Int64Narrow + @"
void main()
{
" + Index + @"
    o[i] = parity_int_narrow(~a[i], t);
}
", int64: true));

    // grid_sample 3-D, as the CPU's GridSampler3D: unnormalize, clip or reflect, then trilinear (8 taps) or nearest
    // (round half to even), skipping out-of-range voxels. mode 0 bilinear, 1 nearest; padding 0 zeros, 1 border,
    // 2 reflection. `slope` is d voxel / d normalized: (size - 1) / 2 aligned, size / 2 otherwise; 0 where border or
    // reflection clipping holds the coordinate; sign-flipped by each reflection.
    private const string GridHelpers = @"
double grid_reflect(double coord, int twiceLow, int twiceHigh, inout double slope)
{
    if (twiceLow == twiceHigh) { slope = 0.0; return 0.0; }
    double mn = double(twiceLow) / 2.0, span = double(twiceHigh - twiceLow) / 2.0, sgn = 1.0;
    coord -= mn;
    if (coord < 0.0) { sgn = -1.0; coord = -coord; }
    double extra = d_fmod(coord, span), flips = floor(coord / span);
    if (d_fmod(flips, 2.0) == 0.0) { slope *= sgn; return extra + mn; }
    slope *= -sgn;
    return span - extra + mn;
}

double grid_source(double normalized, int size, int padding, int align, out double slope)
{
    double coord = align != 0 ? (normalized + 1.0) / 2.0 * double(size - 1) : ((normalized + 1.0) * double(size) - 1.0) / 2.0;
    slope = align != 0 ? double(size - 1) / 2.0 : double(size) / 2.0;
    if (padding == 2) coord = align != 0 ? grid_reflect(coord, 0, 2 * (size - 1), slope) : grid_reflect(coord, -1, 2 * size - 1, slope);
    if (padding >= 1)
    {
        if (coord <= 0.0) { slope = 0.0; return 0.0; }
        if (coord >= double(size - 1)) { slope = 0.0; return double(size - 1); }
    }
    return coord;
}

// A voxel index from a coordinate as the CPU's (int) cast, where NaN and out-of-range values become int.MinValue: here
// a sentinel far enough below zero that neither it nor its +1 neighbour lands inside the volume.
int grid_index(double v) { return (d_isnan(v) || v < -2147483648.0 || v >= 2147483647.0) ? -4 : int(v); }

// The taps of point (cx, cy, cz): offsets into a [depth, height, width] plane, weights, and each weight's derivative
// with respect to the three voxel coordinates.
int grid_taps(double cx, double cy, double cz, int depth, int height, int width, int mode,
    out int offsets[8], out double weights[8], out double gx[8], out double gy[8], out double gz[8])
{
    int taps = 0;
    if (mode == 1)
    {
        int ix = grid_index(d_rint(cx)), iy = grid_index(d_rint(cy)), iz = grid_index(d_rint(cz));
        if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth)
        {
            offsets[0] = (iz * height + iy) * width + ix;
            weights[0] = 1.0; gx[0] = 0.0; gy[0] = 0.0; gz[0] = 0.0;
            taps = 1;
        }
        return taps;
    }
    double fx0 = floor(cx), fy0 = floor(cy), fz0 = floor(cz);
    int x0 = grid_index(fx0), y0 = grid_index(fy0), z0 = grid_index(fz0);
    double fx = cx - fx0, fy = cy - fy0, fz = cz - fz0;
    for (int oz = 0; oz < 2; oz++)
        for (int oy = 0; oy < 2; oy++)
            for (int ox = 0; ox < 2; ox++)
            {
                int ix = x0 + ox, iy = y0 + oy, iz = z0 + oz;
                if (ix < 0 || ix >= width || iy < 0 || iy >= height || iz < 0 || iz >= depth) continue;
                double wx = ox == 1 ? fx : 1.0 - fx, wy = oy == 1 ? fy : 1.0 - fy, wz = oz == 1 ? fz : 1.0 - fz;
                offsets[taps] = (iz * height + iy) * width + ix;
                weights[taps] = wx * wy * wz;
                gx[taps] = (ox == 1 ? 1.0 : -1.0) * wy * wz;
                gy[taps] = (oy == 1 ? 1.0 : -1.0) * wx * wz;
                gz[taps] = (oz == 1 ? 1.0 : -1.0) * wx * wy;
                taps++;
            }
    return taps;
}
";

    private const string GridPush =
        "layout(push_constant) uniform P { int batches; int channels; int depth; int height; int width; int points; int mode; int padding; int align; };\n";

    /// <summary>grid_sample 3-D forward: one thread per (batch, point) locates its taps once and sums every channel.</summary>
    public static string GridSample3D => Cached("gridSample3D", () => DoubleShader(@"
layout(set = 0, binding = 0) readonly buffer X { float x[]; };
layout(set = 0, binding = 1) readonly buffer G { float grid[]; };
layout(set = 0, binding = 2) writeonly buffer O { float o[]; };
" + GridPush, GridHelpers + @"
void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= batches * points) return;
    int b = idx / points, q = idx % points;
    double sx, sy, sz;
    double cx = grid_source(D(grid[idx * 3]), width, padding, align, sx);
    double cy = grid_source(D(grid[idx * 3 + 1]), height, padding, align, sy);
    double cz = grid_source(D(grid[idx * 3 + 2]), depth, padding, align, sz);
    int volume = depth * height * width;
    int offsets[8]; double weights[8], gx[8], gy[8], gz[8];
    int taps = grid_taps(cx, cy, cz, depth, height, width, mode, offsets, weights, gx, gy, gz);
    for (int ch = 0; ch < channels; ch++)
    {
        int plane = (b * channels + ch) * volume;
        double acc = 0.0;
        for (int t = 0; t < taps; t++) acc += weights[t] * D(x[plane + offsets[t]]);
        o[(b * channels + ch) * points + q] = F(acc);
    }
}
", special: false));

    /// <summary>
    /// grid_sample 3-D adjoints: dx gathers w * dy from every point (compare-and-swap adds: points share voxels, and
    /// float atomics are not portable); dgrid is owned by its point's thread.
    /// </summary>
    public static string GridSample3DBackward => Cached("gridSample3DBackward", () => DoubleShader(@"
layout(set = 0, binding = 0) readonly buffer X { float x[]; };
layout(set = 0, binding = 1) readonly buffer G { float grid[]; };
layout(set = 0, binding = 2) readonly buffer Dy { float dy[]; };
layout(set = 0, binding = 3) buffer Dx { uint dx[]; };
layout(set = 0, binding = 4) writeonly buffer Dg { float dgrid[]; };
" + GridPush, GridHelpers + @"
void atomic_add(int index, float value)
{
    uint expected = dx[index];
    while (true)
    {
        uint actual = atomicCompSwap(dx[index], expected, floatBitsToUint(uintBitsToFloat(expected) + value));
        if (actual == expected) break;
        expected = actual;
    }
}

void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= batches * points) return;
    int b = idx / points, q = idx % points;
    double sx, sy, sz;
    double cx = grid_source(D(grid[idx * 3]), width, padding, align, sx);
    double cy = grid_source(D(grid[idx * 3 + 1]), height, padding, align, sy);
    double cz = grid_source(D(grid[idx * 3 + 2]), depth, padding, align, sz);
    int volume = depth * height * width;
    int offsets[8]; double weights[8], gx[8], gy[8], gz[8];
    int taps = grid_taps(cx, cy, cz, depth, height, width, mode, offsets, weights, gx, gy, gz);
    double dgx = 0.0, dgy = 0.0, dgz = 0.0;
    for (int ch = 0; ch < channels; ch++)
    {
        int plane = (b * channels + ch) * volume;
        double up = D(dy[(b * channels + ch) * points + q]);
        for (int t = 0; t < taps; t++)
        {
            atomic_add(plane + offsets[t], F(weights[t] * up));
            double v = D(x[plane + offsets[t]]) * up;
            dgx += v * (gx[t] * sx); dgy += v * (gy[t] * sy); dgz += v * (gz[t] * sz);
        }
    }
    dgrid[idx * 3] = F(dgx);
    dgrid[idx * 3 + 1] = F(dgy);
    dgrid[idx * 3 + 2] = F(dgz);
}
", special: false));
}
