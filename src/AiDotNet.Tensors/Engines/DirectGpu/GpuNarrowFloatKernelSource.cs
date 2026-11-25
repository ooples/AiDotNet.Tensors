// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The 16-bit float kernels of the C-family backends (OpenCL, CUDA, HIP), appended to the integer program so they
/// compile and load with it. Conversions are done on the bits, so they need no half extension and round the same way
/// on every device: to nearest, ties to even, as the host conversions do. On OpenCL the arithmetic is done on the bits
/// too (the packed kernels' binary32 library): an OpenCL device may flush binary32 subnormals with no build option to
/// keep them (AMD's does), and a BFloat16 below 2^-126 widens to one, so 0x0001 + 0x0001 came out 0.
/// </summary>
internal static class GpuNarrowFloatKernelSource
{
    public static readonly GpuElementType[] Types = { GpuElementType.Float16, GpuElementType.BFloat16 };

    private static readonly GpuNarrowFloatBinaryOperation[] BinaryOperations =
    {
        GpuNarrowFloatBinaryOperation.Add, GpuNarrowFloatBinaryOperation.Subtract, GpuNarrowFloatBinaryOperation.Multiply,
        GpuNarrowFloatBinaryOperation.Divide,
    };

    private static readonly GpuNarrowFloatUnaryOperation[] UnaryOperations =
    {
        GpuNarrowFloatUnaryOperation.Negate, GpuNarrowFloatUnaryOperation.Relu, GpuNarrowFloatUnaryOperation.Gelu,
        GpuNarrowFloatUnaryOperation.Abs, GpuNarrowFloatUnaryOperation.Sqrt, GpuNarrowFloatUnaryOperation.Exp,
        GpuNarrowFloatUnaryOperation.Log, GpuNarrowFloatUnaryOperation.Tanh, GpuNarrowFloatUnaryOperation.Sigmoid,
    };

    // The operations computed by a 32-bit function of the widened operand, and that function.
    private static string? FloatFunction(GpuNarrowFloatUnaryOperation operation) => operation switch
    {
        GpuNarrowFloatUnaryOperation.Gelu => "nf_gelu",
        GpuNarrowFloatUnaryOperation.Sqrt => "nf_sqrt",
        GpuNarrowFloatUnaryOperation.Exp => "nf_exp",
        GpuNarrowFloatUnaryOperation.Log => "nf_log",
        GpuNarrowFloatUnaryOperation.Tanh => "nf_tanh",
        GpuNarrowFloatUnaryOperation.Sigmoid => "nf_sigmoid",
        _ => null,
    };

    public static string Suffix(GpuElementType type) => type switch
    {
        GpuElementType.Float16 => "f16",
        GpuElementType.BFloat16 => "bf16",
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Not a 16-bit float element type."),
    };

    public static string Binary(GpuNarrowFloatBinaryOperation operation, GpuElementType type) => operation switch
    {
        GpuNarrowFloatBinaryOperation.Add => "nfadd_",
        GpuNarrowFloatBinaryOperation.Subtract => "nfsub_",
        GpuNarrowFloatBinaryOperation.Multiply => "nfmul_",
        GpuNarrowFloatBinaryOperation.Divide => "nfdiv_",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown 16-bit float operation."),
    } + Suffix(type);

    public static string Unary(GpuNarrowFloatUnaryOperation operation, GpuElementType type) => operation switch
    {
        GpuNarrowFloatUnaryOperation.Negate => "nfneg_",
        GpuNarrowFloatUnaryOperation.Relu => "nfrelu_",
        GpuNarrowFloatUnaryOperation.Gelu => "nfgelu_",
        GpuNarrowFloatUnaryOperation.Abs => "nfabs_",
        GpuNarrowFloatUnaryOperation.Sqrt => "nfsqrt_",
        GpuNarrowFloatUnaryOperation.Exp => "nfexp_",
        GpuNarrowFloatUnaryOperation.Log => "nflog_",
        GpuNarrowFloatUnaryOperation.Tanh => "nftanh_",
        GpuNarrowFloatUnaryOperation.Sigmoid => "nfsig_",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown 16-bit float operation."),
    } + Suffix(type);

    public static string MatrixMultiply(GpuElementType type) => "nfgemm_" + Suffix(type);

    /// <summary>
    /// The first pass of a sum: arguments (a, o, count, blocks), <c>o</c> the binary32 bits of each block's sum.
    /// </summary>
    public static string Sum(GpuElementType type) => "nfsum_" + Suffix(type);

    /// <summary>A later pass of a sum, over the partials of the pass before: arguments (a, o, count, blocks).</summary>
    public const string SumPartials = "nfsum_w";

    /// <summary>The kernel of <c>o[i] = a[i] op y</c> for a 32-bit float <c>y</c>: arguments (a, o, y, size).</summary>
    public static string Scalar(GpuNarrowFloatBinaryOperation operation, GpuElementType type)
    {
        string name = Binary(operation, type);
        return name.Substring(0, name.Length - Suffix(type).Length) + "s_" + Suffix(type);
    }

    public static void AddNames(List<string> names)
    {
        foreach (var type in Types)
        {
            foreach (var operation in BinaryOperations)
            {
                names.Add(Binary(operation, type));
                names.Add(Scalar(operation, type));
            }

            foreach (var operation in UnaryOperations) names.Add(Unary(operation, type));
            names.Add(MatrixMultiply(type));
            names.Add(Sum(type));
            if (!names.Contains(SumPartials)) names.Add(SumPartials);
        }
    }

    // The binary32 result of `x` and `y`.
    private static string Combine(GpuNarrowFloatBinaryOperation operation) => operation switch
    {
        GpuNarrowFloatBinaryOperation.Add => "x + y",
        GpuNarrowFloatBinaryOperation.Subtract => "x - y",
        GpuNarrowFloatBinaryOperation.Divide => "nf_div(x, y)",
        _ => "x * y",
    };

    /// <summary>Appends the kernels to a program that already has the integer program's prelude.</summary>
    public static void Append(StringBuilder source, bool openCl)
    {
        source.Append(openCl
            ? @"
typedef ushort u16t;
#define NF_FN
#define NF_BITS(f) as_uint(f)
#define NF_FLOAT(u) as_float(u)
#define NF_TANH(x) tanh(x)
#define NF_SQRT(x) sqrt(x)
#define NF_EXP(x) exp(x)
#define NF_LOG(x) log(x)
"
            : @"
typedef unsigned short u16t;
#define NF_FN __device__ inline
#define NF_BITS(f) __float_as_uint(f)
#define NF_FLOAT(u) __uint_as_float(u)
#define NF_TANH(x) tanhf(x)
#define NF_SQRT(x) sqrtf(x)
#define NF_EXP(x) expf(x)
#define NF_LOG(x) logf(x)
");
        source.Append(@"
NF_FN float nf_w_f16(u16t h)
{
    const u32t sign = ((u32t)h & 0x8000u) << 16;
    const u32t e = ((u32t)h >> 10) & 0x1Fu;
    const u32t m = (u32t)h & 0x3FFu;
    if (e == 0u)
    {
        /* Zero or subnormal: m units of 2^-24, exact in a float. */
        const float v = (float)m * 5.9604644775390625e-8f;
        return NF_FLOAT(NF_BITS(v) | sign);
    }
    if (e == 31u) return NF_FLOAT(sign | 0x7F800000u | (m << 13));
    return NF_FLOAT(sign | ((e + 112u) << 23) | (m << 13));
}
NF_FN u16t nf_n_f16(float f)
{
    const u32t x = NF_BITS(f);
    const u32t sign = (x >> 16) & 0x8000u;
    const u32t ax = x & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) return (u16t)(sign | 0x7E00u | ((ax >> 13) & 0x1FFu));
    if (ax >= 0x47800000u) return (u16t)(sign | 0x7C00u);
    if (ax >= 0x38800000u)
    {
        /* A normal half: rebias the exponent, round the 13 dropped bits; a carry runs into the exponent (and to infinity). */
        const u32t m = ax - 0x38000000u;
        return (u16t)(sign | ((m + 0xFFFu + ((m >> 13) & 1u)) >> 13));
    }
    if (ax <= 0x33000000u) return (u16t)sign;
    {
        /* A subnormal half: the 24-bit significand shifted down by 14 to 24 bits, rounded on what falls off. */
        const u32t shift = 126u - (ax >> 23);
        const u32t mant = (ax & 0x7FFFFFu) | 0x800000u;
        u32t q = mant >> shift;
        const u32t rem = mant & ((1u << shift) - 1u);
        const u32t mid = 1u << (shift - 1u);
        if (rem > mid || (rem == mid && (q & 1u) != 0u)) q++;
        return (u16t)(sign | q);
    }
}
NF_FN float nf_w_bf16(u16t h) { return NF_FLOAT((u32t)h << 16); }
NF_FN u16t nf_n_bf16(float f)
{
    const u32t x = NF_BITS(f);
    if ((x & 0x7F800000u) == 0x7F800000u && (x & 0x007FFFFFu) != 0u) return (u16t)((x >> 16) | 0x0040u);
    return (u16t)((x + 0x7FFFu + ((x >> 16) & 1u)) >> 16);
}
NF_FN float nf_gelu(float x)
{
    float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    if (z > 20.0f) z = 20.0f;
    if (z < -20.0f) z = -20.0f;
    return 0.5f * x * (1.0f + NF_TANH(z));
}
/* The quotient on the bits, correctly rounded: a device's own division need not round a subnormal quotient as the
   host does (an OpenCL device returned -3.203125 / FLT_MAX one unit off). */
NF_FN float nf_div(float fa, float fb)
{
    const u32t a = NF_BITS(fa), b = NF_BITS(fb);
    const u32t sign = (a ^ b) & 0x80000000u;
    const u32t ea = (a >> 23) & 0xFFu, eb = (b >> 23) & 0xFFu;
    u32t ma = a & 0x7FFFFFu, mb = b & 0x7FFFFFu;
    const int za = ea == 0u && ma == 0u, zb = eb == 0u && mb == 0u;
    int xa = (int)ea, xb = (int)eb, x;
    u64t q, n;
    u32t s, r, tail;
    if ((ea == 255u && ma != 0u) || (eb == 255u && mb != 0u)) return NF_FLOAT(0x7FC00000u);
    if (ea == 255u) return NF_FLOAT(eb == 255u ? 0x7FC00000u : (sign | 0x7F800000u));
    if (eb == 255u) return NF_FLOAT(sign);
    if (zb) return NF_FLOAT(za ? 0x7FC00000u : (sign | 0x7F800000u));
    if (za) return NF_FLOAT(sign);
    if (ea == 0u) { xa = 1; while ((ma & 0x800000u) == 0u) { ma <<= 1; xa--; } } else ma |= 0x800000u;
    if (eb == 0u) { xb = 1; while ((mb & 0x800000u) == 0u) { mb <<= 1; xb--; } } else mb |= 0x800000u;
    x = xa - xb + 127;
    if (ma < mb) { ma <<= 1; x--; }
    /* 27 quotient bits in [2^26, 2^27): 24 and three to round on, whatever remains sticky. */
    n = (u64t)ma << 26;
    q = n / (u64t)mb;
    s = (u32t)q | (q * (u64t)mb != n ? 1u : 0u);
    if (x <= 0)
    {
        const u32t down = (u32t)(1 - x);
        if (down > 27u) s = s != 0u ? 1u : 0u;
        else { const u32t lost = s & ((1u << down) - 1u); s >>= down; if (lost != 0u) s |= 1u; }
        x = 1;
    }
    tail = s & 7u;
    r = s >> 3;
    if (tail > 4u || (tail == 4u && (r & 1u) != 0u)) r++;
    if (r >= 0x1000000u) { r >>= 1; x++; }
    if (r < 0x800000u) return NF_FLOAT(sign | r);
    if (x >= 255) return NF_FLOAT(sign | 0x7F800000u);
    return NF_FLOAT(sign | ((u32t)x << 23) | (r & 0x7FFFFFu));
}
/* The functions below decide on the bits what a device that flushes binary32 subnormals would lose: a subnormal
   operand (a bfloat16 subnormal widens to one) is rebuilt from its integer significand, and a subnormal result is
   rounded from a scaled one. */
NF_FN float nf_sqrt(float x)
{
    const u32t b = NF_BITS(x);
    const u32t m = b & 0x7FFFFFu;
    if ((b & 0x7F800000u) == 0u && m != 0u)
    {
        if ((b >> 31) != 0u) return NF_FLOAT(0x7FC00000u);
        /* m 2^-149 is (2m) 2^-150: the root of the integer, exact to scale by 2^-75. */
        return NF_SQRT((float)(m << 1)) * NF_FLOAT(0x1A000000u);
    }
    return NF_SQRT(x);
}
NF_FN float nf_log(float x)
{
    const u32t b = NF_BITS(x);
    const u32t m = b & 0x7FFFFFu;
    if ((b & 0x7F800000u) == 0u && m != 0u)
        return (b >> 31) != 0u ? NF_FLOAT(0x7FC00000u) : NF_LOG((float)m) - 103.27892990343185f;
    return NF_LOG(x);
}
/* r 2^-64 rounded to nearest even, for a positive normal r. */
NF_FN float nf_down64(float r)
{
    const u32t b = NF_BITS(r);
    const int e = (int)(b >> 23) - 64;
    u32t m, lost, mid;
    int down;
    if (e >= 1) return NF_FLOAT(((u32t)e << 23) | (b & 0x7FFFFFu));
    down = 1 - e;
    if (down > 24) return 0.0f;
    m = (b & 0x7FFFFFu) | 0x800000u;
    mid = 1u << (down - 1);
    lost = m & ((mid << 1) - 1u);
    m >>= down;
    if (lost > mid || (lost == mid && (m & 1u) != 0u)) m++;
    return NF_FLOAT(m);
}
NF_FN float nf_exp(float x)
{
    if ((NF_BITS(x) & 0x7F800000u) == 0u) return 1.0f;
    /* Below -87 the result is subnormal: e^(x + 64 ln 2), scaled down on the bits. */
    if (x < -87.0f) return x < -104.0f ? 0.0f : nf_down64(NF_EXP(x + 44.3614195558365f));
    return NF_EXP(x);
}
/* Below 2^-13 the tangent is its operand to the last bit of either type. */
NF_FN float nf_tanh(float x) { return ((NF_BITS(x) >> 23) & 0xFFu) < 114u ? x : NF_TANH(x); }
/* Below -80, e^-x overflows on its way to the quotient (a zero, where the result is e^x: 8e-40 at -90). */
NF_FN float nf_sigmoid(float x) { return x < -80.0f ? nf_exp(x) : 1.0f / (1.0f + NF_EXP(-x)); }
");
        if (openCl) source.Append(BitExactLibrary());
        foreach (var type in Types)
        {
            string t = Suffix(type);
            string widen = "nf_w_" + t;
            string narrow = "nf_n_" + t;
            if (openCl)
            {
                AppendBitExact(source, type);
                continue;
            }

            foreach (var operation in BinaryOperations)
            {
                source.Append("IK_KERNEL ").Append(Binary(operation, type))
                    .Append("(IK_PTR const u16t* a, IK_PTR const u16t* b, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
                    .Append("    if (i < size) { const float x = ").Append(widen).Append("(a[i]); const float y = ").Append(widen)
                    .Append("(b[i]); const float r = ").Append(Combine(operation)).Append("; o[i] = ").Append(narrow).Append("(r); }\n}\n");
                source.Append("IK_KERNEL ").Append(Scalar(operation, type))
                    .Append("(IK_PTR const u16t* a, IK_PTR u16t* o, const float y, const int size)\n{\n    const int i = IK_GID;\n")
                    .Append("    if (i < size) { const float x = ").Append(widen).Append("(a[i]); const float r = ")
                    .Append(Combine(operation)).Append("; o[i] = ").Append(narrow).Append("(r); }\n}\n");
            }

            source.Append("IK_KERNEL ").Append(Unary(GpuNarrowFloatUnaryOperation.Negate, type))
                .Append("(IK_PTR const u16t* a, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
                .Append("    if (i < size) o[i] = (u16t)(a[i] ^ 0x8000u);\n}\n");
            source.Append("IK_KERNEL ").Append(Unary(GpuNarrowFloatUnaryOperation.Relu, type))
                .Append("(IK_PTR const u16t* a, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
                .Append("    if (i < size) { const u16t h = a[i]; o[i] = ").Append(widen).Append("(h) > 0.0f ? h : (u16t)0; }\n}\n");
            source.Append("IK_KERNEL ").Append(Unary(GpuNarrowFloatUnaryOperation.Abs, type))
                .Append("(IK_PTR const u16t* a, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
                .Append("    if (i < size) o[i] = (u16t)(a[i] & 0x7FFFu);\n}\n");
            foreach (var (operation, function) in UnaryOperations
                .Select(operation => (Operation: operation, Function: FloatFunction(operation)))
                .Where(kernel => kernel.Function is not null))
            {
                source.Append("IK_KERNEL ").Append(Unary(operation, type))
                    .Append("(IK_PTR const u16t* a, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
                    .Append("    if (i < size) o[i] = ").Append(narrow).Append('(').Append(function).Append('(').Append(widen).Append("(a[i])));\n}\n");
            }

            // A block's elements added in order into a binary32 partial, stored as its bits.
            source.Append("IK_KERNEL ").Append(Sum(type)).Append("(IK_PTR const u16t* a, IK_PTR u32t* o, const int count, const int blocks)\n{\n"
                + "    const int i = IK_GID;\n    if (i >= blocks) return;\n    const int first = i * " + GpuNarrowFloatLayout.SumBlock
                + ";\n    const int last = (count - first > " + GpuNarrowFloatLayout.SumBlock + ") ? first + " + GpuNarrowFloatLayout.SumBlock
                + " : count;\n    float s = 0.0f;\n    for (int j = first; j < last; j++) s += " + widen + "(a[j]);\n    o[i] = NF_BITS(s);\n}\n");
            if (type == Types[0])
            {
                source.Append("IK_KERNEL ").Append(SumPartials).Append("(IK_PTR const u32t* a, IK_PTR u32t* o, const int count, const int blocks)\n{\n"
                + "    const int i = IK_GID;\n    if (i >= blocks) return;\n    const int first = i * " + GpuNarrowFloatLayout.SumBlock
                + ";\n    const int last = (count - first > " + GpuNarrowFloatLayout.SumBlock + ") ? first + " + GpuNarrowFloatLayout.SumBlock
                + " : count;\n    float s = 0.0f;\n    for (int j = first; j < last; j++) s += NF_FLOAT(a[j]);\n    o[i] = NF_BITS(s);\n}\n");
            }

            // One invocation per output element: a row of a against a column of b, accumulated in a float.
            source.Append("IK_KERNEL ").Append(MatrixMultiply(type))
                .Append("(IK_PTR const u16t* a, IK_PTR const u16t* b, IK_PTR u16t* c, const int m, const int n, const int k)\n{\n")
                .Append("    const int i = IK_GID;\n    if (i >= m * n) return;\n    const int row = i / n;\n    const int col = i - row * n;\n")
                .Append("    float s = 0.0f;\n    for (int j = 0; j < k; j++) s += ").Append(widen).Append("(a[row * k + j]) * ").Append(widen)
                .Append("(b[j * n + col]);\n    c[i] = ").Append(narrow).Append("(s);\n}\n");
        }
    }

    // The packed kernels' binary32 library up to its gelu (which needs clamp and tanh), spelled as C: a functional cast
    // becomes a C cast and uint the program's u32t.
    private static string BitExactLibrary()
    {
        string text = KernelSourceResources.Read("PackedKernels.float.c");
        text = text.Substring(0, text.IndexOf("float pk_gelu(", StringComparison.Ordinal));
        text = Regex.Replace(text, @"\b(uint|int)\(", "($1)(");
        text = Regex.Replace(text, @"\buint\b", "u32t");
        return Regex.Replace(text, @"^(u32t|int|float|bool) (pk_\w+)\(", "NF_FN $1 $2(", RegexOptions.Multiline);
    }

    // The OpenCL kernels: add, subtract, multiply, relu and the matrix product's accumulation on the bits, as the packed
    // kernels compute them; negate is a sign flip and gelu (a transcendental, checked to a tolerance) stays native.
    private static void AppendBitExact(StringBuilder source, GpuElementType type)
    {
        string t = Suffix(type);
        string widen = "pk_w_" + t, narrow = "pk_n_" + t;
        foreach (var operation in BinaryOperations)
        {
            string value = operation switch
            {
                GpuNarrowFloatBinaryOperation.Add => "pk_fadd(" + widen + "(a[i]), " + widen + "(b[i]))",
                GpuNarrowFloatBinaryOperation.Subtract => "pk_fadd(" + widen + "(a[i]), " + widen + "(b[i]) ^ 0x80000000u)",
                _ => "pk_fmul(" + widen + "(a[i]), " + widen + "(b[i]))",
            };
            source.Append("IK_KERNEL ").Append(Binary(operation, type))
                .Append("(IK_PTR const u16t* a, IK_PTR const u16t* b, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
                .Append("    if (i < size) o[i] = (u16t)").Append(narrow).Append('(').Append(value).Append(");\n}\n");
        }

        // The largest pattern below the sign bit that is not a NaN: infinity.
        string top = type == GpuElementType.Float16 ? "0x7C00u" : "0x7F80u";
        source.Append("IK_KERNEL ").Append(Unary(GpuNarrowFloatUnaryOperation.Negate, type))
            .Append("(IK_PTR const u16t* a, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
            .Append("    if (i < size) o[i] = (u16t)(a[i] ^ 0x8000u);\n}\n");
        source.Append("IK_KERNEL ").Append(Unary(GpuNarrowFloatUnaryOperation.Relu, type))
            .Append("(IK_PTR const u16t* a, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
            .Append("    if (i < size) o[i] = (u16t)pk_relu(a[i], ").Append(top).Append(");\n}\n");
        source.Append("IK_KERNEL ").Append(Unary(GpuNarrowFloatUnaryOperation.Gelu, type))
            .Append("(IK_PTR const u16t* a, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
            .Append("    if (i < size) o[i] = nf_n_").Append(t).Append("(nf_gelu(nf_w_").Append(t).Append("(a[i])));\n}\n");
        // Each product and sum rounded to binary32 in the host's order, as the host's float loop does.
        source.Append("IK_KERNEL ").Append(MatrixMultiply(type))
            .Append("(IK_PTR const u16t* a, IK_PTR const u16t* b, IK_PTR u16t* c, const int m, const int n, const int k)\n{\n")
            .Append("    const int i = IK_GID;\n    if (i >= m * n) return;\n    const int row = i / n;\n    const int col = i - row * n;\n")
            .Append("    u32t s = 0u;\n    for (int j = 0; j < k; j++) s = pk_fadd(s, pk_fmul(").Append(widen).Append("(a[row * k + j]), ")
            .Append(widen).Append("(b[j * n + col])));\n    c[i] = (u16t)").Append(narrow).Append("(s);\n}\n");
    }
}
