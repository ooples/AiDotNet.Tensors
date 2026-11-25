// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
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
    };

    private static readonly GpuNarrowFloatUnaryOperation[] UnaryOperations =
    {
        GpuNarrowFloatUnaryOperation.Negate, GpuNarrowFloatUnaryOperation.Relu, GpuNarrowFloatUnaryOperation.Gelu,
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
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown 16-bit float operation."),
    } + Suffix(type);

    public static string Unary(GpuNarrowFloatUnaryOperation operation, GpuElementType type) => operation switch
    {
        GpuNarrowFloatUnaryOperation.Negate => "nfneg_",
        GpuNarrowFloatUnaryOperation.Relu => "nfrelu_",
        GpuNarrowFloatUnaryOperation.Gelu => "nfgelu_",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown 16-bit float operation."),
    } + Suffix(type);

    public static string MatrixMultiply(GpuElementType type) => "nfgemm_" + Suffix(type);

    public static void AddNames(List<string> names)
    {
        foreach (var type in Types)
        {
            foreach (var operation in BinaryOperations) names.Add(Binary(operation, type));
            foreach (var operation in UnaryOperations) names.Add(Unary(operation, type));
            names.Add(MatrixMultiply(type));
        }
    }

    private static string Symbol(GpuNarrowFloatBinaryOperation operation) => operation switch
    {
        GpuNarrowFloatBinaryOperation.Add => "+",
        GpuNarrowFloatBinaryOperation.Subtract => "-",
        _ => "*",
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
"
            : @"
typedef unsigned short u16t;
#define NF_FN __device__ inline
#define NF_BITS(f) __float_as_uint(f)
#define NF_FLOAT(u) __uint_as_float(u)
#define NF_TANH(x) tanhf(x)
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
                    .Append("(b[i]); const float r = x ").Append(Symbol(operation)).Append(" y; o[i] = ").Append(narrow).Append("(r); }\n}\n");
            }

            source.Append("IK_KERNEL ").Append(Unary(GpuNarrowFloatUnaryOperation.Negate, type))
                .Append("(IK_PTR const u16t* a, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
                .Append("    if (i < size) o[i] = (u16t)(a[i] ^ 0x8000u);\n}\n");
            source.Append("IK_KERNEL ").Append(Unary(GpuNarrowFloatUnaryOperation.Relu, type))
                .Append("(IK_PTR const u16t* a, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
                .Append("    if (i < size) { const u16t h = a[i]; o[i] = ").Append(widen).Append("(h) > 0.0f ? h : (u16t)0; }\n}\n");
            source.Append("IK_KERNEL ").Append(Unary(GpuNarrowFloatUnaryOperation.Gelu, type))
                .Append("(IK_PTR const u16t* a, IK_PTR u16t* o, const int size)\n{\n    const int i = IK_GID;\n")
                .Append("    if (i < size) o[i] = ").Append(narrow).Append("(nf_gelu(").Append(widen).Append("(a[i])));\n}\n");
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
