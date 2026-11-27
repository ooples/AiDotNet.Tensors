// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The packed kernels' binary32 library (<c>PackedKernels.float.c</c>: round-to-nearest-even add, multiply and divide on
/// the bits) spelled as C, for kernels that must keep subnormals on a device that flushes them. An OpenCL device may flush
/// binary32 subnormals with no build option to keep them (AMD's does), in arithmetic and in comparisons alike.
/// </summary>
internal static class PackedFloatLibrary
{
    private static string? s_openCl;

    /// <summary>
    /// The library up to <paramref name="end"/> (by default its gelu, which needs clamp and tanh): a functional cast
    /// becomes a C cast, <c>uint</c> becomes <paramref name="uintName"/> and each function takes
    /// <paramref name="functionPrefix"/>.
    /// </summary>
    internal static string CSpelling(string functionPrefix, string uintName, string end = "float pk_gelu(")
    {
        string text = KernelSourceResources.Read("PackedKernels.float.c");
        text = text.Substring(0, text.IndexOf(end, StringComparison.Ordinal));
        text = Regex.Replace(text, @"\b(uint|int|float)\(", "($1)(");
        // The library's float routines (square root, exponential, logarithm, tangent, sigmoid) read and write the bits
        // through FBITS and UFLOAT, which each shader language spells its own way: as_uint and as_float in OpenCL C.
        text = text.Replace("FBITS(", "as_uint(").Replace("UFLOAT(", "as_float(");
        if (uintName != "uint") text = Regex.Replace(text, @"\buint\b", uintName);
        string type = "(" + Regex.Escape(uintName) + "|int|float|bool)";
        return Regex.Replace(text, "^" + type + @" (pk_\w+)\(", functionPrefix + "$1 $2(", RegexOptions.Multiline);
    }

    /// <summary>
    /// The library's add, multiply and divide for an OpenCL program, defined once however many of the program's sources
    /// include it, followed by <c>pk_add_keep</c> and <c>pk_mul_keep</c>: the native sum and product, except where an operand is
    /// subnormal or the result is below 2^-125 (where the device may have flushed it), which are computed on the bits.
    /// </summary>
    internal static string OpenCl => s_openCl ??= "#ifndef PK_FLOAT_LIBRARY\n#define PK_FLOAT_LIBRARY\n"
        + CSpelling("", "uint", end: "uint pk_usqrt(")
        + @"
inline int pk_tiny(uint m) { return m - 1u < 0x007FFFFFu; }
float pk_add_keep(float x, float y)
{
    float s = x + y;
    uint mx = as_uint(x) & 0x7FFFFFFFu, my = as_uint(y) & 0x7FFFFFFFu, ms = as_uint(s) & 0x7FFFFFFFu;
    return pk_tiny(mx) || pk_tiny(my) || ms < 0x01000000u ? as_float(pk_fadd(as_uint(x), as_uint(y))) : s;
}
float pk_mul_keep(float x, float y)
{
    float p = x * y;
    uint mx = as_uint(x) & 0x7FFFFFFFu, my = as_uint(y) & 0x7FFFFFFFu, mp = as_uint(p) & 0x7FFFFFFFu;
    return pk_tiny(mx) || pk_tiny(my) || mp < 0x01000000u ? as_float(pk_fmul(as_uint(x), as_uint(y))) : p;
}
// x > 0 on the bits: a positive subnormal compares as zero where the device flushes.
inline int pk_positive(float x) { return as_int(x) > 0; }
#endif
";
}
