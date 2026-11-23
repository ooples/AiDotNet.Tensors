// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>The shading language a packed kernel is written out in.</summary>
internal enum GpuPackedKernelDialect
{
    /// <summary>GLSL 4.50 (Vulkan).</summary>
    Glsl,
    /// <summary>WGSL (WebGPU).</summary>
    Wgsl,
    /// <summary>Metal Shading Language.</summary>
    Metal,
}

/// <summary>
/// One kernel over buffers of 32-bit words: its name, how many buffers it binds (the last is the one it writes) and its
/// body in the neutral form <see cref="GpuPackedKernelSource"/> writes out for each shading language.
/// </summary>
internal sealed class GpuPackedKernel
{
    public GpuPackedKernel(string name, int buffers, string library, string body)
    {
        Name = name;
        Buffers = buffers;
        Library = library;
        Body = body;
    }

    /// <summary>The kernel's name: its entry point in a Metal library, its module name elsewhere.</summary>
    public string Name { get; }

    /// <summary>Buffers bound, in order: the operands, then the output.</summary>
    public int Buffers { get; }

    /// <summary>The helper functions the body calls, in the neutral form.</summary>
    public string Library { get; }

    /// <summary>The statements of one invocation, in the neutral form.</summary>
    public string Body { get; }
}

/// <summary>
/// The kernels of the element types narrower or wider than a word on the backends whose buffers are 4-byte words
/// (Vulkan, Metal, WebGPU): Int8, UInt8, Bool, Int16, Int64, Float16 and BFloat16. None of the three shading languages
/// has 8- or 16-bit storage everywhere, and WGSL has no 64-bit integer, so every kernel reads and writes whole words:
/// four 8-bit or two 16-bit elements to a word (the first in the low bits, as the bytes lie in memory), and an Int64
/// as two words, low first.
/// </summary>
/// <remarks>
/// <para>
/// An element-wise kernel runs one invocation per output word, so two invocations never write the same word. The
/// elements of the last word past the requested count keep what the output held.
/// </para>
/// <para>
/// The 16-bit float addition, subtraction and multiplication are computed on the bits in integer arithmetic: the
/// operands widened to binary32, the operation rounded to nearest even as IEEE 754 defines it, and that result
/// rounded once more into the stored type. A device that flushes subnormal floats to zero -- which the shading
/// languages allow, and a BFloat16 subnormal is a binary32 subnormal -- therefore gives the host's bits too.
/// </para>
/// <para>
/// The bodies are written once, in a small C-like form all three languages share after a mechanical rewrite: braces
/// on every block, a declaration with an initialiser on every variable, no conditional operator, and every mixed
/// expression parenthesised (WGSL does not rank the bitwise operators against each other).
/// </para>
/// </remarks>
internal static class GpuPackedKernelSource
{
    /// <summary>Elements one invocation of a sum pass folds.</summary>
    public const int SumBlock = 64;

    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(5);

    private static readonly ConcurrentDictionary<string, GpuPackedKernel> Kernels =
        new ConcurrentDictionary<string, GpuPackedKernel>(StringComparer.Ordinal);

    private static readonly ConcurrentDictionary<string, string> Sources =
        new ConcurrentDictionary<string, string>(StringComparer.Ordinal);

    /// <summary>The integer and bool element types computed by these kernels.</summary>
    public static readonly GpuElementType[] IntegerTypes =
    {
        GpuElementType.Int8, GpuElementType.UInt8, GpuElementType.Int16, GpuElementType.Int64, GpuElementType.Bool,
    };

    /// <summary>The 16-bit float element types computed by these kernels.</summary>
    public static readonly GpuElementType[] FloatTypes = { GpuElementType.Float16, GpuElementType.BFloat16 };

    /// <summary>Whether <paramref name="type"/> is one of the element types these kernels compute.</summary>
    public static bool Computes(GpuElementType type)
        => type is GpuElementType.Int8 or GpuElementType.UInt8 or GpuElementType.Int16 or GpuElementType.Int64
            or GpuElementType.Bool or GpuElementType.Float16 or GpuElementType.BFloat16;

    /// <summary>Elements of <paramref name="type"/> in one word (an Int64 is two words: 0 here).</summary>
    public static int Lanes(GpuElementType type) => GpuElementTypes.SizeInBytes(type) switch
    {
        1 => 4,
        2 => 2,
        _ => 0,
    };

    /// <summary>Words that hold <paramref name="count"/> elements of <paramref name="type"/>.</summary>
    public static int Words(GpuElementType type, int count)
    {
        int lanes = Lanes(type);
        return lanes == 0 ? checked(count * 2) : (int)(((long)count + lanes - 1) / lanes);
    }

    private static string Suffix(GpuElementType type) => type switch
    {
        GpuElementType.Int8 => "i8",
        GpuElementType.UInt8 => "u8",
        GpuElementType.Int16 => "i16",
        GpuElementType.Int64 => "i64",
        GpuElementType.Bool => "b8",
        GpuElementType.Float16 => "f16",
        GpuElementType.BFloat16 => "bf16",
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Not an element type the packed kernels compute."),
    };

    // ------------------------------------------------------------------------------------------------------------
    // Helper functions, in the neutral form. Embedded resources (KernelSources/PackedKernels.*.c), not literals: the
    // assembly's string literals are at the compiler's limit (CS8103).
    // ------------------------------------------------------------------------------------------------------------

    // The high word of a 32 x 32 bit product, from 16-bit pieces (WGSL has no 64-bit integer and no multiply-high).
    private static string IntegerLibrary => KernelSourceResources.Read("PackedKernels.integer.c");

    // binary32 arithmetic on the bits. pk_fpack takes a significand with its leading one at bit 26 (24 significand
    // bits and three below them: guard, round and a sticky bit for everything further down) and the biased exponent
    // of that leading one; or any smaller significand with exponent 1, which is a subnormal.
    private static string FloatLibrary => KernelSourceResources.Read("PackedKernels.float.c");

    // ------------------------------------------------------------------------------------------------------------
    // Kernels.
    // ------------------------------------------------------------------------------------------------------------

    private static GpuPackedKernel Kernel(string name, int buffers, string library, Func<string> body)
        => Kernels.GetOrAdd(name, key => new GpuPackedKernel(key, buffers, library, body()));

    private static string Stem(GpuIntegerBinaryOperation operation) => GpuInt32ShaderSource.Name(operation);

    private static string Stem(GpuIntegerUnaryOperation operation) => GpuInt32ShaderSource.Name(operation);

    private static void RequireInteger(GpuElementType type)
    {
        if (Array.IndexOf(IntegerTypes, type) < 0)
            throw new ArgumentOutOfRangeException(nameof(type), type, "Not an integer element type the packed kernels compute.");
    }

    private static void RequireFloat(GpuElementType type)
    {
        if (!GpuNarrowFloatLayout.IsNarrowFloat(type))
            throw new ArgumentOutOfRangeException(nameof(type), type, "Not a 16-bit float element type.");
    }

    // One invocation per output word: each element of the word that lies inside the count is computed from `x` (and
    // `y`) into `v`; the others keep the output's bits.
    private static string LaneBody(GpuElementType type, bool second, bool scalar, string value)
    {
        int lanes = Lanes(type);
        int bits = 32 / lanes;
        string mask = bits == 8 ? "0xFFu" : "0xFFFFu";
        var body = new StringBuilder();
        body.Append("    uint wa = a[gid];\n");
        if (second) body.Append("    uint wb = b[gid];\n");
        body.Append("    uint old = o[gid];\n    uint r = 0u;\n    uint keep = 0u;\n");
        body.Append("    for (uint l = 0u; l < ").Append(lanes).Append("u; l++) {\n");
        body.Append("        uint sh = l * ").Append(bits).Append("u;\n");
        body.Append("        if (((gid * ").Append(lanes).Append("u) + l) < pcount) {\n");
        body.Append("            uint x = (wa >> sh) & ").Append(mask).Append(";\n");
        if (second) body.Append("            uint y = (wb >> sh) & ").Append(mask).Append(";\n");
        if (scalar) body.Append("            uint y = pb & ").Append(mask).Append(";\n");
        // The operands with whatever lies above the element still in them, for the operations whose low bits depend
        // only on the operands' low bits (see the multiply in LaneValue).
        body.Append("            uint xw = wa >> sh;\n");
        if (second) body.Append("            uint yw = wb >> sh;\n");
        if (scalar) body.Append("            uint yw = pb;\n");
        body.Append("            uint v = ").Append(value).Append(";\n");
        // The element is placed by two shifts, not by a mask: the first drops every bit of `v` above the element.
        body.Append("            r = r | ((v << ").Append(32 - bits).Append("u) >> (").Append(32 - bits).Append("u - sh));\n");
        body.Append("            keep = keep | (").Append(mask).Append(" << sh);\n");
        body.Append("        }\n    }\n    o[gid] = (r & keep) | (old & (~keep));\n");
        return body.ToString();
    }

    private static string Signed(GpuElementType type, string operand)
    {
        string up = type == GpuElementType.Int8 ? "24u" : "16u";
        return "(int(" + operand + " << " + up + ") >> " + up + ")";
    }

    private static string LaneValue(GpuIntegerBinaryOperation operation, GpuElementType type)
    {
        if (type == GpuElementType.Bool)
        {
            return operation switch
            {
                GpuIntegerBinaryOperation.And => "uint((x != 0u) && (y != 0u))",
                GpuIntegerBinaryOperation.Or => "uint((x != 0u) || (y != 0u))",
                GpuIntegerBinaryOperation.Xor => "uint((x != 0u) != (y != 0u))",
                _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Not defined on Bool."),
            };
        }

        bool signed = type != GpuElementType.UInt8;
        return operation switch
        {
            GpuIntegerBinaryOperation.Add => "x + y",
            GpuIntegerBinaryOperation.Subtract => "x - y",
            // On the unmasked operands. One shader compiler (Dawn on Direct3D 12, AMD) treated the product of two
            // operands it knew to be below 2^8 or 2^16 as below that too: it dropped the mask or shift that removes
            // the product's high part, which then spilled into the next element of the word. The low bits of a
            // product depend only on the operands' low bits, so the unmasked product has the same element in it and
            // gives the compiler no range to reason from.
            GpuIntegerBinaryOperation.Multiply => "xw * yw",
            GpuIntegerBinaryOperation.Min => signed ? "uint(min(" + Signed(type, "x") + ", " + Signed(type, "y") + "))" : "min(x, y)",
            GpuIntegerBinaryOperation.Max => signed ? "uint(max(" + Signed(type, "x") + ", " + Signed(type, "y") + "))" : "max(x, y)",
            GpuIntegerBinaryOperation.And => "x & y",
            GpuIntegerBinaryOperation.Or => "x | y",
            GpuIntegerBinaryOperation.Xor => "x ^ y",
            _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
        };
    }

    private static string WideStatements(GpuIntegerBinaryOperation operation) => operation switch
    {
        GpuIntegerBinaryOperation.Add => "    rl = xl + yl;\n    rh = (xh + yh) + uint(rl < xl);\n",
        GpuIntegerBinaryOperation.Subtract => "    rl = xl - yl;\n    rh = (xh - yh) - uint(xl < yl);\n",
        GpuIntegerBinaryOperation.Multiply => "    rl = xl * yl;\n    rh = (pk_mulhi(xl, yl) + (xl * yh)) + (xh * yl);\n",
        GpuIntegerBinaryOperation.Min =>
            "    bool lt = (int(xh) < int(yh)) || ((xh == yh) && (xl < yl));\n    rl = yl;\n    rh = yh;\n    if (lt) {\n        rl = xl;\n        rh = xh;\n    }\n",
        GpuIntegerBinaryOperation.Max =>
            "    bool lt = (int(xh) < int(yh)) || ((xh == yh) && (xl < yl));\n    rl = xl;\n    rh = xh;\n    if (lt) {\n        rl = yl;\n        rh = yh;\n    }\n",
        GpuIntegerBinaryOperation.And => "    rl = xl & yl;\n    rh = xh & yh;\n",
        GpuIntegerBinaryOperation.Or => "    rl = xl | yl;\n    rh = xh | yh;\n",
        GpuIntegerBinaryOperation.Xor => "    rl = xl ^ yl;\n    rh = xh ^ yh;\n",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
    };

    private const string WideHead = "    uint xl = a[gid * 2u];\n    uint xh = a[(gid * 2u) + 1u];\n";

    private const string WideTail = "    o[gid * 2u] = rl;\n    o[(gid * 2u) + 1u] = rh;\n";

    private const string WideResult = "    uint rl = 0u;\n    uint rh = 0u;\n";

    /// <summary><c>o[i] = a[i] op b[i]</c>: buffers (a, b, o), one invocation per output word (per element for Int64).</summary>
    public static GpuPackedKernel Binary(GpuIntegerBinaryOperation operation, GpuElementType type)
    {
        RequireInteger(type);
        if (!GpuIntegerLayout.Defined(operation, type))
            throw new ArgumentException($"{operation} is not defined on {type}.", nameof(operation));
        return Kernel("pk_" + Stem(operation) + "_" + Suffix(type), 3, type == GpuElementType.Int64 ? IntegerLibrary : string.Empty,
            () => type == GpuElementType.Int64
                ? WideHead + "    uint yl = b[gid * 2u];\n    uint yh = b[(gid * 2u) + 1u];\n" + WideResult + WideStatements(operation) + WideTail
                : LaneBody(type, second: true, scalar: false, LaneValue(operation, type)));
    }

    /// <summary><c>o[i] = a[i] op y</c>: buffers (a, o); the scalar's low word in <c>pb</c>, its high word in <c>pc</c>.</summary>
    public static GpuPackedKernel Scalar(GpuIntegerBinaryOperation operation, GpuElementType type)
    {
        RequireInteger(type);
        if (!GpuIntegerLayout.Defined(operation, type))
            throw new ArgumentException($"{operation} is not defined on {type}.", nameof(operation));
        return Kernel("pk_" + Stem(operation) + "_s_" + Suffix(type), 2, type == GpuElementType.Int64 ? IntegerLibrary : string.Empty,
            () => type == GpuElementType.Int64
                ? WideHead + "    uint yl = pb;\n    uint yh = pc;\n" + WideResult + WideStatements(operation) + WideTail
                : LaneBody(type, second: false, scalar: true, LaneValue(operation, type)));
    }

    /// <summary><c>o[i] = op a[i]</c>: buffers (a, o).</summary>
    public static GpuPackedKernel Unary(GpuIntegerUnaryOperation operation, GpuElementType type)
    {
        RequireInteger(type);
        if (!GpuIntegerLayout.Defined(operation, type))
            throw new ArgumentException($"{operation} is not defined on {type}.", nameof(operation));
        return Kernel("pk_" + Stem(operation) + "_" + Suffix(type), 2, string.Empty, () =>
        {
            if (type == GpuElementType.Int64)
            {
                string statements = operation == GpuIntegerUnaryOperation.Negate
                    ? "    rl = 0u - xl;\n    rh = (~xh) + uint(xl == 0u);\n"
                    : "    rl = ~xl;\n    rh = ~xh;\n";
                return WideHead + WideResult + statements + WideTail;
            }

            string value = type == GpuElementType.Bool ? "uint(x == 0u)"
                : operation == GpuIntegerUnaryOperation.Negate ? "0u - x" : "~x";
            return LaneBody(type, second: false, scalar: false, value);
        });
    }

    /// <summary>
    /// A pass of a sum, buffers (a, o): invocation <c>i</c> adds block <c>i</c> of the <c>pcount</c> input elements.
    /// The narrow types add into one word a block (the sum modulo 2^32, whose low bits are the sum in the type), so
    /// the partials are an Int32 buffer; Int64 adds into two words, and its partials are Int64 again.
    /// </summary>
    public static GpuPackedKernel Sum(GpuElementType type)
    {
        RequireInteger(type);
        if (type == GpuElementType.Bool) throw new ArgumentException("Bool has no sum.", nameof(type));
        return Kernel("pk_sum_" + Suffix(type), 2, string.Empty, () =>
        {
            string head = "    uint first = gid * " + SumBlock + "u;\n    uint last = min(first + " + SumBlock + "u, pcount);\n";
            if (type == GpuElementType.Int64)
            {
                return head + "    uint sl = 0u;\n    uint su = 0u;\n    for (uint j = first; j < last; j++) {\n"
                    + "        uint vl = a[j * 2u];\n        uint vh = a[(j * 2u) + 1u];\n        uint t = sl + vl;\n"
                    + "        su = (su + vh) + uint(t < sl);\n        sl = t;\n    }\n"
                    + "    o[gid * 2u] = sl;\n    o[(gid * 2u) + 1u] = su;\n";
            }

            int lanes = Lanes(type);
            int bits = 32 / lanes;
            string mask = bits == 8 ? "0xFFu" : "0xFFFFu";
            string shift = lanes == 4 ? "2u" : "1u";
            return head + "    uint s = 0u;\n    for (uint j = first; j < last; j++) {\n"
                + "        s = s + ((a[j >> " + shift + "] >> ((j & " + (lanes - 1) + "u) * " + bits + "u)) & " + mask + ");\n    }\n"
                + "    o[gid] = s;\n";
        });
    }

    /// <summary>
    /// A copy of <c>pcount</c> elements of <paramref name="elementBytes"/> (1 or 2) bytes from element <c>pa</c> of
    /// buffer a to element <c>pb</c> of buffer o, one invocation per destination word touched. For the ranges a
    /// word-aligned buffer copy cannot express.
    /// </summary>
    public static GpuPackedKernel Copy(int elementBytes)
    {
        if (elementBytes != 1 && elementBytes != 2)
            throw new ArgumentOutOfRangeException(nameof(elementBytes), elementBytes, "The packed copy moves 1- or 2-byte elements.");
        int lanes = 4 / elementBytes;
        int bits = 8 * elementBytes;
        string mask = bits == 8 ? "0xFFu" : "0xFFFFu";
        string shift = lanes == 4 ? "2u" : "1u";
        return Kernel("pk_copy_" + bits, 2, string.Empty, () =>
            "    uint w = (pb >> " + shift + ") + gid;\n    uint r = o[w];\n"
            + "    for (uint l = 0u; l < " + lanes + "u; l++) {\n        uint e = (w << " + shift + ") + l;\n"
            + "        if ((e >= pb) && ((e - pb) < pcount)) {\n            uint se = (e - pb) + pa;\n"
            + "            uint v = (a[se >> " + shift + "] >> ((se & " + (lanes - 1) + "u) * " + bits + "u)) & " + mask + ";\n"
            + "            uint sh = l * " + bits + "u;\n            r = (r & (~(" + mask + " << sh))) | (v << sh);\n        }\n    }\n"
            + "    o[w] = r;\n");
    }

    private static string Widen(GpuElementType type) => "pk_w_" + Suffix(type);

    private static string Narrow(GpuElementType type) => "pk_n_" + Suffix(type);

    /// <summary><c>o[i] = a[i] op b[i]</c> on 16-bit floats: buffers (a, b, o).</summary>
    public static GpuPackedKernel Binary(GpuNarrowFloatBinaryOperation operation, GpuElementType type)
    {
        RequireFloat(type);
        string stem = operation switch
        {
            GpuNarrowFloatBinaryOperation.Add => "nfadd",
            GpuNarrowFloatBinaryOperation.Subtract => "nfsub",
            GpuNarrowFloatBinaryOperation.Multiply => "nfmul",
            _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown 16-bit float operation."),
        };
        string w = Widen(type), n = Narrow(type);
        string value = operation switch
        {
            GpuNarrowFloatBinaryOperation.Add => n + "(pk_fadd(" + w + "(x), " + w + "(y)))",
            GpuNarrowFloatBinaryOperation.Subtract => n + "(pk_fadd(" + w + "(x), " + w + "(y) ^ 0x80000000u))",
            _ => n + "(pk_fmul(" + w + "(x), " + w + "(y)))",
        };
        return Kernel("pk_" + stem + "_" + Suffix(type), 3, FloatLibrary, () => LaneBody(type, second: true, scalar: false, value));
    }

    /// <summary><c>o[i] = op a[i]</c> on 16-bit floats: buffers (a, o).</summary>
    public static GpuPackedKernel Unary(GpuNarrowFloatUnaryOperation operation, GpuElementType type)
    {
        RequireFloat(type);
        string stem = operation switch
        {
            GpuNarrowFloatUnaryOperation.Negate => "nfneg",
            GpuNarrowFloatUnaryOperation.Relu => "nfrelu",
            GpuNarrowFloatUnaryOperation.Gelu => "nfgelu",
            _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown 16-bit float operation."),
        };
        // The largest pattern below the sign bit that is not a NaN: infinity.
        string top = type == GpuElementType.Float16 ? "0x7C00u" : "0x7F80u";
        string value = operation switch
        {
            GpuNarrowFloatUnaryOperation.Negate => "x ^ 0x8000u",
            GpuNarrowFloatUnaryOperation.Relu => "pk_relu(x, " + top + ")",
            _ => Narrow(type) + "(FBITS(pk_gelu(UFLOAT(" + Widen(type) + "(x)))))",
        };
        return Kernel("pk_" + stem + "_" + Suffix(type), 2, FloatLibrary, () => LaneBody(type, second: false, scalar: false, value));
    }

    /// <summary>
    /// <c>c = a b</c> on 16-bit floats, buffers (a, b, c): <c>pcount</c> = m n output elements, <c>pa</c> = n columns,
    /// <c>pb</c> = k. One invocation per output word; each element is a row of a against a column of b accumulated in
    /// binary32 and rounded once into the type. The accumulator is computed on the bits (pk_fmul, pk_fadd), in the
    /// order the host adds in: a BFloat16 operand below 2^-126 widens to a binary32 subnormal, which a device may
    /// flush to zero before a native multiply (0x0001 times 0x7E80 is the normal 0x3C00, not zero).
    /// </summary>
    public static GpuPackedKernel MatrixMultiply(GpuElementType type)
    {
        RequireFloat(type);
        string w = Widen(type);
        return Kernel("pk_nfgemm_" + Suffix(type), 3, FloatLibrary, () =>
            "    uint r = o[gid];\n    for (uint l = 0u; l < 2u; l++) {\n        uint e = (gid * 2u) + l;\n        if (e < pcount) {\n"
            + "            uint row = e / pa;\n            uint col = e - (row * pa);\n            uint s = 0u;\n"
            + "            for (uint j = 0u; j < pb; j++) {\n                uint ia = (row * pb) + j;\n                uint ib = (j * pa) + col;\n"
            + "                uint ha = (a[ia >> 1u] >> ((ia & 1u) * 16u)) & 0xFFFFu;\n"
            + "                uint hb = (b[ib >> 1u] >> ((ib & 1u) * 16u)) & 0xFFFFu;\n"
            + "                s = pk_fadd(s, pk_fmul(" + w + "(ha), " + w + "(hb)));\n            }\n"
            + "            uint v = " + Narrow(type) + "(s);\n            uint sh = l * 16u;\n"
            + "            r = (r & (~(0xFFFFu << sh))) | (v << sh);\n        }\n    }\n    o[gid] = r;\n");
    }

    /// <summary>Every kernel, in a stable order: what a Metal library holds, and what the compilation tests compile.</summary>
    public static IReadOnlyList<GpuPackedKernel> All => AllKernels.Value;

    private static readonly Lazy<List<GpuPackedKernel>> AllKernels = new Lazy<List<GpuPackedKernel>>(() =>
    {
        var all = new List<GpuPackedKernel>();
        foreach (var type in IntegerTypes)
        {
            foreach (var operation in GpuInt32ShaderSource.BinaryOperations.Where(operation => GpuIntegerLayout.Defined(operation, type)))
            {
                all.Add(Binary(operation, type));
                all.Add(Scalar(operation, type));
            }

            all.AddRange(GpuInt32ShaderSource.UnaryOperations.Where(operation => GpuIntegerLayout.Defined(operation, type))
                .Select(operation => Unary(operation, type)));
            if (type != GpuElementType.Bool) all.Add(Sum(type));
        }

        foreach (var type in FloatTypes)
        {
            all.Add(Binary(GpuNarrowFloatBinaryOperation.Add, type));
            all.Add(Binary(GpuNarrowFloatBinaryOperation.Subtract, type));
            all.Add(Binary(GpuNarrowFloatBinaryOperation.Multiply, type));
            all.Add(Unary(GpuNarrowFloatUnaryOperation.Negate, type));
            all.Add(Unary(GpuNarrowFloatUnaryOperation.Relu, type));
            all.Add(Unary(GpuNarrowFloatUnaryOperation.Gelu, type));
            all.Add(MatrixMultiply(type));
        }

        all.Add(Copy(1));
        all.Add(Copy(2));
        return all;
    });

    // ------------------------------------------------------------------------------------------------------------
    // Writing a kernel out.
    // ------------------------------------------------------------------------------------------------------------

    private static string BufferName(int index, int buffers) => index == buffers - 1 ? "o" : index == 0 ? "a" : "b";

    private static string Bits(string text, GpuPackedKernelDialect dialect) => dialect switch
    {
        GpuPackedKernelDialect.Glsl => text.Replace("FBITS(", "floatBitsToUint(").Replace("UFLOAT(", "uintBitsToFloat("),
        GpuPackedKernelDialect.Metal => text.Replace("FBITS(", "as_type<uint>(").Replace("UFLOAT(", "as_type<float>("),
        _ => text.Replace("FBITS(", "bitcast<u32>(").Replace("UFLOAT(", "bitcast<f32>("),
    };

    private static string WgslType(string type) => type switch
    {
        "uint" => "u32",
        "int" => "i32",
        "float" => "f32",
        _ => type,
    };

    // The neutral form as WGSL: function headers, declarations and conversions are the only spellings that differ.
    internal static string ToWgsl(string text)
    {
        text = Regex.Replace(text, @"^(uint|int|float|bool) (\w+)\(([^)]*)\) \{", match =>
        {
            var parameters = match.Groups[3].Value.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(parameter => parameter.Trim().Split(' '))
                .Select(parts => parts[1] + ": " + WgslType(parts[0]));
            return "fn " + match.Groups[2].Value + "(" + string.Join(", ", parameters) + ") -> " + WgslType(match.Groups[1].Value) + " {";
        }, RegexOptions.Multiline, RegexTimeout);
        text = Regex.Replace(text, @"\b(uint|int|float|bool) (\w+) = ",
            match => "var " + match.Groups[2].Value + ": " + WgslType(match.Groups[1].Value) + " = ", RegexOptions.None, RegexTimeout);
        text = Regex.Replace(text, @"\b(uint|int|float)\(", match => WgslType(match.Groups[1].Value) + "(", RegexOptions.None, RegexTimeout);
        text = Regex.Replace(text, @"\bpcount\b", "params.p0", RegexOptions.None, RegexTimeout);
        text = Regex.Replace(text, @"\bpa\b", "params.p1", RegexOptions.None, RegexTimeout);
        text = Regex.Replace(text, @"\bpb\b", "(params.s0 | (params.s1 << 16u))", RegexOptions.None, RegexTimeout);
        text = Regex.Replace(text, @"\bpc\b", "(params.s2 | (params.s3 << 16u))", RegexOptions.None, RegexTimeout);
        return text;
    }

    /// <summary>The GLSL compute shader of <paramref name="kernel"/>: buffers at bindings 0.., five push-constant words.</summary>
    public static string Glsl(GpuPackedKernel kernel, int localSize)
        => Sources.GetOrAdd("glsl:" + localSize + ":" + kernel.Name, _ =>
        {
            var source = new StringBuilder("#version 450\nlayout(local_size_x = ").Append(localSize).Append(") in;\n");
            for (int i = 0; i < kernel.Buffers; i++)
            {
                string name = BufferName(i, kernel.Buffers);
                source.Append("layout(set = 0, binding = ").Append(i).Append(") ").Append(name == "o" ? string.Empty : "readonly ")
                    .Append("buffer B").Append(i).Append(" { uint ").Append(name).Append("[]; };\n");
            }

            source.Append("layout(push_constant) uniform P { uint pn; uint pcount; uint pa; uint pb; uint pc; };\n");
            source.Append(Bits(kernel.Library, GpuPackedKernelDialect.Glsl));
            source.Append("void main()\n{\n    uint gid = gl_GlobalInvocationID.x;\n    if (gid >= pn) {\n        return;\n    }\n");
            source.Append(Bits(kernel.Body, GpuPackedKernelDialect.Glsl)).Append("}\n");
            return source.ToString();
        });

    /// <summary>
    /// The WGSL module of <paramref name="kernel"/> (entry point <c>main</c>): buffers at bindings 0.., then the
    /// uniform block every typed kernel of this backend takes (<c>n, p0, p1, pad, s0..s3</c>, the last four 16-bit
    /// pieces of a 64-bit value).
    /// </summary>
    public static string Wgsl(GpuPackedKernel kernel)
        => Sources.GetOrAdd("wgsl:" + kernel.Name, _ =>
        {
            var source = new StringBuilder();
            for (int i = 0; i < kernel.Buffers; i++)
            {
                string name = BufferName(i, kernel.Buffers);
                source.Append("@group(0) @binding(").Append(i).Append(") var<storage, ").Append(name == "o" ? "read_write" : "read")
                    .Append("> ").Append(name).Append(": array<u32>;\n");
            }

            source.Append("struct Params { n: u32, p0: u32, p1: u32, pad0: u32, s0: u32, s1: u32, s2: u32, s3: u32 }\n");
            source.Append("@group(0) @binding(").Append(kernel.Buffers).Append(") var<uniform> params: Params;\n");
            source.Append(ToWgsl(Bits(kernel.Library, GpuPackedKernelDialect.Wgsl)));
            source.Append("@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) invocation: vec3<u32>) {\n");
            source.Append("    let gid = invocation.x;\n    if (gid >= params.n) {\n        return;\n    }\n");
            source.Append(ToWgsl(Bits(kernel.Body, GpuPackedKernelDialect.Wgsl))).Append("}\n");
            return source.ToString();
        });

    /// <summary>
    /// One Metal library with every kernel: buffers at indexes 0.., then the five parameter words, each a
    /// <c>constant uint&amp;</c>.
    /// </summary>
    public static string MetalLibrary => MetalSource.Value;

    private static readonly Lazy<string> MetalSource = new Lazy<string>(() =>
    {
        var source = new StringBuilder("#include <metal_stdlib>\nusing namespace metal;\n");
        source.Append(IntegerLibrary).Append(Bits(FloatLibrary, GpuPackedKernelDialect.Metal));
        foreach (var kernel in All)
        {
            source.Append("\nkernel void ").Append(kernel.Name).Append("(\n");
            for (int i = 0; i < kernel.Buffers; i++)
            {
                string name = BufferName(i, kernel.Buffers);
                source.Append("    device ").Append(name == "o" ? string.Empty : "const ").Append("uint* ").Append(name)
                    .Append(" [[buffer(").Append(i).Append(")]],\n");
            }

            string[] parameters = { "pn", "pcount", "pa", "pb", "pc" };
            for (int i = 0; i < parameters.Length; i++)
                source.Append("    constant uint& ").Append(parameters[i]).Append(" [[buffer(").Append(kernel.Buffers + i).Append(")]],\n");
            source.Append("    uint gid [[thread_position_in_grid]])\n{\n    if (gid >= pn) {\n        return;\n    }\n");
            source.Append(Bits(kernel.Body, GpuPackedKernelDialect.Metal)).Append("}\n");
        }

        return source.ToString();
    });
}
