// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>The C dialect an integer kernel program is generated for.</summary>
internal enum GpuIntegerKernelDialect
{
    /// <summary>OpenCL C.</summary>
    OpenCl,
    /// <summary>CUDA C++ (also what HIP compiles).</summary>
    Cuda,
}

/// <summary>
/// The integer and bool kernels of the backends that compile C (OpenCL, CUDA, HIP), generated from one description so
/// the three run the same arithmetic. Signed overflow is undefined in C, so every wrapping operation is computed in an
/// unsigned type at least as wide as the element and truncated on the store.
/// </summary>
internal static class GpuIntegerKernelSource
{
    /// <summary>Elements one invocation of the sum kernel folds into a 64-bit partial.</summary>
    public const int SumBlock = 256;

    public static readonly GpuElementType[] Types =
    {
        GpuElementType.Int8, GpuElementType.UInt8, GpuElementType.Int16, GpuElementType.Int32, GpuElementType.Int64,
        GpuElementType.Bool,
    };

    /// <summary>The kernel-name suffix of <paramref name="type"/>.</summary>
    public static string Suffix(GpuElementType type) => type switch
    {
        GpuElementType.Int8 => "i8",
        GpuElementType.UInt8 => "u8",
        GpuElementType.Int16 => "i16",
        GpuElementType.Int32 => "i32",
        GpuElementType.Int64 => "i64",
        GpuElementType.Bool => "b8",
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Not an integer element type."),
    };

    private static string Stem(GpuIntegerBinaryOperation operation) => operation switch
    {
        GpuIntegerBinaryOperation.Add => "iadd",
        GpuIntegerBinaryOperation.Subtract => "isub",
        GpuIntegerBinaryOperation.Multiply => "imul",
        GpuIntegerBinaryOperation.Min => "imin",
        GpuIntegerBinaryOperation.Max => "imax",
        GpuIntegerBinaryOperation.And => "iand",
        GpuIntegerBinaryOperation.Or => "ior",
        GpuIntegerBinaryOperation.Xor => "ixor",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
    };

    private static string Stem(GpuIntegerUnaryOperation operation) => operation switch
    {
        GpuIntegerUnaryOperation.Negate => "ineg",
        GpuIntegerUnaryOperation.Not => "inot",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
    };

    /// <summary>The kernel computing <c>o[i] = a[i] op b[i]</c>: <c>(a, b, o, int size)</c>.</summary>
    public static string Binary(GpuIntegerBinaryOperation operation, GpuElementType type) => Stem(operation) + "_" + Suffix(type);

    /// <summary>The kernel computing <c>o[i] = a[i] op y</c>: <c>(a, o, 64-bit y, int size)</c>.</summary>
    public static string Scalar(GpuIntegerBinaryOperation operation, GpuElementType type) => Stem(operation) + "_scalar_" + Suffix(type);

    /// <summary>The kernel computing <c>o[i] = op a[i]</c>: <c>(a, o, int size)</c>.</summary>
    public static string Unary(GpuIntegerUnaryOperation operation, GpuElementType type) => Stem(operation) + "_" + Suffix(type);

    /// <summary>
    /// The kernel summing blocks of <see cref="SumBlock"/> elements into 64-bit partials:
    /// <c>(a, 64-bit o, int count, int blocks)</c>, one invocation per block.
    /// </summary>
    public static string Sum(GpuElementType type) => "isum_" + Suffix(type);

    /// <summary>Every kernel name of the program, in source order.</summary>
    public static IReadOnlyList<string> KernelNames => Names.Value;

    // Lazy: the operation tables below are initialised after this member in textual order.
    private static readonly Lazy<List<string>> Names = new Lazy<List<string>>(BuildNames);

    private static readonly GpuIntegerBinaryOperation[] BinaryOperations =
    {
        GpuIntegerBinaryOperation.Add, GpuIntegerBinaryOperation.Subtract, GpuIntegerBinaryOperation.Multiply,
        GpuIntegerBinaryOperation.Min, GpuIntegerBinaryOperation.Max, GpuIntegerBinaryOperation.And,
        GpuIntegerBinaryOperation.Or, GpuIntegerBinaryOperation.Xor,
    };

    private static readonly GpuIntegerUnaryOperation[] UnaryOperations = { GpuIntegerUnaryOperation.Negate, GpuIntegerUnaryOperation.Not };

    private static List<string> BuildNames()
    {
        var names = new List<string>();
        foreach (var type in Types)
        {
            foreach (var operation in BinaryOperations)
            {
                if (!GpuIntegerLayout.Defined(operation, type)) continue;
                names.Add(Binary(operation, type));
                names.Add(Scalar(operation, type));
            }

            foreach (var operation in UnaryOperations)
                if (GpuIntegerLayout.Defined(operation, type)) names.Add(Unary(operation, type));
            if (type != GpuElementType.Bool) names.Add(Sum(type));
        }

        GpuNarrowFloatKernelSource.AddNames(names);
        return names;
    }

    private static string Expression(GpuIntegerBinaryOperation operation, bool logical) => operation switch
    {
        GpuIntegerBinaryOperation.Add => "(W)x + (W)y",
        GpuIntegerBinaryOperation.Subtract => "(W)x - (W)y",
        GpuIntegerBinaryOperation.Multiply => "(W)x * (W)y",
        GpuIntegerBinaryOperation.Min => "x < y ? x : y",
        GpuIntegerBinaryOperation.Max => "x > y ? x : y",
        GpuIntegerBinaryOperation.And => logical ? "(x != 0) && (y != 0)" : "x & y",
        GpuIntegerBinaryOperation.Or => logical ? "(x != 0) || (y != 0)" : "x | y",
        GpuIntegerBinaryOperation.Xor => logical ? "(x != 0) != (y != 0)" : "x ^ y",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
    };

    /// <summary>The whole program for <paramref name="dialect"/>.</summary>
    public static string Build(GpuIntegerKernelDialect dialect)
    {
        bool openCl = dialect == GpuIntegerKernelDialect.OpenCl;
        var source = new StringBuilder(16384);
        source.Append(openCl
            ? @"
typedef char i8t; typedef uchar u8t; typedef short i16t; typedef int i32t; typedef long i64t;
typedef uint u32t; typedef ulong u64t;
#define IK_KERNEL __kernel void
#define IK_PTR __global
#define IK_GID ((int)get_global_id(0))
"
            : @"
typedef signed char i8t; typedef unsigned char u8t; typedef short i16t; typedef int i32t; typedef long long i64t;
typedef unsigned int u32t; typedef unsigned long long u64t;
#define IK_KERNEL extern ""C"" __global__ void
#define IK_PTR
#define IK_GID ((int)(blockIdx.x * blockDim.x + threadIdx.x))
");
        foreach (var type in Types)
        {
            bool logical = type == GpuElementType.Bool;
            string t = type switch
            {
                GpuElementType.Int8 => "i8t",
                GpuElementType.Int16 => "i16t",
                GpuElementType.Int32 => "i32t",
                GpuElementType.Int64 => "i64t",
                _ => "u8t",
            };
            // The unsigned type the wrapping operations are computed in.
            string w = type == GpuElementType.Int64 ? "u64t" : "u32t";
            foreach (var operation in BinaryOperations)
            {
                if (!GpuIntegerLayout.Defined(operation, type)) continue;
                string expression = Expression(operation, logical).Replace("(W)", "(" + w + ")");
                source.Append("IK_KERNEL ").Append(Binary(operation, type))
                    .Append("(IK_PTR const ").Append(t).Append("* a, IK_PTR const ").Append(t).Append("* b, IK_PTR ").Append(t)
                    .Append("* o, const int size)\n{\n    const int i = IK_GID;\n    if (i < size) { const ").Append(t)
                    .Append(" x = a[i]; const ").Append(t).Append(" y = b[i]; o[i] = (").Append(t).Append(")(").Append(expression)
                    .Append("); }\n}\n");
                source.Append("IK_KERNEL ").Append(Scalar(operation, type))
                    .Append("(IK_PTR const ").Append(t).Append("* a, IK_PTR ").Append(t)
                    .Append("* o, const i64t y64, const int size)\n{\n    const int i = IK_GID;\n    if (i < size) { const ").Append(t)
                    .Append(" x = a[i]; const ").Append(t).Append(" y = (").Append(t).Append(")y64; o[i] = (").Append(t).Append(")(")
                    .Append(expression).Append("); }\n}\n");
            }

            foreach (var operation in UnaryOperations)
            {
                if (!GpuIntegerLayout.Defined(operation, type)) continue;
                string expression = operation == GpuIntegerUnaryOperation.Negate ? "(" + w + ")0 - (" + w + ")x" : logical ? "x == 0" : "~x";
                source.Append("IK_KERNEL ").Append(Unary(operation, type))
                    .Append("(IK_PTR const ").Append(t).Append("* a, IK_PTR ").Append(t)
                    .Append("* o, const int size)\n{\n    const int i = IK_GID;\n    if (i < size) { const ").Append(t)
                    .Append(" x = a[i]; o[i] = (").Append(t).Append(")(").Append(expression).Append("); }\n}\n");
            }

            if (logical) continue;
            // Each element is widened to 64 bits by its own signedness, then added modulo 2^64.
            source.Append("IK_KERNEL ").Append(Sum(type))
                .Append("(IK_PTR const ").Append(t).Append("* a, IK_PTR i64t* o, const int count, const int blocks)\n{\n")
                .Append("    const int i = IK_GID;\n    if (i >= blocks) return;\n    const int first = i * ").Append(SumBlock)
                .Append(";\n    const int last = (first + ").Append(SumBlock).Append(" < count) ? first + ").Append(SumBlock)
                .Append(" : count;\n    u64t s = 0;\n    for (int j = first; j < last; j++) s += (u64t)(i64t)a[j];\n    o[i] = (i64t)s;\n}\n");
        }

        // The 16-bit float kernels ride in the same program: one compile, one module.
        GpuNarrowFloatKernelSource.Append(source, openCl);
        return source.ToString();
    }
}
