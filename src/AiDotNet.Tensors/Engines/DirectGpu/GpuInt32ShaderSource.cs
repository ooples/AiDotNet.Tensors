// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>The shading language a 32-bit integer expression is written for.</summary>
internal enum GpuInt32ShaderDialect
{
    /// <summary>GLSL (Vulkan).</summary>
    Glsl,
    /// <summary>WGSL (WebGPU).</summary>
    Wgsl,
    /// <summary>Metal Shading Language.</summary>
    Metal,
}

/// <summary>
/// The 32-bit integer expressions of the backends whose buffers are 4-byte words (Vulkan, WebGPU, Metal): one
/// description, so the three compute the same thing. Operands are the signed <c>x</c> and <c>y</c>; the result is a
/// signed 32-bit value that wraps as unchecked host arithmetic does.
/// </summary>
internal static class GpuInt32ShaderSource
{
    /// <summary>Elements one invocation of a sum pass folds.</summary>
    public const int SumBlock = 64;

    /// <summary>The kernel-name stem of <paramref name="operation"/>.</summary>
    public static string Name(GpuIntegerBinaryOperation operation) => operation switch
    {
        GpuIntegerBinaryOperation.Add => "add",
        GpuIntegerBinaryOperation.Subtract => "sub",
        GpuIntegerBinaryOperation.Multiply => "mul",
        GpuIntegerBinaryOperation.Min => "min",
        GpuIntegerBinaryOperation.Max => "max",
        GpuIntegerBinaryOperation.And => "and",
        GpuIntegerBinaryOperation.Or => "or",
        GpuIntegerBinaryOperation.Xor => "xor",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
    };

    /// <summary>The kernel-name stem of <paramref name="operation"/>.</summary>
    public static string Name(GpuIntegerUnaryOperation operation) => operation switch
    {
        GpuIntegerUnaryOperation.Negate => "neg",
        GpuIntegerUnaryOperation.Not => "not",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
    };

    public static readonly GpuIntegerBinaryOperation[] BinaryOperations =
    {
        GpuIntegerBinaryOperation.Add, GpuIntegerBinaryOperation.Subtract, GpuIntegerBinaryOperation.Multiply,
        GpuIntegerBinaryOperation.Min, GpuIntegerBinaryOperation.Max, GpuIntegerBinaryOperation.And,
        GpuIntegerBinaryOperation.Or, GpuIntegerBinaryOperation.Xor,
    };

    public static readonly GpuIntegerUnaryOperation[] UnaryOperations = { GpuIntegerUnaryOperation.Negate, GpuIntegerUnaryOperation.Not };

    // WGSL defines i32 arithmetic as wrapping. GLSL and MSL leave signed overflow to the compiler, so there the
    // wrapping operations are computed on the unsigned bits.
    private static string Wrapping(GpuInt32ShaderDialect dialect, string symbol, string left, string right)
        => dialect == GpuInt32ShaderDialect.Wgsl
            ? left + " " + symbol + " " + right
            : "int(uint(" + left + ") " + symbol + " uint(" + right + "))";

    /// <summary><c>x op y</c>.</summary>
    public static string Expression(GpuIntegerBinaryOperation operation, GpuInt32ShaderDialect dialect) => operation switch
    {
        GpuIntegerBinaryOperation.Add => Wrapping(dialect, "+", "x", "y"),
        GpuIntegerBinaryOperation.Subtract => Wrapping(dialect, "-", "x", "y"),
        GpuIntegerBinaryOperation.Multiply => Wrapping(dialect, "*", "x", "y"),
        GpuIntegerBinaryOperation.Min => "min(x, y)",
        GpuIntegerBinaryOperation.Max => "max(x, y)",
        GpuIntegerBinaryOperation.And => "x & y",
        GpuIntegerBinaryOperation.Or => "x | y",
        GpuIntegerBinaryOperation.Xor => "x ^ y",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
    };

    /// <summary><c>op x</c>.</summary>
    public static string Expression(GpuIntegerUnaryOperation operation, GpuInt32ShaderDialect dialect) => operation switch
    {
        GpuIntegerUnaryOperation.Negate => dialect == GpuInt32ShaderDialect.Wgsl ? "-x" : "int(0u - uint(x))",
        GpuIntegerUnaryOperation.Not => "~x",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
    };

    /// <summary>Validates a sum's operand: an Int32 buffer with <paramref name="size"/> elements.</summary>
    public static void RequireSum(IGpuBuffer a, int size)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (a.ElementType != GpuElementType.Int32)
            throw new ArgumentException($"This device reduces {GpuElementType.Int32} buffers, got {a.ElementType}.", nameof(a));
        if (size < 0 || size > a.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside the buffer's {a.Size} elements.");
    }

    /// <summary>Validates an element-wise launch on a device that computes Int32 only.</summary>
    public static void Require(string operation, int size, IGpuBuffer output, params IGpuBuffer[] inputs)
    {
        var type = GpuIntegerLayout.Require(operation, size, output, inputs);
        if (type != GpuElementType.Int32)
            throw new ArgumentException($"This device computes {GpuElementType.Int32} buffers, got {type}.", nameof(output));
    }
}
