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
        GpuIntegerBinaryOperation.Divide => "div",
        GpuIntegerBinaryOperation.Remainder => "rem",
        GpuIntegerBinaryOperation.Equal => "eq",
        GpuIntegerBinaryOperation.NotEqual => "ne",
        GpuIntegerBinaryOperation.Greater => "gt",
        GpuIntegerBinaryOperation.Less => "lt",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
    };

    /// <summary>The kernel-name stem of <paramref name="operation"/>.</summary>
    public static string Name(GpuIntegerUnaryOperation operation) => operation switch
    {
        GpuIntegerUnaryOperation.Negate => "neg",
        GpuIntegerUnaryOperation.Not => "not",
        GpuIntegerUnaryOperation.Abs => "abs",
        GpuIntegerUnaryOperation.Sign => "sign",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
    };

    /// <summary>The kernel stem of a reduction (<c>min</c> and <c>max</c> are the element-wise ones).</summary>
    public static string Name(GpuIntegerReduction reduction) => reduction switch
    {
        GpuIntegerReduction.Sum => "sum",
        GpuIntegerReduction.Min => "rmin",
        GpuIntegerReduction.Max => "rmax",
        _ => throw new ArgumentOutOfRangeException(nameof(reduction), reduction, "Unknown integer reduction."),
    };

    public static readonly GpuIntegerReduction[] Reductions = { GpuIntegerReduction.Sum, GpuIntegerReduction.Min, GpuIntegerReduction.Max };

    public static readonly GpuIntegerBinaryOperation[] BinaryOperations =
    {
        GpuIntegerBinaryOperation.Add, GpuIntegerBinaryOperation.Subtract, GpuIntegerBinaryOperation.Multiply,
        GpuIntegerBinaryOperation.Min, GpuIntegerBinaryOperation.Max, GpuIntegerBinaryOperation.And,
        GpuIntegerBinaryOperation.Or, GpuIntegerBinaryOperation.Xor, GpuIntegerBinaryOperation.Divide,
        GpuIntegerBinaryOperation.Remainder, GpuIntegerBinaryOperation.Equal, GpuIntegerBinaryOperation.NotEqual,
        GpuIntegerBinaryOperation.Greater, GpuIntegerBinaryOperation.Less,
    };

    public static readonly GpuIntegerUnaryOperation[] UnaryOperations = 
    {
        GpuIntegerUnaryOperation.Negate, GpuIntegerUnaryOperation.Not, GpuIntegerUnaryOperation.Abs, GpuIntegerUnaryOperation.Sign,
    };

    // WGSL defines i32 arithmetic as wrapping. GLSL and MSL leave signed overflow to the compiler, so there the
    // wrapping operations are computed on the unsigned bits.
    private static string Wrapping(GpuInt32ShaderDialect dialect, string symbol, string left, string right)
        => dialect == GpuInt32ShaderDialect.Wgsl
            ? left + " " + symbol + " " + right
            : "int(uint(" + left + ") " + symbol + " uint(" + right + "))";

    /// <summary><c>x op y</c>.</summary>
    private static string Flag(GpuInt32ShaderDialect dialect, string condition)
        => (dialect == GpuInt32ShaderDialect.Wgsl ? "i32(" : "int(") + condition + ")";

    // `whenTrue` where `condition` holds, else `whenFalse`. WGSL has no conditional operator; its select evaluates
    // both, which is why the operands below are made safe to evaluate either way.
    private static string Choose(GpuInt32ShaderDialect dialect, string condition, string whenTrue, string whenFalse)
        => dialect == GpuInt32ShaderDialect.Wgsl
            ? "select(" + whenFalse + ", " + whenTrue + ", " + condition + ")"
            : "((" + condition + ") ? " + whenTrue + " : " + whenFalse + ")";

    // The quotient rounded toward zero, or its remainder, of `x` and `y`: on the magnitudes as unsigned values, where
    // every dialect defines the division (signed division of negative operands is not defined by GLSL, and the most
    // negative value over -1 by none of them), then signed. The divisor is never zero where the division is evaluated.
    private static string Quotient(GpuInt32ShaderDialect dialect, bool remainder)
    {
        string u = dialect == GpuInt32ShaderDialect.Wgsl ? "u32(" : "uint(";
        string i = dialect == GpuInt32ShaderDialect.Wgsl ? "i32(" : "int(";
        string ax = Choose(dialect, "x < 0", "(0u - " + u + "x))", u + "x)");
        string ay = "max(" + Choose(dialect, "y < 0", "(0u - " + u + "y))", u + "y)") + ", 1u)";
        string magnitude = "(" + ax + (remainder ? " % " : " / ") + ay + ")";
        string negative = remainder ? "x < 0" : "(x < 0) != (y < 0)";
        string signed = i + Choose(dialect, negative, "(0u - " + magnitude + ")", magnitude) + ")";
        return Choose(dialect, "y == 0", "0", signed);
    }

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
        GpuIntegerBinaryOperation.Divide => Quotient(dialect, remainder: false),
        GpuIntegerBinaryOperation.Remainder => Quotient(dialect, remainder: true),
        GpuIntegerBinaryOperation.Equal => Flag(dialect, "x == y"),
        GpuIntegerBinaryOperation.NotEqual => Flag(dialect, "x != y"),
        GpuIntegerBinaryOperation.Greater => Flag(dialect, "x > y"),
        GpuIntegerBinaryOperation.Less => Flag(dialect, "x < y"),
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown integer operation."),
    };

    /// <summary><c>op x</c>.</summary>
    public static string Expression(GpuIntegerUnaryOperation operation, GpuInt32ShaderDialect dialect) => operation switch
    {
        GpuIntegerUnaryOperation.Negate => dialect == GpuInt32ShaderDialect.Wgsl ? "-x" : "int(0u - uint(x))",
        GpuIntegerUnaryOperation.Not => "~x",
        // WGSL defines abs of the most negative i32 as itself; the others get it from the unsigned negation.
        GpuIntegerUnaryOperation.Abs => dialect == GpuInt32ShaderDialect.Wgsl ? "abs(x)" : "((x < 0) ? int(0u - uint(x)) : x)",
        GpuIntegerUnaryOperation.Sign => Flag(dialect, "x > 0") + " - " + Flag(dialect, "x < 0"),
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
