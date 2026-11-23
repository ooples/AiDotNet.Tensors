// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>An element-wise operation over two integer or <see cref="GpuElementType.Bool"/> buffers of one type.</summary>
public enum GpuIntegerBinaryOperation
{
    /// <summary><c>a[i] + b[i]</c>, wrapping in the element type.</summary>
    Add,
    /// <summary><c>a[i] - b[i]</c>, wrapping in the element type.</summary>
    Subtract,
    /// <summary><c>a[i] * b[i]</c>, wrapping in the element type.</summary>
    Multiply,
    /// <summary>The smaller of <c>a[i]</c> and <c>b[i]</c>.</summary>
    Min,
    /// <summary>The larger of <c>a[i]</c> and <c>b[i]</c>.</summary>
    Max,
    /// <summary>Bitwise and; the logical and of two <see cref="GpuElementType.Bool"/> elements.</summary>
    And,
    /// <summary>Bitwise or; the logical or of two <see cref="GpuElementType.Bool"/> elements.</summary>
    Or,
    /// <summary>Bitwise exclusive or; the logical exclusive or of two <see cref="GpuElementType.Bool"/> elements.</summary>
    Xor,
}

/// <summary>An element-wise operation over one integer or <see cref="GpuElementType.Bool"/> buffer.</summary>
public enum GpuIntegerUnaryOperation
{
    /// <summary><c>-x</c>, wrapping in the element type (the most negative value is its own negation).</summary>
    Negate,
    /// <summary>Bitwise complement; the logical negation of a <see cref="GpuElementType.Bool"/> element.</summary>
    Not,
}

/// <summary>
/// Element-wise integer arithmetic computed on the device in the buffers' own element type. Results are exact and
/// wrap on overflow as unchecked host arithmetic in that type does, so they equal the host's bit for bit; nothing is
/// routed through a float.
/// </summary>
public interface IGpuIntegerElementwise
{
    /// <summary>
    /// Whether this device computes element-wise operations on <paramref name="type"/> buffers. For
    /// <see cref="GpuElementType.Bool"/> that is the logical operations only
    /// (<see cref="GpuIntegerBinaryOperation.And"/>, <see cref="GpuIntegerBinaryOperation.Or"/>,
    /// <see cref="GpuIntegerBinaryOperation.Xor"/> and <see cref="GpuIntegerUnaryOperation.Not"/>).
    /// </summary>
    bool SupportsIntegerElementwise(GpuElementType type);

    /// <summary>
    /// Applies <paramref name="operation"/> to the first <paramref name="size"/> elements of <paramref name="a"/> and
    /// <paramref name="b"/>, writing <paramref name="output"/>. The three buffers hold one element type, and
    /// <paramref name="output"/> is not one of the operands.
    /// </summary>
    void BinaryInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);

    /// <summary>
    /// Applies <paramref name="operation"/> to the first <paramref name="size"/> elements of <paramref name="a"/>,
    /// writing <paramref name="output"/>, a different buffer of the same element type.
    /// </summary>
    void UnaryInteger(GpuIntegerUnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size);

    /// <summary>
    /// Applies <paramref name="operation"/> between each of the first <paramref name="size"/> elements of
    /// <paramref name="a"/> (the left operand) and <paramref name="scalar"/> truncated to the element type (the right
    /// one), writing <paramref name="output"/>, a different buffer of the same element type.
    /// </summary>
    void ScalarInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, long scalar, IGpuBuffer output, int size);
}

/// <summary>Integer reductions computed on the device in the buffer's own element type.</summary>
public interface IGpuIntegerReductions
{
    /// <summary>Whether this device reduces <paramref name="type"/> buffers.</summary>
    bool SupportsIntegerReductions(GpuElementType type);

    /// <summary>
    /// The sum of the first <paramref name="size"/> elements of <paramref name="a"/>, wrapped to the element type's
    /// width as an unchecked host sum in that type is, and widened to <see cref="long"/>.
    /// </summary>
    long SumInteger(IGpuBuffer a, int size);
}

/// <summary>What the integer kernels of every backend share.</summary>
internal static class GpuIntegerLayout
{
    /// <summary>Whether <paramref name="type"/> is one the integer interfaces are about.</summary>
    public static bool IsIntegerOrBool(GpuElementType type)
        => type is GpuElementType.Int8 or GpuElementType.UInt8 or GpuElementType.Int16 or GpuElementType.Int32
            or GpuElementType.Int64 or GpuElementType.Bool;

    /// <summary>Whether <paramref name="operation"/> is defined for <paramref name="type"/>.</summary>
    public static bool Defined(GpuIntegerBinaryOperation operation, GpuElementType type)
        => type != GpuElementType.Bool
            ? IsIntegerOrBool(type)
            : operation is GpuIntegerBinaryOperation.And or GpuIntegerBinaryOperation.Or or GpuIntegerBinaryOperation.Xor;

    /// <summary>Whether <paramref name="operation"/> is defined for <paramref name="type"/>.</summary>
    public static bool Defined(GpuIntegerUnaryOperation operation, GpuElementType type)
        => type != GpuElementType.Bool ? IsIntegerOrBool(type) : operation == GpuIntegerUnaryOperation.Not;

    /// <summary>
    /// Validates the operands of one integer kernel launch and returns their element type: every buffer holds that
    /// type, <paramref name="size"/> elements fit in each, and the output is not an operand.
    /// </summary>
    public static GpuElementType Require(string operation, int size, IGpuBuffer output, params IGpuBuffer[] inputs)
    {
        if (output is null) throw new System.ArgumentNullException(nameof(output));
        var type = output.ElementType;
        if (!IsIntegerOrBool(type))
            throw new System.ArgumentException($"{operation} computes on integer buffers, got {type}.", nameof(output));
        if (size < 0 || size > output.Size)
            throw new System.ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside the output's {output.Size} elements.");
        foreach (var input in inputs)
        {
            if (input is null) throw new System.ArgumentNullException(nameof(inputs));
            if (input.ElementType != type)
                throw new System.ArgumentException($"{operation} cannot mix {input.ElementType} and {type} buffers.", nameof(inputs));
            if (size > input.Size)
                throw new System.ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside an operand's {input.Size} elements.");
            if (ReferenceEquals(input, output) || (input.Handle != System.IntPtr.Zero && input.Handle == output.Handle))
                throw new System.ArgumentException($"{operation}: the output buffer must not be one of the operands.", nameof(output));
        }

        return type;
    }

    /// <summary>
    /// How many blocks of <paramref name="block"/> hold <paramref name="count"/> elements, rounded up in 64 bits: the
    /// largest buffers are within a block of <see cref="int.MaxValue"/>, where a 32-bit round-up wraps negative.
    /// </summary>
    public static int Blocks(int count, int block) => (int)(((long)count + block - 1) / block);

    /// <summary><paramref name="value"/> wrapped to <paramref name="type"/>'s width and widened again.</summary>
    public static long Wrap(long value, GpuElementType type) => type switch
    {
        GpuElementType.Int8 => unchecked((sbyte)value),
        GpuElementType.UInt8 => unchecked((byte)value),
        GpuElementType.Int16 => unchecked((short)value),
        GpuElementType.Int32 => unchecked((int)value),
        GpuElementType.Int64 => value,
        _ => throw new System.ArgumentOutOfRangeException(nameof(type), type, "Not an integer element type."),
    };
}
