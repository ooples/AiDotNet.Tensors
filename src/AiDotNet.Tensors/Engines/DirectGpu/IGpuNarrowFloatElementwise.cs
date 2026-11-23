// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Element-wise operations on two 16-bit float buffers.</summary>
public enum GpuNarrowFloatBinaryOperation
{
    /// <summary><c>a + b</c>.</summary>
    Add,

    /// <summary><c>a - b</c>.</summary>
    Subtract,

    /// <summary><c>a * b</c>.</summary>
    Multiply,
}

/// <summary>Element-wise operations on one 16-bit float buffer.</summary>
public enum GpuNarrowFloatUnaryOperation
{
    /// <summary><c>-a</c>: the sign bit flipped.</summary>
    Negate,

    /// <summary><c>max(a, 0)</c>.</summary>
    Relu,

    /// <summary>GELU in its tanh form, <c>0.5 a (1 + tanh(sqrt(2/pi) (a + 0.044715 a^3)))</c>.</summary>
    Gelu,
}

/// <summary>
/// Element-wise kernels over 16-bit float buffers (<see cref="GpuElementType.Float16"/> and
/// <see cref="GpuElementType.BFloat16"/>). Each element is widened to a 32-bit float, the operation is done there, and
/// the result is rounded to the storage type to nearest, ties to even.
/// </summary>
/// <remarks>
/// The buffers hold <paramref name="type"/> elements; the output may be one of the operands. Which (operation, type)
/// pairs a device has is asked per pair: a backend whose buffers are 4-byte words has the half kernels it always had
/// and no BFloat16 ones.
/// </remarks>
public interface IGpuNarrowFloatElementwise
{
    /// <summary>Whether <see cref="BinaryNarrowFloat"/> runs <paramref name="operation"/> on <paramref name="type"/> buffers.</summary>
    bool SupportsNarrowFloatBinary(GpuNarrowFloatBinaryOperation operation, GpuElementType type);

    /// <summary>Whether <see cref="UnaryNarrowFloat"/> runs <paramref name="operation"/> on <paramref name="type"/> buffers.</summary>
    bool SupportsNarrowFloatUnary(GpuNarrowFloatUnaryOperation operation, GpuElementType type);

    /// <summary><c>output[i] = a[i] (operation) b[i]</c> for the first <paramref name="size"/> elements.</summary>
    void BinaryNarrowFloat(GpuNarrowFloatBinaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);

    /// <summary><c>output[i] = (operation) a[i]</c> for the first <paramref name="size"/> elements.</summary>
    void UnaryNarrowFloat(GpuNarrowFloatUnaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer output, int size);
}

/// <summary>
/// Matrix multiply over 16-bit float buffers: products are accumulated in 32-bit floats and the result is rounded to
/// the storage type.
/// </summary>
public interface IGpuNarrowFloatMatrixMultiply
{
    /// <summary>Whether <see cref="MatrixMultiplyNarrowFloat"/> runs on <paramref name="type"/> buffers.</summary>
    bool SupportsNarrowFloatMatrixMultiply(GpuElementType type);

    /// <summary>
    /// <c>c = a b</c> for row-major <paramref name="a"/> (<paramref name="m"/> by <paramref name="k"/>),
    /// <paramref name="b"/> (<paramref name="k"/> by <paramref name="n"/>) and <paramref name="c"/>
    /// (<paramref name="m"/> by <paramref name="n"/>), all of <paramref name="type"/>. The output is not an operand.
    /// </summary>
    void MatrixMultiplyNarrowFloat(GpuElementType type, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k);
}

/// <summary>What the 16-bit float kernels of every backend share.</summary>
internal static class GpuNarrowFloatLayout
{
    /// <summary>Whether <paramref name="type"/> is a 16-bit float type.</summary>
    public static bool IsNarrowFloat(GpuElementType type) => type is GpuElementType.Float16 or GpuElementType.BFloat16;

    /// <summary>
    /// Validates one launch: <paramref name="type"/> is a 16-bit float type, <paramref name="count"/> is not negative,
    /// and no buffer is typed as the other 16-bit float type or holds fewer than its share of elements. (A raw byte
    /// buffer carries no element type; the device bounds it by its byte capacity.)
    /// </summary>
    public static void Require(string operation, GpuElementType type, int count, params (IGpuBuffer Buffer, int Elements)[] buffers)
    {
        if (!IsNarrowFloat(type))
            throw new System.ArgumentException($"{operation} computes on 16-bit float buffers, got {type}.", nameof(type));
        if (count < 0) throw new System.ArgumentOutOfRangeException(nameof(count), $"{operation}: a negative element count ({count}).");
        foreach (var (buffer, elements) in buffers)
        {
            if (buffer is null) throw new System.ArgumentNullException(nameof(buffers));
            if (!IsNarrowFloat(buffer.ElementType)) continue;
            if (buffer.ElementType != type)
                throw new System.ArgumentException($"{operation} on {type} was given a {buffer.ElementType} buffer.", nameof(buffers));
            if (elements > buffer.Size)
                throw new System.ArgumentOutOfRangeException(nameof(count), $"{operation}: {elements} elements are outside a buffer of {buffer.Size}.");
        }
    }

    /// <summary>Whether <paramref name="backend"/> multiplies <see cref="GpuElementType.Float16"/> matrices.</summary>
    public static bool HasHalfMatrixMultiply(object? backend)
        => backend is IGpuNarrowFloatMatrixMultiply multiply && multiply.SupportsNarrowFloatMatrixMultiply(GpuElementType.Float16);

    /// <summary>
    /// A matrix multiply reads its operands while it writes: the output may not be one of them, as the same object or
    /// as another buffer over the same device memory.
    /// </summary>
    public static void RequireDistinctOutput(IGpuBuffer output, params IGpuBuffer[] operands)
    {
        if (output is null) throw new System.ArgumentNullException(nameof(output));
        foreach (var operand in operands)
        {
            if (operand is null) throw new System.ArgumentNullException(nameof(operands));
            if (ReferenceEquals(operand, output) || (operand.Handle != System.IntPtr.Zero && operand.Handle == output.Handle))
                throw new System.ArgumentException("The output of a matrix multiply must not be one of its operands.", nameof(output));
        }
    }

    /// <summary>The <paramref name="m"/> by <paramref name="n"/> by <paramref name="k"/> element counts of a matrix multiply, checked.</summary>
    public static (int A, int B, int C) MatrixCounts(int m, int n, int k)
    {
        if (m <= 0 || n <= 0 || k <= 0)
            throw new System.ArgumentOutOfRangeException(nameof(m), $"Matrix dimensions must be positive, got {m} x {k} times {k} x {n}.");
        return (checked(m * k), checked(k * n), checked(m * n));
    }
}
