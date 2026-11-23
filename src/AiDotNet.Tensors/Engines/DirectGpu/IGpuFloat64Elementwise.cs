// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>An element-wise operation over two <see cref="GpuElementType.Float64"/> buffers.</summary>
public enum GpuFloat64BinaryOperation
{
    /// <summary><c>output[i] = a[i] + b[i]</c>.</summary>
    Add,
    /// <summary><c>output[i] = a[i] - b[i]</c>.</summary>
    Subtract,
    /// <summary><c>output[i] = a[i] * b[i]</c>.</summary>
    Multiply,
    /// <summary><c>output[i] = a[i] / b[i]</c>.</summary>
    Divide,
}

/// <summary>An element-wise operation over one <see cref="GpuElementType.Float64"/> buffer.</summary>
public enum GpuFloat64UnaryOperation
{
    /// <summary><c>-x</c>.</summary>
    Negate,
    /// <summary><c>|x|</c>.</summary>
    Abs,
    /// <summary>The square root of <c>x</c>.</summary>
    Sqrt,
    /// <summary><c>e^x</c>.</summary>
    Exp,
    /// <summary>The natural logarithm of <c>x</c>.</summary>
    Log,
    /// <summary>The hyperbolic tangent of <c>x</c>.</summary>
    Tanh,
    /// <summary><c>1 / (1 + e^-x)</c>.</summary>
    Sigmoid,
    /// <summary><c>max(x, 0)</c>.</summary>
    Relu,
}

/// <summary>
/// Element-wise arithmetic computed in double precision on the device, over <see cref="GpuElementType.Float64"/>
/// buffers. The arithmetic operations, negation, absolute value, square root and ReLU are single correctly rounded
/// IEEE operations per element, so their results equal the host's bit for bit, apart from the payload of a NaN result,
/// which a backend may canonicalize; the exponential, logarithm, hyperbolic
/// tangent and sigmoid are the device's double library functions, accurate to a few units in the last place.
/// <see cref="SupportsFloat64Elementwise"/> says whether this device computes in double.
/// </summary>
public interface IGpuFloat64Elementwise
{
    /// <summary>Whether the double-precision element-wise kernels are available on this device.</summary>
    bool SupportsFloat64Elementwise { get; }

    /// <summary>
    /// Applies <paramref name="operation"/> to the first <paramref name="size"/> elements of <paramref name="a"/> and
    /// <paramref name="b"/>, writing <paramref name="output"/>. <paramref name="output"/> may be one of the operands.
    /// </summary>
    void BinaryFloat64(GpuFloat64BinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);

    /// <summary>
    /// Applies <paramref name="operation"/> to the first <paramref name="size"/> elements of <paramref name="a"/>,
    /// writing <paramref name="output"/>, which may be <paramref name="a"/>.
    /// </summary>
    void UnaryFloat64(GpuFloat64UnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size);

    /// <summary>
    /// Applies <paramref name="operation"/> between each of the first <paramref name="size"/> elements of
    /// <paramref name="a"/> (the left operand) and <paramref name="scalar"/> (the right one), writing
    /// <paramref name="output"/>, which may be <paramref name="a"/>.
    /// </summary>
    void ScalarFloat64(GpuFloat64BinaryOperation operation, IGpuBuffer a, double scalar, IGpuBuffer output, int size);
}
