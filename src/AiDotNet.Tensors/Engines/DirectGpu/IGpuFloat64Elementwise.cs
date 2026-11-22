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

/// <summary>
/// Element-wise arithmetic computed in double precision on the device, over <see cref="GpuElementType.Float64"/>
/// buffers. Each operation is a single correctly rounded IEEE operation per element, so the result equals the host's
/// bit for bit. <see cref="SupportsFloat64Elementwise"/> says whether this device computes in double.
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
}
