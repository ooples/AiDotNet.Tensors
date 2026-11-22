// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Matrix multiplication computed in double precision on the device, over <see cref="GpuElementType.Float64"/> buffers.
/// <see cref="SupportsFloat64MatrixMultiply"/> says whether this device has the routine.
/// </summary>
public interface IGpuFloat64MatrixMultiply
{
    /// <summary>Whether double-precision matrix multiplication is available on this device.</summary>
    bool SupportsFloat64MatrixMultiply { get; }

    /// <summary>
    /// <c>c = a · b</c> for row-major <paramref name="a"/> of <paramref name="m"/> x <paramref name="k"/> and
    /// <paramref name="b"/> of <paramref name="k"/> x <paramref name="n"/>; <paramref name="c"/> is
    /// <paramref name="m"/> x <paramref name="n"/> and must not be one of the operands.
    /// </summary>
    void MatrixMultiplyFloat64(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k);
}
