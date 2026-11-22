// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Reductions computed in double precision on the device, over <see cref="GpuElementType.Float64"/> buffers. A backend
/// implements this only where the device computes in double; <see cref="SupportsFloat64Reductions"/> says whether this
/// device does.
/// </summary>
public interface IGpuFloat64Reductions
{
    /// <summary>Whether the double-precision reduction kernels are available on this device.</summary>
    bool SupportsFloat64Reductions { get; }

    /// <summary>The sum of the first <paramref name="size"/> elements of <paramref name="a"/>.</summary>
    double SumFloat64(IGpuBuffer a, int size);

    /// <summary>The dot product of the first <paramref name="size"/> elements of <paramref name="a"/> and <paramref name="b"/>.</summary>
    double DotFloat64(IGpuBuffer a, IGpuBuffer b, int size);
}
