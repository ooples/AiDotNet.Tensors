// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>A reduction of a double buffer to one value, computed a block a pass.</summary>
internal enum GpuFloat64Reduction
{
    /// <summary>The sum.</summary>
    Sum,
    /// <summary>The smallest element.</summary>
    Min,
    /// <summary>The largest element.</summary>
    Max,
}

/// <summary>
/// Reductions computed in double precision on the device, over <see cref="GpuElementType.Float64"/> buffers. A backend
/// implements this only where the device computes in double; <see cref="IGpuFloat64Reductions.SupportsFloat64Reductions"/>
/// says whether this device does.
/// </summary>
public interface IGpuFloat64Reductions
{
    /// <summary>Whether the double-precision reduction kernels are available on this device.</summary>
    bool SupportsFloat64Reductions { get; }

    /// <summary>The sum of the first <paramref name="size"/> elements of <paramref name="a"/>.</summary>
    double SumFloat64(IGpuBuffer a, int size);

    /// <summary>
    /// The largest of the first <paramref name="size"/> elements of <paramref name="a"/>; NaN when any of them is NaN
    /// (as <see cref="System.Math.Max(double, double)"/> gives). Zeros of either sign compare equal, so the sign of a
    /// zero result is not specified. An empty range has none and is refused.
    /// </summary>
    double MaxFloat64(IGpuBuffer a, int size);

    /// <summary>
    /// The smallest of the first <paramref name="size"/> elements of <paramref name="a"/>; NaN when any of them is
    /// NaN. The sign of a zero result is not specified. An empty range has none and is refused.
    /// </summary>
    double MinFloat64(IGpuBuffer a, int size);

    /// <summary>The dot product of the first <paramref name="size"/> elements of <paramref name="a"/> and <paramref name="b"/>.</summary>
    double DotFloat64(IGpuBuffer a, IGpuBuffer b, int size);
}
