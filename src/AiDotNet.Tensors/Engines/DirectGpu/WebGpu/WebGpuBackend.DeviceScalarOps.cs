#if NET7_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IDeviceScalarOps
{
    /// <inheritdoc/>
    /// <remarks>A [1, size] broadcast multiply against the one-element scalar, in place.</remarks>
    void IDeviceScalarOps.ScaleByDeviceScalar(IGpuBuffer buffer, IGpuBuffer scalar, int size)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (scalar is null) throw new ArgumentNullException(nameof(scalar));
        if (size <= 0) return;
        BroadcastMultiplyFirstAxis(buffer, scalar, buffer, 1, size);
    }

    /// <remarks>Reduced on the device (WebGpuFiniteGuardKernels), so the compiled plan's finiteness check needs no
    /// host read (#1522).</remarks>
    bool IDeviceScalarOps.SupportsMultiTensorSumOfSquares => IsAvailable;

    void IDeviceScalarOps.MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
        => MultiTensorSumOfSquares(tensors, sizes, sumOfSquares);
}
#endif
