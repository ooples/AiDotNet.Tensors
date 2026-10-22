// Copyright (c) AiDotNet. All rights reserved.
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IDeviceScalarOps
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

    /// <remarks>multi_tensor_sum_squares writes the double on the device, so the compiled plan's finiteness check needs
    /// no host read (#1519); host-staged when that kernel did not build.</remarks>
    bool IDeviceScalarOps.SupportsMultiTensorSumOfSquares => IsAvailable && _kernelCache.ContainsKey("multi_tensor_sum_squares");

    void IDeviceScalarOps.MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
    {
        if (_kernelCache.ContainsKey("multi_tensor_sum_squares")) MultiTensorSumOfSquares(tensors, sizes, sumOfSquares);
        else PortableDeviceScalarOps.MultiTensorSumOfSquares(this, tensors, sizes, sumOfSquares);
    }
}
