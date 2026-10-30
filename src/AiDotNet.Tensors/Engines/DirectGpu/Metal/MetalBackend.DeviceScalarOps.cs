// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IDeviceScalarOps
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

    /// <remarks>Reduced on the device by the resident kernels, so the compiled plan's finiteness check needs no host
    /// read (#1520); host-staged when that library did not build.</remarks>
    bool IDeviceScalarOps.SupportsMultiTensorSumOfSquares => IsAvailable && _residentLibrary != IntPtr.Zero;

    void IDeviceScalarOps.MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
    {
        if (_residentLibrary != IntPtr.Zero) MultiTensorSumOfSquares(tensors, sizes, sumOfSquares);
        else PortableDeviceScalarOps.MultiTensorSumOfSquares(this, tensors, sizes, sumOfSquares);
    }
}
