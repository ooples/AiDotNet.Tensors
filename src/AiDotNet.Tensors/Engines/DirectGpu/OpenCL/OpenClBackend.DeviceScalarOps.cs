// Copyright (c) AiDotNet. All rights reserved.
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IDeviceScalarOps
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

    /// <remarks>No fused multi-tensor launch here; the reduction below is host-staged.</remarks>
    bool IDeviceScalarOps.SupportsMultiTensorSumOfSquares => false;

    void IDeviceScalarOps.MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
        => PortableDeviceScalarOps.MultiTensorSumOfSquares(this, tensors, sizes, sumOfSquares);
}
