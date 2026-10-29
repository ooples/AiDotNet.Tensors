// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Scaling by a scalar that stays on the device, and the fused multi-tensor sum of squares the optimizer's
/// global-norm and finiteness checks read (#1154).
/// </summary>
internal interface IDeviceScalarOps
{
    /// <summary>buffer[i] *= scalar[0] for the first <paramref name="size"/> elements; the scalar never leaves the device.</summary>
    void ScaleByDeviceScalar(IGpuBuffer buffer, IGpuBuffer scalar, int size);

    /// <summary>True when <see cref="MultiTensorSumOfSquares"/> runs as one fused launch on this backend.</summary>
    bool SupportsMultiTensorSumOfSquares { get; }

    /// <summary>The fused sum of squares over many tensors; only after <see cref="SupportsMultiTensorSumOfSquares"/>.</summary>
    void MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares);
}