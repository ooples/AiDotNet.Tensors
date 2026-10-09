using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// The global-norm clip as one threadgroup reduction per tensor, the Metal port of CudaBackend's IMultiTensorKernels.
// acc[0] holds the float sum of squares.
public sealed partial class MetalBackend : IMultiTensorKernels
{
    public void MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
    {
        ThrowIfDisposed();
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        MultiTensorArgs.Validate(tensors, sizes);
        Fill(sumOfSquares, 0f, 2);
        for (int t = 0; t < tensors.Count; t++)
            DispatchResidentMetal("tensor_sum_squares_accumulate", MultiTensorArgs.ReductionGroupSize,
                new[] { tensors[t], sumOfSquares }, (uint)sizes[t]);
    }

    public void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale)
    {
        ThrowIfDisposed();
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        DispatchResidentMetal("clip_scale_from_sum_squares", 1, new[] { sumOfSquares, scale },
            unchecked((uint)SingleToInt32BitsCompat(maxNorm)));
    }

    public void MultiTensorScaleByDeviceScalar(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale)
    {
        ThrowIfDisposed();
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        MultiTensorArgs.Validate(tensors, sizes);
        for (int t = 0; t < tensors.Count; t++)
            DispatchResidentMetal("scale_by_device_scalar_inplace", sizes[t], new[] { tensors[t], scale }, (uint)sizes[t]);
    }
}