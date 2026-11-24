// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The in-place transforms run the reference kernels only, never a declared program, a direct-PTX path or a tuned slot
// (whose winner depends on the machine and the shape): each reference kernel reads an element (and its row's
// statistics) before it writes the same element, and declares no __restrict__ on the aliased pointers.
// FusedLinearCrossEntropy_index_chunked_in_place_row_transforms_match checks them bit for bit against the out-of-place
// calls at the chunk shapes the fused cross-entropy uses.
public sealed partial class CudaBackend : IInPlaceRowTransformsBackend
{
    void IInPlaceRowTransformsBackend.BiasAddInPlace(IGpuBuffer data, IGpuBuffer bias, int rows, int cols)
        => LaunchBiasAddReference(data, bias, data, rows, cols, relu: false);

    unsafe void IInPlaceRowTransformsBackend.ExpInPlace(IGpuBuffer data, int size)
    {
        if (size <= 0) return;
        if (!_kernelCache.TryGetValue("exp_vector", out var kernel))
            throw new GpuCapabilityUnavailableException("CUDA kernel not found: exp_vector");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr ptr = data.Handle;
        int n = size;
        void** args = stackalloc void*[3];
        args[0] = &ptr;
        args[1] = &ptr;
        args[2] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    void IInPlaceRowTransformsBackend.SoftmaxInPlace(IGpuBuffer data, int rows, int cols)
        => LaunchSoftmaxReference(data, data, rows, cols);

    void IInPlaceRowTransformsBackend.BroadcastMultiplyLastAxisInPlace(IGpuBuffer data, IGpuBuffer factors, int outerSize, int innerSize)
        => BroadcastMultiplyLastAxis(data, factors, data, outerSize, innerSize);

    /// <summary>Tests: the reference softmax kernel out of place (the route the in-place form runs).</summary>
    internal void SoftmaxReferenceForTest(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
        => LaunchSoftmaxReference(input, output, rows, cols);

    /// <summary>Tests: the reference bias-add kernel out of place (the route the in-place form runs).</summary>
    internal void BiasAddReferenceForTest(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer output, int rows, int cols)
        => LaunchBiasAddReference(input, bias, output, rows, cols, relu: false);
}
