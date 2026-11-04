using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// The global-norm clip as chunk reductions per tensor, the Metal port of CudaBackend's IMultiTensorKernels: the
// deterministic record (Gpu.DeterministicGradientNorm). The public MultiTensorSumOfSquares is the finiteness reduction
// (a float sum and a non-finite mark finished as a double) that IDeviceScalarOps and the guarded updates use (#1520).
public sealed partial class MetalBackend : IMultiTensorKernels
{
    private IGpuBuffer? _normPartials;

    // Exactly `groups` threadgroups of `groupSize` threads (the norm reductions need whole 256-thread groups).
    private void DispatchNormGroups(string kernelName, uint groups, uint groupSize, IGpuBuffer[] buffers, params uint[] values)
    {
        if (groups == 0) return;
        if (_normLibrary == IntPtr.Zero)
            throw new GpuCapabilityUnavailableException("Metal deterministic norm kernels are unavailable.");
        var pipeline = GetPipeline("DeterministicNorm", _normLibrary, kernelName);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; ++i)
        {
            if (buffers[i] is not MetalGpuBuffer buffer)
                throw new ArgumentException("Buffers must be MetalGpuBuffer.", nameof(buffers));
            encoder.SetBuffer(buffer, i);
        }
        for (int i = 0; i < values.Length; ++i)
            encoder.SetBytes(values[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(new MetalNativeBindings.MTLSize(groups, 1, 1), new MetalNativeBindings.MTLSize(groupSize, 1, 1));
    }

    // The deterministic norm record (Gpu.DeterministicGradientNorm): per-tensor chunk reductions, then one combine.
    void IMultiTensorKernels.MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer record)
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            MultiTensorArgs.ValidateRecordBuffer(record);
            MultiTensorArgs.Validate(tensors, sizes);
            int total = 0;
            for (int t = 0; t < sizes.Count; t++) total += Gpu.DeterministicGradientNorm.ChunkCount(sizes[t]);
            int words = Math.Max(1, total) * Gpu.DeterministicGradientNorm.PartialWords;
            if (_normPartials is not { } partials || partials.Size < words)
            {
                if (_normPartials is not null) Synchronize();
                var grown = AllocateBuffer(Math.Max(words, 64 * Gpu.DeterministicGradientNorm.PartialWords));
                _normPartials?.Dispose();
                partials = _normPartials = grown;
            }
            uint first = 0;
            for (int t = 0; t < tensors.Count; t++)
            {
                uint chunks = (uint)Gpu.DeterministicGradientNorm.ChunkCount(sizes[t]);
                DispatchNormGroups("dgn_chunk_partials_tensor", chunks, DeterministicNormKernels.GroupSize,
                    new[] { tensors[t], partials }, (uint)sizes[t], (uint)t, first);
                first += chunks;
            }
            DispatchNormGroups("dgn_finish", 1, DeterministicNormKernels.GroupSize, new[] { partials, record },
                (uint)total, (uint)Gpu.DeterministicGradientNorm.Headroom(total));
        }
    }

    // The finiteness reduction IDeviceScalarOps promises (a double in two slots).
    public void MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
    {
        ThrowIfDisposed();
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        MultiTensorArgs.Validate(tensors, sizes);
        Fill(sumOfSquares, 0f, 2);
        for (int t = 0; t < tensors.Count; t++)
            DispatchResidentMetal("tensor_sum_squares_accumulate", MultiTensorArgs.ReductionGroupSize,
                new[] { tensors[t], sumOfSquares }, (uint)sizes[t]);
        FinishSumOfSquaresDouble(sumOfSquares);
    }

    // Turns the float sum and non-finite mark in the buffer's two slots into the double they stand for.
    internal void FinishSumOfSquaresDouble(IGpuBuffer accumulator)
        => DispatchResidentMetal("sum_squares_finish_double", 1, new[] { accumulator });

    public void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale)
        => ((IMultiTensorKernels)this).ClipScaleFromRecord(sumOfSquares, maxNorm, scale, Gpu.NonFiniteGradientPolicy.LeaveUnscaled);

    void IMultiTensorKernels.ClipScaleFromRecord(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale,
        Gpu.NonFiniteGradientPolicy policy)
    {
        ThrowIfDisposed();
        MultiTensorArgs.ValidateRecordBuffer(sumOfSquares);
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        DispatchNormGroups("dgn_clip_scale", 1, 1, new[] { sumOfSquares, scale },
            Gpu.DeterministicGradientNorm.SingleToUInt32Bits(maxNorm), (uint)policy);
    }

    public void MultiTensorScaleByDeviceScalar(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale)
        => ((IMultiTensorKernels)this).MultiTensorClipScale(tensors, sizes, scale, false);

    void IMultiTensorKernels.MultiTensorClipScale(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale, bool zeroNonFinite)
    {
        ThrowIfDisposed();
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        MultiTensorArgs.Validate(tensors, sizes);
        // dgn_mul: the IEEE product with subnormals kept (Metal flushes them in float arithmetic).
        for (int t = 0; t < tensors.Count; t++)
            DispatchNormGroups("dgn_scale_tensor", (uint)((sizes[t] + 255) / 256), 256, new[] { tensors[t], scale }, (uint)sizes[t], zeroNonFinite ? 1u : 0u);
    }

    /// <inheritdoc />
    double IMultiTensorKernels.ReadSumOfSquares(IGpuBuffer sumOfSquares)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        return MultiTensorArgs.DoubleFromSlots(DownloadBuffer(sumOfSquares));
    }
}
