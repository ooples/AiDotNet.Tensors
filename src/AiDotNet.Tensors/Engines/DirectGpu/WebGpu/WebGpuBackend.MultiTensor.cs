#if NET7_0_OR_GREATER
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The global-norm clip as chunk reductions per tensor (WGSL cannot address a buffer from a table), the WebGPU port of
// CudaBackend's IMultiTensorKernels: the deterministic record (Gpu.DeterministicGradientNorm) in u32-only WGSL
// (WebGpuDeterministicNormKernels). The public MultiTensorSumOfSquares is the finiteness reduction (the double the
// guard and IDeviceScalarOps read, WebGpuFiniteGuardKernels).
public sealed partial class WebGpuBackend : IMultiTensorKernels
{
    private IGpuBuffer? _normPartials;

    // The finite guard's uniform (WebGpuBackend.FiniteGuard): n and the norm bound.
    private static float[] MultiTensorParams(int n, float maxNorm) =>
        new[] { BitConverter.Int32BitsToSingle(n), maxNorm, 0f, 0f };

    private static float[] NormParams(uint p0, uint p1 = 0, uint p2 = 0) =>
        new[] { BitConverter.Int32BitsToSingle(unchecked((int)p0)), BitConverter.Int32BitsToSingle(unchecked((int)p1)),
                BitConverter.Int32BitsToSingle(unchecked((int)p2)), 0f };

    private void NormDispatch(string kernel, IGpuBuffer a, IGpuBuffer b, float[] parameters, int workSize) =>
        Dispatch2BufferAsync(WebGpuDeterministicNormKernels.ModuleName, WebGpuDeterministicNormKernels.Source, kernel,
            a, b, parameters, workSize).GetAwaiter().GetResult();

    void IMultiTensorKernels.MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer record)
    {
        MultiTensorArgs.ValidateRecordBuffer(record);
        MultiTensorArgs.Validate(tensors, sizes);
        int total = 0;
        for (int t = 0; t < sizes.Count; t++) total += Gpu.DeterministicGradientNorm.ChunkCount(sizes[t]);
        int words = Math.Max(1, total) * Gpu.DeterministicGradientNorm.PartialWords;
        if (_normPartials is not { } partials || partials.Size < words)
        {
            _normPartials?.Dispose();
            partials = _normPartials = AllocateBuffer(Math.Max(words, 64 * Gpu.DeterministicGradientNorm.PartialWords));
        }
        uint first = 0;
        for (int t = 0; t < tensors.Count; t++)
        {
            int chunks = Gpu.DeterministicGradientNorm.ChunkCount(sizes[t]);
            NormDispatch("dgn_chunk_partials", tensors[t], partials, NormParams((uint)sizes[t], (uint)t, first),
                chunks * DeterministicNormKernels.GroupSize);
            first += (uint)chunks;
        }
        NormDispatch("dgn_finish", partials, record, NormParams((uint)total, (uint)Gpu.DeterministicGradientNorm.Headroom(total)),
            DeterministicNormKernels.GroupSize);
    }

    public void MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        MultiTensorArgs.Validate(tensors, sizes);
        Fill(sumOfSquares, 0f, 2);
        // The double the guard and IDeviceScalarOps read (#1522): a float sum and non-finite mark, then finished.
        for (int t = 0; t < tensors.Count; t++)
            GuardDispatch("guard_sum_squares_accumulate", tensors[t], sumOfSquares, sizes[t], 0f,
                MultiTensorArgs.ReductionGroupSize);
        FinishSumOfSquaresDouble(sumOfSquares);
    }

    public void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale)
        => ((IMultiTensorKernels)this).ClipScaleFromRecord(sumOfSquares, maxNorm, scale, Gpu.NonFiniteGradientPolicy.LeaveUnscaled);

    void IMultiTensorKernels.ClipScaleFromRecord(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale, Gpu.NonFiniteGradientPolicy policy)
    {
        MultiTensorArgs.ValidateRecordBuffer(sumOfSquares);
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        NormDispatch("dgn_clip_scale", sumOfSquares, scale,
            NormParams(Gpu.DeterministicGradientNorm.SingleToUInt32Bits(maxNorm), (uint)policy), 1);
    }

    public void MultiTensorScaleByDeviceScalar(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale)
        => ((IMultiTensorKernels)this).MultiTensorClipScale(tensors, sizes, scale, false);

    void IMultiTensorKernels.MultiTensorClipScale(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale, bool zeroNonFinite)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        MultiTensorArgs.Validate(tensors, sizes);
        // dgn_mul: the IEEE product with subnormals kept, so a flush-to-zero device still matches the others.
        for (int t = 0; t < tensors.Count; t++)
            NormDispatch("dgn_scale", scale, tensors[t], NormParams((uint)sizes[t], zeroNonFinite ? 1u : 0u), sizes[t]);
    }

    /// <inheritdoc />
    double IMultiTensorKernels.ReadSumOfSquares(IGpuBuffer sumOfSquares)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        return MultiTensorArgs.DoubleFromSlots(DownloadBuffer(sumOfSquares));
    }
}
#endif