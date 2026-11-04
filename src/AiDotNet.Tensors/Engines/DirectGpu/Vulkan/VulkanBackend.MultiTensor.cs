using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// The global-norm clip as chunk reductions per tensor (no device address tables in core Vulkan compute), the Vulkan
// port of CudaBackend's IMultiTensorKernels: the deterministic record (Gpu.DeterministicGradientNorm) in 32-bit integer
// GLSL (VulkanDeterministicNormKernels). The finite guard's own reduction (MultiTensorSumOfSquaresOnDevice) still
// serves IDeviceScalarOps and the guarded optimizer updates.
public sealed partial class VulkanBackend : IMultiTensorKernels, IFusedNormClipKernels
{
    // The reduction here is one dispatch per tensor (no buffer table), so the record already takes N + 1 launches; the
    // coefficient follows as one more. CUDA and HIP fold all of it into one launch (dgn_norm_clip).
    void IFusedNormClipKernels.MultiTensorNormAndClipScale(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes,
        IGpuBuffer record, float maxNorm, IGpuBuffer? scale, Gpu.NonFiniteGradientPolicy policy)
    {
        IMultiTensorKernels multi = this;
        multi.MultiTensorSumOfSquares(tensors, sizes, record);
        if (scale is not null) multi.ClipScaleFromRecord(record, maxNorm, scale, policy);
    }

    private IGpuBuffer? _normPartials;

    public void MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
    {
        lock (_computeLock)
        {
            EnsureInitialized();
            MultiTensorArgs.ValidateRecordBuffer(sumOfSquares);
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
                int chunks = Gpu.DeterministicGradientNorm.ChunkCount(sizes[t]);
                GlslUnaryOp(VulkanDeterministicNormKernels.ChunkPartials, tensors[t], partials,
                    chunks * DeterministicNormKernels.GroupSize, new[] { (uint)sizes[t], (uint)t, first }, 3 * sizeof(uint));
                first += (uint)chunks;
            }
            GlslUnaryOp(VulkanDeterministicNormKernels.Finish, partials, sumOfSquares, DeterministicNormKernels.GroupSize,
                new[] { (uint)total, (uint)Gpu.DeterministicGradientNorm.Headroom(total) }, 2 * sizeof(uint));
        }
    }

    public void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale)
        => ((IMultiTensorKernels)this).ClipScaleFromRecord(sumOfSquares, maxNorm, scale, Gpu.NonFiniteGradientPolicy.LeaveUnscaled);

    void IMultiTensorKernels.ClipScaleFromRecord(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale, Gpu.NonFiniteGradientPolicy policy)
    {
        EnsureInitialized();
        MultiTensorArgs.ValidateRecordBuffer(sumOfSquares);
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        GlslUnaryOp(VulkanDeterministicNormKernels.ClipScale, sumOfSquares, scale, 1,
            new[] { Gpu.DeterministicGradientNorm.SingleToUInt32Bits(maxNorm), (uint)policy }, 2 * sizeof(uint));
    }

    public void MultiTensorScaleByDeviceScalar(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale)
        => ((IMultiTensorKernels)this).MultiTensorClipScale(tensors, sizes, scale, false);

    void IMultiTensorKernels.MultiTensorClipScale(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale, bool zeroNonFinite)
    {
        EnsureInitialized();
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        MultiTensorArgs.Validate(tensors, sizes);
        // dgn_mul: the IEEE product with subnormals kept, so a flush-to-zero device still matches the others.
        for (int t = 0; t < tensors.Count; t++)
            GlslUnaryOp(VulkanDeterministicNormKernels.Scale, scale, tensors[t], Math.Min(sizes[t], MultiTensorArgs.MaxStridedScaleThreads), new[] { (uint)sizes[t], zeroNonFinite ? 1u : 0u }, 2 * sizeof(uint));
    }

    /// <inheritdoc />
    double IMultiTensorKernels.ReadSumOfSquares(IGpuBuffer sumOfSquares)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        return MultiTensorArgs.DoubleFromSlots(DownloadBuffer(sumOfSquares));
    }
}