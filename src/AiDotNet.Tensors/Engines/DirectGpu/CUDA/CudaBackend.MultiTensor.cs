using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The clip's coefficient and scale kernels (dgn_clip_scale, dgn_multi_tensor_scale) are launched from CudaBackend.cs;
// the deterministic sum of squares is here. The public MultiTensorSumOfSquares in CudaBackend.cs is the finiteness
// reduction (IDeviceScalarOps, captured step guards), which only needs the double's finiteness.
public sealed partial class CudaBackend : IMultiTensorKernels, IFusedNormClipKernels
{
    private IGpuBuffer? _normPartials;

    /// <summary>The deterministic norm record (<see cref="Gpu.DeterministicGradientNorm"/>): one launch writes every
    /// chunk's partial, a second combines them, both over the gradient table.</summary>
    unsafe void IMultiTensorKernels.MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer record)
    {
        MultiTensorArgs.ValidateRecordBuffer(record);
        if (tensors is null) throw new ArgumentNullException(nameof(tensors));
        if (!_kernelCache.TryGetValue("dgn_chunk_partials", out var chunkKernel) || !_kernelCache.TryGetValue("dgn_finish", out var finish))
            throw new GpuCapabilityUnavailableException("CUDA kernel not found: dgn_chunk_partials / dgn_finish");
        lock (_gradientTableSync)
        {
            int total = 0;
            IntPtr recH = record.Handle, partH;
            using var _ = PushContext();
            if (tensors.Count > 0)
            {
                var table = GetGradientTable(tensors, sizes);
                total = table.TotalChunks;
                var partials = EnsureNormPartials(total);
                partH = partials.Handle;
                IntPtr pH = table.Pointers!.Handle, sH = table.SizesBuffer!.Handle, ctH = table.ChunkTensor!.Handle, csH = table.ChunkStart!.Handle;
                void** a = stackalloc void*[6];
                a[0] = &pH; a[1] = &sH; a[2] = &ctH; a[3] = &csH; a[4] = &total; a[5] = &partH;
                LaunchKernel(chunkKernel, (uint)total, DeterministicNormKernels.GroupSize, a);
            }
            else partH = EnsureNormPartials(1).Handle;
            int headroom = Gpu.DeterministicGradientNorm.Headroom(total);
            void** f = stackalloc void*[4];
            f[0] = &partH; f[1] = &total; f[2] = &headroom; f[3] = &recH;
            LaunchKernel(finish, 1, DeterministicNormKernels.GroupSize, f);
        }
    }

    private IGpuBuffer? _normTicket;

    unsafe void IFusedNormClipKernels.MultiTensorNormAndClipScale(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes,
        IGpuBuffer record, float maxNorm, IGpuBuffer? scale, Gpu.NonFiniteGradientPolicy policy)
    {
        MultiTensorArgs.ValidateRecordBuffer(record);
        if (tensors is null) throw new ArgumentNullException(nameof(tensors));
        IMultiTensorKernels multi = this;
        if (tensors.Count == 0)
        {
            multi.MultiTensorSumOfSquares(tensors, sizes, record);
            if (scale is not null) multi.ClipScaleFromRecord(record, maxNorm, scale, policy);
            return;
        }
        if (!_kernelCache.TryGetValue("dgn_norm_clip", out var kernel))
            throw new GpuCapabilityUnavailableException("CUDA kernel not found: dgn_norm_clip");
        lock (_gradientTableSync)
        {
            using var _ = PushContext();
            var table = GetGradientTable(tensors, sizes);
            int total = table.TotalChunks;
            var partials = EnsureNormPartials(total);
            // The ticket starts at zero and the launch's last block resets it.
            var ticket = _normTicket ??= AllocateBuffer(new float[1]);
            IntPtr pH = table.Pointers!.Handle, sH = table.SizesBuffer!.Handle, ctH = table.ChunkTensor!.Handle, csH = table.ChunkStart!.Handle;
            IntPtr partH = partials.Handle, tH = ticket.Handle, recH = record.Handle, scH = scale?.Handle ?? record.Handle;
            int headroom = Gpu.DeterministicGradientNorm.Headroom(total);
            uint bits = Gpu.DeterministicGradientNorm.SingleToUInt32Bits(maxNorm);
            uint pol = scale is null ? DeterministicNormKernels.RecordOnly : (uint)policy;
            void** a = stackalloc void*[12];
            a[0] = &pH; a[1] = &sH; a[2] = &ctH; a[3] = &csH; a[4] = &total; a[5] = &partH; a[6] = &tH; a[7] = &headroom;
            a[8] = &recH; a[9] = &bits; a[10] = &pol; a[11] = &scH;
            LaunchKernel(kernel, (uint)Math.Min(total, FusedNormMaxBlocks), DeterministicNormKernels.GroupSize, a);
        }
    }

    // Blocks of the fused norm launch (each grid-strides over chunks): enough to fill the device, few enough that the
    // last block's combine does not wait on a long tail of ticket increments.
    private const int FusedNormMaxBlocks = 4096;

    // Grow-only scratch for the chunk partials; launches on this stream are ordered, so the next call may reuse it.
    private IGpuBuffer EnsureNormPartials(int chunks)
    {
        int words = Math.Max(1, chunks) * Gpu.DeterministicGradientNorm.PartialWords;
        if (_normPartials is { } p && p.Size >= words) return p;
        var grown = AllocateBuffer(Math.Max(words, 64 * Gpu.DeterministicGradientNorm.PartialWords));
        _normPartials?.Dispose();
        return _normPartials = grown;
    }

    /// <inheritdoc />
    double IMultiTensorKernels.ReadSumOfSquares(IGpuBuffer sumOfSquares)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        return MultiTensorArgs.DoubleFromSlots(DownloadBuffer(sumOfSquares));
    }
}
