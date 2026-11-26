// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IUniquePositionAddBackend
{
    // One thread per value, an atomic add into its own cell. The positions are distinct, so each cell takes at most one
    // contribution and the result is the same in any order: the deterministic ScatterAdd routes (a stable sort sized to
    // the destination, or one thread per destination cell) are not needed. On the chunked fused cross-entropy the sort
    // kept a start slot per logit of a chunk, 128 MiB at 672 rows x 49152 classes, for the life of the backend.
    unsafe void IUniquePositionAddBackend.AddAtUniquePositions(IGpuBuffer values, IGpuBuffer positionBits,
        IGpuBuffer destination, int count, int destinationSize)
    {
        if (values is null) throw new System.ArgumentNullException(nameof(values));
        if (positionBits is null) throw new System.ArgumentNullException(nameof(positionBits));
        if (destination is null) throw new System.ArgumentNullException(nameof(destination));
        if (count <= 0) return;
        if (values.Size < count || positionBits.Size < count || destination.Size < destinationSize)
            throw new System.ArgumentOutOfRangeException(nameof(count));
        if (!_kernelCache.TryGetValue("embedding_backward", out var kernel))
            throw new GpuCapabilityUnavailableException("CUDA kernel not found: embedding_backward");
        using var _ = PushContext();
        System.IntPtr src = values.Handle, idx = positionBits.Handle, dst = destination.Handle;
        int n = count, one = 1, size = destinationSize;
        void** args = stackalloc void*[6];
        args[0] = &src;
        args[1] = &idx;
        args[2] = &dst;
        args[3] = &n;
        args[4] = &one;
        args[5] = &size;
        LaunchKernel(kernel, (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    /// <summary>Bytes the stable-sort scatter scratch holds now (tests).</summary>
    internal long SortedScatterScratchBytes
    {
        get
        {
            long bytes = 0;
            foreach (var buffer in new[] { _sortKeysA, _sortValuesA, _sortKeysB, _sortValuesB, _sortTileCounts, _sortStarts })
                if (buffer is not null) bytes += buffer.SizeInBytes;
            return bytes;
        }
    }
}
