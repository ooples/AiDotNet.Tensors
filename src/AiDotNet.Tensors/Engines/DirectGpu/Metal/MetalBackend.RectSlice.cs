using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// One-launch rectangular N-d slice, the Metal port of CudaBackend.RectSlice (rect_slice_nd in the resident library).
public sealed partial class MetalBackend : IRectSliceKernels
{
    public void RectSlice(IGpuBuffer full, IGpuBuffer slice, int[] fullShape, int[] start, int[] length, bool scatter)
    {
        ThrowIfDisposed();
        var outDims = new int[RectSliceLimits.MaxRank];
        var fullStrides = new int[RectSliceLimits.MaxRank];
        var starts = new int[RectSliceLimits.MaxRank];
        RectSliceGeometry.Build(full, slice, fullShape, start, length, outDims, fullStrides, starts,
            out int rank, out int total);
        var meta = new int[3 * RectSliceLimits.MaxRank];
        Array.Copy(outDims, 0, meta, 0, RectSliceLimits.MaxRank);
        Array.Copy(fullStrides, 0, meta, RectSliceLimits.MaxRank, RectSliceLimits.MaxRank);
        Array.Copy(starts, 0, meta, 2 * RectSliceLimits.MaxRank, RectSliceLimits.MaxRank);
        using var metaBuffer = AllocateIntBuffer(meta);
        DispatchResidentMetal("rect_slice_nd", total, new[] { full, slice, metaBuffer },
            (uint)rank, (uint)total, scatter ? 1u : 0u);
    }
}