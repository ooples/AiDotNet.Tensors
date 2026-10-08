using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // One-launch rectangular N-d slice, the OpenCL port of CudaBackend.RectSlice.
    public sealed partial class OpenClBackend : IRectSliceKernels
    {
        public void RectSlice(IGpuBuffer full, IGpuBuffer slice, int[] fullShape, int[] start, int[] length, bool scatter)
        {
            var outDims = new int[RectSliceLimits.MaxRank];
            var fullStrides = new int[RectSliceLimits.MaxRank];
            var starts = new int[RectSliceLimits.MaxRank];
            RectSliceGeometry.Build(full, slice, fullShape, start, length, outDims, fullStrides, starts,
                out int rank, out int total);
            // The tables repeat every training step (same shapes, same windows), so they come from the
            // constant-table cache: a warm step uploads nothing for its slices.
            var outDimsBuffer = ConstantIntTable(outDims);
            var fullStridesBuffer = ConstantIntTable(fullStrides);
            var startsBuffer = ConstantIntTable(starts);
            var kernel = _kernelCache["rect_slice_nd"];
            kernel.SetArg(0, ((DirectOpenClGpuBuffer)full).Buffer.Handle);
            kernel.SetArg(1, ((DirectOpenClGpuBuffer)slice).Buffer.Handle);
            kernel.SetArg(2, ((DirectOpenClGpuBuffer)outDimsBuffer).Buffer.Handle);
            kernel.SetArg(3, ((DirectOpenClGpuBuffer)fullStridesBuffer).Buffer.Handle);
            kernel.SetArg(4, ((DirectOpenClGpuBuffer)startsBuffer).Buffer.Handle);
            kernel.SetArg(5, rank);
            kernel.SetArg(6, total);
            kernel.SetArg(7, scatter ? 1 : 0);
            kernel.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
        }
    }
}