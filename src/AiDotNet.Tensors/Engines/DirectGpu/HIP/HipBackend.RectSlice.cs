using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// One-launch rectangular N-d slice, the HIP port of CudaBackend.RectSlice (rect_slice_nd in the neural-net module).
public sealed partial class HipBackend : IRectSliceKernels
{
    // Mirrors `struct RectSliceMeta` in the kernel source; passed by value as a kernel parameter.
    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct RectSliceMeta
    {
        public int Rank;
        public int Total;
        public fixed int OutDims[RectSliceLimits.MaxRank];
        public fixed int FullStrides[RectSliceLimits.MaxRank];
        public fixed int Starts[RectSliceLimits.MaxRank];
    }

    public unsafe void RectSlice(IGpuBuffer full, IGpuBuffer slice, int[] fullShape, int[] start, int[] length, bool scatter)
    {
        var meta = new RectSliceMeta();
        RectSliceGeometry.Build(full, slice, fullShape, start, length, meta.OutDims, meta.FullStrides, meta.Starts,
            out meta.Rank, out meta.Total);
        if (!_kernelCache.TryGetValue("rect_slice_nd", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: rect_slice_nd");

        uint gridDim = (uint)((meta.Total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr f = full.Handle, s = slice.Handle;
        int mode = scatter ? 1 : 0;
        void** args = stackalloc void*[4];
        args[0] = &f; args[1] = &s; args[2] = &meta; args[3] = &mode;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
    }
}