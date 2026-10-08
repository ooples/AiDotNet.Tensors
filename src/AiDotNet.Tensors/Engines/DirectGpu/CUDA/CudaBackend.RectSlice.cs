using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// One-launch rectangular N-d slice (rect_slice_nd in the neural-net module). The per-row copy loop it replaces
// made slice-heavy training host-bound: 1.2 M cuMemcpyDtoDAsync for 10 CNN steps, ~1 s of API time per step.
public sealed partial class CudaBackend : IRectSliceKernels
{
    // Mirrors `struct RectSliceMeta` in the kernel source; passed BY VALUE as a kernel parameter, so the launch
    // needs no device metadata buffer and records cleanly into a captured graph.
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
        int rank = fullShape.Length;
        if (rank < 1 || rank > RectSliceLimits.MaxRank || start.Length != rank || length.Length != rank)
            throw new ArgumentException($"RectSlice supports rank 1..{RectSliceLimits.MaxRank} with matching start/length.");
        if (!_kernelCache.TryGetValue("rect_slice_nd", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rect_slice_nd");

        var meta = new RectSliceMeta { Rank = rank };
        int total = 1, stride = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            if (start[d] < 0 || length[d] < 1 || start[d] + length[d] > fullShape[d])
                throw new ArgumentOutOfRangeException(nameof(start), $"Axis {d}: [{start[d]}, +{length[d]}) is outside {fullShape[d]}.");
            meta.OutDims[d] = length[d];
            meta.FullStrides[d] = stride;
            meta.Starts[d] = start[d];
            stride = checked(stride * fullShape[d]);
            total = checked(total * length[d]);
        }
        meta.Total = total;
        // An out-of-bounds launch is a sticky CUDA error 700 that poisons the shared primary context for every engine
        // in the process, so the buffers are checked against the shapes before it, not just the shapes themselves.
        if (full.Size < stride)
            throw new ArgumentException($"RectSlice: the full buffer holds {full.Size} elements, the shape needs {stride}.", nameof(full));
        if (slice.Size < total)
            throw new ArgumentException($"RectSlice: the slice buffer holds {slice.Size} elements, the slice needs {total}.", nameof(slice));

        using var _ = PushContext();
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr f = full.Handle, s = slice.Handle;
        int mode = scatter ? 1 : 0;
        void** args = stackalloc void*[4];
        args[0] = &f; args[1] = &s; args[2] = &meta; args[3] = &mode;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
    }
}
