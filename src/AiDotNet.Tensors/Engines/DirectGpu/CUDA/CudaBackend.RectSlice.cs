// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaBackend));
        if (!IsAvailable)
            throw new GpuCapabilityUnavailableException("CUDA backend is not available.");
        if (full is null) throw new ArgumentNullException(nameof(full));
        if (slice is null) throw new ArgumentNullException(nameof(slice));
        if (fullShape is null) throw new ArgumentNullException(nameof(fullShape));
        if (start is null) throw new ArgumentNullException(nameof(start));
        if (length is null) throw new ArgumentNullException(nameof(length));
        int rank = fullShape.Length;
        if (rank < 1 || rank > RectSliceLimits.MaxRank || start.Length != rank || length.Length != rank)
            throw new ArgumentException($"RectSlice supports rank 1..{RectSliceLimits.MaxRank} with matching start/length.");
        if (!_kernelCache.TryGetValue("rect_slice_nd", out var kernel))
            throw new GpuCapabilityUnavailableException("CUDA kernel not found: rect_slice_nd");

        var meta = new RectSliceMeta { Rank = rank };
        int total = 1, stride = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            if (fullShape[d] < 1 || start[d] < 0 || length[d] < 1 || (long)start[d] + length[d] > fullShape[d])
                throw new ArgumentOutOfRangeException(nameof(start), $"Axis {d}: [{start[d]}, +{length[d]}) is outside {fullShape[d]}.");
            meta.OutDims[d] = length[d];
            meta.FullStrides[d] = stride;
            meta.Starts[d] = start[d];
            stride = checked(stride * fullShape[d]);
            total = checked(total * length[d]);
        }
        meta.Total = total;
        // Reject released buffers before reading offsets (otherwise IntPtr.Zero + baseOffset * 4 forms a non-null
        // wild device address) and check buffer capacities before launching to prevent sticky CUDA error 700.
        if (IsReleased(full, out IntPtr f))
            throw new ObjectDisposedException(nameof(full), "GPU buffer was released before RectSlice.");
        if (IsReleased(slice, out IntPtr s))
            throw new ObjectDisposedException(nameof(slice), "GPU buffer was released before RectSlice.");
        if (full.Size < stride)
            throw new ArgumentException($"RectSlice: the full buffer holds {full.Size} elements, the shape needs {stride}.", nameof(full));
        if (slice.Size < total)
            throw new ArgumentException($"RectSlice: the slice buffer holds {slice.Size} elements, the slice needs {total}.", nameof(slice));

        using var _ = PushContext();
        // A slice whose axes collapse to rows of contiguous runs (a channel slice of NCHW: N rows of size*H*W, C*H*W
        // apart) is a strided row copy: no per-element index decomposition, float4 moves where aligned.
        if (TryCollapseToRows(fullShape, start, length, out long baseOffset, out int rows, out int cols, out int rowStride))
        {
            IntPtr fullAt = (IntPtr)((long)f + baseOffset * sizeof(float));
            if (scatter) CopyRows(s, fullAt, rows, cols, cols, rowStride);
            else CopyRows(fullAt, s, rows, cols, rowStride, cols);
            return;
        }
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        int mode = scatter ? 1 : 0;
        void** args = stackalloc void*[4];
        args[0] = &f; args[1] = &s; args[2] = &meta; args[3] = &mode;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
    }

    private static bool IsReleased(IGpuBuffer buffer, out IntPtr handle)
    {
        handle = buffer.Handle;
        return handle == IntPtr.Zero;
    }

    // The slice as `rows` runs of `cols` contiguous elements, `rowStride` apart in the full tensor from `baseOffset`:
    // unit-length axes drop out, and an axis merges into the next inner run when that run covers its own axis whole.
    private static bool TryCollapseToRows(int[] fullShape, int[] start, int[] length,
        out long baseOffset, out int rows, out int cols, out int rowStride)
    {
        baseOffset = 0; rows = 1; cols = 1; rowStride = 0;
        int rank = fullShape.Length;
        Span<long> lens = stackalloc long[rank];
        Span<long> strides = stackalloc long[rank];
        int count = 0;
        long stride = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            baseOffset += start[d] * stride;
            if (length[d] != 1)
            {
                if (count > 0 && strides[count - 1] * lens[count - 1] == stride)
                    lens[count - 1] *= length[d];
                else { lens[count] = length[d]; strides[count] = stride; count++; }
            }
            stride *= fullShape[d];
        }
        if (count == 0) return true;
        if (strides[0] != 1 || count > 2 || lens[0] > int.MaxValue) return false;
        cols = (int)lens[0];
        if (count == 2)
        {
            if (lens[1] > int.MaxValue || strides[1] > int.MaxValue) return false;
            rows = (int)lens[1]; rowStride = (int)strides[1];
        }
        return true;
    }
}
