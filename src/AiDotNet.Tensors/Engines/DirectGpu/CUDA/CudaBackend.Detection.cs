// Copyright (c) AiDotNet. All rights reserved.
// CUDA launcher shims for the vision detection kernels (Issue #217).
// Pattern matches CudaBackend.Parity210.cs: kernel resolved from
// _kernelCache, dispatched via 256-thread block / grid-ceil.

using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IDetectionBackend
{
    private IntPtr ResolveDetectionKernel(string name)
    {
        if (_detectionModule == IntPtr.Zero)
            throw new InvalidOperationException(
                "Detection CUDA module was not compiled (older toolkit?). Falling back to CPU reference.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        return kernel;
    }

    private unsafe void DispatchPairwiseIou(
        string kernelName, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, int m)
    {
        if (n <= 0 || m <= 0) return;
        var kernel = ResolveDetectionKernel(kernelName);
        using var _ = PushContext();
        int total = n * m;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = a.Handle, bPtr = b.Handle, oPtr = output.Handle;
        int nn = n, mm = m;
        void** args = stackalloc void*[5];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &oPtr;
        args[3] = &nn; args[4] = &mm;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void BoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou("detection_box_iou", boxesA, boxesB, output, n, m);

    public unsafe void GeneralizedBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou("detection_generalized_box_iou", boxesA, boxesB, output, n, m);

    public unsafe void DistanceBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou("detection_distance_box_iou", boxesA, boxesB, output, n, m);

    public unsafe void CompleteBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou("detection_complete_box_iou", boxesA, boxesB, output, n, m);

    public unsafe void BoxArea(IGpuBuffer boxes, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        var kernel = ResolveDetectionKernel("detection_box_area");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr bPtr = boxes.Handle, oPtr = output.Handle;
        int nn = n;
        void** args = stackalloc void*[3];
        args[0] = &bPtr; args[1] = &oPtr; args[2] = &nn;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void BoxConvert(IGpuBuffer boxes, IGpuBuffer output, int n, int fromFormat, int toFormat)
    {
        if (n <= 0) return;
        var kernel = ResolveDetectionKernel("detection_box_convert");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr bPtr = boxes.Handle, oPtr = output.Handle;
        int nn = n, ff = fromFormat, tf = toFormat;
        void** args = stackalloc void*[5];
        args[0] = &bPtr; args[1] = &oPtr; args[2] = &nn; args[3] = &ff; args[4] = &tf;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void IouFamilyBackward(
        IGpuBuffer gradOutput, IGpuBuffer boxesA, IGpuBuffer boxesB,
        IGpuBuffer gradA, IGpuBuffer gradB,
        int n, int m, int variant)
    {
        // Don't short-circuit on n==0 or m==0: the kernel's inner for-loop
        // over the other dim is empty in that case and each thread writes
        // zero, which is the correct gradient. Skipping would leak stale
        // pooled-buffer values into autodiff.
        if (n <= 0 && m <= 0) return;
        var kernelA = ResolveDetectionKernel("detection_iou_backward_a");
        var kernelB = ResolveDetectionKernel("detection_iou_backward_b");
        using var _ = PushContext();
        IntPtr goPtr = gradOutput.Handle, aPtr = boxesA.Handle, bPtr = boxesB.Handle;
        IntPtr gAPtr = gradA.Handle, gBPtr = gradB.Handle;
        int nn = n, mm = m, vv = variant;

        if (n > 0)
        {
            uint gridA = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
            void** argsA = stackalloc void*[7];
            argsA[0] = &goPtr; argsA[1] = &aPtr; argsA[2] = &bPtr;
            argsA[3] = &gAPtr; argsA[4] = &nn; argsA[5] = &mm; argsA[6] = &vv;
            LaunchKernel(kernelA, gridA, DefaultBlockSize, argsA);
        }

        if (m > 0)
        {
            uint gridB = (uint)((m + DefaultBlockSize - 1) / DefaultBlockSize);
            void** argsB = stackalloc void*[7];
            argsB[0] = &goPtr; argsB[1] = &aPtr; argsB[2] = &bPtr;
            argsB[3] = &gBPtr; argsB[4] = &nn; argsB[5] = &mm; argsB[6] = &vv;
            LaunchKernel(kernelB, gridB, DefaultBlockSize, argsB);
        }
    }
}
