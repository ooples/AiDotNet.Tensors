// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// 16-bit float compute (Float16 and BFloat16) on typed or raw half-width buffers. Every element-wise operation runs the
// kernels generated with the integer program (GpuNarrowFloatKernelSource), which CUDA and OpenCL also run: they compute in
// FP32 once and round to nearest even, as the host does. The older half add and gelu kernels do not, so they are not
// behind this interface.
public sealed partial class HipBackend : IGpuNarrowFloatElementwise, IGpuNarrowFloatMatrixMultiply, IGpuNarrowFloatReductions
{
    private bool NarrowFloatKernelsReady(GpuElementType type)
        => IsAvailable && GpuNarrowFloatLayout.IsNarrowFloat(type) && IntegerKernelsLoaded();



    /// <inheritdoc/>
    public bool SupportsNarrowFloatBinary(GpuNarrowFloatBinaryOperation operation, GpuElementType type)
        => NarrowFloatKernelsReady(type);

    /// <inheritdoc/>
    public bool SupportsNarrowFloatUnary(GpuNarrowFloatUnaryOperation operation, GpuElementType type)
        => NarrowFloatKernelsReady(type);

    /// <inheritdoc/>
    public bool SupportsNarrowFloatMatrixMultiply(GpuElementType type)
        => NarrowFloatKernelsReady(type) || (type == GpuElementType.Float16 && SupportsHgemm);

    /// <inheritdoc/>
    public unsafe void BinaryNarrowFloat(GpuNarrowFloatBinaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(BinaryNarrowFloat), type, size, (a, size), (b, size), (output, size));
        IntPtr kernel = IntegerKernel(GpuNarrowFloatKernelSource.Binary(operation, type));
        if (size == 0) return;
        IntPtr aPtr = IntegerHandle(a);
        IntPtr bPtr = IntegerHandle(b);
        IntPtr outputPtr = IntegerHandle(output);
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &outputPtr;
        args[3] = &n;
        LaunchKernel(kernel, IntegerGrid(size), IntegerBlock, args);
    }

    /// <inheritdoc/>
    public bool SupportsNarrowFloatSum(GpuElementType type) => NarrowFloatKernelsReady(type);

    /// <inheritdoc/>
    public unsafe float SumNarrowFloat(GpuElementType type, IGpuBuffer a, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(SumNarrowFloat), type, size, (a, size));
        IntPtr kernel = IntegerKernel(GpuNarrowFloatKernelSource.Sum(type));
        if (size == 0) return 0f;
        IntPtr later = IntegerKernel(GpuNarrowFloatKernelSource.SumPartials);
        int block = GpuNarrowFloatLayout.SumBlock;
        int partials = GpuIntegerLayout.Blocks(size, block);
        // Partials are binary32 bits, a word each. Two scratch buffers, alternated: a pass never reads the buffer it writes.
        using var even = AllocateBuffer(partials, GpuElementType.Int32);
        using var odd = AllocateBuffer(Math.Max(1, GpuIntegerLayout.Blocks(partials, block)), GpuElementType.Int32);
        IGpuBuffer current = a;
        IGpuBuffer target = even;
        int count = size;
        void** args = stackalloc void*[4];
        while (true)
        {
            int blocks = GpuIntegerLayout.Blocks(count, block);
            // (a, o as binary32 bits, count, blocks), one thread per block.
            IntPtr inputPtr = IntegerHandle(current);
            IntPtr outputPtr = IntegerHandle(target);
            int n = count;
            int groups = blocks;
            args[0] = &inputPtr;
            args[1] = &outputPtr;
            args[2] = &n;
            args[3] = &groups;
            LaunchKernel(kernel, IntegerGrid(blocks), IntegerBlock, args);
            if (blocks == 1)
                return BitConverter.ToSingle(DownloadByteBuffer(target, sizeof(float)), 0);
            current = target;
            target = ReferenceEquals(target, even) ? odd : even;
            count = blocks;
            kernel = later;
        }
    }

    /// <inheritdoc/>
    public unsafe void ScalarNarrowFloat(GpuNarrowFloatBinaryOperation operation, GpuElementType type, IGpuBuffer a, float scalar,
        IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(ScalarNarrowFloat), type, size, (a, size), (output, size));
        IntPtr kernel = IntegerKernel(GpuNarrowFloatKernelSource.Scalar(operation, type));
        if (size == 0) return;
        // (a, o, float y, size)
        IntPtr aPtr = IntegerHandle(a);
        IntPtr outputPtr = IntegerHandle(output);
        float y = scalar;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &outputPtr;
        args[2] = &y;
        args[3] = &n;
        LaunchKernel(kernel, IntegerGrid(size), IntegerBlock, args);
    }

    /// <inheritdoc/>
    public unsafe void UnaryNarrowFloat(GpuNarrowFloatUnaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(UnaryNarrowFloat), type, size, (a, size), (output, size));
        IntPtr kernel = IntegerKernel(GpuNarrowFloatKernelSource.Unary(operation, type));
        if (size == 0) return;
        IntPtr aPtr = IntegerHandle(a);
        IntPtr outputPtr = IntegerHandle(output);
        int n = size;
        void** args = stackalloc void*[3];
        args[0] = &aPtr;
        args[1] = &outputPtr;
        args[2] = &n;
        LaunchKernel(kernel, IntegerGrid(size), IntegerBlock, args);
    }

    /// <inheritdoc/>
    public unsafe void MatrixMultiplyNarrowFloat(GpuElementType type, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
    {
        var (aCount, bCount, cCount) = GpuNarrowFloatLayout.MatrixCounts(m, n, k);
        GpuNarrowFloatLayout.Require(nameof(MatrixMultiplyNarrowFloat), type, cCount, (a, aCount), (b, bCount), (c, cCount));
        GpuNarrowFloatLayout.RequireDistinctOutput(c, a, b);
        if (type == GpuElementType.Float16 && SupportsHgemm)
        {
            Hgemm(a, b, c, m, n, k);
            return;
        }

        IntPtr kernel = IntegerKernel(GpuNarrowFloatKernelSource.MatrixMultiply(type));
        IntPtr aPtr = IntegerHandle(a);
        IntPtr bPtr = IntegerHandle(b);
        IntPtr cPtr = IntegerHandle(c);
        int rows = m, columns = n, inner = k;
        void** args = stackalloc void*[6];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &rows;
        args[4] = &columns;
        args[5] = &inner;
        LaunchKernel(kernel, IntegerGrid(cCount), IntegerBlock, args);
    }
}
