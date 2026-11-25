// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

// 16-bit float compute (Float16 and BFloat16) on typed or raw half-width buffers. Every element-wise operation runs the
// kernels generated with the integer program (GpuNarrowFloatKernelSource), which CUDA and HIP also run: they compute in
// FP32 once and round to nearest even, as the host does. The older half add and gelu kernels do not, so they are not
// behind this interface.
public sealed partial class OpenClBackend : IGpuNarrowFloatElementwise, IGpuNarrowFloatMatrixMultiply, IGpuNarrowFloatReductions
{
    private bool NarrowFloatKernelsReady(GpuElementType type)
        => GpuNarrowFloatLayout.IsNarrowFloat(type) && EnsureIntegerKernels() is not null;



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
    public void BinaryNarrowFloat(GpuNarrowFloatBinaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(BinaryNarrowFloat), type, size, (a, size), (b, size), (output, size));
        var kernel = IntegerKernel(GpuNarrowFloatKernelSource.Binary(operation, type));
        if (size == 0) return;
        kernel.SetArg(0, IntegerHandle(a));
        kernel.SetArg(1, IntegerHandle(b));
        kernel.SetArg(2, IntegerHandle(output));
        kernel.SetArg(3, size);
        kernel.Execute1D(size, CalculateOptimalWorkGroupSize1D(size));
    }

    /// <inheritdoc/>
    public bool SupportsNarrowFloatSum(GpuElementType type) => NarrowFloatKernelsReady(type);

    /// <inheritdoc/>
    public float SumNarrowFloat(GpuElementType type, IGpuBuffer a, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(SumNarrowFloat), type, size, (a, size));
        var kernel = IntegerKernel(GpuNarrowFloatKernelSource.Sum(type));
        if (size == 0) return 0f;
        var later = IntegerKernel(GpuNarrowFloatKernelSource.SumPartials);
        int block = GpuNarrowFloatLayout.SumBlock;
        int partials = GpuIntegerLayout.Blocks(size, block);
        // Partials are binary32 bits, a word each. Two scratch buffers, alternated: a pass never reads the buffer it writes.
        using var even = AllocateBuffer(partials, GpuElementType.Int32);
        using var odd = AllocateBuffer(Math.Max(1, GpuIntegerLayout.Blocks(partials, block)), GpuElementType.Int32);
        IGpuBuffer current = a;
        IGpuBuffer target = even;
        int count = size;
        while (true)
        {
            int blocks = GpuIntegerLayout.Blocks(count, block);
            kernel.SetArg(0, IntegerHandle(current));
            kernel.SetArg(1, IntegerHandle(target));
            kernel.SetArg(2, count);
            kernel.SetArg(3, blocks);
            kernel.Execute1D(blocks, CalculateOptimalWorkGroupSize1D(blocks));
            if (blocks == 1)
                return BitConverter.ToSingle(DownloadByteBuffer(target, sizeof(float)), 0);
            current = target;
            target = ReferenceEquals(target, even) ? odd : even;
            count = blocks;
            kernel = later;
        }
    }

    /// <inheritdoc/>
    public void ScalarNarrowFloat(GpuNarrowFloatBinaryOperation operation, GpuElementType type, IGpuBuffer a, float scalar,
        IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(ScalarNarrowFloat), type, size, (a, size), (output, size));
        var kernel = IntegerKernel(GpuNarrowFloatKernelSource.Scalar(operation, type));
        if (size == 0) return;
        kernel.SetArg(0, IntegerHandle(a));
        kernel.SetArg(1, IntegerHandle(output));
        kernel.SetArg(2, scalar);
        kernel.SetArg(3, size);
        kernel.Execute1D(size, CalculateOptimalWorkGroupSize1D(size));
    }

    /// <inheritdoc/>
    public void UnaryNarrowFloat(GpuNarrowFloatUnaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(UnaryNarrowFloat), type, size, (a, size), (output, size));
        var kernel = IntegerKernel(GpuNarrowFloatKernelSource.Unary(operation, type));
        if (size == 0) return;
        kernel.SetArg(0, IntegerHandle(a));
        kernel.SetArg(1, IntegerHandle(output));
        kernel.SetArg(2, size);
        kernel.Execute1D(size, CalculateOptimalWorkGroupSize1D(size));
    }

    /// <inheritdoc/>
    public void MatrixMultiplyNarrowFloat(GpuElementType type, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
    {
        var (aCount, bCount, cCount) = GpuNarrowFloatLayout.MatrixCounts(m, n, k);
        GpuNarrowFloatLayout.Require(nameof(MatrixMultiplyNarrowFloat), type, cCount, (a, aCount), (b, bCount), (c, cCount));
        GpuNarrowFloatLayout.RequireDistinctOutput(c, a, b);
        if (type == GpuElementType.Float16 && SupportsHgemm)
        {
            Hgemm(a, b, c, m, n, k);
            return;
        }

        var kernel = IntegerKernel(GpuNarrowFloatKernelSource.MatrixMultiply(type));
        kernel.SetArg(0, IntegerHandle(a));
        kernel.SetArg(1, IntegerHandle(b));
        kernel.SetArg(2, IntegerHandle(c));
        kernel.SetArg(3, m);
        kernel.SetArg(4, n);
        kernel.SetArg(5, k);
        kernel.Execute1D(cCount, CalculateOptimalWorkGroupSize1D(cCount));
    }
}
