// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

// 16-bit float compute (Float16 and BFloat16) on typed or raw half-width buffers. The half add, relu, gelu and matrix
// multiply this backend already had keep their kernels; everything else, and all of BFloat16, runs the kernels generated
// with the integer program (GpuNarrowFloatKernelSource), which CUDA and HIP also run.
public sealed partial class OpenClBackend : IGpuNarrowFloatElementwise, IGpuNarrowFloatMatrixMultiply
{
    private bool NarrowFloatKernelsReady(GpuElementType type)
        => GpuNarrowFloatLayout.IsNarrowFloat(type) && EnsureIntegerKernels() is not null;

    private bool HalfKernel(GpuNarrowFloatBinaryOperation operation, GpuElementType type)
        => type == GpuElementType.Float16 && operation == GpuNarrowFloatBinaryOperation.Add && SupportsFp16NativeOps;

    private bool HalfKernel(GpuNarrowFloatUnaryOperation operation, GpuElementType type)
        => type == GpuElementType.Float16 && SupportsFp16NativeOps
           && operation is GpuNarrowFloatUnaryOperation.Relu or GpuNarrowFloatUnaryOperation.Gelu;

    /// <inheritdoc/>
    public bool SupportsNarrowFloatBinary(GpuNarrowFloatBinaryOperation operation, GpuElementType type)
        => NarrowFloatKernelsReady(type) || HalfKernel(operation, type);

    /// <inheritdoc/>
    public bool SupportsNarrowFloatUnary(GpuNarrowFloatUnaryOperation operation, GpuElementType type)
        => NarrowFloatKernelsReady(type) || HalfKernel(operation, type);

    /// <inheritdoc/>
    public bool SupportsNarrowFloatMatrixMultiply(GpuElementType type)
        => NarrowFloatKernelsReady(type) || (type == GpuElementType.Float16 && SupportsHgemm);

    /// <inheritdoc/>
    public void BinaryNarrowFloat(GpuNarrowFloatBinaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(BinaryNarrowFloat), type, size, (a, size), (b, size), (output, size));
        if (HalfKernel(operation, type))
        {
            if (size > 0) Fp16Add(a, b, output, size);
            return;
        }

        var kernel = IntegerKernel(GpuNarrowFloatKernelSource.Binary(operation, type));
        if (size == 0) return;
        kernel.SetArg(0, IntegerHandle(a));
        kernel.SetArg(1, IntegerHandle(b));
        kernel.SetArg(2, IntegerHandle(output));
        kernel.SetArg(3, size);
        kernel.Execute1D(size, CalculateOptimalWorkGroupSize1D(size));
    }

    /// <inheritdoc/>
    public void UnaryNarrowFloat(GpuNarrowFloatUnaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(UnaryNarrowFloat), type, size, (a, size), (output, size));
        if (HalfKernel(operation, type))
        {
            if (size == 0) return;
            if (operation == GpuNarrowFloatUnaryOperation.Relu) Fp16Relu(a, output, size);
            else Fp16Gelu(a, output, size);
            return;
        }

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
