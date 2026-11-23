// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// 16-bit float compute (Float16 and BFloat16) on typed or raw half-width buffers. The half add, relu, gelu and matrix
// multiply this backend already had keep their kernels; everything else, and all of BFloat16, runs the kernels generated
// with the integer program (GpuNarrowFloatKernelSource), which CUDA and OpenCL also run.
public sealed partial class HipBackend : IGpuNarrowFloatElementwise, IGpuNarrowFloatMatrixMultiply
{
    private bool NarrowFloatKernelsReady(GpuElementType type)
        => IsAvailable && GpuNarrowFloatLayout.IsNarrowFloat(type) && IntegerKernelsLoaded();

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
    public unsafe void BinaryNarrowFloat(GpuNarrowFloatBinaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(BinaryNarrowFloat), type, size, (a, size), (b, size), (output, size));
        if (HalfKernel(operation, type))
        {
            if (size > 0) Fp16Add(a, b, output, size);
            return;
        }

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
    public unsafe void UnaryNarrowFloat(GpuNarrowFloatUnaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(UnaryNarrowFloat), type, size, (a, size), (output, size));
        if (HalfKernel(operation, type))
        {
            if (size == 0) return;
            if (operation == GpuNarrowFloatUnaryOperation.Relu) Fp16Relu(a, output, size);
            else Fp16Gelu(a, output, size);
            return;
        }

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
