// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// 16-bit float compute (Float16 and BFloat16) on typed or raw half-width buffers. Every element-wise operation runs the
// kernels generated with the integer program (GpuNarrowFloatKernelSource), which HIP and OpenCL also run: they compute
// in FP32 once and round to nearest even, which is what the host does. The half add and gelu this backend had before
// do not (the add rounds a tie away from zero, the gelu is another formula), so they are not behind this interface.
public sealed partial class CudaBackend : IGpuNarrowFloatElementwise, IGpuNarrowFloatMatrixMultiply
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
        using var _ = PushContext();
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
    public unsafe void ScalarNarrowFloat(GpuNarrowFloatBinaryOperation operation, GpuElementType type, IGpuBuffer a, float scalar,
        IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(ScalarNarrowFloat), type, size, (a, size), (output, size));
        IntPtr kernel = IntegerKernel(GpuNarrowFloatKernelSource.Scalar(operation, type));
        if (size == 0) return;
        using var _ = PushContext();
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
        using var _ = PushContext();
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
        using var _ = PushContext();
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
