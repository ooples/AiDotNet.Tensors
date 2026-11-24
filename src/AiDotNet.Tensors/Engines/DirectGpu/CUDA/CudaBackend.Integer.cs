// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Integer and bool compute in the buffers' own element type, from the kernel program OpenCL and HIP also run
// (GpuIntegerKernelSource).
public sealed partial class CudaBackend : IGpuIntegerElementwise, IGpuIntegerReductions
{
    private const int IntegerBlock = 256;

    private bool IntegerKernelsLoaded()
    {
        foreach (string name in GpuIntegerKernelSource.KernelNames)
            if (!_kernelCache.ContainsKey(name)) return false;
        return true;
    }

    /// <inheritdoc/>
    public bool SupportsIntegerElementwise(GpuElementType type)
        => IsAvailable && GpuIntegerLayout.IsIntegerOrBool(type) && SupportsElementType(type) && IntegerKernelsLoaded();

    /// <inheritdoc/>
    public bool SupportsIntegerReductions(GpuElementType type)
        => type != GpuElementType.Bool && SupportsIntegerElementwise(type);

    private IntPtr IntegerKernel(string name)
    {
        if (!IsAvailable || !_kernelCache.TryGetValue(name, out IntPtr kernel))
            throw new GpuCapabilityUnavailableException("Integer kernels are not available on this device.");
        return kernel;
    }

    // The handle is launched as a device pointer of this API: a buffer of another backend carries a handle that is
    // not one, however well its type and size fit.
    private static IntPtr IntegerHandle(IGpuBuffer buffer)
    {
        if (buffer is not CudaGpuBuffer && buffer is not CudaGpuByteBuffer)
            throw new ArgumentException("The buffer does not belong to the CUDA backend.", nameof(buffer));
        IntPtr handle = buffer.Handle;
        if (handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(buffer));
        return handle;
    }

    private static uint IntegerGrid(int work) => (uint)GpuIntegerLayout.Blocks(work, IntegerBlock);

    /// <inheritdoc/>
    public unsafe void BinaryInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        var type = GpuIntegerLayout.Require(nameof(BinaryInteger), size, output, a, b);
        if (!GpuIntegerLayout.Defined(operation, type))
            throw new ArgumentException($"{operation} is not defined for {type} elements.", nameof(operation));
        IntPtr kernel = IntegerKernel(GpuIntegerKernelSource.Binary(operation, type));
        if (size == 0) return;
        using var _ = PushContext();
        // (a, b, o, size)
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
    public unsafe void UnaryInteger(GpuIntegerUnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size)
    {
        var type = GpuIntegerLayout.Require(nameof(UnaryInteger), size, output, a);
        if (!GpuIntegerLayout.Defined(operation, type))
            throw new ArgumentException($"{operation} is not defined for {type} elements.", nameof(operation));
        IntPtr kernel = IntegerKernel(GpuIntegerKernelSource.Unary(operation, type));
        if (size == 0) return;
        using var _ = PushContext();
        // (a, o, size)
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
    public unsafe void ScalarInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, long scalar, IGpuBuffer output, int size)
    {
        var type = GpuIntegerLayout.Require(nameof(ScalarInteger), size, output, a);
        if (!GpuIntegerLayout.Defined(operation, type))
            throw new ArgumentException($"{operation} is not defined for {type} elements.", nameof(operation));
        IntPtr kernel = IntegerKernel(GpuIntegerKernelSource.Scalar(operation, type));
        if (size == 0) return;
        using var _ = PushContext();
        // (a, o, 64-bit y, size)
        IntPtr aPtr = IntegerHandle(a);
        IntPtr outputPtr = IntegerHandle(output);
        long y = scalar;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &outputPtr;
        args[2] = &y;
        args[3] = &n;
        LaunchKernel(kernel, IntegerGrid(size), IntegerBlock, args);
    }

    /// <inheritdoc/>
    public unsafe long SumInteger(IGpuBuffer a, int size)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        var type = a.ElementType;
        if (!GpuIntegerLayout.IsIntegerOrBool(type) || type == GpuElementType.Bool)
            throw new ArgumentException($"{nameof(SumInteger)} reduces integer buffers, got {type}.", nameof(a));
        if (size < 0 || size > a.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside the buffer's {a.Size} elements.");
        IntPtr kernel = IntegerKernel(GpuIntegerKernelSource.Sum(type));
        if (size == 0) return 0;
        IntPtr later = IntegerKernel(GpuIntegerKernelSource.Sum(GpuElementType.Int64));
        using var _ = PushContext();
        int block = GpuIntegerKernelSource.SumBlock;
        int partials = GpuIntegerLayout.Blocks(size, block);
        // Two scratch buffers, alternated: a pass never reads the buffer it writes.
        using var even = AllocateBuffer(partials, GpuElementType.Int64);
        using var odd = AllocateBuffer(Math.Max(1, GpuIntegerLayout.Blocks(partials, block)), GpuElementType.Int64);
        IGpuBuffer current = a;
        IGpuBuffer target = even;
        int count = size;
        void** args = stackalloc void*[4];
        while (true)
        {
            int blocks = GpuIntegerLayout.Blocks(count, block);
            // (a, 64-bit o, count, blocks), one thread per block.
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
                return GpuIntegerLayout.Wrap(BitConverter.ToInt64(DownloadByteBuffer(target, sizeof(long)), 0), type);
            current = target;
            target = ReferenceEquals(target, even) ? odd : even;
            count = blocks;
            kernel = later;
        }
    }
}
