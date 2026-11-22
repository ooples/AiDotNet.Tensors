// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGpuFloat64Reductions, IGpuFloat64Elementwise
{
    private const int Float64ReductionBlock = 256;

    /// <inheritdoc/>
    public bool SupportsFloat64Reductions
        => IsAvailable && _kernelCache.ContainsKey("reduce_sum_f64") && _kernelCache.ContainsKey("reduce_dot_f64");

    /// <inheritdoc/>
    public double SumFloat64(IGpuBuffer a, int size)
    {
        CheckFloat64Operand(a, size, nameof(a));
        return size == 0 ? 0.0 : ReduceFloat64(a, null, size);
    }

    /// <inheritdoc/>
    public double DotFloat64(IGpuBuffer a, IGpuBuffer b, int size)
    {
        CheckFloat64Operand(a, size, nameof(a));
        CheckFloat64Operand(b, size, nameof(b));
        return size == 0 ? 0.0 : ReduceFloat64(a, b, size);
    }

    /// <inheritdoc/>
    public bool SupportsFloat64Elementwise
        => IsAvailable && _kernelCache.ContainsKey("add_f64") && _kernelCache.ContainsKey("subtract_f64")
            && _kernelCache.ContainsKey("multiply_f64") && _kernelCache.ContainsKey("divide_f64");

    /// <inheritdoc/>
    public unsafe void BinaryFloat64(GpuFloat64BinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        if (!SupportsFloat64Elementwise)
            throw new GpuCapabilityUnavailableException("Double-precision element-wise kernels are not available on this device.");
        CheckFloat64Operand(a, size, nameof(a));
        CheckFloat64Operand(b, size, nameof(b));
        CheckFloat64Operand(output, size, nameof(output));
        if (size == 0) return;
        IntPtr kernel = _kernelCache[operation switch
        {
            GpuFloat64BinaryOperation.Add => "add_f64",
            GpuFloat64BinaryOperation.Subtract => "subtract_f64",
            GpuFloat64BinaryOperation.Multiply => "multiply_f64",
            GpuFloat64BinaryOperation.Divide => "divide_f64",
            _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown double-precision operation."),
        }];
        using var _ = PushContext();
        // Every element-wise double kernel takes (a, b, output, size).
        IntPtr aPtr = a.Handle;
        IntPtr bPtr = b.Handle;
        IntPtr outputPtr = output.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &outputPtr;
        args[3] = &n;
        LaunchKernel(kernel, (uint)((size + Float64ReductionBlock - 1) / Float64ReductionBlock), Float64ReductionBlock, args);
    }

    private void CheckFloat64Operand(IGpuBuffer buffer, int size, string name)
    {
        if (!SupportsFloat64Reductions)
            throw new GpuCapabilityUnavailableException("Double-precision reductions are not available on this CUDA device.");
        if (buffer is null) throw new ArgumentNullException(name);
        if (buffer.ElementType != GpuElementType.Float64)
            throw new ArgumentException($"Expected a {GpuElementType.Float64} buffer, got {buffer.ElementType}.", name);
        if (size < 0 || size > buffer.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside the buffer's {buffer.Size} elements.");
    }

    // Reduces to one partial per block until one value is left. The first pass multiplies by `multiplier` when it is
    // given (the dot product); every later pass sums partials. A single element still takes one pass, so the product is
    // formed and the result lands in a buffer this method owns.
    private unsafe double ReduceFloat64(IGpuBuffer input, IGpuBuffer? multiplier, int size)
    {
        IntPtr sumKernel = _kernelCache["reduce_sum_f64"];
        IntPtr dotKernel = _kernelCache["reduce_dot_f64"];
        using var _ = PushContext();
        int maximumPartials = (size + Float64ReductionBlock - 1) / Float64ReductionBlock;
        using var temporaryA = AllocateBuffer(maximumPartials, GpuElementType.Float64);
        using var temporaryB = AllocateBuffer(maximumPartials, GpuElementType.Float64);
        IGpuBuffer current = input;
        int currentSize = size;
        bool writeA = true;
        bool first = true;
        uint sharedBytes = Float64ReductionBlock * sizeof(double);
        while (currentSize > 1 || first)
        {
            int partialCount = (currentSize + Float64ReductionBlock - 1) / Float64ReductionBlock;
            IGpuBuffer next = writeA ? temporaryA : temporaryB;
            if (first && multiplier is not null)
                LaunchFloat64Dot(dotKernel, current, multiplier, next, currentSize, (uint)partialCount, sharedBytes);
            else
                LaunchFloat64Sum(sumKernel, current, next, currentSize, (uint)partialCount, sharedBytes);

            current = next;
            currentSize = partialCount;
            writeA = !writeA;
            first = false;
        }

        // DownloadByteBuffer counts the readback.
        return BitConverter.ToDouble(DownloadByteBuffer(current, sizeof(double)), 0);
    }

    // Each launch fills exactly its kernel's parameters: reduce_sum_f64(input, output, size).
    private unsafe void LaunchFloat64Sum(IntPtr sumKernel, IGpuBuffer input, IGpuBuffer output, int size, uint grid, uint sharedBytes)
    {
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int n = size;
        void** args = stackalloc void*[3];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &n;
        LaunchKernelWithSharedMem(sumKernel, grid, Float64ReductionBlock, sharedBytes, args);
    }

    // reduce_dot_f64(a, b, output, size).
    private unsafe void LaunchFloat64Dot(IntPtr dotKernel, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size, uint grid, uint sharedBytes)
    {
        IntPtr aPtr = a.Handle;
        IntPtr bPtr = b.Handle;
        IntPtr outputPtr = output.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &outputPtr;
        args[3] = &n;
        LaunchKernelWithSharedMem(dotKernel, grid, Float64ReductionBlock, sharedBytes, args);
    }
}
