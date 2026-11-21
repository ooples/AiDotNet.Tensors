// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGpuFloat64Reductions
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
        void** args = stackalloc void*[4];
        while (currentSize > 1 || first)
        {
            int partialCount = (currentSize + Float64ReductionBlock - 1) / Float64ReductionBlock;
            IGpuBuffer next = writeA ? temporaryA : temporaryB;
            IntPtr inputPtr = current.Handle;
            IntPtr outputPtr = next.Handle;
            int n = currentSize;
            if (first && multiplier is not null)
            {
                IntPtr multiplierPtr = multiplier.Handle;
                args[0] = &inputPtr;
                args[1] = &multiplierPtr;
                args[2] = &outputPtr;
                args[3] = &n;
                LaunchKernelWithSharedMem(dotKernel, (uint)partialCount, Float64ReductionBlock, sharedBytes, args);
            }
            else
            {
                args[0] = &inputPtr;
                args[1] = &outputPtr;
                args[2] = &n;
                LaunchKernelWithSharedMem(sumKernel, (uint)partialCount, Float64ReductionBlock, sharedBytes, args);
            }

            current = next;
            currentSize = partialCount;
            writeA = !writeA;
            first = false;
        }

        double scalar = BitConverter.ToDouble(DownloadByteBuffer(current, sizeof(double)), 0);
        GpuLaunchProbe.OnReadback(sizeof(double), GpuBackendType.Cuda);
        return scalar;
    }
}
