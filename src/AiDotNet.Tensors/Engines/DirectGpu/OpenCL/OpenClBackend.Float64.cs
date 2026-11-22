// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IGpuFloat64Reductions, IGpuFloat64Elementwise
    {
        private bool _float64ReductionsReady;

        /// <inheritdoc/>
        public bool SupportsFloat64Reductions => _float64ReductionsReady;

        /// <inheritdoc/>
        public double SumFloat64(IGpuBuffer a, int size)
        {
            var input = Float64Operand(a, size, nameof(a));
            if (size == 0) return 0.0;
            return ReduceFloat64(input, size, firstPass: null);
        }

        /// <inheritdoc/>
        public double DotFloat64(IGpuBuffer a, IGpuBuffer b, int size)
        {
            var left = Float64Operand(a, size, nameof(a));
            var right = Float64Operand(b, size, nameof(b));
            if (size == 0) return 0.0;
            return ReduceFloat64(left, size, firstPass: right);
        }

        /// <inheritdoc/>
        public bool SupportsFloat64Elementwise => _float64ReductionsReady;

        /// <inheritdoc/>
        public void BinaryFloat64(GpuFloat64BinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        {
            var left = Float64Operand(a, size, nameof(a));
            var right = Float64Operand(b, size, nameof(b));
            var destination = Float64Operand(output, size, nameof(output));
            if (size == 0) return;
            var kernel = _kernelCache[operation switch
            {
                GpuFloat64BinaryOperation.Add => "add_f64",
                GpuFloat64BinaryOperation.Subtract => "subtract_f64",
                GpuFloat64BinaryOperation.Multiply => "multiply_f64",
                GpuFloat64BinaryOperation.Divide => "divide_f64",
                _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown double-precision operation."),
            }];
            kernel.SetArg(0, left.Buffer.Handle);
            kernel.SetArg(1, right.Buffer.Handle);
            kernel.SetArg(2, destination.Buffer.Handle);
            kernel.SetArg(3, size);
            kernel.Execute1D(size, CalculateOptimalWorkGroupSize1D(size));
        }

        private DirectOpenClGpuByteBuffer Float64Operand(IGpuBuffer buffer, int size, string name)
        {
            if (_context == null || !_float64ReductionsReady)
                throw new GpuCapabilityUnavailableException("Double-precision kernels are not available on this OpenCL device.");
            if (buffer is null) throw new ArgumentNullException(name);
            if (buffer.ElementType != GpuElementType.Float64 || buffer is not DirectOpenClGpuByteBuffer typed)
                throw new ArgumentException($"Expected a {GpuElementType.Float64} buffer of this backend, got {buffer.ElementType}.", name);
            if (size < 0 || size > buffer.Size)
                throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside the buffer's {buffer.Size} elements.");
            return typed;
        }

        // Reduces to one partial per work-group until one value is left. The first pass multiplies by `firstPass` when
        // it is given (the dot product); every later pass sums partials.
        private double ReduceFloat64(DirectOpenClGpuByteBuffer input, int size, DirectOpenClGpuByteBuffer? firstPass)
        {
            var sumKernel = _kernelCache["reduce_sum_f64"];
            var dotKernel = _kernelCache["reduce_dot_f64"];
            int firstLocal = Float64LocalSize(firstPass is null ? sumKernel : dotKernel, size);
            int maximumPartials = Math.Max(1, (size + firstLocal - 1) / firstLocal);
            using var temporaryA = AllocateBuffer(maximumPartials, GpuElementType.Float64);
            using var temporaryB = AllocateBuffer(maximumPartials, GpuElementType.Float64);

            var current = input;
            int currentSize = size;
            bool writeA = true;
            bool first = true;
            // A single element still takes one pass: the dot product has to multiply it, and the result must land in a
            // buffer this method owns.
            while (currentSize > 1 || first)
            {
                var kernel = first && firstPass is not null ? dotKernel : sumKernel;
                int localSize = Float64LocalSize(kernel, currentSize);
                int partialCount = (currentSize + localSize - 1) / localSize;
                if (partialCount >= currentSize && currentSize > 1)
                    throw new NotSupportedException(
                        $"OpenCL cannot reduce {currentSize} double values because the device only supports a local size of one.");
                var next = (DirectOpenClGpuByteBuffer)(writeA ? temporaryA : temporaryB);
                uint arg = 0;
                kernel.SetArg(arg++, current.Buffer.Handle);
                if (first && firstPass is not null) kernel.SetArg(arg++, firstPass.Buffer.Handle);
                kernel.SetArg(arg++, next.Buffer.Handle);
                kernel.SetLocalArg(arg++, localSize * sizeof(double));
                kernel.SetArg(arg, currentSize);
                kernel.Execute1D(currentSize, localSize);
                current = next;
                currentSize = partialCount;
                writeA = !writeA;
                first = false;
            }

            // DownloadByteBuffer counts the readback.
            return BitConverter.ToDouble(DownloadByteBuffer(current, sizeof(double)), 0);
        }

        private int Float64LocalSize(DirectOpenClKernel kernel, int size)
            => Math.Max(1, ClampLocalSizeForKernel(kernel, CalculateOptimalWorkGroupSize1D(size), sizeof(double)));
    }
}
