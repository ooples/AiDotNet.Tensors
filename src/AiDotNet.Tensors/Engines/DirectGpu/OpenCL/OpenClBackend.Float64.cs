// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IGpuFloat64Reductions, IGpuFloat64Elementwise, IGpuFloat64MatrixMultiply
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

        // The CLBlast direct GEMM kernels built over double. Compiled on first use; a device without fp64 fails the
        // build and reports the routine unavailable.
        private DirectOpenClKernel? _float64GemmKernel;
        private bool _float64GemmFailed;

        /// <inheritdoc/>
        public bool SupportsFloat64MatrixMultiply => _float64ReductionsReady && EnsureFloat64GemmKernel();

        private bool EnsureFloat64GemmKernel()
        {
            if (_float64GemmKernel is not null) return true;
            if (_float64GemmFailed || _context == null) return false;
            lock (_clblastLock)
            {
                if (_float64GemmKernel is not null) return true;
                if (_float64GemmFailed) return false;
                try
                {
                    if (_clblastDirectParams.Wgd <= 0 || _clblastDirectParams.MdimCd <= 0 || _clblastDirectParams.NdimCd <= 0)
                        _clblastDirectParams = ClBlastXgemmDirectDatabase.GetParameters(ClBlastDeviceInfo.FromContext(_context));
                    var program = CompileOrLoadCached(
                        ClBlastXgemmDirectKernel.BuildSource(_clblastDirectParams, doublePrecision: true),
                        OpenClBuildOptions.OptimizationFlags,
                        "CLBlast direct kernels (double)");
                    _programs.Add(program);
                    // Row-major operands: A and C are read as transposed, B is not.
                    _float64GemmKernel = new DirectOpenClKernel(_context, program, "XgemmDirectTN");
                    return true;
                }
                catch (InvalidOperationException buildFailure)
                {
                    _float64GemmFailed = true;
                    System.Diagnostics.Trace.TraceWarning(
                        "OpenCL double-precision matrix multiply is unavailable: {0}", buildFailure);
                    return false;
                }
            }
        }

        /// <inheritdoc/>
        public void MatrixMultiplyFloat64(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
        {
            if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), m, "Matrix dimensions must be positive.");
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), n, "Matrix dimensions must be positive.");
            if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), k, "Matrix dimensions must be positive.");
            var left = Float64Operand(a, checked(m * k), nameof(a));
            var right = Float64Operand(b, checked(k * n), nameof(b));
            var product = Float64Operand(c, checked(m * n), nameof(c));
            if (ReferenceEquals(c, a) || ReferenceEquals(c, b))
                throw new ArgumentException("The product buffer must not be one of the operands.", nameof(c));
            if (!EnsureFloat64GemmKernel() || _float64GemmKernel is not { } kernel)
                throw new GpuCapabilityUnavailableException("Double-precision matrix multiply is not available on this OpenCL device.");

            kernel.SetArg(0, m);
            kernel.SetArg(1, n);
            kernel.SetArg(2, k);
            kernel.SetArg(3, 1.0);
            kernel.SetArg(4, 0.0);
            kernel.SetArg(5, left.Buffer.Handle);
            kernel.SetArg(6, 0);
            kernel.SetArg(7, k);
            kernel.SetArg(8, right.Buffer.Handle);
            kernel.SetArg(9, 0);
            kernel.SetArg(10, n);
            kernel.SetArg(11, product.Buffer.Handle);
            kernel.SetArg(12, 0);
            kernel.SetArg(13, n);
            kernel.SetArg(14, 1);
            kernel.SetArg(15, 0);
            kernel.SetArg(16, 0);
            int mCeiled = CeilToMultiple(m, _clblastDirectParams.Wgd);
            int nCeiled = CeilToMultiple(n, _clblastDirectParams.Wgd);
            int globalX = (mCeiled * _clblastDirectParams.MdimCd) / _clblastDirectParams.Wgd;
            int globalY = (nCeiled * _clblastDirectParams.NdimCd) / _clblastDirectParams.Wgd;
            kernel.Execute2D(globalX, globalY, _clblastDirectParams.MdimCd, _clblastDirectParams.NdimCd);
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
