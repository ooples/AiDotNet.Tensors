// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // Integer and bool compute in the buffers' own element type. Every OpenCL device has the integer types, so the
    // program compiles wherever the backend does.
    public sealed partial class OpenClBackend : IGpuIntegerElementwise, IGpuIntegerReductions
    {
        private readonly object _integerKernelLock = new object();
        private bool _integerKernelsReady;
        private bool _integerKernelsFailed;

        private bool EnsureIntegerKernels()
        {
            if (_integerKernelsReady) return true;
            if (_integerKernelsFailed || _context == null) return false;
            lock (_integerKernelLock)
            {
                if (_integerKernelsReady) return true;
                if (_integerKernelsFailed) return false;
                try
                {
                    var program = CompileOrLoadCached(
                        GpuIntegerKernelSource.Build(GpuIntegerKernelDialect.OpenCl),
                        OpenClBuildOptions.SafeMathFlags,
                        "Integer kernels");
                    _programs.Add(program);
                    foreach (string name in GpuIntegerKernelSource.KernelNames)
                        _kernelCache[name] = new DirectOpenClKernel(_context, program, name);
                    _integerKernelsReady = true;
                    return true;
                }
                catch (InvalidOperationException buildFailure)
                {
                    _integerKernelsFailed = true;
                    System.Diagnostics.Trace.TraceWarning("OpenCL integer kernels are unavailable: {0}", buildFailure);
                    return false;
                }
            }
        }

        /// <inheritdoc/>
        public bool SupportsIntegerElementwise(GpuElementType type)
            => GpuIntegerLayout.IsIntegerOrBool(type) && SupportsElementType(type) && EnsureIntegerKernels();

        /// <inheritdoc/>
        public bool SupportsIntegerReductions(GpuElementType type)
            => type != GpuElementType.Bool && SupportsIntegerElementwise(type);

        private DirectOpenClKernel IntegerKernel(string name)
        {
            if (!EnsureIntegerKernels())
                throw new GpuCapabilityUnavailableException("Integer kernels are not available on this OpenCL device.");
            return _kernelCache[name];
        }

        private static IntPtr IntegerHandle(IGpuBuffer buffer)
        {
            if (buffer is not DirectOpenClGpuBuffer && buffer is not DirectOpenClGpuByteBuffer)
                throw new ArgumentException("The buffer does not belong to the OpenCL backend.", nameof(buffer));
            IntPtr handle = buffer.Handle;
            if (handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(buffer));
            return handle;
        }

        /// <inheritdoc/>
        public void BinaryInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        {
            var type = GpuIntegerLayout.Require(nameof(BinaryInteger), size, output, a, b);
            if (!GpuIntegerLayout.Defined(operation, type))
                throw new ArgumentException($"{operation} is not defined for {type} elements.", nameof(operation));
            var kernel = IntegerKernel(GpuIntegerKernelSource.Binary(operation, type));
            if (size == 0) return;
            kernel.SetArg(0, IntegerHandle(a));
            kernel.SetArg(1, IntegerHandle(b));
            kernel.SetArg(2, IntegerHandle(output));
            kernel.SetArg(3, size);
            kernel.Execute1D(size, CalculateOptimalWorkGroupSize1D(size));
        }

        /// <inheritdoc/>
        public void UnaryInteger(GpuIntegerUnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size)
        {
            var type = GpuIntegerLayout.Require(nameof(UnaryInteger), size, output, a);
            if (!GpuIntegerLayout.Defined(operation, type))
                throw new ArgumentException($"{operation} is not defined for {type} elements.", nameof(operation));
            var kernel = IntegerKernel(GpuIntegerKernelSource.Unary(operation, type));
            if (size == 0) return;
            kernel.SetArg(0, IntegerHandle(a));
            kernel.SetArg(1, IntegerHandle(output));
            kernel.SetArg(2, size);
            kernel.Execute1D(size, CalculateOptimalWorkGroupSize1D(size));
        }

        /// <inheritdoc/>
        public void ScalarInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, long scalar, IGpuBuffer output, int size)
        {
            var type = GpuIntegerLayout.Require(nameof(ScalarInteger), size, output, a);
            if (!GpuIntegerLayout.Defined(operation, type))
                throw new ArgumentException($"{operation} is not defined for {type} elements.", nameof(operation));
            var kernel = IntegerKernel(GpuIntegerKernelSource.Scalar(operation, type));
            if (size == 0) return;
            kernel.SetArg(0, IntegerHandle(a));
            kernel.SetArg(1, IntegerHandle(output));
            kernel.SetArg(2, unchecked((ulong)scalar));
            kernel.SetArg(3, size);
            kernel.Execute1D(size, CalculateOptimalWorkGroupSize1D(size));
        }

        /// <inheritdoc/>
        public long SumInteger(IGpuBuffer a, int size)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            var type = a.ElementType;
            if (!GpuIntegerLayout.IsIntegerOrBool(type) || type == GpuElementType.Bool)
                throw new ArgumentException($"{nameof(SumInteger)} reduces integer buffers, got {type}.", nameof(a));
            if (size < 0 || size > a.Size)
                throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside the buffer's {a.Size} elements.");
            var first = IntegerKernel(GpuIntegerKernelSource.Sum(type));
            if (size == 0) return 0;
            var later = IntegerKernel(GpuIntegerKernelSource.Sum(GpuElementType.Int64));
            int block = GpuIntegerKernelSource.SumBlock;
            int partials = (size + block - 1) / block;
            // Two scratch buffers, alternated: a pass never reads the buffer it writes.
            using var even = AllocateBuffer(partials, GpuElementType.Int64);
            using var odd = AllocateBuffer(Math.Max(1, (partials + block - 1) / block), GpuElementType.Int64);
            IGpuBuffer current = a;
            IGpuBuffer target = even;
            int count = size;
            var kernel = first;
            while (true)
            {
                int blocks = (count + block - 1) / block;
                kernel.SetArg(0, IntegerHandle(current));
                kernel.SetArg(1, IntegerHandle(target));
                kernel.SetArg(2, count);
                kernel.SetArg(3, blocks);
                kernel.Execute1D(blocks, CalculateOptimalWorkGroupSize1D(blocks));
                if (blocks == 1)
                    return GpuIntegerLayout.Wrap(BitConverter.ToInt64(DownloadByteBuffer(target, sizeof(long)), 0), type);
                current = target;
                target = ReferenceEquals(target, even) ? odd : even;
                count = blocks;
                kernel = later;
            }
        }
    }
}
