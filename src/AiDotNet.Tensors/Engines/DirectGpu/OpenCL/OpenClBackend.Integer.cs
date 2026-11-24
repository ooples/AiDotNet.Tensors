// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // Integer and bool compute in the buffers' own element type. Every OpenCL device has the integer types, so the
    // program compiles wherever the backend does.
    public sealed partial class OpenClBackend : IGpuIntegerElementwise, IGpuIntegerReductions
    {
        private readonly object _integerKernelLock = new object();
        // The integer kernels have their own registry: the shared kernel cache is read without a lock by every other
        // operation, so nothing is added to it after construction. The registry is built off to the side and
        // published whole; a reader sees none of it or all of it.
        private System.Collections.Generic.Dictionary<string, DirectOpenClKernel>? _integerKernels;
        private bool _integerKernelsFailed;

        private System.Collections.Generic.Dictionary<string, DirectOpenClKernel>? EnsureIntegerKernels()
        {
            // A disposed backend has released them (ReleaseIntegerKernels), whatever was published before.
            if (!IsAvailable) return null;
            if (System.Threading.Volatile.Read(ref _integerKernels) is { } ready) return ready;
            if (_integerKernelsFailed || _context == null) return null;
            lock (_integerKernelLock)
            {
                if (!IsAvailable) return null;
                if (_integerKernels is { } published) return published;
                if (_integerKernelsFailed) return null;
                var built = new System.Collections.Generic.Dictionary<string, DirectOpenClKernel>(StringComparer.Ordinal);
                try
                {
                    var program = CompileOrLoadCached(
                        GpuIntegerKernelSource.Build(GpuIntegerKernelDialect.OpenCl),
                        OpenClBuildOptions.SafeMathFlags,
                        "Integer kernels");
                    _programs.Add(program);
                    foreach (string name in GpuIntegerKernelSource.KernelNames)
                        built[name] = new DirectOpenClKernel(_context, program, name);
                    System.Threading.Volatile.Write(ref _integerKernels, built);
                    return built;
                }
                catch (InvalidOperationException buildFailure)
                {
                    foreach (var kernel in built.Values) kernel.Dispose();
                    _integerKernelsFailed = true;
                    System.Diagnostics.Trace.TraceWarning("OpenCL integer kernels are unavailable: {0}", buildFailure);
                    return null;
                }
            }
        }

        // Called by Dispose once IsAvailable is false: no registry is published past this point.
        private void ReleaseIntegerKernels()
        {
            lock (_integerKernelLock)
            {
                if (_integerKernels is not { } kernels) return;
                System.Threading.Volatile.Write(ref _integerKernels, null);
                foreach (var kernel in kernels.Values) kernel.Dispose();
            }
        }

        /// <inheritdoc/>
        public bool SupportsIntegerElementwise(GpuElementType type)
            => GpuIntegerLayout.IsIntegerOrBool(type) && SupportsElementType(type) && EnsureIntegerKernels() is not null;

        /// <inheritdoc/>
        public bool SupportsIntegerReductions(GpuElementType type)
            => type != GpuElementType.Bool && SupportsIntegerElementwise(type);

        private DirectOpenClKernel IntegerKernel(string name)
        {
            if (EnsureIntegerKernels() is not { } kernels || !kernels.TryGetValue(name, out var kernel))
                throw new GpuCapabilityUnavailableException("Integer kernels are not available on this OpenCL device.");
            return kernel;
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
        public long SumInteger(IGpuBuffer a, int size) => ReduceInteger(GpuIntegerReduction.Sum, a, size);

        /// <inheritdoc/>
        public long MinInteger(IGpuBuffer a, int size) => ReduceInteger(GpuIntegerReduction.Min, a, size);

        /// <inheritdoc/>
        public long MaxInteger(IGpuBuffer a, int size) => ReduceInteger(GpuIntegerReduction.Max, a, size);

        private long ReduceInteger(GpuIntegerReduction reduction, IGpuBuffer a, int size)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            var type = a.ElementType;
            if (!GpuIntegerLayout.IsIntegerOrBool(type) || type == GpuElementType.Bool)
                throw new ArgumentException($"{nameof(SumInteger)} reduces integer buffers, got {type}.", nameof(a));
            if (size < 0 || size > a.Size)
                throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside the buffer's {a.Size} elements.");
            var first = IntegerKernel(GpuIntegerKernelSource.Reduce(reduction, type));
            GpuIntegerLayout.RequireElements(reduction, size);
            if (size == 0) return 0;
            var later = IntegerKernel(GpuIntegerKernelSource.Reduce(reduction, GpuElementType.Int64));
            int block = GpuIntegerKernelSource.SumBlock;
            int partials = GpuIntegerLayout.Blocks(size, block);
            // Two scratch buffers, alternated: a pass never reads the buffer it writes.
            using var even = AllocateBuffer(partials, GpuElementType.Int64);
            using var odd = AllocateBuffer(Math.Max(1, GpuIntegerLayout.Blocks(partials, block)), GpuElementType.Int64);
            IGpuBuffer current = a;
            IGpuBuffer target = even;
            int count = size;
            var kernel = first;
            while (true)
            {
                int blocks = GpuIntegerLayout.Blocks(count, block);
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
