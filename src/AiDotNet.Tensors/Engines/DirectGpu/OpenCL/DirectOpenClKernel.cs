// Copyright (c) AiDotNet. All rights reserved.
// Pure P/Invoke OpenCL kernel - no managed GPU runtime dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// OpenCL kernel wrapper using pure P/Invoke. No managed GPU runtime dependency.
    /// </summary>
    internal sealed class DirectOpenClKernel : IDisposable
    {
        private IntPtr _kernel;
        private readonly DirectOpenClContext _context;
        private bool _disposed;

        public IntPtr Handle => _kernel;

        public DirectOpenClKernel(DirectOpenClContext context, DirectOpenClProgram program, string kernelName)
        {
            _context = context;

            _kernel = OpenClNativeBindings.CreateKernel(program.Handle, kernelName, out int err);
            if (err != OpenClNativeBindings.CL_SUCCESS || _kernel == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL kernel '{kernelName}': {err}");
        }

        // ── Thread-safety: a cl_kernel object is SHARED (cached per name) across all
        // threads, and clSetKernelArg mutates it. Two threads doing SetArg+Execute
        // concurrently would stomp each other's args (undefined behavior / crashes —
        // the root cause of concurrent-tape GPU corruption). Fix: SetArg only RECORDS
        // the argument into a per-thread pending list; Execute then applies the args
        // and enqueues ATOMICALLY under a single process-wide submit lock. Because
        // clEnqueueNDRangeKernel snapshots the kernel's current args into the queued
        // command, the lock only needs to span apply-args + enqueue — the GPU work
        // itself still runs asynchronously, so submission is serialized but execution
        // pipelines. A single GPU has one in-order queue, so this is the natural model.
        private enum ArgKind { Buffer, Int32, Float, UInt64, Local }
        private readonly struct PendingArg
        {
            public readonly uint Index;
            public readonly ArgKind Kind;
            public readonly long Raw; // buffer handle / int / float-bits / ulong / local byte size
            public PendingArg(uint index, ArgKind kind, long raw) { Index = index; Kind = kind; Raw = raw; }
        }

        // Per-thread pending args. Each op sets a kernel's args then immediately
        // Executes it on the same thread, so the list holds exactly that kernel's
        // args at Execute time; Execute clears it.
        [ThreadStatic] private static System.Collections.Generic.List<PendingArg>? _pendingArgs;

        // Serializes the apply-args + enqueue critical section across ALL kernels
        // (the shared cl_kernel arg state and the shared command queue both require it).
        private static readonly object _submitLock = new object();

        private static System.Collections.Generic.List<PendingArg> Pending
            => _pendingArgs ??= new System.Collections.Generic.List<PendingArg>(8);

        #region SetArg Overloads

        public void SetArg(uint index, IntPtr bufferHandle)
            => Pending.Add(new PendingArg(index, ArgKind.Buffer, (long)bufferHandle));

        public void SetArg(uint index, int value)
            => Pending.Add(new PendingArg(index, ArgKind.Int32, value));

        public void SetArg(uint index, float value)
            => Pending.Add(new PendingArg(index, ArgKind.Float, BitConverter.ToInt32(BitConverter.GetBytes(value), 0)));

        public void SetArg(uint index, ulong value)
            => Pending.Add(new PendingArg(index, ArgKind.UInt64, unchecked((long)value)));

        /// <summary>
        /// Sets a local memory argument (for shared memory allocation).
        /// </summary>
        public void SetLocalArg(uint index, int sizeInBytes)
            => Pending.Add(new PendingArg(index, ArgKind.Local, sizeInBytes));

        // Applies the per-thread pending args to the shared kernel. MUST be called
        // while holding _submitLock and immediately before the matching enqueue.
        private void ApplyPendingArgsLocked()
        {
            var pending = _pendingArgs;
            if (pending == null) return;
            for (int i = 0; i < pending.Count; i++)
            {
                var a = pending[i];
                int err;
                switch (a.Kind)
                {
                    case ArgKind.Local:
                        err = OpenClNativeBindings.SetKernelArg(_kernel, a.Index, (UIntPtr)a.Raw, IntPtr.Zero);
                        break;
                    case ArgKind.Buffer:
                    {
                        IntPtr ptr = Marshal.AllocHGlobal(IntPtr.Size);
                        try { Marshal.WriteIntPtr(ptr, (IntPtr)a.Raw); err = OpenClNativeBindings.SetKernelArg(_kernel, a.Index, (UIntPtr)IntPtr.Size, ptr); }
                        finally { Marshal.FreeHGlobal(ptr); }
                        break;
                    }
                    case ArgKind.Int32:
                    {
                        IntPtr ptr = Marshal.AllocHGlobal(sizeof(int));
                        try { Marshal.WriteInt32(ptr, (int)a.Raw); err = OpenClNativeBindings.SetKernelArg(_kernel, a.Index, (UIntPtr)sizeof(int), ptr); }
                        finally { Marshal.FreeHGlobal(ptr); }
                        break;
                    }
                    case ArgKind.Float:
                    {
                        IntPtr ptr = Marshal.AllocHGlobal(sizeof(float));
                        try { Marshal.Copy(new float[] { BitConverter.ToSingle(BitConverter.GetBytes((int)a.Raw), 0) }, 0, ptr, 1); err = OpenClNativeBindings.SetKernelArg(_kernel, a.Index, (UIntPtr)sizeof(float), ptr); }
                        finally { Marshal.FreeHGlobal(ptr); }
                        break;
                    }
                    default: // UInt64
                    {
                        IntPtr ptr = Marshal.AllocHGlobal(sizeof(ulong));
                        try { Marshal.WriteInt64(ptr, a.Raw); err = OpenClNativeBindings.SetKernelArg(_kernel, a.Index, (UIntPtr)sizeof(ulong), ptr); }
                        finally { Marshal.FreeHGlobal(ptr); }
                        break;
                    }
                }
                if (err != OpenClNativeBindings.CL_SUCCESS)
                {
                    pending.Clear();
                    throw new InvalidOperationException($"Failed to set kernel arg {a.Index}: {err}");
                }
            }
        }

        #endregion

        #region Execution

        /// <summary>
        /// Executes kernel with 1D work distribution.
        /// </summary>
        public void Execute1D(int globalSize, int localSize)
        {
            GpuLaunchProbe.OnLaunch();
            // Round up global size to multiple of local size
            int alignedGlobal = ((globalSize + localSize - 1) / localSize) * localSize;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobal };
            var localSizes = new UIntPtr[] { (UIntPtr)localSize };

            int err;
            lock (_submitLock)
            {
                ApplyPendingArgsLocked();
                err = OpenClNativeBindings.EnqueueNDRangeKernel(
                    _context.CommandQueue,
                    _kernel,
                    1, // work_dim
                    null, // global_work_offset
                    globalSizes,
                    localSizes,
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);
                _pendingArgs?.Clear();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to enqueue kernel: {err}");
        }

        /// <summary>
        /// Executes kernel with 2D work distribution.
        /// </summary>
        public void Execute2D(int globalSizeX, int globalSizeY, int localSizeX, int localSizeY)
        {
            GpuLaunchProbe.OnLaunch();
            // Round up global sizes to multiples of local sizes
            int alignedGlobalX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
            int alignedGlobalY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobalX, (UIntPtr)alignedGlobalY };
            var localSizes = new UIntPtr[] { (UIntPtr)localSizeX, (UIntPtr)localSizeY };

            int err;
            lock (_submitLock)
            {
                ApplyPendingArgsLocked();
                err = OpenClNativeBindings.EnqueueNDRangeKernel(
                    _context.CommandQueue,
                    _kernel,
                    2, // work_dim
                    null, // global_work_offset
                    globalSizes,
                    localSizes,
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);
                _pendingArgs?.Clear();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to enqueue kernel: {err}");
        }

        /// <summary>
        /// Executes kernel with 3D work distribution.
        /// </summary>
        public void Execute3D(int globalSizeX, int globalSizeY, int globalSizeZ, int localSizeX, int localSizeY, int localSizeZ)
        {
            GpuLaunchProbe.OnLaunch();
            // Round up global sizes to multiples of local sizes
            int alignedGlobalX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
            int alignedGlobalY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;
            int alignedGlobalZ = ((globalSizeZ + localSizeZ - 1) / localSizeZ) * localSizeZ;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobalX, (UIntPtr)alignedGlobalY, (UIntPtr)alignedGlobalZ };
            var localSizes = new UIntPtr[] { (UIntPtr)localSizeX, (UIntPtr)localSizeY, (UIntPtr)localSizeZ };

            int err;
            lock (_submitLock)
            {
                ApplyPendingArgsLocked();
                err = OpenClNativeBindings.EnqueueNDRangeKernel(
                    _context.CommandQueue,
                    _kernel,
                    3, // work_dim
                    null, // global_work_offset
                    globalSizes,
                    localSizes,
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);
                _pendingArgs?.Clear();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to enqueue kernel: {err}");
        }

        #endregion

        #region Stream-Specific Execution

        /// <summary>
        /// Executes kernel with 1D work distribution on a specific command queue.
        /// </summary>
        /// <param name="commandQueue">The command queue handle to execute on.</param>
        /// <param name="globalSize">The global work size.</param>
        /// <param name="localSize">The local work size.</param>
        public void Execute1DOnQueue(IntPtr commandQueue, int globalSize, int localSize)
        {
            GpuLaunchProbe.OnLaunch();
            // Round up global size to multiple of local size
            int alignedGlobal = ((globalSize + localSize - 1) / localSize) * localSize;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobal };
            var localSizes = new UIntPtr[] { (UIntPtr)localSize };

            int err;
            lock (_submitLock)
            {
                ApplyPendingArgsLocked();
                err = OpenClNativeBindings.EnqueueNDRangeKernel(
                    commandQueue,
                    _kernel,
                    1, // work_dim
                    null, // global_work_offset
                    globalSizes,
                    localSizes,
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);
                _pendingArgs?.Clear();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to enqueue kernel on queue: {err}");
        }

        /// <summary>
        /// Executes kernel with 2D work distribution on a specific command queue.
        /// </summary>
        /// <param name="commandQueue">The command queue handle to execute on.</param>
        /// <param name="globalSizeX">The global work size in X dimension.</param>
        /// <param name="globalSizeY">The global work size in Y dimension.</param>
        /// <param name="localSizeX">The local work size in X dimension.</param>
        /// <param name="localSizeY">The local work size in Y dimension.</param>
        public void Execute2DOnQueue(IntPtr commandQueue, int globalSizeX, int globalSizeY, int localSizeX, int localSizeY)
        {
            GpuLaunchProbe.OnLaunch();
            // Round up global sizes to multiples of local sizes
            int alignedGlobalX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
            int alignedGlobalY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobalX, (UIntPtr)alignedGlobalY };
            var localSizes = new UIntPtr[] { (UIntPtr)localSizeX, (UIntPtr)localSizeY };

            int err;
            lock (_submitLock)
            {
                ApplyPendingArgsLocked();
                err = OpenClNativeBindings.EnqueueNDRangeKernel(
                    commandQueue,
                    _kernel,
                    2, // work_dim
                    null, // global_work_offset
                    globalSizes,
                    localSizes,
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);
                _pendingArgs?.Clear();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to enqueue kernel on queue: {err}");
        }

        /// <summary>
        /// Executes kernel with 3D work distribution on a specific command queue.
        /// </summary>
        /// <param name="commandQueue">The command queue handle to execute on.</param>
        /// <param name="globalSizeX">The global work size in X dimension.</param>
        /// <param name="globalSizeY">The global work size in Y dimension.</param>
        /// <param name="globalSizeZ">The global work size in Z dimension.</param>
        /// <param name="localSizeX">The local work size in X dimension.</param>
        /// <param name="localSizeY">The local work size in Y dimension.</param>
        /// <param name="localSizeZ">The local work size in Z dimension.</param>
        public void Execute3DOnQueue(IntPtr commandQueue, int globalSizeX, int globalSizeY, int globalSizeZ,
            int localSizeX, int localSizeY, int localSizeZ)
        {
            GpuLaunchProbe.OnLaunch();
            // Round up global sizes to multiples of local sizes
            int alignedGlobalX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
            int alignedGlobalY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;
            int alignedGlobalZ = ((globalSizeZ + localSizeZ - 1) / localSizeZ) * localSizeZ;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobalX, (UIntPtr)alignedGlobalY, (UIntPtr)alignedGlobalZ };
            var localSizes = new UIntPtr[] { (UIntPtr)localSizeX, (UIntPtr)localSizeY, (UIntPtr)localSizeZ };

            int err;
            lock (_submitLock)
            {
                ApplyPendingArgsLocked();
                err = OpenClNativeBindings.EnqueueNDRangeKernel(
                    commandQueue,
                    _kernel,
                    3, // work_dim
                    null, // global_work_offset
                    globalSizes,
                    localSizes,
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);
                _pendingArgs?.Clear();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to enqueue kernel on queue: {err}");
        }

        #endregion

        #region Profiled Execution

        /// <summary>
        /// Executes kernel with 2D work distribution on the profiling queue and returns an event handle.
        /// The caller must release the event after getting profiling info.
        /// </summary>
        /// <returns>Event handle for profiling, or IntPtr.Zero if profiling is not available.</returns>
        public IntPtr Execute2DProfiled(int globalSizeX, int globalSizeY, int localSizeX, int localSizeY)
        {
            if (!_context.IsProfilingEnabled)
            {
                // Fall back to non-profiled execution
                Execute2D(globalSizeX, globalSizeY, localSizeX, localSizeY);
                return IntPtr.Zero;
            }

            // Round up global sizes to multiples of local sizes
            int alignedGlobalX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
            int alignedGlobalY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobalX, (UIntPtr)alignedGlobalY };
            var localSizes = new UIntPtr[] { (UIntPtr)localSizeX, (UIntPtr)localSizeY };

            // Allocate event handle
            IntPtr eventHandle = Marshal.AllocHGlobal(IntPtr.Size);
            try
            {
                int err;
                lock (_submitLock)
                {
                    ApplyPendingArgsLocked();
                    err = OpenClNativeBindings.EnqueueNDRangeKernel(
                        _context.ProfilingCommandQueue,
                        _kernel,
                        2, // work_dim
                        null, // global_work_offset
                        globalSizes,
                        localSizes,
                        0,
                        IntPtr.Zero,
                        eventHandle);
                    _pendingArgs?.Clear();
                }

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to enqueue kernel: {err}");

                // Read the event pointer from the allocated memory
                IntPtr eventPtr = Marshal.ReadIntPtr(eventHandle);
                return eventPtr;
            }
            finally
            {
                Marshal.FreeHGlobal(eventHandle);
            }
        }

        /// <summary>
        /// Executes kernel with 1D work distribution on the profiling queue and returns an event handle.
        /// </summary>
        public IntPtr Execute1DProfiled(int globalSize, int localSize)
        {
            if (!_context.IsProfilingEnabled)
            {
                Execute1D(globalSize, localSize);
                return IntPtr.Zero;
            }

            int alignedGlobal = ((globalSize + localSize - 1) / localSize) * localSize;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobal };
            var localSizes = new UIntPtr[] { (UIntPtr)localSize };

            IntPtr eventHandle = Marshal.AllocHGlobal(IntPtr.Size);
            try
            {
                int err;
                lock (_submitLock)
                {
                    ApplyPendingArgsLocked();
                    err = OpenClNativeBindings.EnqueueNDRangeKernel(
                        _context.ProfilingCommandQueue,
                        _kernel,
                        1,
                        null,
                        globalSizes,
                        localSizes,
                        0,
                        IntPtr.Zero,
                        eventHandle);
                    _pendingArgs?.Clear();
                }

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to enqueue kernel: {err}");

                return Marshal.ReadIntPtr(eventHandle);
            }
            finally
            {
                Marshal.FreeHGlobal(eventHandle);
            }
        }

        #endregion

        public void Dispose()
        {
            if (_disposed) return;

            if (_kernel != IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseKernel(_kernel);
                _kernel = IntPtr.Zero;
            }

            _disposed = true;
        }
    }
}
