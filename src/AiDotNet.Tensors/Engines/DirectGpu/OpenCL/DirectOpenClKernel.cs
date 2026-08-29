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
        private readonly string _kernelName;
        private bool _disposed;

        public IntPtr Handle => _kernel;

        public DirectOpenClKernel(DirectOpenClContext context, DirectOpenClProgram program, string kernelName)
        {
            _context = context;
            _kernelName = kernelName;

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

        /// <summary>Which kernel the thread's staged args belong to.</summary>
        /// <remarks>
        /// The pending list is per-thread but shared across every kernel, and it was cleared only on
        /// the success path of Execute. So any throw between SetArg and Execute -- a guard rejecting
        /// an argument, an enqueue failure, an abandoned call -- left those args staged, and the NEXT
        /// kernel this thread executed applied them: another kernel's signature, at another kernel's
        /// arg indices, carrying buffer handles that may since have been disposed. Passing a released
        /// cl_mem to clSetKernelArg faults inside the driver, which is what an 0xC0000005 in
        /// SetKernelArg looks like from managed code.
        /// </remarks>
        [ThreadStatic] private static DirectOpenClKernel? _pendingOwner;

        // Serializes the apply-args + enqueue critical section across ALL kernels
        // (the shared cl_kernel arg state and the shared command queue both require it).
        private static readonly object _submitLock = new object();

        private static System.Collections.Generic.List<PendingArg> Pending
            => _pendingArgs ??= new System.Collections.Generic.List<PendingArg>(8);

        /// <summary>Begins (or continues) staging for <paramref name="owner"/>, dropping any args
        /// abandoned by a previous kernel on this thread.</summary>
        private static System.Collections.Generic.List<PendingArg> Stage(DirectOpenClKernel owner, uint index)
        {
            var pending = Pending;

            // Index 0 starts a launch sequence -- every launcher sets its arguments from 0 upwards
            // -- so this is where a previous sequence's leftovers are dropped. That makes each
            // launch self-cleaning even when the last one threw between SetArg and Execute, without
            // depending on the clear at the end of Execute having been reached.
            if (index == 0 || !ReferenceEquals(_pendingOwner, owner))
            {
                pending.Clear();
                _pendingOwner = owner;
            }

            return pending;
        }

        /// <summary>Drops the thread's staged args; safe to call from a finally.</summary>
        private static void DiscardPendingArgs()
        {
            _pendingArgs?.Clear();
            _pendingOwner = null;
        }

        #region SetArg Overloads

        public void SetArg(uint index, IntPtr bufferHandle)
            => Stage(this, index).Add(new PendingArg(index, ArgKind.Buffer, (long)bufferHandle));

        public void SetArg(uint index, int value)
            => Stage(this, index).Add(new PendingArg(index, ArgKind.Int32, value));

        public void SetArg(uint index, float value)
            => Stage(this, index).Add(new PendingArg(index, ArgKind.Float, BitConverter.ToInt32(BitConverter.GetBytes(value), 0)));

        public void SetArg(uint index, ulong value)
            => Stage(this, index).Add(new PendingArg(index, ArgKind.UInt64, unchecked((long)value)));

        /// <summary>
        /// Sets a local memory argument (for shared memory allocation).
        /// </summary>
        public void SetLocalArg(uint index, int sizeInBytes)
            => Stage(this, index).Add(new PendingArg(index, ArgKind.Local, sizeInBytes));

        // Applies the per-thread pending args to the shared kernel. MUST be called
        // while holding _submitLock and immediately before the matching enqueue.
        private void ApplyPendingArgsLocked()
        {
            ThrowIfUnusable();

            var pending = _pendingArgs;
            if (pending == null) return;

            // Never apply args staged for a different kernel. Indices and sizes belong to that
            // kernel's signature, not this one.
            if (!ReferenceEquals(_pendingOwner, this))
            {
                pending.Clear();
                return;
            }
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
            ThrowIfUnusable();
            GpuLaunchProbe.OnLaunch();
            // Round up global size to multiple of local size
            int alignedGlobal = ((globalSize + localSize - 1) / localSize) * localSize;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobal };
            var localSizes = new UIntPtr[] { (UIntPtr)localSize };

            BeginLaunch(TotalOf(globalSizes), TotalOf(localSizes));

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
                DiscardPendingArgs();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
            {
                throw new InvalidOperationException(
                    $"Failed to enqueue kernel '{_kernelName}': {err}. Recent launches, most recent "
                        + "last: " + Environment.NewLine + GpuKernelDiagnostics.DescribeRecentLaunches());
            }

            EndLaunch();
        }

        /// <summary>
        /// Executes kernel with 2D work distribution.
        /// </summary>
        public void Execute2D(int globalSizeX, int globalSizeY, int localSizeX, int localSizeY)
        {
            ThrowIfUnusable();
            GpuLaunchProbe.OnLaunch();
            // Round up global sizes to multiples of local sizes
            int alignedGlobalX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
            int alignedGlobalY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobalX, (UIntPtr)alignedGlobalY };
            var localSizes = new UIntPtr[] { (UIntPtr)localSizeX, (UIntPtr)localSizeY };

            BeginLaunch(TotalOf(globalSizes), TotalOf(localSizes));

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
                DiscardPendingArgs();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
            {
                throw new InvalidOperationException(
                    $"Failed to enqueue kernel '{_kernelName}': {err}. Recent launches, most recent "
                        + "last: " + Environment.NewLine + GpuKernelDiagnostics.DescribeRecentLaunches());
            }

            EndLaunch();
        }

        /// <summary>
        /// Executes kernel with 3D work distribution.
        /// </summary>
        public void Execute3D(int globalSizeX, int globalSizeY, int globalSizeZ, int localSizeX, int localSizeY, int localSizeZ)
        {
            ThrowIfUnusable();
            GpuLaunchProbe.OnLaunch();
            // Round up global sizes to multiples of local sizes
            int alignedGlobalX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
            int alignedGlobalY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;
            int alignedGlobalZ = ((globalSizeZ + localSizeZ - 1) / localSizeZ) * localSizeZ;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobalX, (UIntPtr)alignedGlobalY, (UIntPtr)alignedGlobalZ };
            var localSizes = new UIntPtr[] { (UIntPtr)localSizeX, (UIntPtr)localSizeY, (UIntPtr)localSizeZ };

            BeginLaunch(TotalOf(globalSizes), TotalOf(localSizes));

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
                DiscardPendingArgs();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
            {
                throw new InvalidOperationException(
                    $"Failed to enqueue kernel '{_kernelName}': {err}. Recent launches, most recent "
                        + "last: " + Environment.NewLine + GpuKernelDiagnostics.DescribeRecentLaunches());
            }

            EndLaunch();
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
            ThrowIfUnusable();
            GpuLaunchProbe.OnLaunch();
            // Round up global size to multiple of local size
            int alignedGlobal = ((globalSize + localSize - 1) / localSize) * localSize;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobal };
            var localSizes = new UIntPtr[] { (UIntPtr)localSize };

            BeginLaunch(TotalOf(globalSizes), TotalOf(localSizes));

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
                DiscardPendingArgs();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to enqueue kernel on queue: {err}");

            // Finish THIS queue, not the default one. Under AIDOTNET_GPU_SYNC_LAUNCHES the whole
            // point is that a device fault is attributed to the launch that caused it; a path that
            // enqueues and returns without finishing leaves its faults asynchronous and blames
            // whatever synchronises next, which is the behaviour the flag exists to remove.
            EndLaunch(commandQueue);
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
            ThrowIfUnusable();
            GpuLaunchProbe.OnLaunch();
            // Round up global sizes to multiples of local sizes
            int alignedGlobalX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
            int alignedGlobalY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobalX, (UIntPtr)alignedGlobalY };
            var localSizes = new UIntPtr[] { (UIntPtr)localSizeX, (UIntPtr)localSizeY };

            BeginLaunch(TotalOf(globalSizes), TotalOf(localSizes));

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
                DiscardPendingArgs();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to enqueue kernel on queue: {err}");

            // Finish THIS queue, not the default one. Under AIDOTNET_GPU_SYNC_LAUNCHES the whole
            // point is that a device fault is attributed to the launch that caused it; a path that
            // enqueues and returns without finishing leaves its faults asynchronous and blames
            // whatever synchronises next, which is the behaviour the flag exists to remove.
            EndLaunch(commandQueue);
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
            ThrowIfUnusable();
            GpuLaunchProbe.OnLaunch();
            // Round up global sizes to multiples of local sizes
            int alignedGlobalX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
            int alignedGlobalY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;
            int alignedGlobalZ = ((globalSizeZ + localSizeZ - 1) / localSizeZ) * localSizeZ;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobalX, (UIntPtr)alignedGlobalY, (UIntPtr)alignedGlobalZ };
            var localSizes = new UIntPtr[] { (UIntPtr)localSizeX, (UIntPtr)localSizeY, (UIntPtr)localSizeZ };

            BeginLaunch(TotalOf(globalSizes), TotalOf(localSizes));

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
                DiscardPendingArgs();
            }

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to enqueue kernel on queue: {err}");

            // Finish THIS queue, not the default one. Under AIDOTNET_GPU_SYNC_LAUNCHES the whole
            // point is that a device fault is attributed to the launch that caused it; a path that
            // enqueues and returns without finishing leaves its faults asynchronous and blames
            // whatever synchronises next, which is the behaviour the flag exists to remove.
            EndLaunch(commandQueue);
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

            BeginLaunch(TotalOf(globalSizes), TotalOf(localSizes));

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
                    DiscardPendingArgs();
                }

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to enqueue kernel: {err}");

                // The enqueue created the event, so from here on WE own it. If the synchronise
                // throws -- which is exactly what sync-launch mode exists to make happen at the
                // faulting launch -- we never return, so the caller never receives the handle and
                // can never release it. Release it before the exception leaves.
                try
                {
                    EndLaunch(_context.ProfilingCommandQueue);
                }
                catch
                {
                    IntPtr orphaned = Marshal.ReadIntPtr(eventHandle);
                    if (orphaned != IntPtr.Zero) OpenClNativeBindings.ReleaseEvent(orphaned);
                    throw;
                }

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

            BeginLaunch(TotalOf(globalSizes), TotalOf(localSizes));

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
                    DiscardPendingArgs();
                }

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to enqueue kernel: {err}");

                // The enqueue created the event, so from here on WE own it. If the synchronise
                // throws -- which is exactly what sync-launch mode exists to make happen at the
                // faulting launch -- we never return, so the caller never receives the handle and
                // can never release it. Release it before the exception leaves.
                try
                {
                    EndLaunch(_context.ProfilingCommandQueue);
                }
                catch
                {
                    IntPtr orphaned = Marshal.ReadIntPtr(eventHandle);
                    if (orphaned != IntPtr.Zero) OpenClNativeBindings.ReleaseEvent(orphaned);
                    throw;
                }

                return Marshal.ReadIntPtr(eventHandle);
            }
            finally
            {
                Marshal.FreeHGlobal(eventHandle);
            }
        }

        #endregion

        /// <summary>The kernel's own argument count, queried once. -1 when unavailable.</summary>
        private int _declaredArgCount = -2;   // -2 = not yet queried, -1 = device would not say

        private int DeclaredArgCount
        {
            get
            {
                if (_declaredArgCount == -2)
                {
                    _declaredArgCount = OpenClNativeBindings.GetKernelNumArgs(_kernel);
                }

                return _declaredArgCount;
            }
        }

        /// <summary>
        /// Validates the launch, records it in the diagnostics journal, and -- under
        /// AIDOTNET_GPU_SYNC_LAUNCHES -- finishes the queue so an asynchronous fault is attributed
        /// to THIS launch rather than to whatever synchronises next.
        /// </summary>
        private void BeginLaunch(long globalSize, long localSize)
        {
            int staged = _pendingArgs is null || !ReferenceEquals(_pendingOwner, this)
                ? 0
                : _pendingArgs.Count;

            GpuKernelDiagnostics.ValidateLaunch(
                _kernelName,
                handleIsValid: !_disposed && _kernel != IntPtr.Zero,
                stagedArgCount: staged,
                declaredArgCount: DeclaredArgCount,
                globalSize: globalSize,
                localSize: localSize);

            GpuKernelDiagnostics.RecordLaunch(_kernelName, globalSize, localSize, staged);
        }

        /// <summary>Finishes the queue when synchronous diagnostics are on, so a device fault is
        /// reported against the launch that caused it.</summary>
        /// <summary>Work-item count across however many dimensions the launch uses.</summary>
        private static long TotalOf(UIntPtr[] sizes)
        {
            long total = 1;
            foreach (var size in sizes) total *= (long)(ulong)size;
            return total;
        }

        private void EndLaunch() => EndLaunch(_context.CommandQueue);

        /// <summary>
        /// Finishes THE QUEUE THIS LAUNCH USED. The OnQueue and profiled paths submit to a different
        /// queue than <c>_context.CommandQueue</c>, so finishing the default one would attribute a
        /// fault to the wrong work — or miss it entirely.
        /// </summary>
        private void EndLaunch(IntPtr commandQueue)
        {
            if (!GpuKernelDiagnostics.SynchronousLaunches) return;

            int err = OpenClNativeBindings.Finish(commandQueue);
            if (err != OpenClNativeBindings.CL_SUCCESS)
            {
                throw new InvalidOperationException(
                    $"GPU fault after launching '{_kernelName}' (clFinish returned {err}). Recent "
                        + "launches, most recent last: " + Environment.NewLine
                        + GpuKernelDiagnostics.DescribeRecentLaunches());
            }
        }

        /// <summary>Refuses to launch a kernel whose handle has been released.</summary>
        /// <remarks>
        /// Dispose sets _kernel to IntPtr.Zero, and nothing on the launch path checked it, so a
        /// kernel used after its backend was disposed handed NULL to clSetKernelArg. The OpenCL spec
        /// says that returns CL_INVALID_KERNEL; drivers are not obliged to be careful about it, and
        /// a dereference inside the runtime surfaces as an 0xC0000005 that kills the process with no
        /// managed frame to blame. This turns that into a named, catchable failure that says which
        /// kernel and can be attributed to a test.
        /// </remarks>
        private void ThrowIfUnusable()
        {
            if (_disposed || _kernel == IntPtr.Zero)
            {
                throw new ObjectDisposedException(
                    nameof(DirectOpenClKernel),
                    $"OpenCL kernel '{_kernelName}' was used after its handle was released. The "
                        + "backend that owns it has been disposed while something still held a "
                        + "reference to this kernel.");
            }
        }

        public void Dispose()
        {
            // Released under the SUBMIT LOCK. ThrowIfUnusable is otherwise a time-of-check to
            // time-of-use window: a launch validates the handle, Dispose calls clReleaseKernel, and
            // the launch then hands the released handle to clSetKernelArg. Taking the same lock the
            // apply-and-enqueue critical section holds makes release and submission mutually
            // exclusive, so a validated handle stays valid until its enqueue completes.
            lock (_submitLock)
            {
                DisposeLocked();
            }
        }

        private void DisposeLocked()
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
