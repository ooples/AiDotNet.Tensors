// Copyright (c) AiDotNet. All rights reserved.
// Pure P/Invoke OpenCL buffer - no managed GPU runtime dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// Helper for issue #285 — validates a requested allocation size against
    /// the device's <c>CL_DEVICE_MAX_MEM_ALLOC_SIZE</c> before calling
    /// <c>clCreateBuffer</c>. Allocations above the cap return
    /// <c>CL_INVALID_BUFFER_SIZE (-61)</c> with no further information; this
    /// throws a typed <see cref="GpuBufferTooLargeException"/> so dispatch
    /// shims can catch it and fall back to a CPU path or chunked execution.
    /// </summary>
    internal static class DirectOpenClBufferGuards
    {
        internal static void EnsureFits(DirectOpenClContext context, long requestedBytes)
        {
            ulong cap = context.MaxMemAllocSize;
            // MaxMemAllocSize == 0 means the query failed at context init —
            // skip the guard rather than wrongly reject every allocation. The
            // underlying clCreateBuffer call will still surface any error.
            GpuBufferSizeGuard.EnsureFits(
                backend: "OpenCL",
                requestedBytes: requestedBytes,
                deviceCap: cap > 0 ? (long)cap : 0,
                deviceName: !string.IsNullOrEmpty(context.DeviceBoardName)
                    ? context.DeviceBoardName
                    : context.DeviceName);
        }
    }

    internal static class DirectOpenClHostTransfer
    {
        internal static void WaitAndRelease(IntPtr completionEvent, string operation)
        {
            if (completionEvent == IntPtr.Zero)
                throw new InvalidOperationException($"{operation} did not return a completion event.");
            try
            {
                int waitError = OpenClNativeBindings.WaitForEvents(1, new[] { completionEvent });
                if (waitError != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to wait for {operation}: {waitError}");
            }
            finally
            {
                OpenClNativeBindings.ReleaseEvent(completionEvent);
            }
        }
    }

    /// <summary>
    /// OpenCL buffer wrapper using pure P/Invoke. No managed GPU runtime dependency.
    /// </summary>
    internal sealed class DirectOpenClBuffer : IDisposable, IDirectOpenClMemoryObject
    {
        private IntPtr _buffer;
        private readonly DirectOpenClContext _context;
        private readonly int _length;

        /// <summary>Bytes this buffer holds on the device, for residency accounting.</summary>
        private long ByteSize => (long)_length * sizeof(float);
        private bool _disposed;

        public IntPtr Handle => _buffer;
        public int Length => _length;
        public IntPtr NativeHandle => _buffer;
        public DirectOpenClContext OwningContext => _context;
        public IntPtr LastSubmissionQueue { get; set; }

        /// <summary>
        /// Creates a buffer and uploads data from host.
        /// </summary>
        public DirectOpenClBuffer(DirectOpenClContext context, float[] data)
        {
            _context = context;
            _length = data.Length;
            DirectOpenClBufferGuards.EnsureFits(context, (long)data.Length * sizeof(float));

            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                _buffer = OpenClNativeBindings.CreateBuffer(
                    context.Context,
                    OpenClNativeBindings.CL_MEM_READ_WRITE | OpenClNativeBindings.CL_MEM_COPY_HOST_PTR,
                    (UIntPtr)(data.Length * sizeof(float)),
                    handle.AddrOfPinnedObject(),
                    out int err);

                if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                    throw new InvalidOperationException($"Failed to create OpenCL buffer: {err}");

                GpuKernelDiagnostics.RecordBufferAllocated(ByteSize);
                _context.RegisterMemoryObject(this);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Creates an empty buffer of specified size.
        /// </summary>
        public DirectOpenClBuffer(DirectOpenClContext context, int size)
        {
            _context = context;
            _length = size;
            DirectOpenClBufferGuards.EnsureFits(context, (long)size * sizeof(float));

            _buffer = OpenClNativeBindings.CreateBuffer(
                context.Context,
                OpenClNativeBindings.CL_MEM_READ_WRITE,
                (UIntPtr)(size * sizeof(float)),
                IntPtr.Zero,
                out int err);

            if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL buffer: {err}");

            GpuKernelDiagnostics.RecordBufferAllocated(ByteSize);
            _context.RegisterMemoryObject(this);
        }

        /// <summary>
        /// Downloads buffer contents to a new array.
        /// </summary>
        public float[] ToArray()
        {
            var result = new float[_length];
            CopyToHost(result);
            return result;
        }

        /// <summary>
        /// Downloads buffer contents to existing array.
        /// </summary>
        public void CopyToHost(float[] destination)
        {
            if (destination.Length < _length)
                throw new ArgumentException("Destination array too small");

            GCHandle handle = GCHandle.Alloc(destination, GCHandleType.Pinned);
            IntPtr transferEvent = IntPtr.Zero;
            try
            {
                IntPtr queue = _context.CommandQueue;
                var memories = DirectOpenClSubmission.GetDirectSubmissionMemories(this);
                int err;
                try
                {
                    lock (DirectOpenClSubmission.Gate)
                    {
                        using var waits = DirectOpenClSubmission.PrepareLocked(queue, memories);
                        err = OpenClNativeBindings.EnqueueReadBufferWithEvent(
                            queue, _buffer, 0, UIntPtr.Zero,
                            (UIntPtr)(_length * sizeof(float)), handle.AddrOfPinnedObject(),
                            waits.Count, waits.Pointer, out transferEvent);
                        if (err == OpenClNativeBindings.CL_SUCCESS)
                            DirectOpenClSubmission.CommitLocked(queue, memories);
                    }
                }
                finally
                {
                    DirectOpenClSubmission.ReleaseDirectSubmissionMemories(memories);
                }

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to read OpenCL buffer: {err}");
                IntPtr completionEvent = transferEvent;
                transferEvent = IntPtr.Zero;
                DirectOpenClHostTransfer.WaitAndRelease(completionEvent, "OpenCL buffer read");
            }
            finally
            {
                try
                {
                    if (transferEvent != IntPtr.Zero)
                        DirectOpenClHostTransfer.WaitAndRelease(transferEvent, "OpenCL buffer read cleanup");
                }
                finally { handle.Free(); }
            }
        }

        /// <summary>
        /// Uploads data to buffer.
        /// </summary>
        public void CopyFromHost(float[] source)
        {
            if (source.Length > _length)
                throw new ArgumentException("Source array too large");

            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr transferEvent = IntPtr.Zero;
            try
            {
                IntPtr queue = _context.CommandQueue;
                var memories = DirectOpenClSubmission.GetDirectSubmissionMemories(this);
                int err;
                try
                {
                    lock (DirectOpenClSubmission.Gate)
                    {
                        using var waits = DirectOpenClSubmission.PrepareLocked(queue, memories);
                        err = OpenClNativeBindings.EnqueueWriteBufferWithEvent(
                            queue, _buffer, 0, UIntPtr.Zero,
                            (UIntPtr)(source.Length * sizeof(float)), handle.AddrOfPinnedObject(),
                            waits.Count, waits.Pointer, out transferEvent);
                        if (err == OpenClNativeBindings.CL_SUCCESS)
                            DirectOpenClSubmission.CommitLocked(queue, memories);
                    }
                }
                finally
                {
                    DirectOpenClSubmission.ReleaseDirectSubmissionMemories(memories);
                }

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to write OpenCL buffer: {err}");
                IntPtr completionEvent = transferEvent;
                transferEvent = IntPtr.Zero;
                DirectOpenClHostTransfer.WaitAndRelease(completionEvent, "OpenCL buffer write");
            }
            finally
            {
                try
                {
                    if (transferEvent != IntPtr.Zero)
                        DirectOpenClHostTransfer.WaitAndRelease(transferEvent, "OpenCL buffer write cleanup");
                }
                finally { handle.Free(); }
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            if (_buffer != IntPtr.Zero)
            {
                IntPtr memoryObject = _buffer;
                _buffer = IntPtr.Zero;
                _context.RetireMemoryObject(this, memoryObject, ByteSize);
            }

            _disposed = true;
        }
    }

    /// <summary>
    /// OpenCL byte buffer wrapper using pure P/Invoke.
    /// Used for storing packed sparse indices (1 byte per group of 4 elements).
    /// </summary>
    internal sealed class DirectOpenClByteBuffer : IDisposable, IDirectOpenClMemoryObject
    {
        private IntPtr _buffer;
        private readonly DirectOpenClContext _context;
        private readonly int _length;

        /// <summary>Bytes this buffer holds on the device, for residency accounting.</summary>
        private long ByteSize => (long)_length * 1;
        private bool _disposed;

        public IntPtr Handle => _buffer;
        public int Length => _length;
        public IntPtr NativeHandle => _buffer;
        public DirectOpenClContext OwningContext => _context;
        public IntPtr LastSubmissionQueue { get; set; }

        /// <summary>
        /// Creates a byte buffer and uploads data from host.
        /// </summary>
        public DirectOpenClByteBuffer(DirectOpenClContext context, byte[] data)
        {
            _context = context;
            _length = data.Length;
            DirectOpenClBufferGuards.EnsureFits(context, data.Length);

            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                _buffer = OpenClNativeBindings.CreateBuffer(
                    context.Context,
                    OpenClNativeBindings.CL_MEM_READ_WRITE | OpenClNativeBindings.CL_MEM_COPY_HOST_PTR,
                    (UIntPtr)data.Length,
                    handle.AddrOfPinnedObject(),
                    out int err);

                if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                    throw new InvalidOperationException($"Failed to create OpenCL byte buffer: {err}");

                GpuKernelDiagnostics.RecordBufferAllocated(ByteSize);
                _context.RegisterMemoryObject(this);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Creates an empty byte buffer of specified size.
        /// </summary>
        public DirectOpenClByteBuffer(DirectOpenClContext context, int size)
        {
            _context = context;
            _length = size;
            DirectOpenClBufferGuards.EnsureFits(context, size);

            _buffer = OpenClNativeBindings.CreateBuffer(
                context.Context,
                OpenClNativeBindings.CL_MEM_READ_WRITE,
                (UIntPtr)size,
                IntPtr.Zero,
                out int err);

            if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL byte buffer: {err}");

            GpuKernelDiagnostics.RecordBufferAllocated(ByteSize);
            _context.RegisterMemoryObject(this);
        }

        /// <summary>
        /// Downloads buffer contents to a new array.
        /// </summary>
        public byte[] ToArray()
        {
            var result = new byte[_length];
            CopyToHost(result);
            return result;
        }

        /// <summary>
        /// Downloads buffer contents to existing array.
        /// </summary>
        public void CopyToHost(byte[] destination)
        {
            if (destination.Length < _length)
                throw new ArgumentException("Destination array too small");

            GCHandle handle = GCHandle.Alloc(destination, GCHandleType.Pinned);
            IntPtr transferEvent = IntPtr.Zero;
            try
            {
                IntPtr queue = _context.CommandQueue;
                var memories = DirectOpenClSubmission.GetDirectSubmissionMemories(this);
                int err;
                try
                {
                    lock (DirectOpenClSubmission.Gate)
                    {
                        using var waits = DirectOpenClSubmission.PrepareLocked(queue, memories);
                        err = OpenClNativeBindings.EnqueueReadBufferWithEvent(
                            queue, _buffer, 0, UIntPtr.Zero, (UIntPtr)_length,
                            handle.AddrOfPinnedObject(), waits.Count, waits.Pointer, out transferEvent);
                        if (err == OpenClNativeBindings.CL_SUCCESS)
                            DirectOpenClSubmission.CommitLocked(queue, memories);
                    }
                }
                finally
                {
                    DirectOpenClSubmission.ReleaseDirectSubmissionMemories(memories);
                }

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to read OpenCL byte buffer: {err}");
                IntPtr completionEvent = transferEvent;
                transferEvent = IntPtr.Zero;
                DirectOpenClHostTransfer.WaitAndRelease(completionEvent, "OpenCL byte-buffer read");
            }
            finally
            {
                try
                {
                    if (transferEvent != IntPtr.Zero)
                        DirectOpenClHostTransfer.WaitAndRelease(transferEvent, "OpenCL byte-buffer read cleanup");
                }
                finally { handle.Free(); }
            }
        }

        /// <summary>
        /// Uploads byte contents from an existing array.
        /// </summary>
        public void CopyFromHost(byte[] source)
        {
            if (source.Length > _length)
                throw new ArgumentException("Source array too large");
            if (source.Length == 0)
                return;

            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr transferEvent = IntPtr.Zero;
            try
            {
                IntPtr queue = _context.CommandQueue;
                var memories = DirectOpenClSubmission.GetDirectSubmissionMemories(this);
                int err;
                try
                {
                    lock (DirectOpenClSubmission.Gate)
                    {
                        using var waits = DirectOpenClSubmission.PrepareLocked(queue, memories);
                        err = OpenClNativeBindings.EnqueueWriteBufferWithEvent(
                            queue, _buffer, 0, UIntPtr.Zero, (UIntPtr)source.Length,
                            handle.AddrOfPinnedObject(), waits.Count, waits.Pointer, out transferEvent);
                        if (err == OpenClNativeBindings.CL_SUCCESS)
                            DirectOpenClSubmission.CommitLocked(queue, memories);
                    }
                }
                finally
                {
                    DirectOpenClSubmission.ReleaseDirectSubmissionMemories(memories);
                }

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to write OpenCL byte buffer: {err}");
                IntPtr completionEvent = transferEvent;
                transferEvent = IntPtr.Zero;
                DirectOpenClHostTransfer.WaitAndRelease(completionEvent, "OpenCL byte-buffer write");
            }
            finally
            {
                try
                {
                    if (transferEvent != IntPtr.Zero)
                        DirectOpenClHostTransfer.WaitAndRelease(transferEvent, "OpenCL byte-buffer write cleanup");
                }
                finally { handle.Free(); }
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            if (_buffer != IntPtr.Zero)
            {
                IntPtr memoryObject = _buffer;
                _buffer = IntPtr.Zero;
                _context.RetireMemoryObject(this, memoryObject, ByteSize);
            }

            _disposed = true;
        }
    }

    /// <summary>
    /// OpenCL int buffer wrapper using pure P/Invoke.
    /// Used for atomic counters in work-stealing kernels.
    /// </summary>
    internal sealed class DirectOpenClIntBuffer : IDisposable, IDirectOpenClMemoryObject
    {
        private IntPtr _buffer;
        private readonly DirectOpenClContext _context;
        private readonly int _length;

        /// <summary>Bytes this buffer holds on the device, for residency accounting.</summary>
        private long ByteSize => (long)_length * sizeof(int);
        private bool _disposed;

        public IntPtr Handle => _buffer;
        public int Length => _length;
        public IntPtr NativeHandle => _buffer;
        public DirectOpenClContext OwningContext => _context;
        public IntPtr LastSubmissionQueue { get; set; }

        /// <summary>
        /// Creates an int buffer and uploads data from host.
        /// </summary>
        public DirectOpenClIntBuffer(DirectOpenClContext context, int[] data)
        {
            _context = context;
            _length = data.Length;
            DirectOpenClBufferGuards.EnsureFits(context, (long)data.Length * sizeof(int));

            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                _buffer = OpenClNativeBindings.CreateBuffer(
                    context.Context,
                    OpenClNativeBindings.CL_MEM_READ_WRITE | OpenClNativeBindings.CL_MEM_COPY_HOST_PTR,
                    (UIntPtr)(data.Length * sizeof(int)),
                    handle.AddrOfPinnedObject(),
                    out int err);

                if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                    throw new InvalidOperationException($"Failed to create OpenCL int buffer: {err}");

                GpuKernelDiagnostics.RecordBufferAllocated(ByteSize);
                _context.RegisterMemoryObject(this);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Creates an empty int buffer of specified size.
        /// </summary>
        public DirectOpenClIntBuffer(DirectOpenClContext context, int size)
        {
            _context = context;
            _length = size;
            DirectOpenClBufferGuards.EnsureFits(context, (long)size * sizeof(int));

            _buffer = OpenClNativeBindings.CreateBuffer(
                context.Context,
                OpenClNativeBindings.CL_MEM_READ_WRITE,
                (UIntPtr)(size * sizeof(int)),
                IntPtr.Zero,
                out int err);

            if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL int buffer: {err}");

            GpuKernelDiagnostics.RecordBufferAllocated(ByteSize);
            _context.RegisterMemoryObject(this);
        }

        /// <summary>
        /// Downloads buffer contents to a new array.
        /// </summary>
        public int[] ToArray()
        {
            var result = new int[_length];
            CopyToHost(result);
            return result;
        }

        /// <summary>
        /// Downloads buffer contents to existing array.
        /// </summary>
        public void CopyToHost(int[] destination)
        {
            if (destination.Length < _length)
                throw new ArgumentException("Destination array too small");

            GCHandle handle = GCHandle.Alloc(destination, GCHandleType.Pinned);
            IntPtr transferEvent = IntPtr.Zero;
            try
            {
                IntPtr queue = _context.CommandQueue;
                var memories = DirectOpenClSubmission.GetDirectSubmissionMemories(this);
                int err;
                try
                {
                    lock (DirectOpenClSubmission.Gate)
                    {
                        using var waits = DirectOpenClSubmission.PrepareLocked(queue, memories);
                        err = OpenClNativeBindings.EnqueueReadBufferWithEvent(
                            queue, _buffer, 0, UIntPtr.Zero,
                            (UIntPtr)(_length * sizeof(int)), handle.AddrOfPinnedObject(),
                            waits.Count, waits.Pointer, out transferEvent);
                        if (err == OpenClNativeBindings.CL_SUCCESS)
                            DirectOpenClSubmission.CommitLocked(queue, memories);
                    }
                }
                finally
                {
                    DirectOpenClSubmission.ReleaseDirectSubmissionMemories(memories);
                }

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to read OpenCL int buffer: {err}");
                IntPtr completionEvent = transferEvent;
                transferEvent = IntPtr.Zero;
                DirectOpenClHostTransfer.WaitAndRelease(completionEvent, "OpenCL int-buffer read");
            }
            finally
            {
                try
                {
                    if (transferEvent != IntPtr.Zero)
                        DirectOpenClHostTransfer.WaitAndRelease(transferEvent, "OpenCL int-buffer read cleanup");
                }
                finally { handle.Free(); }
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            if (_buffer != IntPtr.Zero)
            {
                IntPtr memoryObject = _buffer;
                _buffer = IntPtr.Zero;
                _context.RetireMemoryObject(this, memoryObject, ByteSize);
            }

            _disposed = true;
        }
    }
}
