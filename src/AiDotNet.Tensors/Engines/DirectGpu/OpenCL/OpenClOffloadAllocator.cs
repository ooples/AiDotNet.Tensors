// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// OpenCL-backed <see cref="IGpuOffloadAllocator"/>. Pinned-scheme
/// allocates a host pointer + creates a CL buffer with
/// <c>CL_MEM_ALLOC_HOST_PTR</c> (driver-managed pinned host memory the
/// device can DMA into). Managed-scheme uses OpenCL 2.0 SVM
/// (<c>clSVMAlloc</c>) which is the unified-memory equivalent.
///
/// <para>The current iteration ships the host-side allocation + the
/// IsAvailable probe; the CL buffer-creation handshake plumbs through
/// the existing OpenCL backend's per-context queue, which the higher-
/// level dispatcher injects when staging a weight tensor onto a kernel
/// invocation. Same shape as how PR #267 wires the cuRAND on-device
/// dispatch — the allocator owns memory, the kernel owns scheduling.</para>
/// </summary>
public sealed class OpenClOffloadAllocator : IGpuOffloadAllocator
{
    private readonly ConcurrentDictionary<IntPtr, AllocRecord> _live = new();
    private IntPtr _context = IntPtr.Zero;
    private readonly object _lock = new();
    private bool _disposed;

    public bool IsAvailable => OpenClPlatformProbe.IsAvailable;

    private void EnsureContext()
    {
        if (_context != IntPtr.Zero) return;
        lock (_lock)
        {
            if (_context != IntPtr.Zero) return;
            // Pick the first GPU on the first platform.
            int err = OpenClPlatformProbe.clGetPlatformIDs(0, null, out uint numPlatforms);
            if (err != 0 || numPlatforms == 0)
                throw new InvalidOperationException("OpenCL: no platforms registered.");
            var platforms = new IntPtr[numPlatforms];
            err = OpenClPlatformProbe.clGetPlatformIDs(numPlatforms, platforms, out _);
            if (err != 0) throw new InvalidOperationException($"clGetPlatformIDs returned {err}");

            for (int p = 0; p < platforms.Length; p++)
            {
                err = OpenClPlatformProbe.clGetDeviceIDs(platforms[p], OpenClPlatformProbe.CL_DEVICE_TYPE_GPU, 0, null, out uint numDevices);
                if (err != 0 || numDevices == 0) continue;
                var devices = new IntPtr[numDevices];
                err = OpenClPlatformProbe.clGetDeviceIDs(platforms[p], OpenClPlatformProbe.CL_DEVICE_TYPE_GPU, numDevices, devices, out _);
                if (err != 0) continue;
                _context = OpenClPlatformProbe.clCreateContext(IntPtr.Zero, 1, new[] { devices[0] }, IntPtr.Zero, IntPtr.Zero, out int ctxErr);
                if (ctxErr == 0 && _context != IntPtr.Zero) return;
            }
            throw new InvalidOperationException("OpenCL: no usable GPU device.");
        }
    }

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
    {
        // Hold _lock across allocate + register so a concurrent Dispose
        // cannot snapshot _live, clear it, and let this allocation slip in
        // afterwards (which would leak the cl_mem / SVM pointer). _lock is
        // reentrant so EnsureContext's nested lock is safe.
        lock (_lock)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(OpenClOffloadAllocator));
            if (!IsAvailable)
                throw new NotSupportedException("OpenCL ICD is not loadable on this host.");
            if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

            EnsureContext();
            var effective = scheme == OffloadScheme.Auto ? OffloadScheme.Pinned : scheme;
            if (effective == OffloadScheme.Managed)
            {
                // SVM (OpenCL 2.0+).
                var svmPtr = OpenClPlatformProbe.clSVMAlloc(_context,
                    OpenClPlatformProbe.CL_MEM_READ_WRITE | OpenClPlatformProbe.CL_MEM_SVM_FINE_GRAIN_BUFFER,
                    (UIntPtr)bytes, alignment: 0);
                if (svmPtr == IntPtr.Zero)
                    throw new InvalidOperationException("clSVMAlloc returned null (OpenCL 2.0 SVM not supported on this device).");
                var rec = new AllocRecord { Scheme = effective, IsSvm = true, HostPtr = svmPtr };
                _live[svmPtr] = rec;
                return new GpuOffloadHandle(svmPtr, svmPtr, bytes, effective);
            }
            else
            {
                // Pinned-host: clCreateBuffer with CL_MEM_ALLOC_HOST_PTR.
                // The driver allocates pinned host memory and returns a cl_mem
                // handle; we ALSO keep an aligned host buffer for the user-
                // visible HostPointer.
                IntPtr hostBuf = Marshal.AllocHGlobal((IntPtr)bytes);
                var memObj = OpenClPlatformProbe.clCreateBuffer(_context,
                    OpenClPlatformProbe.CL_MEM_READ_WRITE | OpenClPlatformProbe.CL_MEM_ALLOC_HOST_PTR,
                    (UIntPtr)bytes, IntPtr.Zero, out int err);
                if (err != 0 || memObj == IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(hostBuf);
                    throw new InvalidOperationException($"clCreateBuffer returned errcode {err}.");
                }
                var rec = new AllocRecord { Scheme = effective, IsSvm = false, HostPtr = hostBuf, MemObject = memObj };
                _live[hostBuf] = rec;
                return new GpuOffloadHandle(hostBuf, hostBuf, bytes, effective, memObj);
            }
        }
    }

    public void Free(GpuOffloadHandle handle)
    {
        if (handle.HostPointer == IntPtr.Zero) return;
        if (!_live.TryRemove(handle.HostPointer, out var rec)) return;
        FreeRecord(rec, _context);
    }

    private static void FreeRecord(AllocRecord rec, IntPtr context)
    {
        try
        {
            if (rec.IsSvm)
            {
                // clSVMFree requires the context that owned the allocation;
                // skipping when context is zero (already released in Dispose).
                if (context != IntPtr.Zero) OpenClPlatformProbe.clSVMFree(context, rec.HostPtr);
            }
            else
            {
                if (rec.MemObject != IntPtr.Zero) OpenClPlatformProbe.clReleaseMemObject(rec.MemObject);
                Marshal.FreeHGlobal(rec.HostPtr);
            }
        }
        catch { /* best-effort */ }
    }

    public void Dispose()
    {
        AllocRecord[] snapshot;
        IntPtr ctx;
        lock (_lock)
        {
            if (_disposed) return;
            // Flip _disposed under the lock so any racing Allocate observes
            // the flip on entry and throws before adding new records.
            _disposed = true;
            snapshot = System.Linq.Enumerable.ToArray(_live.Values);
            _live.Clear();
            ctx = _context;
        }
        // Free the records using the still-valid context. clSVMFree must run
        // BEFORE clReleaseContext, so we don't null _context until after.
        foreach (var rec in snapshot) FreeRecord(rec, ctx);
        if (ctx != IntPtr.Zero)
        {
            try { OpenClPlatformProbe.clReleaseContext(ctx); } catch { }
        }
        lock (_lock) { _context = IntPtr.Zero; }
    }

    private sealed class AllocRecord
    {
        public OffloadScheme Scheme;
        public bool IsSvm;
        public IntPtr HostPtr;
        public IntPtr MemObject;
    }
}
