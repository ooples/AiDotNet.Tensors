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

    public void Free(GpuOffloadHandle handle)
    {
        if (_disposed) return;
        if (handle.HostPointer == IntPtr.Zero) return;
        if (!_live.TryRemove(handle.HostPointer, out var rec)) return;
        if (rec.IsSvm)
        {
            OpenClPlatformProbe.clSVMFree(_context, handle.HostPointer);
        }
        else
        {
            if (rec.MemObject != IntPtr.Zero) OpenClPlatformProbe.clReleaseMemObject(rec.MemObject);
            Marshal.FreeHGlobal(handle.HostPointer);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var rec in _live.Values)
        {
            try
            {
                if (rec.IsSvm) OpenClPlatformProbe.clSVMFree(_context, rec.HostPtr);
                else
                {
                    if (rec.MemObject != IntPtr.Zero) OpenClPlatformProbe.clReleaseMemObject(rec.MemObject);
                    Marshal.FreeHGlobal(rec.HostPtr);
                }
            }
            catch { /* best-effort */ }
        }
        _live.Clear();
        if (_context != IntPtr.Zero)
        {
            try { OpenClPlatformProbe.clReleaseContext(_context); } catch { }
            _context = IntPtr.Zero;
        }
    }

    private sealed class AllocRecord
    {
        public OffloadScheme Scheme;
        public bool IsSvm;
        public IntPtr HostPtr;
        public IntPtr MemObject;
    }
}
