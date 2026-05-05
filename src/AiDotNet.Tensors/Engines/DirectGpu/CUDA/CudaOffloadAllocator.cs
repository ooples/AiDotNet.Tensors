// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// CUDA-backed <see cref="IGpuOffloadAllocator"/>. Pinned-scheme uses
/// <c>cuMemAllocHost</c> (mapped + portable); Managed-scheme uses
/// <c>cuMemAllocManaged</c> with <c>CU_MEM_ATTACH_GLOBAL</c>.
/// </summary>
public sealed class CudaOffloadAllocator : IGpuOffloadAllocator
{
    private readonly ConcurrentDictionary<IntPtr, GpuOffloadHandle> _live = new();
    private readonly object _lifecycleLock = new();
    // Context owned by this allocator. cuMemAllocHost / cuMemAllocManaged
    // require a current CUDA context on the calling thread; an allocator
    // standalone of CudaBackend (the offload-only path #1222 weight
    // streaming hits when no GPU compute is wired up) has none, so we
    // create + push our own. Field stays IntPtr.Zero until first
    // Allocate; Dispose destroys it iff non-zero.
    private IntPtr _context;
    private bool _disposed;

    public bool IsAvailable => CudaNativeBindings.IsAvailable;

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
    {
        // Hold _lifecycleLock across the entire allocate+register so a
        // concurrent Dispose cannot snapshot _live, clear it, and let this
        // allocation slip in afterwards (which would leak the device handle
        // since Dispose has already returned).
        lock (_lifecycleLock)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CudaOffloadAllocator));
            if (!IsAvailable)
                throw new NotSupportedException("CUDA driver is not loadable on this host.");
            if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

            EnsureContext();
            using (PushContextScope())
            {
                IntPtr ptr;
                OffloadScheme effective = scheme == OffloadScheme.Auto ? OffloadScheme.Pinned : scheme;
                switch (effective)
                {
                    case OffloadScheme.Pinned:
                        {
                            var rc = CuBlasNative.cuMemAllocHost(out ptr, (ulong)bytes);
                            CuBlasNative.CheckCudaResult(rc, "cuMemAllocHost(offload)");
                            break;
                        }
                    case OffloadScheme.Managed:
                        {
                            var rc = CudaNativeBindings.cuMemAllocManaged(out ptr, (ulong)bytes, CudaNativeBindings.CU_MEM_ATTACH_GLOBAL);
                            if (rc != CudaResult.Success)
                                throw new InvalidOperationException($"cuMemAllocManaged returned {rc}");
                            break;
                        }
                    default:
                        throw new ArgumentOutOfRangeException(nameof(scheme), scheme, "Unknown offload scheme.");
                }
                var h = new GpuOffloadHandle(ptr, ptr, bytes, effective);
                _live[ptr] = h;
                return h;
            }
        }
    }

    public void Free(GpuOffloadHandle handle)
    {
        if (handle.HostPointer == IntPtr.Zero) return;
        // Only call native free for handles WE own — a foreign handle or
        // double-free would corrupt the heap on the second native release.
        // Lock TryRemove against Dispose's snapshot+clear so a concurrent
        // Dispose can't free the same pointer twice.
        lock (_lifecycleLock)
        {
            if (!_live.TryRemove(handle.HostPointer, out _)) return;
            // Native free needs our context current; push before, pop
            // after so the thread's context stack is restored to its
            // prior state.
            using (PushContextScope())
            {
                FreeNative(handle);
            }
        }
    }

    private void EnsureContext()
    {
        // Caller holds _lifecycleLock. Lazily creates the context on
        // first use. cuCtxCreate makes the new context current on the
        // calling thread AND pushes it on the thread's context stack
        // — we immediately pop it so the stack is restored. Subsequent
        // allocate/free calls use PushContextScope to push/pop around
        // their native calls cleanly.
        if (_context != IntPtr.Zero) return;
        CuBlasNative.CheckCudaResult(CuBlasNative.cuInit(0), "cuInit(offload)");
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuDeviceGet(out int device, 0),
            "cuDeviceGet(offload)");
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuCtxCreate(out _context, 0, device),
            "cuCtxCreate(offload)");
        // Pop the just-pushed context so we don't leak our context onto
        // the caller's thread stack. If we left it pushed, a sibling
        // CudaBackend's later cuCtxPopCurrent would pop our context
        // instead of its own — exactly the finalizer crash mode this
        // method exists to prevent.
        CuBlasNative.cuCtxPopCurrent(out _);
    }

    private CudaContextPushScope PushContextScope() => new(_context);

    private readonly struct CudaContextPushScope : IDisposable
    {
        private readonly bool _pushed;

        public CudaContextPushScope(IntPtr context)
        {
            _pushed = false;
            if (context == IntPtr.Zero) return;
            // Push our context onto the thread's stack. On Dispose we
            // pop, restoring the stack to what the caller had before.
            // This matches CudaBackend.CudaContextScope's pattern so
            // both can coexist without trampling each other's contexts.
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuCtxPushCurrent(context),
                "cuCtxPushCurrent(offload)");
            _pushed = true;
        }

        public void Dispose()
        {
            if (_pushed)
            {
                // Best-effort pop on failure: a throwing pop here would
                // mask the original native error and leave the stack
                // permanently corrupted. The CUDA driver only returns
                // non-success here on disposed contexts, so we'd
                // already be in a fault path.
                CuBlasNative.cuCtxPopCurrent(out _);
            }
        }
    }

    private static void FreeNative(GpuOffloadHandle handle)
    {
        switch (handle.Scheme)
        {
            case OffloadScheme.Pinned:
                CuBlasNative.cuMemFreeHost(handle.HostPointer);
                break;
            case OffloadScheme.Managed:
                CudaNativeBindings.cuMemFree(handle.DevicePointer);
                break;
        }
    }

    public void Dispose()
    {
        GpuOffloadHandle[] snapshot;
        IntPtr ctxToDestroy;
        lock (_lifecycleLock)
        {
            if (_disposed) return;
            // Flip _disposed under the lock so any Allocate that's already
            // waiting on _lifecycleLock observes the flip and throws,
            // and any Allocate that hasn't yet entered the lock cannot race
            // past us with a fresh allocation.
            _disposed = true;
            snapshot = _live.Values.ToArray();
            _live.Clear();
            ctxToDestroy = _context;
            _context = IntPtr.Zero;

            if (ctxToDestroy != IntPtr.Zero)
            {
                // Push our context to free its allocations cleanly,
                // then pop and destroy. Push/pop discipline matters
                // even at dispose time: a CudaBackend running on the
                // same thread might have its own context current, and
                // a free issued against the wrong context would be
                // INVALID_CONTEXT or worse.
                using (var scope = new CudaContextPushScope(ctxToDestroy))
                {
                    foreach (var h in snapshot) FreeNative(h);
                }
                CuBlasNative.cuCtxDestroy(ctxToDestroy);
            }
            else
            {
                // No context was ever created (no Allocate ran), so
                // _live must be empty. Defensive — free anything that
                // somehow got registered without an alloc going
                // through.
                foreach (var h in snapshot) FreeNative(h);
            }
        }
    }
}
