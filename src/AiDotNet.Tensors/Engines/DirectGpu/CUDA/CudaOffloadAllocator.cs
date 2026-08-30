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
public sealed class CudaOffloadAllocator : IGpuOffloadAllocator, IGpuDevicePointerWrapper
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
    // Device whose PRIMARY context we retained (owned path). -1 until EnsureContext runs. Needed because
    // cuDevicePrimaryCtxRelease is keyed by device, not by context handle.
    private int _device = -1;
    // True when this allocator created _context itself (standalone offload path)
    // and must destroy it on Dispose. False when _context is SHARED with an
    // existing CudaBackend — then the offload buffers live in the SAME context
    // as the compute kernels (no cross-context access), so the optimizer kernel
    // reading the pinned moments and the stream sync that orders it are in one
    // context. Sharing must NOT destroy a context the backend still owns.
    private readonly bool _ownsContext = true;
    private bool _disposed;

    /// <summary>Standalone offload allocator — lazily creates and owns its own CUDA context.</summary>
    public CudaOffloadAllocator() { }

    /// <summary>
    /// Offload allocator that SHARES an existing CUDA context (the compute
    /// backend's). Pinned moment/weight buffers then live in the same context as
    /// the kernels that read them, so cross-context access and its murky
    /// cross-context synchronization (the source of the GPU-resident-optimizer
    /// host-readback flakiness) are eliminated. The shared context is owned by
    /// the caller and is NOT destroyed on <see cref="Dispose"/>.
    /// </summary>
    public CudaOffloadAllocator(IntPtr sharedContext)
    {
        _context = sharedContext;
        _ownsContext = sharedContext == IntPtr.Zero; // zero -> behave like the default ctor (create our own)
    }

    internal static bool IsCudaUsable(bool driverAvailable, bool circuitBroken)
        => driverAvailable && !circuitBroken;

    internal static bool ShouldPopContext(bool pushed, bool circuitBroken)
        => pushed && !circuitBroken;

    public bool IsAvailable
    {
        get
        {
            if (!IsCudaUsable(CudaNativeBindings.IsAvailable, AiDotNetEngine.GpuCircuitBroken))
                return false;

            lock (_lifecycleLock)
            {
                if (_disposed) return false;
                lock (CudaBackend.ContextLifecycleLock)
                    return IsContextUsableUnderLifecycleLock();
            }
        }
    }

    // Caller must hold CudaBackend.ContextLifecycleLock. For a shared backend context, membership
    // is the authoritative lifetime signal: CudaBackend removes it under the same lock immediately
    // before cuCtxDestroy. Owned primary contexts are protected by this allocator's lifecycle lock.
    private bool IsContextUsableUnderLifecycleLock()
        => _ownsContext || (_context != IntPtr.Zero && CudaBackend.LiveContexts.ContainsKey(_context));

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
    {
        // Hold _lifecycleLock across the entire allocate+register so a concurrent Dispose cannot
        // let this allocation escape its snapshot. ContextLifecycleLock is nested inside it and held
        // across {live check + push + native call + pop}, making backend context destruction mutually
        // exclusive with shared-context use.
        lock (_lifecycleLock)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CudaOffloadAllocator));
            if (!IsCudaUsable(CudaNativeBindings.IsAvailable, AiDotNetEngine.GpuCircuitBroken))
                throw new NotSupportedException("CUDA offload is unavailable on this host.");
            if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

            lock (CudaBackend.ContextLifecycleLock)
            {
                if (!IsContextUsableUnderLifecycleLock())
                    throw new NotSupportedException("The CUDA context backing this offload allocator is no longer live.");

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
                                var rc = CudaNativeBindings.cuMemAllocManaged(
                                    out ptr, (ulong)bytes, CudaNativeBindings.CU_MEM_ATTACH_GLOBAL);
                                CuBlasNative.CheckCudaResult(rc, "cuMemAllocManaged(offload)");
                                break;
                            }
                        default:
                            throw new ArgumentOutOfRangeException(nameof(scheme), scheme, "Unknown offload scheme.");
                    }
                    var handle = new GpuOffloadHandle(ptr, ptr, bytes, effective);
                    _live[ptr] = handle;
                    return handle;
                }
            }
        }
    }

    public void Free(GpuOffloadHandle handle)
    {
        if (handle.HostPointer == IntPtr.Zero) return;
        // Only call native free for handles WE own. Serialize the live-context check and CUDA
        // cleanup against CudaBackend's {deregister + destroy} critical section.
        lock (_lifecycleLock)
        {
            if (!_live.TryRemove(handle.HostPointer, out _)) return;
            if (!IsCudaUsable(CudaNativeBindings.IsAvailable, AiDotNetEngine.GpuCircuitBroken))
                return;

            lock (CudaBackend.ContextLifecycleLock)
            {
                if (!IsContextUsableUnderLifecycleLock()) return;
                using (PushContextScope())
                {
                    FreeNative(handle);
                }
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
        // RETAIN the device primary context instead of cuCtxCreate. A fresh cuCtxCreate reserves a whole
        // new ~200 MB context per allocator; under memory pressure (or many allocators) that OOMs. The
        // primary context is SHARED with every other retainer (including the compute backend's primary-ctx
        // consumers), so this adds no meaningful device memory. cuDevicePrimaryCtxRetain does NOT push the
        // context onto the calling thread's stack, so — unlike cuCtxCreate — there is nothing to pop here;
        // allocate/free still push+pop it via PushContextScope around their native calls.
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuDevicePrimaryCtxRetain(out _context, device),
            "cuDevicePrimaryCtxRetain(offload)");
        _device = device;
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
            if (ShouldPopContext(_pushed, AiDotNetEngine.GpuCircuitBroken))
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

    /// <summary>
    /// Issue #336: wraps a device pointer (returned by <see cref="Allocate"/>)
    /// as a non-owning <see cref="IGpuBuffer"/>. Use case: cuBLAS-backed
    /// Adam / SGD optimizer kernels reading a <see cref="WeightLifetime.GpuPinned"/>
    /// tensor's <c>OffloadDevicePointer</c> need the buffer abstraction
    /// without taking ownership — the buffer's <c>Dispose</c> is a no-op
    /// because the allocation is owned by this allocator's
    /// <c>_live</c> map (freed via <see cref="Free"/>).
    /// </summary>
    public IGpuBuffer? WrapDevicePointerAsBuffer(IntPtr devicePointer, int elementCount, int elementByteSize)
    {
        if (devicePointer == IntPtr.Zero) return null;
        if (elementCount <= 0 || elementByteSize <= 0) return null;
        if (!_live.ContainsKey(devicePointer))
        {
            // Pointer wasn't allocated by this instance — refuse rather
            // than fabricate a buffer over unknown memory. Caller should
            // have used the same allocator that produced the pointer.
            return null;
        }
        return new CudaNonOwningBuffer(devicePointer, elementCount, (long)elementCount * elementByteSize);
    }

    /// <summary>
    /// Non-owning <see cref="IGpuBuffer"/> that wraps a device pointer
    /// allocated elsewhere. Dispose is a no-op; the owning
    /// <see cref="CudaOffloadAllocator"/> frees the underlying allocation
    /// when <see cref="Free"/> is called for the matching handle.
    /// </summary>
    private sealed class CudaNonOwningBuffer : IGpuBuffer
    {
        public IntPtr Handle { get; }
        public int Size { get; }
        public long SizeInBytes { get; }

        public CudaNonOwningBuffer(IntPtr handle, int size, long sizeInBytes)
        {
            Handle = handle;
            Size = size;
            SizeInBytes = sizeInBytes;
        }

        public void Dispose() { /* non-owning */ }
    }

    private static void FreeNative(GpuOffloadHandle handle)
    {
        switch (handle.Scheme)
        {
            case OffloadScheme.Pinned:
                CuBlasNative.CheckCudaResult(
                    CuBlasNative.cuMemFreeHost(handle.HostPointer),
                    "cuMemFreeHost(offload)");
                break;
            case OffloadScheme.Managed:
                CuBlasNative.CheckCudaResult(
                    CudaNativeBindings.cuMemFree(handle.DevicePointer),
                    "cuMemFree(offload)");
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

            if (AiDotNetEngine.GpuCircuitBroken)
            {
                // CUDA 700/709/719 can invalidate the context and even make a nominal cleanup call
                // fault natively. The circuit breaker is process-lifetime, so abandon native teardown;
                // the OS reclaims these resources when the process exits.
                return;
            }

            if (ctxToDestroy != IntPtr.Zero && _ownsContext)
            {
                // WE retained the device primary context, so it is guaranteed valid here. Push it to free
                // our allocations cleanly, then pop and RELEASE our retain reference. Push/pop discipline
                // matters even at dispose time: a CudaBackend running on the same thread might have its own
                // context current, and a free issued against the wrong context would be INVALID_CONTEXT or
                // worse. cuDevicePrimaryCtxRelease only drops OUR reference — the primary context is torn
                // down by the driver only when the last retainer releases it, so this never pulls the context
                // out from under a sibling consumer that still holds it.
                using (var scope = new CudaContextPushScope(ctxToDestroy))
                {
                    foreach (var h in snapshot) FreeNative(h);
                }
                if (_device >= 0)
                    CuBlasNative.cuDevicePrimaryCtxRelease(_device);
            }
            else if (ctxToDestroy != IntPtr.Zero)
            {
                // SHARED (non-owning) context: free our outstanding allocations while its owner
                // still has it live, but serialize the live check + push/free/pop against the
                // backend's deregister + cuCtxDestroy critical section. If the backend already
                // destroyed the context (for example ResetToCpu demoted it first), touching the raw
                // handle is a use-after-free that can fault the process natively; skip cleanup in
                // that case because cuCtxDestroy has already reclaimed the context's resources.
                lock (CudaBackend.ContextLifecycleLock)
                {
                    if (CudaBackend.LiveContexts.ContainsKey(ctxToDestroy))
                    {
                        using var scope = new CudaContextPushScope(ctxToDestroy);
                        foreach (var h in snapshot) FreeNative(h);
                    }
                }
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
