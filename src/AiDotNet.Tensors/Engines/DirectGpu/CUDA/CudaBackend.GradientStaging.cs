// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Threading;
using AiDotNet.Tensors.Engines.DevicePrimitives;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGradientHostStagingSource
{
    IGradientHostStaging? IGradientHostStagingSource.TryCreateGradientHostStaging(int floats)
    {
        if (floats <= 0 || _disposed || !IsAvailable) return null;
        // Its own page-locked allocation, not a pool bucket: the area lives as long as the plan, can exceed the pool's
        // power-of-two buckets (which stop at 1 GiB), and must be freed even after the pool is gone.
        return new PinnedGradientStaging(this, floats);
    }

    IGradientHostStaging? IGradientHostStagingSource.TryRegisterGradientHostStaging(IntPtr host, int floats,
        out string? failure)
    {
        failure = null;
        if (host == IntPtr.Zero || floats <= 0) { failure = "no host memory"; return null; }
        if (_disposed || !IsAvailable) { failure = "the CUDA backend is not available"; return null; }
        ulong bytes = (ulong)floats * sizeof(float);
        using var _ = PushContext();
        var result = CudaNativeBindings.cuMemHostRegister(host, bytes, 1u /* CU_MEMHOSTREGISTER_PORTABLE */);
        if (result != CudaResult.Success) { failure = "cuMemHostRegister returned " + result; return null; }
        return new PinnedGradientStaging(this, host, floats);
    }

    /// <summary>Releases skipped because their context was gone or the runtime was tearing down (tests).</summary>
    internal static long HostReleasesSkipped;

    /// <summary>Releases that reached the driver (tests).</summary>
    internal static long HostReleasesAttempted;

    /// <summary>
    /// Whether a release's bytes are gone from the process: freed, or skipped because the context (or the process)
    /// took them. Accounting decrements on either; a failed free leaves them counted.
    /// </summary>
    internal static bool HostReleaseReturnedMemory(CudaResult? released) => released is null || released == CudaResult.Success;

    /// <summary>
    /// Frees (or unregisters) page-locked host memory in <paramref name="context"/>, only while it is still the
    /// registration <paramref name="generation"/> named and the runtime is not tearing down: a driver call against a
    /// destroyed context, during teardown, or on a context the circuit breaker found corrupted, can be an access violation
    /// no catch traps. Returns null when it was skipped (the context's destruction, or the process's exit, takes the
    /// memory back).
    /// </summary>
    internal static CudaResult? ReleaseHostInLiveContext(IntPtr host, IntPtr context, long generation, bool unregister)
    {
        if (host == IntPtr.Zero) return null;
        // A context the circuit breaker found corrupted is not called into either: the exit of the process takes the memory.
        if (IsRuntimeTearingDown || ProcessExiting || AiDotNetEngine.GpuCircuitBroken) { Interlocked.Increment(ref HostReleasesSkipped); return null; }
        lock (ContextLifecycleLock)   // no destroy between the check and the call
        {
            if (!IsLiveContext(context, generation)) { Interlocked.Increment(ref HostReleasesSkipped); return null; }
            var pushed = CuBlasNative.cuCtxPushCurrent(context);
            if (pushed != CudaResult.Success) return pushed;
            Interlocked.Increment(ref HostReleasesAttempted);
            try { return unregister ? CudaNativeBindings.cuMemHostUnregister(host) : CuBlasNative.cuMemFreeHost(host); }
            finally { CuBlasNative.cuCtxPopCurrent(out _); }
        }
    }

    private sealed unsafe class PinnedGradientStaging : IGradientHostStaging
    {
        private readonly CudaBackend _owner;
        // The context lifetime the memory was allocated (or registered) in: it is released only while that lifetime lasts.
        private readonly IntPtr _context;
        private readonly long _contextGeneration;
        private readonly int _floats;
        private IntPtr _host;
        // Registered: the memory is the caller's (page-locked in place); disposing unregisters it. Otherwise this
        // staging allocated it (cuMemHostAlloc, portable) and frees it.
        private readonly bool _registered;

        internal PinnedGradientStaging(CudaBackend owner, int floats)
        {
            _owner = owner;
            _floats = floats;
            _context = owner._cudaContext;
            _contextGeneration = owner._contextGeneration;
            ulong bytes = (ulong)floats * sizeof(float);
            // Portable only: no fallback to cuMemAllocHost, whose memory is page-locked for one context alone.
            using (owner.PushContext())
                CuBlasNative.CheckCudaResult(CudaPinnedBufferPool.AllocatePortablePinnedHostOnly(out _host, bytes),
                    "cuMemHostAlloc(gradient staging)");
            GpuMemoryStats.RecordAllocation("cuda_pinned_host", (long)bytes);
        }

        internal PinnedGradientStaging(CudaBackend owner, IntPtr registeredHost, int floats)
        {
            _owner = owner;
            _floats = floats;
            _context = owner._cudaContext;
            _contextGeneration = owner._contextGeneration;
            _host = registeredHost;
            _registered = true;
        }

        // No finalizer: the driver is never called on the finalizer thread (see CudaGpuBuffer's). A staging that is
        // never disposed keeps its memory until its context is destroyed or the process exits.

        public Span<float> Host
        {
            get
            {
                ThrowIfUnusable();
                return new Span<float>((void*)_host, _floats);
            }
        }

        private void ThrowIfUnusable()
        {
            if (_host == IntPtr.Zero) throw new ObjectDisposedException(nameof(PinnedGradientStaging));
            if (_owner._disposed) throw new ObjectDisposedException(nameof(CudaBackend), "The staging's backend was disposed.");
        }

        private void Check(IGpuBuffer buffer, int offset, int count)
        {
            ThrowIfUnusable();
            if (buffer is null) throw new ArgumentNullException(nameof(buffer));
            if (buffer.Handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(buffer), "GPU buffer was released before its staged transfer.");
            if (offset < 0 || count < 0 || (long)offset + count > _floats)
                throw new ArgumentOutOfRangeException(nameof(count), "The transfer does not fit the staging area.");
            if (count > buffer.Size)
                throw new ArgumentOutOfRangeException(nameof(count), "The transfer is longer than the device buffer.");
            GpuFloatLayout.Require(buffer, nameof(IGradientHostStaging));
        }

        public void EnqueueDownload(IGpuBuffer buffer, int offset, int count)
        {
            Check(buffer, offset, count);
            if (count == 0) return;
            using var _ = _owner.PushContext();
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuMemcpyDtoHAsync(_host + offset * sizeof(float), buffer.Handle,
                    (ulong)count * sizeof(float), _owner._stream),
                "cuMemcpyDtoHAsync(gradient staging)");
        }

        public void EnqueueUpload(IGpuBuffer buffer, int offset, int count) => EnqueueUploadAt(buffer, 0, offset, count);

        public void EnqueueUploadAt(IGpuBuffer buffer, int bufferOffset, int hostOffset, int count)
        {
            Check(buffer, hostOffset, count);
            if (bufferOffset < 0 || (long)bufferOffset + count > buffer.Size)
                throw new ArgumentOutOfRangeException(nameof(bufferOffset), "The transfer does not fit the device buffer.");
            if (count == 0) return;
            using var _ = _owner.PushContext();
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuMemcpyHtoDAsync(buffer.Handle + bufferOffset * sizeof(float),
                    _host + hostOffset * sizeof(float), (ulong)count * sizeof(float), _owner._stream),
                "cuMemcpyHtoDAsync(gradient staging)");
        }

        public void WaitForTransfers()
        {
            ThrowIfUnusable();
            using var _ = _owner.PushContext();
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamSynchronize(_owner._stream),
                "cuStreamSynchronize(gradient staging)");
        }

        public void Dispose()
        {
            IntPtr host = System.Threading.Interlocked.Exchange(ref _host, IntPtr.Zero);
            if (host == IntPtr.Zero) return;
            if (!_owner._disposed)
            {
                // In-flight copies still use the memory: it is released only after they finish. A failed wait (a
                // sticky device error) means the context is lost and no copy will complete; say so, then release.
                try
                {
                    using var _ = _owner.PushContext();
                    var synced = CudaNativeBindings.cuStreamSynchronize(_owner._stream);
                    if (synced != CudaResult.Success)
                        System.Diagnostics.Trace.TraceWarning(
                            "Gradient staging: the stream wait before releasing the page-locked memory returned " + synced + ".");
                }
                catch (ObjectDisposedException)
                {
                    // The backend finished its teardown meanwhile: its stream is gone, so nothing is in flight.
                }
            }
            // Released only in the context lifetime it belongs to (a backend disposed while others hold the shared
            // primary context still frees it); otherwise that context's destruction already took it back.
            var released = ReleaseHostInLiveContext(host, _context, _contextGeneration, _registered);
            if (released is { } result && result != CudaResult.Success)
                System.Diagnostics.Trace.TraceWarning("Gradient staging: releasing the page-locked memory returned " + result + ".");
            if (!_registered && HostReleaseReturnedMemory(released))
                GpuMemoryStats.RecordFree("cuda_pinned_host", (long)_floats * sizeof(float));
        }
    }
}
