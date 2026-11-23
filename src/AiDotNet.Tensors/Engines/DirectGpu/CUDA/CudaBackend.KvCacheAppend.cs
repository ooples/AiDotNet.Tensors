// Copyright (c) CrowdTrainer LLC. All rights reserved.
// CudaBackend.IKvCacheAppendBackend: a decode cache append at a device-side position in one launch.
using System;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IKvCacheAppendBackend
{
    // The kernel's fault record (CudaKvCacheAppend.c): page-locked, mapped into the device, so a captured graph's
    // append reports a bad position with no copy node and the host reads it after any synchronization.
    private readonly object _kvFaultLock = new();
    private IntPtr _kvFaultHost, _kvFaultDevice;
    private int _kvAppendLaunched;
    private const int KvFaultWords = 3;

    /// <inheritdoc />
    unsafe void IKvCacheAppendBackend.KvCacheAppend(IGpuBuffer cache, IGpuBuffer position, IGpuBuffer value, IGpuBuffer mask,
        int batch, int capacity, int width)
    {
        if (batch <= 0 || capacity <= 0 || width <= 0) throw new ArgumentOutOfRangeException(nameof(batch), "batch, capacity and width must be positive.");
        if (capacity > KvCacheAppendLimits.MaxCapacity)
            throw new ArgumentOutOfRangeException(nameof(capacity), $"A float position is exact up to {KvCacheAppendLimits.MaxCapacity}; the capacity is {capacity}.");
        long rowValues = (long)batch * width;
        GpuBufferCapacity.Require(cache, rowValues * capacity, nameof(cache));
        GpuBufferCapacity.Require(position, 1, nameof(position));
        GpuBufferCapacity.Require(value, rowValues, nameof(value));
        GpuBufferCapacity.Require(mask, capacity, nameof(mask));
        if (!IsAvailable) throw new GpuCapabilityUnavailableException("CUDA backend is not available.");
        if (!_kernelCache.TryGetValue("kv_cache_append", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: kv_cache_append");
        using var _ = PushContext();
        bool capturing = IsStreamCapturing();
        IntPtr f = EnsureKvFaultRecord(capturing);
        IntPtr c = cache.Handle, p = position.Handle, v = value.Handle, m = mask.Handle;
        int n = batch, cap = capacity, w = width;
        void** args = stackalloc void*[8];
        args[0] = &c; args[1] = &p; args[2] = &v; args[3] = &m; args[4] = &f; args[5] = &n; args[6] = &cap; args[7] = &w;
        long work = Math.Max(rowValues, capacity);
        uint threads = (uint)Math.Min(1024L, (work + 31) / 32 * 32);
        Volatile.Write(ref _kvAppendLaunched, 1);
        LaunchKernel(kernel, 1, threads, args);
        // An eager append reports a bad position as the host implementation does. A captured one cannot raise here;
        // the plan replaying it calls ThrowIfAppendFaulted after the replay.
        if (!capturing) ((IKvCacheAppendBackend)this).ThrowIfAppendFaulted();
    }

    /// <inheritdoc />
    unsafe void IKvCacheAppendBackend.ThrowIfAppendFaulted()
    {
        if (Volatile.Read(ref _kvAppendLaunched) == 0 || !IsAvailable) return;
        int code, bits, capacity;
        lock (_kvFaultLock)
        {
            if (_kvFaultHost == IntPtr.Zero) return;
            using (PushContext())
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamSynchronize(_mainStream), "cuStreamSynchronize(kv fault)");
            var record = (int*)_kvFaultHost;
            code = Volatile.Read(ref record[0]);
            if (code == 0) return;
            bits = record[1];
            capacity = record[2];
            record[1] = 0; record[2] = 0;
            Volatile.Write(ref record[0], 0);
        }
        if (code == 1) throw new InvalidOperationException($"The cache is full: {capacity} positions.");
        float raw = *(float*)&bits;
        throw new InvalidOperationException($"The cache position must be a non-negative integer; it is {raw}.");
    }

    // Allocated outside a capture (an allocation inside one would be recorded into it); an append first reached
    // inside a capture fails it, and the plan runs eagerly.
    private unsafe IntPtr EnsureKvFaultRecord(bool capturing)
    {
        lock (_kvFaultLock)
        {
            if (_kvFaultDevice != IntPtr.Zero) return _kvFaultDevice;
            if (capturing) throw new InvalidOperationException("The cache append's fault record is allocated by an eager append first.");
            CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAllocHost(out var host, KvFaultWords * sizeof(int)), "cuMemAllocHost(kv fault)");
            new Span<int>((void*)host, KvFaultWords).Clear();
            var result = CudaNativeBindings.cuMemHostGetDevicePointer(out var device, host, 0);
            if (result != CudaResult.Success)
            {
                CuBlasNative.cuMemFreeHost(host);
                CuBlasNative.CheckCudaResult(result, "cuMemHostGetDevicePointer(kv fault)");
            }
            _kvFaultHost = host;
            _kvFaultDevice = device;
            return device;
        }
    }

    // Backend Dispose, with the context current and the stream drained.
    private void DisposeKvFaultRecord()
    {
        lock (_kvFaultLock)
        {
            if (_kvFaultHost != IntPtr.Zero) CuBlasNative.cuMemFreeHost(_kvFaultHost);
            _kvFaultHost = IntPtr.Zero;
            _kvFaultDevice = IntPtr.Zero;
        }
    }
}
