// Copyright (c) CrowdTrainer LLC. All rights reserved.
// CudaBackend.IKvCacheAppendBackend: a decode cache append at a device-side position in one launch.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IKvCacheAppendBackend
{
    /// <inheritdoc />
    unsafe void IKvCacheAppendBackend.KvCacheAppend(IGpuBuffer cache, IGpuBuffer position, IGpuBuffer value, IGpuBuffer mask,
        int batch, int capacity, int width)
    {
        if (batch <= 0 || capacity <= 0 || width <= 0) throw new ArgumentOutOfRangeException(nameof(batch), "batch, capacity and width must be positive.");
        if (!IsAvailable) throw new GpuCapabilityUnavailableException("CUDA backend is not available.");
        if (!_kernelCache.TryGetValue("kv_cache_append", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: kv_cache_append");
        using var _ = PushContext();
        IntPtr c = cache.Handle, p = position.Handle, v = value.Handle, m = mask.Handle;
        int n = batch, cap = capacity, w = width;
        void** args = stackalloc void*[7];
        args[0] = &c; args[1] = &p; args[2] = &v; args[3] = &m; args[4] = &n; args[5] = &cap; args[6] = &w;
        long work = Math.Max((long)batch * width, capacity);
        uint threads = (uint)Math.Min(1024L, (work + 31) / 32 * 32);
        LaunchKernel(kernel, 1, threads, args);
    }
}
