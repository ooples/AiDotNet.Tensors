// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Process-wide registry that ties <see cref="WeightLifetime"/> hints to
/// the matching backing infrastructure: streaming pool for
/// <see cref="WeightLifetime.Streaming"/>, GPU offload allocator for
/// <see cref="WeightLifetime.GpuOffload"/> / <see cref="WeightLifetime.GpuManaged"/>.
///
/// <para>Model authors call <see cref="RegisterWeight{T}"/> with their
/// chosen lifetime; the registry routes to the right backend without
/// the model code knowing which GPU runtime is loaded. Frameworks (DDP,
/// FSDP, the engine dispatcher) read <see cref="Tensor{T}.Lifetime"/>
/// to pick fast paths in their kernel hot path.</para>
/// </summary>
public static class WeightRegistry
{
    private static readonly object _lock = new();
    private static StreamingTensorPool? _streamingPool;
    private static IGpuOffloadAllocator? _offloadAllocator;
    private static GpuOffloadOptions _options = new();

    /// <summary>Replaces the active options; must be called before any
    /// <see cref="WeightLifetime.Streaming"/> / GpuOffload registration.</summary>
    public static void Configure(GpuOffloadOptions options, IGpuOffloadAllocator? offloadAllocator = null)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        lock (_lock)
        {
            // Dispose the previous allocator before swapping — otherwise its
            // outstanding pinned/managed allocations + native context are
            // leaked. Skip disposal when the caller passes the same instance
            // back in (defensive: lets a caller re-Configure with new options
            // without losing live allocations).
            if (!ReferenceEquals(_offloadAllocator, offloadAllocator))
                _offloadAllocator?.Dispose();
            _options = options;
            _offloadAllocator = offloadAllocator;
            _streamingPool?.Dispose();
            _streamingPool = null; // lazy-create on first Streaming registration
        }
    }

    /// <summary>The active streaming pool, lazily constructed.</summary>
    public static StreamingTensorPool StreamingPool
    {
        get
        {
            lock (_lock)
            {
                _streamingPool ??= new StreamingTensorPool(_options);
                return _streamingPool;
            }
        }
    }

    /// <summary>The active offload allocator, or null if not configured.
    /// The model dispatcher reads this and falls back to default allocation
    /// when null + a tensor's Lifetime is GpuOffload / GpuManaged.</summary>
    public static IGpuOffloadAllocator? OffloadAllocator
    {
        get { lock (_lock) return _offloadAllocator; }
    }

    /// <summary>Registers a weight tensor with the registry. Routes to
    /// the streaming pool / offload allocator based on the tensor's
    /// <see cref="Tensor{T}.Lifetime"/>.</summary>
    public static void RegisterWeight<T>(Tensor<T> weight)
    {
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        // Hold _lock across the whole register operation so a concurrent
        // Configure / Reset can't dispose the allocator or pool out from
        // under us mid-allocate. Lock order is always [_lock, then any
        // backend-internal lock] — no inverse pattern exists, so no
        // deadlock. The allocator's Allocate is atomic via its own
        // _lifecycleLock; we hold _lock just to keep the allocator
        // reference and StreamingPool field stable for the duration.
        lock (_lock)
        {
            switch (weight.Lifetime)
            {
                case WeightLifetime.Default:
                    return;
                case WeightLifetime.Streaming:
                    {
                        int byteCount = weight.Length * ElementSize<T>();
                        var bytes = new byte[byteCount];
                        SerializeToBytes(weight, bytes);
                        long handle = StreamingPoolUnlocked().Register(bytes);
                        weight.StreamingPoolHandle = handle;
                        return;
                    }
                case WeightLifetime.GpuOffload:
                case WeightLifetime.GpuManaged:
                    {
                        var alloc = _offloadAllocator;
                        if (alloc is null || !alloc.IsAvailable)
                        {
                            // Backend not loadable on this host — fall through to default.
                            weight.Lifetime = WeightLifetime.Default;
                            return;
                        }
                        int byteCount = weight.Length * ElementSize<T>();
                        var scheme = weight.Lifetime == WeightLifetime.GpuManaged
                            ? OffloadScheme.Managed : OffloadScheme.Pinned;
                        // Serialize FIRST so a SerializeToBytes failure (e.g.,
                        // unsupported element type) doesn't leak a native
                        // allocation. Then allocate, copy, and only on success
                        // commit the metadata onto the tensor — wrap copy in
                        // try/catch to free the handle if Marshal.Copy throws.
                        var stageBytes = new byte[byteCount];
                        SerializeToBytes(weight, stageBytes);
                        var h = alloc.Allocate(byteCount, scheme);
                        try
                        {
                            System.Runtime.InteropServices.Marshal.Copy(stageBytes, 0, h.HostPointer, byteCount);
                            // Persist the full handle so UnregisterWeight can
                            // recreate it correctly. We track HostPointer
                            // separately because the allocator's _live dict
                            // keys by HostPointer; for Vulkan / OpenCL
                            // pinned, host != device, and reconstructing
                            // with device-as-host would silently leak.
                            weight.OffloadHostPointer = h.HostPointer;
                            weight.OffloadDevicePointer = h.DevicePointer;
                            weight.OffloadOpaqueHandle = h.BackendOpaque;
                            weight.OffloadByteCount = byteCount;
                        }
                        catch
                        {
                            alloc.Free(h);
                            throw;
                        }
                        return;
                    }
                default:
                    throw new ArgumentOutOfRangeException(nameof(weight.Lifetime));
            }
        }
    }

    /// <summary>Unregisters and frees backing storage.</summary>
    public static void UnregisterWeight<T>(Tensor<T> weight)
    {
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        // Hold _lock for the same reason RegisterWeight does — Configure /
        // Reset must not dispose the allocator/pool while we're freeing
        // outstanding allocations.
        lock (_lock)
        {
            if (weight.StreamingPoolHandle >= 0)
            {
                _streamingPool?.Unregister(weight.StreamingPoolHandle);
                weight.StreamingPoolHandle = -1;
            }
            if (weight.OffloadDevicePointer != IntPtr.Zero || weight.OffloadHostPointer != IntPtr.Zero)
            {
                var alloc = _offloadAllocator;
                if (alloc is not null)
                {
                    var scheme = weight.Lifetime == WeightLifetime.GpuManaged
                        ? OffloadScheme.Managed : OffloadScheme.Pinned;
                    // Reconstruct the full GpuOffloadHandle from persisted
                    // metadata so the allocator's Free path sees the same
                    // host / device / opaque it allocated. Host pointer is
                    // load-bearing — Free's TryRemove keys by HostPointer.
                    // Fall back to DevicePointer for pre-existing tensors
                    // that don't have OffloadHostPointer set (CUDA/HIP
                    // pinned where host==device).
                    IntPtr host = weight.OffloadHostPointer != IntPtr.Zero
                        ? weight.OffloadHostPointer
                        : weight.OffloadDevicePointer;
                    alloc.Free(new GpuOffloadHandle(
                        host: host,
                        device: weight.OffloadDevicePointer,
                        bytes: weight.OffloadByteCount,
                        scheme: scheme,
                        opaque: weight.OffloadOpaqueHandle));
                }
                weight.OffloadHostPointer = IntPtr.Zero;
                weight.OffloadDevicePointer = IntPtr.Zero;
                weight.OffloadOpaqueHandle = null;
                weight.OffloadByteCount = 0;
            }
        }
    }

    /// <summary>Caller must hold <see cref="_lock"/> — returns the lazy
    /// streaming pool without retaking the lock so RegisterWeight can hold
    /// it across the whole register operation.</summary>
    private static StreamingTensorPool StreamingPoolUnlocked()
    {
        _streamingPool ??= new StreamingTensorPool(_options);
        return _streamingPool;
    }

    private static int ElementSize<T>() => System.Runtime.InteropServices.Marshal.SizeOf<T>();

    /// <summary>Serializes a tensor's elements to a raw byte buffer. We
    /// can't constrain T : unmanaged at the API level (Tensor&lt;T&gt; is
    /// generic over arbitrary numeric types including Complex / Multivector),
    /// so this routes through reflection-friendly per-type fast paths and
    /// a generic byte-by-byte fallback.</summary>
    private static void SerializeToBytes<T>(Tensor<T> tensor, byte[] dst)
    {
        if (typeof(T) == typeof(float))
        {
            var src = (float[])(object)tensor.AsSpan().ToArray();
            Buffer.BlockCopy(src, 0, dst, 0, dst.Length);
            return;
        }
        if (typeof(T) == typeof(double))
        {
            var src = (double[])(object)tensor.AsSpan().ToArray();
            Buffer.BlockCopy(src, 0, dst, 0, dst.Length);
            return;
        }
        if (typeof(T) == typeof(int))
        {
            var src = (int[])(object)tensor.AsSpan().ToArray();
            Buffer.BlockCopy(src, 0, dst, 0, dst.Length);
            return;
        }
        if (typeof(T) == typeof(long))
        {
            var src = (long[])(object)tensor.AsSpan().ToArray();
            Buffer.BlockCopy(src, 0, dst, 0, dst.Length);
            return;
        }
        if (typeof(T) == typeof(Half))
        {
            var arr = (Half[])(object)tensor.AsSpan().ToArray();
            for (int i = 0; i < arr.Length; i++)
            {
                ushort raw = AiDotNet.Tensors.NumericOperations.HalfBits.GetBits(arr[i]);
                dst[i * 2 + 0] = (byte)(raw & 0xFF);
                dst[i * 2 + 1] = (byte)((raw >> 8) & 0xFF);
            }
            return;
        }
        if (typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16))
        {
            var arr = (AiDotNet.Tensors.NumericOperations.BFloat16[])(object)tensor.AsSpan().ToArray();
            for (int i = 0; i < arr.Length; i++)
            {
                ushort raw = arr[i].RawValue;
                dst[i * 2 + 0] = (byte)(raw & 0xFF);
                dst[i * 2 + 1] = (byte)((raw >> 8) & 0xFF);
            }
            return;
        }
        // No exact serializer for T. Streaming pool / GPU offload would
        // silently lose data through a lossy round-trip; refuse instead so
        // the caller knows to convert to a supported element type first
        // (or extend this method with a new fast path).
        throw new NotSupportedException(
            $"WeightRegistry: no exact serializer for element type {typeof(T).Name}. " +
            "Supported types: float, double, int, long, Half, BFloat16. " +
            "For other types, convert weights to a supported representation before tagging Lifetime.");
    }

    /// <summary>For tests: drops all configured backends + flushes pool.</summary>
    public static void Reset()
    {
        lock (_lock)
        {
            _streamingPool?.Dispose();
            _streamingPool = null;
            _offloadAllocator?.Dispose();
            _offloadAllocator = null;
            _options = new GpuOffloadOptions();
        }
    }
}
