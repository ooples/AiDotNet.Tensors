// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
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
    /// <see cref="WeightLifetime.Streaming"/> / GpuOffload registration.
    /// Throws <see cref="InvalidOperationException"/> when the existing pool
    /// has live registered handles — disposing it would orphan those
    /// handles and any subsequent Materialize would throw "handle unknown".
    /// Call <see cref="UnregisterWeight{T}"/> on all live tensors first,
    /// or <see cref="Reset"/> to forcibly drop them (test path).
    /// Throws <see cref="PlatformNotSupportedException"/> on big-endian
    /// hosts — the streaming pool's serialization mixes Buffer.BlockCopy
    /// (native endian) with hand-LE Half/BFloat16 codecs and is only
    /// well-defined on little-endian platforms (x86/x64/ARM-LE).</summary>
    public static void Configure(GpuOffloadOptions options, IGpuOffloadAllocator? offloadAllocator = null)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException(
                "WeightRegistry / StreamingTensorPool requires a little-endian host. " +
                "Big-endian platforms (legacy IBM Power, certain ARM modes) are not supported in v1.");
        lock (_lock)
        {
            // Mid-flight guard: refuse to dispose a pool that holds live
            // entries. Otherwise tensors registered against the old pool
            // would silently break on Materialize.
            if (_streamingPool is not null && _streamingPool.RegisteredEntryCount > 0)
                throw new InvalidOperationException(
                    $"WeightRegistry.Configure: existing streaming pool has {_streamingPool.RegisteredEntryCount} " +
                    "registered entries. Unregister all weights first, or call Reset() to forcibly drop them.");

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
                        // Use long arithmetic so we surface an OOM-style
                        // error explicitly instead of a silent int wrap.
                        // byte[] is itself bounded to ~2.15 GB, so streaming
                        // a single tensor > 2 GB is impossible regardless;
                        // we throw a clear NotSupportedException instead of
                        // the runtime's OutOfMemoryException so callers
                        // know to chunk the tensor at the consumer side.
                        long byteCountLong = (long)weight.Length * ElementSize<T>();
                        if (byteCountLong > int.MaxValue)
                            throw new NotSupportedException(
                                $"Streaming registration requires per-tensor size <= {int.MaxValue} bytes " +
                                $"(byte[] limit). Tensor has {weight.Length} elements × {ElementSize<T>()} bytes = " +
                                $"{byteCountLong} bytes. Chunk the tensor into smaller pool entries on the consumer side.");
                        int byteCount = (int)byteCountLong;
                        var bytes = new byte[byteCount];
                        SerializeToBytes(weight, bytes);
                        long handle = StreamingPoolUnlocked().Register(bytes);
                        weight.StreamingPoolHandle = handle;
                        // Drop the tensor's in-memory data: pool now owns the
                        // canonical copy. Without this, registration just
                        // duplicates memory (tensor + pool entry both
                        // resident) and Streaming mode can't actually save
                        // RAM. Materialize() restores _data on demand.
                        weight.DropStorageForStreaming();
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
                        long offloadByteCountLong = (long)weight.Length * ElementSize<T>();
                        if (offloadByteCountLong > int.MaxValue)
                            throw new NotSupportedException(
                                $"GpuOffload registration requires per-tensor size <= {int.MaxValue} bytes " +
                                $"(stage byte[] limit). Tensor has {weight.Length} elements × {ElementSize<T>()} bytes = " +
                                $"{offloadByteCountLong} bytes. Chunk the tensor on the consumer side.");
                        int byteCount = (int)offloadByteCountLong;
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

    private static int ElementSize<T>()
    {
        // Use typed fast-path sizes that agree with SerializeToBytes /
        // TensorBase.ElementSizeForStreaming. Marshal.SizeOf<Half>() is
        // host-dependent (returns 2 on .NET 5+, 4 on net471 polyfills),
        // so deferring to it would mismatch the byte layout.
        if (typeof(T) == typeof(float)) return sizeof(float);
        if (typeof(T) == typeof(double)) return sizeof(double);
        if (typeof(T) == typeof(int)) return sizeof(int);
        if (typeof(T) == typeof(long)) return sizeof(long);
        if (typeof(T) == typeof(Half)) return 2;
        if (typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16)) return 2;
        // Anything else: fall through to Marshal.SizeOf — these types
        // can't actually be serialized (SerializeToBytes throws), but
        // ElementSize is also used for the GpuOffload byteCount which
        // applies to a wider set.
        return System.Runtime.InteropServices.Marshal.SizeOf<T>();
    }

    /// <summary>Serializes a tensor's elements to a raw byte buffer. We
    /// can't constrain T : unmanaged at the API level (Tensor&lt;T&gt; is
    /// generic over arbitrary numeric types including Complex / Multivector),
    /// so this dispatches via per-type fast paths. For unsupported element
    /// types (Complex, Multivector, etc.) we throw <see cref="NotSupportedException"/>
    /// at the bottom — there is intentionally no generic fallback because a
    /// byte-by-byte memcpy would silently lose data for non-blittable types.
    /// </summary>
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

    /// <summary>
    /// Restores a streaming weight's in-memory data from the pool. If the
    /// pool has already paged the bytes out to disk, the backing file is
    /// read first. No-op for tensors that aren't <see cref="WeightLifetime.Streaming"/>
    /// or whose data is already resident.
    /// </summary>
    /// <remarks>
    /// Used by <c>NeuralNetworkBase.Predict</c> and <c>Backpropagate</c>:
    /// before each layer's Forward / Backward, materialize that layer's
    /// trainable tensors via <see cref="MaterializeMany{T}"/> (which wraps
    /// this in a using-scope). After the layer finishes, call
    /// <see cref="ReleaseToPool{T}"/> to free the bytes again.
    /// </remarks>
    public static void Materialize<T>(Tensor<T> weight)
    {
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (weight.Lifetime != WeightLifetime.Streaming) return;
        if (weight.StreamingPoolHandle < 0) return;
        if (weight.DataVector.Length == weight.Length) return; // already resident

        // Two-phase: snapshot bytes under the registry lock (short — pool
        // does the disk read inside its own lock), then drop the lock and
        // do the deserialize-into-tensor memcpy unsynchronized. The memcpy
        // can be tens of milliseconds for hundreds-of-MB weights — holding
        // the registry lock across it would serialize all concurrent
        // Materialize / PrefetchAsync workers and defeat the W=2 prefetch
        // overlap. RehydrateInto returns a caller-owned byte[] so we no
        // longer race against a concurrent eviction nulling entry.Data.
        byte[] snapshot;
        lock (_lock)
        {
            snapshot = StreamingPoolUnlocked().RehydrateInto(weight.StreamingPoolHandle);
        }
        weight.RestoreStorageFromBytes(snapshot);
    }

    /// <summary>
    /// Drops a streaming weight's in-memory data, leaving the canonical
    /// copy with the streaming pool. The tensor's <see cref="Tensor{T}.Length"/>
    /// stays unchanged; subsequent reads must call <see cref="Materialize{T}"/>
    /// first. No-op if the weight isn't <see cref="WeightLifetime.Streaming"/>
    /// or hasn't been registered.
    /// </summary>
    public static void ReleaseToPool<T>(Tensor<T> weight)
    {
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (weight.Lifetime != WeightLifetime.Streaming) return;
        if (weight.StreamingPoolHandle < 0) return;
        if (weight.DataVector.Length == 0) return; // already released
        weight.DropStorageForStreaming();
    }

    /// <summary>
    /// Returns an <see cref="IDisposable"/> scope that materializes
    /// <paramref name="weights"/> on construction and releases them on
    /// disposal. Use around a layer's Forward to guarantee the tensors
    /// are resident only for the duration of the call:
    /// <code>
    /// using (WeightRegistry.MaterializeMany(layer.GetParameterChunks()))
    /// {
    ///     output = layer.Forward(input);
    /// }
    /// </code>
    /// </summary>
    public static MaterializeScope<T> MaterializeMany<T>(IEnumerable<Tensor<T>> weights)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        return new MaterializeScope<T>(weights);
    }

    /// <summary>
    /// IDisposable scope returned by <see cref="MaterializeMany{T}"/>.
    /// Materializes its tensors on construction; releases on disposal.
    /// </summary>
    public sealed class MaterializeScope<T> : IDisposable
    {
        // Rented from the array pool to avoid per-scope allocation. Each
        // Predict layer instantiates a scope, so on a 64-decoder PaLME
        // forward pass we'd otherwise allocate 64 Tensor<T>[] arrays per
        // step — pooling is meaningful at that frequency.
        private readonly Tensor<T>[] _weights;
        private readonly int _count;
        private int _disposed; // 0 = live, 1 = disposed (Interlocked-guarded)

        internal MaterializeScope(IEnumerable<Tensor<T>> weights)
        {
            // Snapshot to an array so disposal sees the same set even if
            // the source enumerable was lazy / mutated mid-scope. We
            // count first, allocate exactly, then materialize — avoids
            // the List<T> intermediate growth and resize copies.
            int count = 0;
            if (weights is ICollection<Tensor<T>> c) count = c.Count;
            else { foreach (var _w in weights) count++; }

            _weights = count == 0
                ? Array.Empty<Tensor<T>>()
                : System.Buffers.ArrayPool<Tensor<T>>.Shared.Rent(count);
            int idx = 0;
            foreach (var w in weights)
            {
                if (w is null) continue;
                Materialize(w);
                _weights[idx++] = w;
            }
            _count = idx;
        }

        public void Dispose()
        {
            // Interlocked guard against double-Dispose from racing threads.
            // The standard `using` pattern is single-threaded but the
            // scope's reference can leak into other threads via DI etc.
            if (System.Threading.Interlocked.Exchange(ref _disposed, 1) != 0) return;
            for (int i = 0; i < _count; i++)
                ReleaseToPool(_weights[i]);
            // Clear the slots before returning to pool — the rented array
            // may outlive this scope inside the pool, and we don't want to
            // root the tensors past Dispose.
            for (int i = 0; i < _count; i++)
                _weights[i] = null!;
            if (_weights.Length > 0)
                System.Buffers.ArrayPool<Tensor<T>>.Shared.Return(_weights);
        }
    }

    /// <summary>
    /// Returns true when this weight's bytes are currently resident in
    /// the streaming pool (no disk read needed on next
    /// <see cref="Materialize{T}"/>). Used by
    /// <c>NeuralNetworkBase.Backpropagate</c> to skip unnecessary
    /// materialize calls during the forward→backward LRU bridge — after
    /// forward, layers are at LRU head and backward starts there.
    /// Returns false for non-Streaming or unregistered tensors.
    /// </summary>
    public static bool IsResidentInPool<T>(Tensor<T> weight)
    {
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (weight.Lifetime != WeightLifetime.Streaming) return false;
        if (weight.StreamingPoolHandle < 0) return false;
        lock (_lock)
        {
            return _streamingPool?.IsResident(weight.StreamingPoolHandle) ?? false;
        }
    }

    /// <summary>
    /// Issues a background read of <paramref name="weight"/>'s bytes from
    /// the streaming pool's backing store into the resident set. Returns
    /// immediately; the next <see cref="Materialize{T}"/> call should hit
    /// the resident set.
    /// </summary>
    /// <remarks>
    /// Called by <c>NeuralNetworkBase.Predict</c> for layer N+W while
    /// layer N is computing — overlap of disk I/O with compute is the
    /// primary perf win vs. PyTorch FSDP's synchronous all-gather. No-op
    /// if the tensor isn't <see cref="WeightLifetime.Streaming"/>.
    /// </remarks>
    public static void PrefetchAsync<T>(Tensor<T> weight)
    {
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (weight.Lifetime != WeightLifetime.Streaming) return;
        if (weight.StreamingPoolHandle < 0) return;

        long handle = weight.StreamingPoolHandle;
        // Bound prefetch worker concurrency to PrefetchMaxConcurrency
        // outstanding workers (default 8). Unbounded queueing could fill
        // the ThreadPool with workers all blocked on the registry lock if
        // a caller spams PrefetchAsync; the semaphore caps the queue
        // depth. Default 8 is well above the typical W=2 schedule so
        // legitimate use never sees Wait(0) fail; pathological callers
        // see the prefetch dropped (next Materialize does the disk read).
        if (!_prefetchSemaphore.Wait(0)) return;

        // Fire-and-forget on the threadpool. The pool's Rehydrate handles
        // the read; callers that race a concurrent Materialize will hit
        // the resident set instead of double-reading.
        // UnsafeQueueUserWorkItem skips ExecutionContext capture — any
        // AsyncLocal<T> values in the calling context are NOT preserved
        // into this worker. That's an intentional perf optimization for
        // the prefetch hot path; if telemetry / logging needs context,
        // capture it explicitly via the closure.
        System.Threading.ThreadPool.UnsafeQueueUserWorkItem(_ =>
        {
            try
            {
                lock (_lock)
                {
                    // Pool may be null if Configure was called between
                    // queue + execute; pool may be the OLD pool if a new
                    // one was swapped in mid-flight (handle would belong
                    // to the old pool, throw "handle unknown" — caught).
                    _streamingPool?.Rehydrate(handle, isPrefetch: true);
                }
            }
            catch (System.IO.IOException) { /* disk error mid-prefetch */ }
            catch (ObjectDisposedException) { /* pool disposed mid-prefetch */ }
            catch (InvalidOperationException) { /* handle unknown after Configure */ }
            catch (UnauthorizedAccessException) { /* permissions changed */ }
            // Don't catch OOM, ThreadAbort, AccessViolation — those are
            // process-level signals that swallowing would hide. Let them
            // surface as TaskScheduler.UnobservedTaskException.
            finally
            {
                _prefetchSemaphore.Release();
            }
        }, state: null);
    }

    // Caps in-flight prefetch workers. 8 is well above typical W=2 schedule
    // so the semaphore is invisible to legitimate callers but bounds queue
    // depth under pathological load. See PrefetchAsync for rationale.
    private const int PrefetchMaxConcurrency = 8;
    private static readonly System.Threading.SemaphoreSlim _prefetchSemaphore =
        new(initialCount: PrefetchMaxConcurrency, maxCount: PrefetchMaxConcurrency);

    /// <summary>
    /// Returns a snapshot of streaming-pool telemetry counters. Caller
    /// typically reads this at end of inference / training pass and
    /// surfaces it in <c>PredictionModelResult.StreamingReport</c>.
    /// When no pool has been lazily-allocated yet (no streaming
    /// registration has happened), returns a zeroed report seeded with
    /// the configured <see cref="GpuOffloadOptions.EnableCompression"/>
    /// flag so consumers can distinguish "compression-on-but-no-traffic"
    /// from "compression-off".
    /// </summary>
    public static StreamingPoolReport GetStreamingReport()
    {
        lock (_lock)
        {
            if (_streamingPool is not null)
                return _streamingPool.GetReport();
            // Seed the CompressionEnabled flag from current options so a
            // user who set EnableCompression=true but hasn't registered
            // anything yet sees the right surface in the report.
            return new StreamingPoolReport(
                ResidentBytes: 0,
                ResidentBytesPeak: 0,
                RegisteredEntryCount: 0,
                DiskReadCount: 0,
                DiskReadBytes: 0,
                DiskWriteBytes: 0,
                EvictionCount: 0,
                CompressionRatio: 1.0,
                CompressionEnabled: _options.EnableCompression,
                PrefetchHitCount: 0,
                PrefetchMissCount: 0,
                PrefetchIssueCount: 0);
        }
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
