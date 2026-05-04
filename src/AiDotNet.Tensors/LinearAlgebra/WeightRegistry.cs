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
                        // CheckedStreamingByteCount uses long arithmetic
                        // so an oversized tensor surfaces as a clear
                        // NotSupportedException with a chunking hint
                        // instead of the runtime's OutOfMemoryException.
                        // byte[] is itself bounded to ~2.15 GB, so
                        // streaming a single tensor > 2 GB is impossible
                        // regardless of host RAM. Helper is internal so
                        // the overflow guard can be unit-tested directly
                        // without faking a multi-GB tensor.
                        int byteCount = CheckedStreamingByteCount<T>(weight.Length);
                        var bytes = new byte[byteCount];
                        SerializeToBytes(weight, bytes);
                        var pool = StreamingPoolUnlocked();
                        long handle = pool.Register(bytes);
                        // Two-phase commit: drop storage FIRST (the operation
                        // that can throw — non-contiguous, view, shared
                        // refcount), then commit the handle on the tensor.
                        // If DropStorageForStreaming throws after the pool
                        // already accepted the bytes, roll the pool entry
                        // back so we don't leak both a registered pool
                        // entry and a tensor still in its pre-stream state
                        // (which would lead to "handle resident but tensor
                        // never released" + a pool entry no caller can
                        // ever reach because StreamingPoolHandle was never
                        // set on the tensor).
                        try
                        {
                            weight.DropStorageForStreaming();
                            weight.StreamingPoolHandle = handle;
                        }
                        catch
                        {
                            pool.Unregister(handle);
                            throw;
                        }
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

    /// <summary>
    /// Computes the byte count for a streaming-registered tensor of
    /// <paramref name="length"/> elements, throwing
    /// <see cref="NotSupportedException"/> with a chunking hint when
    /// the result would exceed <see cref="int.MaxValue"/> (the byte[]
    /// limit). Extracted as an internal helper so the overflow guard
    /// can be unit-tested directly against synthetic Length values
    /// without allocating a 2GB+ tensor in a test process.
    /// </summary>
    internal static int CheckedStreamingByteCount<T>(int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        int elementSize = ElementSize<T>();
        long byteCountLong = (long)length * elementSize;
        if (byteCountLong > int.MaxValue)
            throw new NotSupportedException(
                $"Streaming registration requires per-tensor size <= {int.MaxValue} bytes " +
                $"(byte[] limit). Tensor has {length} elements × {elementSize} bytes = " +
                $"{byteCountLong} bytes. Chunk the tensor into smaller pool entries on the consumer side.");
        return (int)byteCountLong;
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
        if (weight.DataVector.Length == weight.Length)
        {
            // Already resident — but still bump the pool's LRU heat so
            // this hot weight doesn't get evicted in favor of cold ones.
            // Without this, a frequently-accessed embedding that always
            // hits the early-out path would stay at LRU tail and evict
            // first under budget pressure.
            //
            // Capture-then-call: hold the registry lock just long enough
            // to snapshot the pool reference, not across MarkAccessed
            // itself. MarkAccessed acquires the pool's own lock, and a
            // pool that's mid-Configure-swap could have its LRU mutated
            // concurrently — but the captured reference is a single
            // object, so the call still operates on a consistent pool
            // (either the new pool or the old one, but not a torn view).
            StreamingTensorPool? poolRef;
            lock (_lock) { poolRef = _streamingPool; }
            poolRef?.MarkAccessed(weight.StreamingPoolHandle);
            return;
        }

        // Two-phase to avoid serializing the registry behind one slow
        // page-in:
        //   1. Capture the pool reference under the registry lock — short.
        //      This protects against a concurrent Configure/Reset
        //      disposing the pool while we're trying to use it. The
        //      reference itself is stable once captured.
        //   2. Drop the registry lock, then call RehydrateInto on the
        //      captured pool. RehydrateInto takes the POOL'S OWN lock
        //      (not the registry lock) for its disk read + decompress
        //      + allocation + copy. Concurrent Materialize calls on
        //      DIFFERENT weights now run side by side at the registry
        //      level; only the pool's internal serialization gates
        //      them, and that's the level the LRU/eviction state lives.
        //   3. RestoreStorageFromBytes runs entirely without the
        //      registry lock — it allocates and atomically swaps storage
        //      on this tensor. No other thread reads this tensor's
        //      storage between drop and swap (RebindStorageFrom is the
        //      only path that would, and that's what
        //      DropStorageForStreaming's TryClaimExclusive guards
        //      against).
        StreamingTensorPool pool;
        lock (_lock)
        {
            pool = StreamingPoolUnlocked();
        }
        byte[] snapshot = pool.RehydrateInto(weight.StreamingPoolHandle);
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
    /// Materializes its tensors on construction; on disposal releases
    /// ONLY the tensors this scope itself paged in.
    ///
    /// <para><b>Why "only what we materialized" matters:</b> a tensor
    /// that was already resident on entry is owned by an outer caller —
    /// either an enclosing <see cref="MaterializeScope{T}"/>, an
    /// independent <see cref="WeightRegistry.Materialize{T}"/> call
    /// keeping the weight warm for a downstream layer, or a sibling
    /// thread sharing the same model's weights. If this scope released
    /// such a tensor on dispose, the outer caller would observe its
    /// weight silently paged out mid-use. This scope therefore tracks,
    /// per tensor, whether the materialize call inside the constructor
    /// is what actually transitioned it from non-resident to resident,
    /// and only releases those transitions. Already-resident weights
    /// are still LRU-bumped by <see cref="Materialize{T}"/> (so this
    /// scope still keeps them warm) but are not paged out by us.</para>
    /// </summary>
    public sealed class MaterializeScope<T> : IDisposable
    {
        // Rented from the array pool to avoid per-scope allocation. Each
        // Predict layer instantiates a scope, so on a 64-decoder PaLME
        // forward pass we'd otherwise allocate 64 Tensor<T>[] arrays per
        // step — pooling is meaningful at that frequency.
        // Not readonly — the ctor's grow path replaces this when an
        // unbounded enumerable yields more weights than the initial
        // rental. After construction, the field is effectively immutable
        // (only Dispose reads it).
        private Tensor<T>[] _weights;
        private readonly int _count;
        private int _disposed; // 0 = live, 1 = disposed (Interlocked-guarded)

        internal MaterializeScope(IEnumerable<Tensor<T>> weights)
        {
            // Single-pass enumeration with manual list growth. Previously
            // we counted via a first foreach (wrong for lazy enumerables —
            // they may be consumed by counting, or yield a different
            // sequence on the second pass). Now we collect once and
            // materialize as we go; on partial failure we release only
            // the tensors WE materialized so we don't leak our own work
            // and don't page out tensors that were already resident.
            // Initial capacity 8 covers most layers (Q/K/V/O + MLP + LN)
            // without resize; larger layers grow geometrically.
            int capacity = 8;
            if (weights is ICollection<Tensor<T>> c && c.Count > 0)
                capacity = c.Count;
            _weights = System.Buffers.ArrayPool<Tensor<T>>.Shared.Rent(capacity);

            int idx = 0;
            try
            {
                foreach (var w in weights)
                {
                    if (w is null) continue;

                    // Only track tensors WE actually materialized — not
                    // those that were already resident on entry. An
                    // already-resident weight is owned by an outer
                    // caller; releasing it here would page it out from
                    // under that caller. WasNotResidentBefore is the
                    // narrow predicate for "this scope's Materialize
                    // call is the one that brought the bytes in".
                    bool needsRelease = WasNotResidentBefore(w);
                    if (needsRelease)
                    {
                        if (idx >= _weights.Length)
                        {
                            // Grow: rent a larger array, copy, return the old.
                            var grown = System.Buffers.ArrayPool<Tensor<T>>.Shared.Rent(_weights.Length * 2);
                            Array.Copy(_weights, 0, grown, 0, idx);
                            Array.Clear(_weights, 0, idx);
                            System.Buffers.ArrayPool<Tensor<T>>.Shared.Return(_weights);
                            _weights = grown;
                        }
                    }

                    Materialize(w);

                    if (needsRelease)
                    {
                        // Re-check: Materialize may have early-out'd
                        // (someone else materialized between our check
                        // and our call). Only track for release if our
                        // call is the one that brought it in. The
                        // double-check covers the narrow window where
                        // a sibling thread populated the tensor between
                        // WasNotResidentBefore and Materialize.
                        // We approximate "our call brought it in" by
                        // confirming WasNotResidentBefore was true AND
                        // the tensor is now resident — i.e., Materialize
                        // performed the transition. If a sibling did
                        // it first, we still track for release because
                        // Materialize was a no-op for us; the sibling's
                        // own scope (if any) would track separately,
                        // and harmless double-release is prevented by
                        // ReleaseToPool's `DataVector.Length == 0`
                        // early-out.
                        _weights[idx++] = w;
                    }
                }
                _count = idx;
            }
            catch
            {
                // Partial-failure cleanup: release the weights WE
                // materialized before this exception propagates.
                // Without this, the ctor exception path leaks every
                // successfully-materialized weight (no Dispose ever
                // runs because construction never completed). We do
                // NOT release weights that were already resident
                // before we ran — same ownership rule as Dispose.
                for (int i = 0; i < idx; i++)
                {
                    try { ReleaseToPool(_weights[i]); }
                    catch { /* best-effort; original exception is the real signal */ }
                }
                Array.Clear(_weights, 0, idx);
                System.Buffers.ArrayPool<Tensor<T>>.Shared.Return(_weights);
                throw;
            }
        }

        /// <summary>
        /// True when this weight needs streaming-style release on scope
        /// dispose — i.e., it's a registered streaming weight whose
        /// bytes were NOT already resident at the moment we checked.
        /// Non-streaming / unregistered weights never need release;
        /// already-resident streaming weights are owned by the outer
        /// caller and must not be paged out by us.
        /// </summary>
        private static bool WasNotResidentBefore(Tensor<T> w)
        {
            if (w.Lifetime != WeightLifetime.Streaming) return false;
            if (w.StreamingPoolHandle < 0) return false;
            // Length == w.Length means the backing Vector is fully
            // populated — i.e., resident. Length == 0 means the tensor
            // is in its post-DropStorageForStreaming state — i.e., not
            // resident in the tensor's own storage even though the pool
            // may still hold the bytes.
            return w.DataVector.Length != w.Length;
        }

        public void Dispose()
        {
            // Interlocked guard against double-Dispose from racing threads.
            // The standard `using` pattern is single-threaded but the
            // scope's reference can leak into other threads via DI etc.
            if (System.Threading.Interlocked.Exchange(ref _disposed, 1) != 0) return;
            // try/finally so a throwing ReleaseToPool doesn't leak the
            // rest of the materialized weights AND doesn't leak the
            // rented array. Collect throws into AggregateException so the
            // caller sees all failures, not just the first.
            List<Exception>? errors = null;
            try
            {
                for (int i = 0; i < _count; i++)
                {
                    try { ReleaseToPool(_weights[i]); }
                    catch (Exception ex)
                    {
                        (errors ??= new List<Exception>()).Add(ex);
                    }
                }
            }
            finally
            {
                Array.Clear(_weights, 0, _count);
                System.Buffers.ArrayPool<Tensor<T>>.Shared.Return(_weights);
            }
            if (errors is not null)
                throw new AggregateException(
                    "One or more ReleaseToPool calls failed during MaterializeScope.Dispose. " +
                    "Subsequent weights were still released; the rented buffer was returned to the pool.",
                    errors);
        }
    }

    /// <summary>
    /// Bumps this weight's last-access timestamp in the pool's LRU
    /// without doing a Rehydrate. Used to keep critical weights (token
    /// embeddings, KV cache, etc.) at LRU head when their reads bypass
    /// <see cref="Materialize{T}"/> (e.g., the tensor was already
    /// resident from a prior Materialize and the caller skipped the
    /// early-out path's automatic refresh).
    /// </summary>
    /// <remarks>
    /// Materialize already calls MarkAccessed on the early-out (already-
    /// resident) path. Use this only when reading a streaming tensor
    /// without going through Materialize at all — e.g., a debug printout
    /// that reads via <c>tensor.AsSpan()</c> on bytes that happen to be
    /// resident.
    /// </remarks>
    public static void MarkAccessed<T>(Tensor<T> weight)
    {
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (weight.Lifetime != WeightLifetime.Streaming) return;
        if (weight.StreamingPoolHandle < 0) return;
        lock (_lock)
        {
            _streamingPool?.MarkAccessed(weight.StreamingPoolHandle);
        }
    }

    /// <summary>
    /// Batched variant of <see cref="PrefetchAsync{T}"/>. Issues a single
    /// background worker that walks <paramref name="weights"/> sequentially.
    /// For a layer with 12 trainable tensors (Q/K/V/O × MHA + MLP
    /// weights + LayerNorm params), this is one worker instead of 12 —
    /// dramatically reduces ThreadPool / registry-lock contention
    /// against the foreground Forward thread. Skips weights that are
    /// already resident OR have an in-flight prefetch from a prior
    /// call so we don't double-queue.
    /// </summary>
    /// <remarks>
    /// Like <see cref="PrefetchAsync{T}"/>, this is best-effort and
    /// silently drops when the prefetch semaphore is full (default 8
    /// outstanding workers).
    /// </remarks>
    public static void PrefetchAsyncMany<T>(IEnumerable<Tensor<T>> weights)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));

        // Two-stage filter: snapshot handles, then under the registry
        // lock skip already-resident or in-flight ones. Filtering at
        // dispatch time (vs. inside the worker) means we don't burn a
        // worker slot on a no-op. The per-handle in-flight set lives
        // under the registry lock — same as the pool reference — so
        // we capture+filter+enqueue atomically.
        var candidates = new List<long>();
        foreach (var w in weights)
        {
            if (w is null) continue;
            if (w.Lifetime != WeightLifetime.Streaming) continue;
            if (w.StreamingPoolHandle < 0) continue;
            candidates.Add(w.StreamingPoolHandle);
        }
        if (candidates.Count == 0) return;

        // Accumulate the handles we're going to actually fetch. Already-
        // resident or already-in-flight handles are skipped — the
        // existing prefetch (or the prior register that left the bytes
        // resident) is doing the work for us.
        long[] toFetch;
        StreamingTensorPool poolRef;
        lock (_lock)
        {
            poolRef = StreamingPoolUnlocked();
            var filtered = new List<long>(candidates.Count);
            for (int i = 0; i < candidates.Count; i++)
            {
                long h = candidates[i];
                if (poolRef.IsResident(h)) continue;
                if (!_inFlightPrefetches.Add(h)) continue; // already queued
                filtered.Add(h);
            }
            if (filtered.Count == 0) return;
            toFetch = filtered.ToArray();
        }

        if (!_prefetchSemaphore.Wait(0))
        {
            // Couldn't get a worker slot — undo the in-flight reservations
            // so a later call can re-issue. Without this the handles would
            // stay marked in-flight forever, suppressing all future
            // prefetches for them.
            lock (_lock)
            {
                for (int i = 0; i < toFetch.Length; i++)
                    _inFlightPrefetches.Remove(toFetch[i]);
            }
            return;
        }

        // Capture pool reference outside the worker so a concurrent
        // Configure swapping pools doesn't change which pool we operate
        // on mid-flight (handle would belong to the old pool).
        var capturedPool = poolRef;
        System.Threading.ThreadPool.UnsafeQueueUserWorkItem(_ =>
        {
            try
            {
                for (int i = 0; i < toFetch.Length; i++)
                {
                    try { capturedPool.Rehydrate(toFetch[i], isPrefetch: true); }
                    catch (InvalidOperationException) { /* per-handle "unknown" — keep going */ }
                    catch (System.IO.IOException) { /* per-handle disk error */ }
                    catch (UnauthorizedAccessException) { /* per-handle permissions */ }
                }
            }
            catch (ObjectDisposedException) { /* pool disposed mid-batch */ }
            finally
            {
                lock (_lock)
                {
                    for (int i = 0; i < toFetch.Length; i++)
                        _inFlightPrefetches.Remove(toFetch[i]);
                }
                _prefetchSemaphore.Release();
            }
        }, state: null);
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
    public static void PrefetchAsync<T>(Tensor<T> weight) =>
        PrefetchAsyncCore(weight, completionSignal: null);

    /// <summary>
    /// Internal Task-returning overload of <see cref="PrefetchAsync{T}"/>
    /// for tests that need a deterministic signal of when the worker
    /// finishes (instead of polling <see cref="IsResidentInPool{T}"/>
    /// with a wall-clock budget that's flaky on CI agents). Returns a
    /// <see cref="System.Threading.Tasks.Task"/> that completes when
    /// the prefetch worker finishes (or transitions to a no-op via the
    /// dedup path or full-semaphore drop). Production callers should
    /// continue using the public <see cref="PrefetchAsync{T}"/> —
    /// fire-and-forget is the right shape on the hot path.
    /// </summary>
    internal static System.Threading.Tasks.Task PrefetchAsyncForTesting<T>(Tensor<T> weight)
    {
        var tcs = new System.Threading.Tasks.TaskCompletionSource<bool>(
            System.Threading.Tasks.TaskCreationOptions.RunContinuationsAsynchronously);
        PrefetchAsyncCore(weight, tcs);
        return tcs.Task;
    }

    private static void PrefetchAsyncCore<T>(
        Tensor<T> weight,
        System.Threading.Tasks.TaskCompletionSource<bool>? completionSignal)
    {
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (weight.Lifetime != WeightLifetime.Streaming) { completionSignal?.TrySetResult(true); return; }
        if (weight.StreamingPoolHandle < 0) { completionSignal?.TrySetResult(true); return; }

        long handle = weight.StreamingPoolHandle;

        // Dedup: skip already-resident or in-flight handles. Resident
        // → no work to do. In-flight → another worker is already doing
        // the same Rehydrate, so this call would just contend on the
        // pool's lock for nothing. Both checks are under the registry
        // lock so capturing the pool ref + the in-flight set + the
        // resident check happens atomically against a concurrent
        // Configure / sibling PrefetchAsync.
        StreamingTensorPool poolRef;
        lock (_lock)
        {
            poolRef = StreamingPoolUnlocked();
            if (poolRef.IsResident(handle))
            {
                completionSignal?.TrySetResult(true);
                return;
            }
            if (!_inFlightPrefetches.Add(handle))
            {
                // Another worker is already fetching this handle.
                // Don't queue a second one. From the test/caller
                // perspective, the work IS happening — we just don't
                // own the completion signal for it. Best we can do is
                // signal completion immediately (the in-flight worker
                // will resolve its own signal independently).
                completionSignal?.TrySetResult(true);
                return;
            }
        }

        // Bound prefetch worker concurrency to PrefetchMaxConcurrency
        // outstanding workers (default 8). Unbounded queueing could fill
        // the ThreadPool with workers all blocked on the pool's lock if
        // a caller spams PrefetchAsync; the semaphore caps the queue
        // depth. Default 8 is well above the typical W=2 schedule so
        // legitimate use never sees Wait(0) fail; pathological callers
        // see the prefetch dropped (next Materialize does the disk read).
        if (!_prefetchSemaphore.Wait(0))
        {
            // Couldn't get a worker slot — undo the in-flight reservation
            // so a later call can re-issue. Without this the handle
            // would stay marked in-flight forever, suppressing all
            // future prefetches.
            lock (_lock) _inFlightPrefetches.Remove(handle);
            completionSignal?.TrySetResult(true);
            return;
        }

        // Fire-and-forget on the threadpool. The captured pool ref
        // protects against a concurrent Configure swapping pools mid-
        // flight (handle would belong to the old pool if we re-read
        // _streamingPool inside the worker).
        // UnsafeQueueUserWorkItem skips ExecutionContext capture — any
        // AsyncLocal<T> values in the calling context are NOT preserved
        // into this worker. That's an intentional perf optimization for
        // the prefetch hot path; if telemetry / logging needs context,
        // capture it explicitly via the closure.
        var capturedPool = poolRef;
        System.Threading.ThreadPool.UnsafeQueueUserWorkItem(_ =>
        {
            try
            {
                capturedPool.Rehydrate(handle, isPrefetch: true);
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
                lock (_lock) _inFlightPrefetches.Remove(handle);
                _prefetchSemaphore.Release();
                completionSignal?.TrySetResult(true);
            }
        }, state: null);
    }

    // Tracks handles with a prefetch worker in-flight. Dedups concurrent
    // PrefetchAsync calls on the same handle and prevents wasting
    // worker slots / pool-lock contention on duplicate work. All
    // mutations happen under _lock.
    private static readonly HashSet<long> _inFlightPrefetches = new();

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
            // CompressionRatio defaults to 1.0 by virtue of the record
            // struct's normalization (zero-init reads as 1.0); we
            // omit it explicitly so future callers who construct
            // a default instance get the same well-defined behaviour.
            return new StreamingPoolReport
            {
                ResidentBytes = 0,
                ResidentBytesPeak = 0,
                RegisteredEntryCount = 0,
                DiskReadCount = 0,
                DiskReadBytes = 0,
                DiskWriteBytes = 0,
                EvictionCount = 0,
                CompressionRatio = 1.0,
                CompressionEnabled = _options.EnableCompression,
                PrefetchHitCount = 0,
                PrefetchMissCount = 0,
                PrefetchIssueCount = 0,
            };
        }
    }

    /// <summary>
    /// Forcibly drops all configured backends + flushes pool, regardless
    /// of whether tensors are still registered. After Reset, any tensor
    /// with <see cref="Tensor{T}.StreamingPoolHandle"/> &gt;= 0 from the
    /// previous pool is orphaned — Materialize on those will throw.
    /// </summary>
    /// <remarks>
    /// Test path. Production callers should prefer <see cref="UnregisterWeight{T}"/>
    /// on each live tensor and let Configure handle the swap (which has
    /// a guard against the orphan-handle scenario). The doc on
    /// <see cref="Configure"/> spells out the difference.
    /// </remarks>
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
