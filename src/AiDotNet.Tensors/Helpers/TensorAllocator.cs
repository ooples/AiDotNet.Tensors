using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Tensor allocation with transparent zero-alloc caching via <see cref="ThreadLocalTensorCache{T}"/>.
/// After the first forward pass warms the cache, <see cref="Rent{T}(int[])"/> reuses thread-local
/// buffers — zero allocation, zero GC, zero lock contention, completely invisible to callers.
/// Falls back to ArrayPool for cache misses, then standard allocation when pooling is disabled.
/// Return pooled tensors via <see cref="TensorPool.Return{T}"/> to enable buffer reuse.
/// </summary>
public static class TensorAllocator
{
    /// <summary>
    /// Threshold above which ArrayPool is used instead of standard allocation.
    /// ArrayPool avoids GC pressure for repeated large allocations (e.g., GEMM temporaries).
    /// 256K elements = 1MB for float, 2MB for double.
    /// </summary>
    private const int ArrayPoolThreshold = 256 * 1024;

    /// <summary>
    /// Public accessor for the ArrayPool threshold used by TensorWorkspace and other helpers.
    /// </summary>
    public const int ArrayPoolThresholdValue = ArrayPoolThreshold;

    /// <summary>
    /// Creates a zero-initialized tensor with the given shape.
    /// Large tensors use ArrayPool to reduce GC pressure; small-medium tensors
    /// use standard CLR allocation. All paths return zeroed memory.
    /// </summary>
    public static Tensor<T> Rent<T>(int[] shape)
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);

        // MemoryProfiler hook (#220): only record branches that actually allocate
        // new memory — arena/pool/cache reuse paths do not, and recording them
        // without matching frees would monotonically inflate CurrentBytes. The
        // RecordAllocation calls below are conditional on each tier so the
        // profiler counter reflects real allocations, not Rent invocations.
        // Pairing each record with a Free on tensor disposal is tracked
        // separately (full lifecycle wrapping Tensor<T>).
        long bytesIfTracking = totalSize > 0
            ? (long)totalSize * System.Runtime.CompilerServices.Unsafe.SizeOf<T>()
            : 0L;

        // #318: when consumers enable ForceFreshAllocations they're
        // opting into byte-equal backing arrays across all
        // construction paths (state_dict round-trip determinism,
        // Clone-after-train semantics) at the cost of pool reuse.
        // Same code path as the disabled-pool case — every Rent goes
        // straight to `new Tensor<T>(shape)` which produces a backing
        // array of EXACTLY logical Length.
        if (!TensorPool.Enabled || TensorPool.ForceFreshAllocations || totalSize == 0)
        {
            if (bytesIfTracking > 0)
                AiDotNet.Tensors.Engines.Profiling.Memory.MemoryProfiler.RecordAllocation(
                    "TensorAllocator", bytesIfTracking, shape, typeof(T).Name);
            return new Tensor<T>(shape);
        }

        // Tier 0: Arena allocation — zero GC during training loops.
        var arena = TensorArena.Current;
        if (arena != null)
        {
            T[]? arenaArray = arena.TryAllocate<T>(totalSize);
            if (arenaArray != null)
            {
                var memory = new Memory<T>(arenaArray, 0, totalSize);
                return Tensor<T>.FromMemory(memory, shape);
            }
            // Arena full — fall through to other tiers
        }

#if NET5_0_OR_GREATER
        // Tier 1: Thread-local cache — zero allocation after warmup.
        T[]? cached = ThreadLocalTensorCache<T>.TryRent(totalSize);
        if (cached is null && totalSize >= ArrayPoolThreshold)
            cached = ThreadLocalTensorCache<T>.TryRent(ArrayPoolBucketSize(totalSize));
        if (cached is not null)
        {
            // Issue #311: clear the ENTIRE pooled array, not just the
            // logical portion. The pooled buffer may exceed totalSize
            // (ArrayPool buckets pad to the next power of two; a 401,408-
            // element rent returns a 524,288-element array), and the
            // padding region carries the previous renter's bytes.
            // Downstream kernels that read past the logical extent via
            // SIMD overhang then observe non-zero garbage — making two
            // forward passes through "logically identical" tensors
            // diverge by 3-4% after a couple of layers (DBM clone-after-
            // train). Clearing the whole array makes the zero-init
            // contract layout-invariant: identical logical content +
            // identical padding (= zero) → identical SIMD reduction
            // order across pooled and freshly-allocated tensors.
            Array.Clear(cached, 0, cached.Length);
            var memory = new Memory<T>(cached, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, cached);
        }

        // Tier 3: ArrayPool for large reference types — Rent may return a fresh array
        // or reuse a pooled one; the underlying ArrayPool tracks reuse so we record
        // unconditionally here (RentUninitialized records on the same path).
        if (totalSize >= ArrayPoolThreshold)
        {
            AiDotNet.Tensors.Engines.Profiling.Memory.MemoryProfiler.RecordAllocation(
                "TensorAllocator", bytesIfTracking, shape, typeof(T).Name);
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            // Issue #311: clear the entire array, including the padding
            // beyond totalSize. See the matching comment on the cached
            // path above.
            Array.Clear(pooled, 0, pooled.Length);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

        // Tier 4: Standard managed allocation for small tensors — always a fresh alloc.
        AiDotNet.Tensors.Engines.Profiling.Memory.MemoryProfiler.RecordAllocation(
            "TensorAllocator", bytesIfTracking, shape, typeof(T).Name);
        T[] arr = new T[totalSize];
        var mem = new Memory<T>(arr);
        return Tensor<T>.FromMemory(mem, shape);
#else
        AiDotNet.Tensors.Engines.Profiling.Memory.MemoryProfiler.RecordAllocation(
            "TensorAllocator", bytesIfTracking, shape, typeof(T).Name);
        return new Tensor<T>(shape);
#endif
    }

    /// <summary>
    /// Rents a tensor without zeroing its backing array. The caller MUST write every element
    /// before reading. Use for backward kernels and operations that overwrite the full buffer.
    /// When a TensorArena is active, reuses arena-allocated arrays without clearing — saving
    /// ~0.3ms per 1M-element tensor (the Array.Clear cost).
    /// </summary>
    public static Tensor<T> RentUninitialized<T>(int[] shape)
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);

        // #318: when ForceFreshAllocations is enabled the caller wants
        // backing arrays exactly logical-Length sized. Skip both the
        // arena and pool tiers — go straight to new Tensor<T>(shape).
        if (TensorPool.ForceFreshAllocations)
            return new Tensor<T>(shape);

        // Arena path: reuse the entire Tensor<T> object + backing array — truly zero alloc
        var arena = TensorArena.Current;
        if (arena != null)
        {
            var pooledTensor = arena.TryRentTensor<T>(totalSize, shape);
            if (pooledTensor != null) return pooledTensor;
        }

#if NET5_0_OR_GREATER
        // Thread-local cache: skip Array.Clear
        T[]? cached = ThreadLocalTensorCache<T>.TryRent(totalSize);
        if (cached is null && totalSize >= ArrayPoolThreshold)
            cached = ThreadLocalTensorCache<T>.TryRent(ArrayPoolBucketSize(totalSize));
        if (cached is not null)
        {
            // Reference types: must clear EVERYTHING so we don't retain
            // stale objects in the GC graph.
            // Value types (issue #311): clear only the padding region
            // (totalSize..cached.Length). The caller's contract is that
            // it writes every element of the LOGICAL region, but
            // downstream readers can still SIMD-overhang into padding
            // and observe the prior renter's garbage — that produces
            // ~3-4% drift between original (pooled-padded) and clone
            // (freshly-allocated) Predict outputs after a few layers.
            // Clearing only the padding preserves the "skip clear of
            // logical region" optimization that is the whole point of
            // RentUninitialized; the clear-padding-only cost is the
            // ~25% bucket overhead, not the full buffer.
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
                Array.Clear(cached, 0, cached.Length);
            else if (cached.Length > totalSize)
                Array.Clear(cached, totalSize, cached.Length - totalSize);
            var memory = new Memory<T>(cached, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, cached);
        }

        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            // See matching #311 comment on the cached path above.
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
                Array.Clear(pooled, 0, pooled.Length);
            else if (pooled.Length > totalSize)
                Array.Clear(pooled, totalSize, pooled.Length - totalSize);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

        // Small allocation: new T[] is zeroed by CLR but that's unavoidable
        T[] arr = new T[totalSize];
        return Tensor<T>.FromMemory(new Memory<T>(arr), shape);
#else
        return new Tensor<T>(shape);
#endif
    }

    /// <summary>
    /// Computes the ArrayPool bucket size for a given request.
    /// ArrayPool returns power-of-2 sizes, so we need to match on return.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ArrayPoolBucketSize(int requestedSize)
    {
        // ArrayPool.Shared uses power-of-2 buckets starting at 16
        if (requestedSize <= 16) return 16;
        // Round up to next power of 2
        int v = requestedSize - 1;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

    /// <summary>
    /// Creates a tensor backed by NativeMemory (64-byte aligned, zero GC overhead).
    /// Use for tensors that will ONLY be accessed via Pin()/Memory.Span — never GetDataArray().
    /// oneDNN and VML operations should use this for optimal performance.
    /// GetDataArray() triggers lazy demotion (one-time copy to managed array).
    /// </summary>
    public static Tensor<T> RentNative<T>(int[] shape) where T : unmanaged
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);

        if (totalSize == 0)
            return new Tensor<T>(shape);

#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(float))
        {
            var owner = new NativeMemoryOwner<float>(totalSize, zeroed: true);
            return (Tensor<T>)(object)Tensor<float>.FromMemory(owner.Memory, shape);
        }
        if (typeof(T) == typeof(double))
        {
            var owner = new NativeMemoryOwner<double>(totalSize, zeroed: true);
            return (Tensor<T>)(object)Tensor<double>.FromMemory(owner.Memory, shape);
        }
#endif
        // Fallback for other unmanaged types
        return Rent<T>(shape);
    }

    // RentUninitialized is defined above (line ~95) with arena support

    /// <summary>
    /// Adopts a caller-owned <c>T[]</c> as the backing storage for a new
    /// tensor — zero-copy, no Vector wrapping, no pool churn. Use this on
    /// backward-kernel hot paths where the kernel has already computed
    /// into a fresh array: instead of
    /// <c>Rent&lt;T&gt;(shape, new Vector&lt;T&gt;(arr))</c> (which goes
    /// through <c>IEnumerable&lt;T&gt;.ToArray()</c>, a full copy), call
    /// <see cref="Rent{T}(int[], T[])"/> to skip the duplicate alloc.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Issue #319 Phase 4: this overload is the simplest replacement for
    /// the <c>Rent&lt;T&gt;(shape, new Vector&lt;T&gt;(data))</c> pattern
    /// that appears in ~136 sites in <c>CpuEngine.cs</c>. The Vector
    /// constructor used to call <c>ToArray()</c> on its
    /// <c>IEnumerable&lt;T&gt;</c> input, which means a freshly-computed
    /// gradient array got copied once into the Vector and then again into
    /// the rented tensor (or, on the existing zero-copy fast path, into
    /// a new Vector backing array). This overload skips both steps.
    /// </para>
    /// <para>
    /// <b>Ownership:</b> after this call returns, the caller MUST NOT
    /// retain a reference to <paramref name="data"/>. The tensor wraps
    /// it directly; mutations through the caller's variable become
    /// visible through the tensor and vice versa.
    /// </para>
    /// <para>
    /// <b>Interaction with <see cref="TensorPool.ForceFreshAllocations"/>:</b>
    /// this overload is implicitly compatible — its
    /// <c>data.Length == product(shape)</c> precondition matches the
    /// "backing array length is exactly product(shape)" guarantee that
    /// flag was designed to enforce (no ArrayPool overhang, no pooled
    /// over-sized buffers). Issue #318 callers can use this overload
    /// freely without violating byte-equality contracts.
    /// </para>
    /// </remarks>
    public static Tensor<T> Rent<T>(int[] shape, T[] data)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (data is null) throw new ArgumentNullException(nameof(data));
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);
        if (totalSize != data.Length)
            throw new ArgumentException(
                $"Data length ({data.Length}) must match shape total ({totalSize}).",
                nameof(data));
        return new Tensor<T>(data, shape);
    }

    /// <summary>
    /// Clearer-named alias for <see cref="Rent{T}(int[], T[])"/>. New call
    /// sites should prefer this name — the <c>Rent</c> verb on the other
    /// overload implies pool/arena renting, which can mislead readers of
    /// code that's actually adopting caller-owned storage. <c>Adopt</c>
    /// names the semantic precisely: this overload doesn't rent from any
    /// pool; it takes ownership of an existing <c>T[]</c> and wraps it as
    /// the tensor's backing storage. The other <see cref="Rent{T}(int[], T[])"/>
    /// remains for the ~136 existing call sites; both are zero-copy and
    /// have identical runtime behavior.
    /// </summary>
    /// <param name="shape">The tensor shape; product must equal <c>data.Length</c>.</param>
    /// <param name="data">A caller-owned array. After this call returns, the
    /// caller MUST NOT retain a reference — the tensor wraps it directly.</param>
    public static Tensor<T> Adopt<T>(int[] shape, T[] data)
        => Rent(shape, data);

    /// <summary>
    /// Creates a tensor with the given shape and data from a Vector.
    /// Zero-copy when the Vector's backing array is exactly the right size
    /// (common pattern: caller does new T[n], computes into it, wraps in Vector).
    /// Falls back to copy when the backing array can't be extracted.
    /// </summary>
    public static Tensor<T> Rent<T>(int[] shape, Vector<T> data)
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);

        if (totalSize != data.Length)
            throw new ArgumentException(
                $"Data length ({data.Length}) must match shape total ({totalSize}).",
                nameof(data));

#if NET5_0_OR_GREATER
        // Zero-copy path: if the Vector's backing array is exactly totalSize,
        // wrap it directly — no allocation, no copy. This is the common case when
        // callers do: var arr = new T[n]; compute(arr); Rent(shape, new Vector<T>(arr))
        T[]? backingArray = data._cachedArray;
        if (backingArray is not null && backingArray.Length == totalSize)
        {
            var memory = new Memory<T>(backingArray);
            return Tensor<T>.FromMemory(memory, shape);
        }
#endif

        if (!TensorPool.Enabled || totalSize == 0)
        {
            return new Tensor<T>(shape, data);
        }

#if NET5_0_OR_GREATER
        ReadOnlySpan<T> src = data.AsSpan();
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            src.CopyTo(pooled.AsSpan(0, totalSize));
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>() && pooled.Length > totalSize)
                Array.Clear(pooled, totalSize, pooled.Length - totalSize);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

        T[] array = GC.AllocateUninitializedArray<T>(totalSize);
        src.CopyTo(array);
        var mem = new Memory<T>(array);
        return Tensor<T>.FromMemory(mem, shape);
#else
        return new Tensor<T>(shape, data);
#endif
    }

    /// <summary>
    /// Returns a tensor's backing array to the pool if it was pooled.
    /// SAFETY: The caller MUST ensure the tensor is never accessed after this call.
    /// The tensor's Memory still references the returned array — any access after
    /// Return is undefined behavior (data corruption from reuse).
    /// Only call this for internal temporaries that immediately go out of scope.
    /// External callers should use <see cref="TensorPool.Return{T}"/> instead.
    /// </summary>
    internal static void Return<T>(Tensor<T>? tensor)
    {
        if (tensor == null) return;

        T[]? pooledArray = tensor.PooledArray;
        if (pooledArray != null)
        {
            tensor.DetachPooledArray();
#if NET5_0_OR_GREATER
            // Tier 1: Try thread-local cache first — zero contention, instant reuse.
            if (ThreadLocalTensorCache<T>.TryReturn(pooledArray))
                return;

            // Tier 2: Cache full — fall through to ArrayPool.
            ArrayPool<T>.Shared.Return(pooledArray,
                clearArray: RuntimeHelpers.IsReferenceOrContainsReferences<T>());
#else
            ArrayPool<T>.Shared.Return(pooledArray, clearArray: true);
#endif
        }
    }
}
