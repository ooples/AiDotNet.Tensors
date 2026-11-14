using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

internal interface IPoolableGpuBuffer
{
    /// <summary>Gets the physical allocation capacity in elements.</summary>
    int Capacity { get; }

    /// <summary>Marks the allocation as rented for the requested logical element count.</summary>
    void MarkRented(int size);
    void Release();
}

/// <summary>
/// Typed reuse domain for an asynchronous GPU buffer. The global value preserves the
/// existing single-stream CUDA/HIP pool. Backends with independent command queues use a
/// native-queue affinity so an in-flight allocation is never recycled onto an unordered queue.
/// </summary>
internal readonly struct GpuBufferPoolAffinity : IEquatable<GpuBufferPoolAffinity>
{
    private readonly long _value;

    private GpuBufferPoolAffinity(long value) => _value = value;

    internal static GpuBufferPoolAffinity Global => default;

    internal static GpuBufferPoolAffinity ForNativeQueue(IntPtr queue)
    {
        if (queue == IntPtr.Zero)
            throw new ArgumentException("A GPU buffer-pool queue affinity requires a valid native queue.", nameof(queue));
        return new GpuBufferPoolAffinity(queue.ToInt64());
    }

    public bool Equals(GpuBufferPoolAffinity other) => _value == other._value;
    public override bool Equals(object? obj) => obj is GpuBufferPoolAffinity other && Equals(other);
    public override int GetHashCode() => _value.GetHashCode();
}

internal sealed class GpuBufferPool<TBuffer> : IDisposable where TBuffer : class, IGpuBuffer, IPoolableGpuBuffer
{
    private sealed class Bucket
    {
        // A stack, not a ConcurrentBag: a bag keeps each thread's items in a ThreadLocal, i.e. in that thread's static
        // storage, so a pooled buffer was rooted by every thread that ever returned one - and through the buffer's
        // return-to-pool delegate, the whole backend (its OpenCL context and compiled programs, ~110 MB): a backend
        // dropped without Dispose was never collected.
        public readonly ConcurrentDictionary<GpuBufferPoolAffinity, ConcurrentStack<TBuffer>> Buffers = new();
        public int Count;
    }

    private readonly ConcurrentDictionary<int, Bucket> _buckets = new();
    private readonly int _maxPerSize;
    private readonly int _maxSize;
    // Elements parked in the pool, against MaxPooledElements. A per-size count alone let a training loop return far
    // more buffers of one size than it kept, and every extra one went back to the driver behind a completion marker.
    private long _pooledElements;

    /// <summary>
    /// The most elements the pool keeps parked across all sizes: a caching allocator's budget. A buffer returned past
    /// it is released to the driver. Unlimited by default; a backend sets it from its device memory.
    /// </summary>
    internal long MaxPooledElements { get; set; } = long.MaxValue;
    private readonly object _lifecycleGate = new();
    private long _drainedAtMeasurement;
    // Buffers a thread returned while it measured kernel candidates (their workspaces), until they are rented again or
    // released. The drain after a measurement releases only these.
    private readonly ConcurrentDictionary<TBuffer, byte> _measurementParked =
        new(Autodiff.ReferenceEqualityComparer<TBuffer>.Instance);
    // Entries in _measurementParked, read on every rent/return instead of ConcurrentDictionary.IsEmpty, which takes
    // all of the dictionary's locks (it showed as lock contention on the hot launch path of VGG-11 inference).
    private int _measurementParkedCount;
    private int _disposed;

    private void UnmarkMeasurementParked(TBuffer buffer)
    {
        if (Volatile.Read(ref _measurementParkedCount) != 0 && _measurementParked.TryRemove(buffer, out _))
            Interlocked.Decrement(ref _measurementParkedCount);
    }

    /// <summary>Buffers currently parked in the pool (returned and not yet rented again): device memory the pool holds.</summary>
    internal int PooledCount
    {
        get
        {
            int count = 0;
            foreach (var bucket in _buckets.Values) count += Volatile.Read(ref bucket.Count);
            return count;
        }
    }

    // With exact sizes a bucket holds one capacity only, so a rent never passes over a candidate.
    private readonly bool _exactSizes;

    public GpuBufferPool(int maxPerSize, int maxSize) : this(maxPerSize, maxSize, exactSizes: false)
    {
    }

    /// <summary>
    /// Creates a pool. With <paramref name="exactSizes"/> a buffer is reused only for a request of exactly its
    /// capacity, for a backend whose buffers cannot carry a logical size smaller than their allocation.
    /// </summary>
    public GpuBufferPool(int maxPerSize, int maxSize, bool exactSizes)
    {
        _maxPerSize = maxPerSize > 0 ? maxPerSize : 1;
        _maxSize = maxSize > 0 ? maxSize : int.MaxValue;
        _exactSizes = exactSizes;
        // Tuning measured before this pool existed parked nothing in it. Starting the marker at zero made the first
        // rent of any pool created after a tuning run drain the buffers just returned to it.
        _drainedAtMeasurement = Helpers.Autotune.TunedKernels.TunedKernelRegistry.MeasurementGeneration;
    }

    private int BucketKey(int size) => _exactSizes ? size : NextPowerOfTwo(size);

    /// <summary>
    /// Rounds up to the next power of two for bucket key.
    /// This increases cache hit rate by reducing the number of distinct bucket sizes.
    /// A request for 1000 elements reuses a buffer allocated for 1024 elements.
    /// </summary>
    private static int NextPowerOfTwo(int size)
    {
        if (size <= 0) return 1;
        size--;
        size |= size >> 1;
        size |= size >> 2;
        size |= size >> 4;
        size |= size >> 8;
        size |= size >> 16;
        return size + 1;
    }

    public bool TryRent(int size, out TBuffer? buffer)
        => TryRent(size, GpuBufferPoolAffinity.Global, out buffer);

    public bool TryRent(int size, GpuBufferPoolAffinity affinity, out TBuffer? buffer)
    {
        buffer = null;
        if (Volatile.Read(ref _disposed) != 0 || size <= 0 || size > _maxSize)
        {
            return false;
        }

        // Kernel tuning measured candidates since this pool was last used: their workspaces, parked at sizes nothing
        // asks for again (44 MB idle at a CNN's peak), are released once. Only those: draining everything let tuning on
        // one thread empty the pool between another thread's Return and the TryRent expecting the buffer back.
        long measurements = Helpers.Autotune.TunedKernels.TunedKernelRegistry.MeasurementGeneration;
        if (measurements != Volatile.Read(ref _drainedAtMeasurement)
            && Interlocked.Exchange(ref _drainedAtMeasurement, measurements) != measurements)
            DrainMeasurementParked();

        // Use power-of-two bucket key for higher cache hit rate
        int bucketKey = BucketKey(size);
        if (_buckets.TryGetValue(bucketKey, out var bucket)
            && bucket.Buffers.TryGetValue(affinity, out var affinityBuffers))
        {
            List<TBuffer>? undersized = null;
            while (affinityBuffers.TryPop(out var candidate))
            {
                // Debited as it leaves the bag, like the count: an undersized candidate goes back through Return, which
                // credits it again, so debiting only the chosen one counted every passed-over buffer twice.
                Interlocked.Decrement(ref bucket.Count);
                Interlocked.Add(ref _pooledElements, -candidate.Capacity);

                // Power-of-two bucketing can contain different physical capacities (for example,
                // both 6272 and 8192 elements in bucket 8192). Keep searching instead of turning
                // an undersized top entry into a false miss when a sufficient entry is also pooled.
                if (candidate.Capacity < size)
                {
                    (undersized ??= new List<TBuffer>()).Add(candidate);
                    continue;
                }

                RestoreUndersizedCandidates(undersized, affinity);
                UnmarkMeasurementParked(candidate);
                candidate.MarkRented(size);
                buffer = candidate;
                return true;
            }

            RestoreUndersizedCandidates(undersized, affinity);
        }

        return false;
    }

    private void RestoreUndersizedCandidates(
        List<TBuffer>? candidates,
        GpuBufferPoolAffinity affinity)
    {
        if (candidates is null)
            return;

        // Re-enter through Return so disposal, maximum-size checks, and capacity accounting remain
        // atomic with the lifecycle gate. Adding directly to the bag could race Dispose and orphan
        // an allocation in a bucket that has already been removed.
        foreach (TBuffer candidate in candidates)
            Return(candidate, affinity);
    }

    public void Return(TBuffer buffer)
        => Return(buffer, GpuBufferPoolAffinity.Global);

    public void Return(TBuffer buffer, GpuBufferPoolAffinity affinity)
    {
        if (Volatile.Read(ref _disposed) != 0 || buffer.Capacity > _maxSize)
        {
            UnmarkMeasurementParked(buffer);
            buffer.Release();
            return;
        }

        lock (_lifecycleGate)
        {
            if (Volatile.Read(ref _disposed) != 0)
            {
                UnmarkMeasurementParked(buffer);
                buffer.Release();
                return;
            }

            // Keep bucket creation, capacity accounting, and insertion atomic with Dispose's drain.
            // Otherwise Dispose can clear the dictionary between GetOrAdd and Add, orphaning a live
            // device buffer in a bucket that is no longer reachable.
            int bucketKey = BucketKey(buffer.Capacity);
            var bucket = _buckets.GetOrAdd(bucketKey, _ => new Bucket());
            int count = Interlocked.Increment(ref bucket.Count);
            if (count > _maxPerSize || Interlocked.Add(ref _pooledElements, buffer.Capacity) > MaxPooledElements)
            {
                Interlocked.Decrement(ref bucket.Count);
                if (count <= _maxPerSize) Interlocked.Add(ref _pooledElements, -buffer.Capacity);
                UnmarkMeasurementParked(buffer);
                buffer.Release();
                return;
            }

            // Marked before it is visible to a rent, which clears the mark.
            if (Helpers.Autotune.TunedKernels.TunedKernelRegistry.MeasuringOnThisThread)
                if (_measurementParked.TryAdd(buffer, 0)) Interlocked.Increment(ref _measurementParkedCount);
            bucket.Buffers.GetOrAdd(affinity, _ => new ConcurrentStack<TBuffer>()).Push(buffer);
        }
    }

    /// <summary>
    /// Releases the buffers returned during a kernel-tuning measurement and keeps every other pooled buffer. Returns the
    /// number released.
    /// </summary>
    internal int DrainMeasurementParked()
    {
        if (Volatile.Read(ref _measurementParkedCount) == 0) return 0;
        int released = 0;
        lock (_lifecycleGate)
        {
            if (Volatile.Read(ref _disposed) != 0) return 0;
            var kept = new List<TBuffer>();
            foreach (var bucket in _buckets.Values)
            {
                foreach (var affinityBuffers in bucket.Buffers.Values)
                {
                    kept.Clear();
                    while (affinityBuffers.TryPop(out var buffer))
                    {
                        if (!_measurementParked.TryRemove(buffer, out _))
                        {
                            kept.Add(buffer);
                            continue;
                        }
                        Interlocked.Decrement(ref _measurementParkedCount);
                        Interlocked.Decrement(ref bucket.Count);
                        Interlocked.Add(ref _pooledElements, -buffer.Capacity);
                        buffer.Release();
                        released++;
                    }
                    foreach (var buffer in kept) affinityBuffers.Push(buffer);
                }
            }
        }
        return released;
    }

    /// <summary>
    /// Releases every pooled buffer back to the driver (a real free via
    /// <see cref="IPoolableGpuBuffer.Release"/>, NOT a retain) while leaving the
    /// pool usable for subsequent rents. Called under device-memory pressure —
    /// when an allocation OOMs — to reclaim the bounded-but-real device memory
    /// the pool holds for reuse before the caller gives up. Returns the number
    /// of buffers released.
    /// </summary>
    public int DrainAll()
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            return 0;
        }

        int released = 0;
        foreach (var bucket in _buckets.Values)
        {
            foreach (var affinityBuffers in bucket.Buffers.Values)
            {
                while (affinityBuffers.TryPop(out var buffer))
                {
                    Interlocked.Decrement(ref bucket.Count);
                    Interlocked.Add(ref _pooledElements, -buffer.Capacity);
                    UnmarkMeasurementParked(buffer);
                    buffer.Release();
                    released++;
                }
            }
        }
        return released;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        // At process/AppDomain teardown the GPU context may be gone (buffer.Release would fault). Either is fatal on the
        // finalizer thread. The OS reclaims all device memory at exit, so skip the drain.
        if (AiDotNet.Tensors.Engines.RuntimeShutdown.IsTearingDown)
        {
            return;
        }

        lock (_lifecycleGate)
        {
            try
            {
                foreach (var bucket in _buckets.Values)
                {
                    foreach (var affinityBuffers in bucket.Buffers.Values)
                    {
                        while (affinityBuffers.TryPop(out var buffer))
                        {
                            buffer.Release();
                        }
                    }
                }

                _buckets.Clear();
                _measurementParked.Clear();
                Volatile.Write(ref _measurementParkedCount, 0);
            }
            catch (ObjectDisposedException)
            {
                // A concurrent teardown disposed the bag's ThreadLocal mid-drain. Cleanup is
                // best-effort here (OS reclaims at exit); never let it crash the finalizer.
            }
        }
    }
}
