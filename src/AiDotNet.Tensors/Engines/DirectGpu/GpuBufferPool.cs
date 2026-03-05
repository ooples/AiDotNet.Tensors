using System.Collections.Concurrent;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

internal interface IPoolableGpuBuffer
{
    void MarkRented();
    void Release();
}

internal sealed class GpuBufferPool<TBuffer> : IDisposable where TBuffer : class, IGpuBuffer, IPoolableGpuBuffer
{
    private sealed class Bucket
    {
        public readonly ConcurrentBag<TBuffer> Buffers = new();
        public int Count;
    }

    private readonly ConcurrentDictionary<int, Bucket> _buckets = new();
    private readonly int _maxPerSize;
    private readonly int _maxSize;
    private int _disposed;

    public GpuBufferPool(int maxPerSize, int maxSize)
    {
        _maxPerSize = maxPerSize > 0 ? maxPerSize : 1;
        _maxSize = maxSize > 0 ? maxSize : int.MaxValue;
    }

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
    {
        buffer = null;
        if (Volatile.Read(ref _disposed) != 0 || size <= 0 || size > _maxSize)
        {
            return false;
        }

        // Use power-of-two bucket key for higher cache hit rate
        int bucketKey = NextPowerOfTwo(size);
        if (_buckets.TryGetValue(bucketKey, out var bucket) && bucket.Buffers.TryTake(out var candidate))
        {
            // Verify the pooled buffer's actual allocation is large enough.
            // Power-of-two bucketing can match a smaller buffer (e.g., 6272 → bucket 8192)
            // with a larger request (e.g., 8192 → bucket 8192).
            if (candidate.Size < size)
            {
                // Return the too-small buffer to the pool (count stays consistent since
                // we took one out and are putting the same one back)
                bucket.Buffers.Add(candidate);
                return false;
            }

            Interlocked.Decrement(ref bucket.Count);
            candidate.MarkRented();
            buffer = candidate;
            return true;
        }

        return false;
    }

    public void Return(TBuffer buffer)
    {
        if (Volatile.Read(ref _disposed) != 0 || buffer.Size > _maxSize)
        {
            buffer.Release();
            return;
        }

        // Use power-of-two bucket key matching TryRent
        int bucketKey = NextPowerOfTwo(buffer.Size);
        var bucket = _buckets.GetOrAdd(bucketKey, _ => new Bucket());
        int count = Interlocked.Increment(ref bucket.Count);
        if (Volatile.Read(ref _disposed) != 0)
        {
            Interlocked.Decrement(ref bucket.Count);
            buffer.Release();
            return;
        }
        if (count > _maxPerSize)
        {
            Interlocked.Decrement(ref bucket.Count);
            buffer.Release();
            return;
        }

        bucket.Buffers.Add(buffer);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        foreach (var bucket in _buckets.Values)
        {
            while (bucket.Buffers.TryTake(out var buffer))
            {
                buffer.Release();
            }
        }

        _buckets.Clear();
    }
}
