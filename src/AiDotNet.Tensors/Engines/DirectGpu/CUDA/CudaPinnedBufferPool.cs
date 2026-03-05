using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Pool of CUDA page-locked (pinned) host memory buffers for true async DMA transfers.
/// Pinned memory stays resident in physical RAM, allowing the GPU DMA engine to
/// transfer data without CPU intervention and without requiring a C# fixed block.
/// </summary>
internal sealed class CudaPinnedBufferPool : IDisposable
{
    private readonly ConcurrentDictionary<int, ConcurrentBag<IntPtr>> _buckets = new();
    private bool _disposed;

    /// <summary>
    /// Rents a pinned host buffer of at least <paramref name="sizeInBytes"/> bytes.
    /// Uses power-of-two bucketing to reduce fragmentation.
    /// </summary>
    public IntPtr Rent(int sizeInBytes)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaPinnedBufferPool));
        int bucketSize = NextPowerOfTwo(sizeInBytes);

        if (_buckets.TryGetValue(bucketSize, out var bag) && bag.TryTake(out var ptr))
            return ptr;

        var result = CuBlasNative.cuMemAllocHost(out ptr, (ulong)bucketSize);
        CuBlasNative.CheckCudaResult(result, "cuMemAllocHost");
        return ptr;
    }

    /// <summary>
    /// Returns a pinned host buffer to the pool.
    /// </summary>
    public void Return(IntPtr ptr, int sizeInBytes)
    {
        if (_disposed || ptr == IntPtr.Zero)
            return;

        int bucketSize = NextPowerOfTwo(sizeInBytes);
        var bag = _buckets.GetOrAdd(bucketSize, _ => new ConcurrentBag<IntPtr>());

        // Limit pool size to prevent unbounded growth
        if (bag.Count < 8)
            bag.Add(ptr);
        else
            CuBlasNative.cuMemFreeHost(ptr);
    }

    private static int NextPowerOfTwo(int value)
    {
        if (value <= 0) return 1;
        value--;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        return value + 1;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var kvp in _buckets)
        {
            while (kvp.Value.TryTake(out var ptr))
            {
                CuBlasNative.cuMemFreeHost(ptr);
            }
        }
        _buckets.Clear();
    }
}
