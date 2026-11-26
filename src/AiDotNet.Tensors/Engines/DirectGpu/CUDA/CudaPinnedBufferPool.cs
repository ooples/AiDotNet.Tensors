using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DevicePrimitives;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Pool of CUDA page-locked (pinned) host memory buffers for true async DMA transfers.
/// Pinned memory stays resident in physical RAM, allowing the GPU DMA engine to
/// transfer data without CPU intervention and without requiring a C# fixed block.
/// </summary>
internal sealed class CudaPinnedBufferPool : IDisposable
{
    private const uint CU_MEMHOSTALLOC_PORTABLE = 1;

    [DllImport("nvcuda", EntryPoint = "cuMemHostAlloc")]
    private static extern CudaResult cuMemHostAllocWin(out IntPtr pp, ulong bytesize, uint flags);

    [DllImport("libcuda.so.1", EntryPoint = "cuMemHostAlloc")]
    private static extern CudaResult cuMemHostAllocLinux(out IntPtr pp, ulong bytesize, uint flags);

    internal static CudaResult AllocatePortablePinnedHost(out IntPtr ptr, ulong byteSize)
    {
        try
        {
            var res = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? cuMemHostAllocWin(out ptr, byteSize, CU_MEMHOSTALLOC_PORTABLE)
                : cuMemHostAllocLinux(out ptr, byteSize, CU_MEMHOSTALLOC_PORTABLE);
            if (res == CudaResult.Success)
                return res;
        }
        catch (EntryPointNotFoundException)
        {
            // Fall back to cuMemAllocHost_v2 on drivers without cuMemHostAlloc.
        }
        catch (DllNotFoundException)
        {
            // Fall back to CuBlasNative.cuMemAllocHost so standard loader diagnostics apply.
        }

        return CuBlasNative.cuMemAllocHost(out ptr, byteSize);
    }

    // Stacks, not ConcurrentBags: a bag's per-thread lists live in a ThreadLocal, which the GC finalizes with an abandoned
    // backend, and the deferred Dispose of that backend then threw ObjectDisposedException instead of freeing the blocks.
    private readonly ConcurrentDictionary<int, ConcurrentStack<IntPtr>> _buckets = new();
    private bool _disposed;

    /// <summary>
    /// Rents a pinned host buffer of at least <paramref name="sizeInBytes"/> bytes.
    /// Uses power-of-two bucketing to reduce fragmentation.
    /// </summary>
    public IntPtr Rent(int sizeInBytes)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaPinnedBufferPool));
        // Over 1 GiB the next power of two does not fit an int (the int rounding wrapped to int.MinValue, and the
        // allocation then asked the driver for ~1.8e19 bytes). Such a block is not a pool bucket: refuse it here.
        if (BucketSize(sizeInBytes) > MaxBucketBytes)
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes),
                $"A pooled pinned block is at most {MaxBucketBytes} bytes; {sizeInBytes} bytes needs its own allocation.");
        int bucketSize = NextPowerOfTwo(sizeInBytes);

        if (_buckets.TryGetValue(bucketSize, out var bag) && bag.TryPop(out var ptr))
            return ptr;

        // Allocate with CU_MEMHOSTALLOC_PORTABLE so pinned staging slots remain page-locked and DMA-accessible
        // even when a host thread switches between multiple CUDA device contexts (e.g., GPU 0 <-> GPU 1).
        var result = AllocatePortablePinnedHost(out ptr, (ulong)bucketSize);
        CuBlasNative.CheckCudaResult(result, "cuMemHostAlloc(portable)");
        // Publish allocation to the unified GPU memory ledger so a
        // memory_snapshot during a training step lists this pinned-host
        // bucket alongside CUDA caching-allocator + workspace allocs.
        GpuMemoryStats.RecordAllocation("cuda_pinned_host", bucketSize);
        return ptr;
    }

    /// <summary>
    /// Returns a pinned host buffer to the pool.
    /// </summary>
    public void Return(IntPtr ptr, int sizeInBytes)
    {
        if (ptr == IntPtr.Zero)
            return;
        if (_disposed)
        {
            // The pool is gone: the block is not lost to the process, it is freed (it is portable, so no context is needed).
            CuBlasNative.cuMemFreeHost(ptr);
            if (BucketSize(sizeInBytes) <= MaxBucketBytes) GpuMemoryStats.RecordFree("cuda_pinned_host", BucketSize(sizeInBytes));
            return;
        }

        int bucketSize = NextPowerOfTwo(sizeInBytes);
        var bag = _buckets.GetOrAdd(bucketSize, _ => new ConcurrentStack<IntPtr>());

        // Limit pool size to prevent unbounded growth.
        // Either the buffer goes back to the bucket (still resident, still
        // counted as allocated) or it gets freed — the freed branch is
        // what actually decrements the live byte count for memory_stats.
        if (bag.Count < 8)
        {
            bag.Push(ptr);
        }
        else
        {
            CuBlasNative.cuMemFreeHost(ptr);
            GpuMemoryStats.RecordFree("cuda_pinned_host", bucketSize);
        }
    }

    /// <summary>The largest pooled block: 2^30 bytes (the largest power of two an int bucket size holds).</summary>
    internal const long MaxBucketBytes = 1L << 30;

    /// <summary>The bucket a request rounds to, in long (no wrap above 1 GiB).</summary>
    internal static long BucketSize(int sizeInBytes)
    {
        if (sizeInBytes <= 1) return 1;
        long value = sizeInBytes - 1L;
        int shift = 0;
        while (value > 0) { value >>= 1; shift++; }
        return 1L << shift;
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
            while (kvp.Value.TryPop(out var ptr))
            {
                CuBlasNative.cuMemFreeHost(ptr);
                GpuMemoryStats.RecordFree("cuda_pinned_host", kvp.Key);
            }
        }
        _buckets.Clear();
    }
}
