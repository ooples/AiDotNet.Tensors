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

    private static CudaResult AllocatePortablePinnedHost(out IntPtr ptr, ulong byteSize)
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
        if (_disposed || ptr == IntPtr.Zero)
            return;

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
