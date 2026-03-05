using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// Pool of HIP page-locked (pinned) host memory buffers for true async DMA transfers.
/// </summary>
internal sealed class HipPinnedBufferPool : IDisposable
{
    private readonly ConcurrentDictionary<int, ConcurrentBag<IntPtr>> _buckets = new();
    private bool _disposed;

    public IntPtr Rent(int sizeInBytes)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(HipPinnedBufferPool));
        int bucketSize = NextPowerOfTwo(sizeInBytes);

        if (_buckets.TryGetValue(bucketSize, out var bag) && bag.TryTake(out var ptr))
            return ptr;

        ptr = IntPtr.Zero;
        var result = HipNativeBindings.hipHostMalloc(ref ptr, (UIntPtr)bucketSize, 0);
        HipNativeBindings.CheckError(result, "hipHostMalloc");
        return ptr;
    }

    public void Return(IntPtr ptr, int sizeInBytes)
    {
        if (_disposed || ptr == IntPtr.Zero)
            return;

        int bucketSize = NextPowerOfTwo(sizeInBytes);
        var bag = _buckets.GetOrAdd(bucketSize, _ => new ConcurrentBag<IntPtr>());

        if (bag.Count < 8)
            bag.Add(ptr);
        else
            HipNativeBindings.hipHostFree(ptr);
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
                HipNativeBindings.hipHostFree(ptr);
            }
        }
        _buckets.Clear();
    }
}
