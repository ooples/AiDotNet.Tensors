using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Native memory pool for zero-GC inference. Weights are pinned permanently,
/// activation buffers use NativeMemory.AlignedAlloc (64-byte aligned, outside GC heap).
/// Combined with raw function pointer BLAS, this eliminates ALL managed overhead
/// during inference forward passes.
/// </summary>
public sealed class NativeInferencePool : IDisposable
{
    private readonly List<GCHandle> _pinnedWeights = new();
    private readonly Dictionary<int, IntPtr> _nativeBuffers = new(); // size -> native ptr
    private bool _disposed;

    /// <summary>
    /// Pins a weight array permanently for the lifetime of this pool.
    /// Returns the native pointer for direct BLAS access.
    /// </summary>
    public unsafe float* PinWeights(float[] weights)
    {
        var handle = GCHandle.Alloc(weights, GCHandleType.Pinned);
        _pinnedWeights.Add(handle);
        return (float*)handle.AddrOfPinnedObject();
    }

    /// <summary>
    /// Pins a double weight array permanently.
    /// </summary>
    public unsafe double* PinWeights(double[] weights)
    {
        var handle = GCHandle.Alloc(weights, GCHandleType.Pinned);
        _pinnedWeights.Add(handle);
        return (double*)handle.AddrOfPinnedObject();
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// Gets or allocates a 64-byte aligned native buffer for activations.
    /// Buffers are reused across forward passes (same size returns same pointer).
    /// </summary>
    public unsafe float* GetActivationBuffer(int floatCount)
    {
        int byteCount = floatCount * sizeof(float);
        if (!_nativeBuffers.TryGetValue(byteCount, out var ptr))
        {
            ptr = (IntPtr)NativeMemory.AlignedAlloc((nuint)byteCount, 64);
            _nativeBuffers[byteCount] = ptr;
        }
        return (float*)ptr;
    }

    /// <summary>
    /// Gets or allocates a 64-byte aligned native buffer for double activations.
    /// </summary>
    public unsafe double* GetActivationBufferDouble(int doubleCount)
    {
        int byteCount = doubleCount * sizeof(double);
        if (!_nativeBuffers.TryGetValue(byteCount, out var ptr))
        {
            ptr = (IntPtr)NativeMemory.AlignedAlloc((nuint)byteCount, 64);
            _nativeBuffers[byteCount] = ptr;
        }
        return (double*)ptr;
    }
#endif

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var handle in _pinnedWeights)
        {
            if (handle.IsAllocated)
                handle.Free();
        }
        _pinnedWeights.Clear();

#if NET5_0_OR_GREATER
        foreach (var kvp in _nativeBuffers)
        {
            unsafe { NativeMemory.AlignedFree((void*)kvp.Value); }
        }
        _nativeBuffers.Clear();
#endif
    }
}
