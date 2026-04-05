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
    [ThreadStatic]
    private static NativeInferencePool? _current;

    /// <summary>Gets the active inference pool for this thread, or null.</summary>
    internal static NativeInferencePool? Current => _current;

    private NativeInferencePool? _previous;
    private readonly List<GCHandle> _pinnedWeights = new();
    private readonly Dictionary<int, IntPtr> _nativeBuffers = new();
    // Cache pinned pointers by array reference for O(1) lookup
    private readonly Dictionary<object, IntPtr> _pinnedPtrCache = new();
    private bool _disposed;

    /// <summary>
    /// Creates and activates a new inference pool for this thread.
    /// All inference operations will use pre-pinned/native memory until disposed.
    /// </summary>
    public static NativeInferencePool Create()
    {
        var pool = new NativeInferencePool();
        pool._previous = _current;
        _current = pool;
        return pool;
    }

    private NativeInferencePool() { }

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
    /// Gets a cached pinned pointer for a float array, pinning it on first access.
    /// </summary>
    public unsafe float* GetOrPin(float[] array)
    {
        if (_pinnedPtrCache.TryGetValue(array, out var cached))
            return (float*)cached;
        var ptr = PinWeights(array);
        _pinnedPtrCache[array] = (IntPtr)ptr;
        return ptr;
    }

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
        // Use negative key space for float to avoid collision with double buffers
        int key = -floatCount;
        if (!_nativeBuffers.TryGetValue(key, out var ptr))
        {
            // Use long multiplication to avoid int overflow for large buffers
            long byteCount = (long)floatCount * sizeof(float);
            ptr = (IntPtr)NativeMemory.AlignedAlloc((nuint)byteCount, 64);
            _nativeBuffers[key] = ptr;
        }
        return (float*)ptr;
    }

    /// <summary>
    /// Gets or allocates a 64-byte aligned native buffer for double activations.
    /// </summary>
    public unsafe double* GetActivationBufferDouble(int doubleCount)
    {
        // Use positive key space for double (element count, not byte count)
        int key = doubleCount;
        if (!_nativeBuffers.TryGetValue(key, out var ptr))
        {
            // Use long multiplication to avoid int overflow for large buffers
            long byteCount = (long)doubleCount * sizeof(double);
            ptr = (IntPtr)NativeMemory.AlignedAlloc((nuint)byteCount, 64);
            _nativeBuffers[key] = ptr;
        }
        return (double*)ptr;
    }
#endif

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_current == this)
            _current = _previous;

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
