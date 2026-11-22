namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IEngine.KvCacheAppend{T}"/> as one kernel: the row write at the device-side position, the attention mask
/// and the position increment.
/// </summary>
[BackendSpecificRoute("CUDA one-kernel decode cache append at a device-side position (row write, mask and position increment); other backends run the host implementation over their buffers through Copy",
    nameof(IDirectGpuBackend.Copy))]
internal interface IKvCacheAppendBackend
{
    /// <summary>
    /// <c>cache[b, p, :] = value[b, :]</c> for <c>p = position[0]</c>, <c>mask[s] = s &lt;= p ? 0 : -inf</c>,
    /// <c>position[0] = p + 1</c>. A position that is not a non-negative integer below capacity (a full cache, NaN, an
    /// infinity, a negative or fractional value) writes neither the cache nor the position, fills the mask with NaN and
    /// records a fault: outside a graph capture the call then throws <see cref="System.InvalidOperationException"/> as
    /// the host implementation does; inside one, <see cref="ThrowIfAppendFaulted"/> raises it after the replay.
    /// </summary>
    void KvCacheAppend(IGpuBuffer cache, IGpuBuffer position, IGpuBuffer value, IGpuBuffer mask, int batch, int capacity, int width);

    /// <summary>
    /// Waits for the backend's stream and throws <see cref="System.InvalidOperationException"/> (clearing the record)
    /// when an append since the last check found a bad position; a captured plan calls it after each replay.
    /// </summary>
    void ThrowIfAppendFaulted();
}

/// <summary>The bounds an <see cref="IKvCacheAppendBackend"/> launch accepts.</summary>
internal static class KvCacheAppendLimits
{
    /// <summary>The largest capacity a float position counts exactly (2^24).</summary>
    internal const int MaxCapacity = 1 << 24;
}
