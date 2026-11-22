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
    /// <c>position[0] = p + 1</c>; past capacity nothing is written, the position stays and the mask is all zero.
    /// </summary>
    void KvCacheAppend(IGpuBuffer cache, IGpuBuffer position, IGpuBuffer value, IGpuBuffer mask, int batch, int capacity, int width);
}
