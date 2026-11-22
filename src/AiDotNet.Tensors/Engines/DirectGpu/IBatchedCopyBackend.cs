namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: several independent device-to-device copies as one launch, where each copy would be its own
/// launch (a stateful inference plan's trailing state writes, which a captured graph replays every execution).
/// </summary>
[BackendSpecificRoute("CUDA one-launch batch of device copies (one captured graph node per eight copies, and inference graph capture is CUDA-only); every backend runs each copy natively",
    nameof(IDirectGpuBackend.Copy))]
internal interface IBatchedCopyBackend
{
    /// <summary>
    /// <c>destinations[i][0..lengths[i]) = sources[i][0..lengths[i])</c> for every i, stream-ordered like
    /// <see cref="IDirectGpuBackend.Copy(IGpuBuffer, IGpuBuffer, int)"/>. The copies must not overlap one another.
    /// </summary>
    void CopyMany(IReadOnlyList<IGpuBuffer> sources, IReadOnlyList<IGpuBuffer> destinations, IReadOnlyList<int> lengths);
}
