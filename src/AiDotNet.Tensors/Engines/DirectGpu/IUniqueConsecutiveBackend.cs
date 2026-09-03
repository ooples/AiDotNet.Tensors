namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Native stable consecutive-value compaction with a device-side count result.</summary>
public interface IUniqueConsecutiveBackend
{
    /// <summary>
    /// Writes compacted values into <paramref name="outputCapacity"/> and the resulting length as
    /// one exactly-representable float into <paramref name="outputCount"/>.
    /// </summary>
    void UniqueConsecutive(
        IGpuBuffer input,
        IGpuBuffer outputCapacity,
        IGpuBuffer outputCount,
        int length);

    /// <summary>
    /// Compacts adjacent runs and optionally emits the run index for each input element and the
    /// length of each run. Unrequested metadata buffers are not written.
    /// </summary>
    void UniqueConsecutiveWithInfo(
        IGpuBuffer input,
        IGpuBuffer outputValues,
        IGpuBuffer outputInverse,
        IGpuBuffer outputCounts,
        IGpuBuffer outputCount,
        int length,
        bool returnInverse,
        bool returnCounts);

    /// <summary>
    /// Compacts already-sorted values while mapping inverse indices back through the matching
    /// original-position buffer. Original positions use the DirectGPU exact-float index convention.
    /// </summary>
    void UniqueSortedWithInfo(
        IGpuBuffer sortedInput,
        IGpuBuffer sortedOriginalIndices,
        IGpuBuffer outputValues,
        IGpuBuffer outputInverse,
        IGpuBuffer outputCounts,
        IGpuBuffer outputCount,
        int length,
        bool returnInverse,
        bool returnCounts);
}
