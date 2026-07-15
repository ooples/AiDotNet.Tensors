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
}
