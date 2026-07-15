namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Native GPU primitives used by Instant-NGP operations whose integer hash arithmetic cannot be
/// represented by the generic floating-point tensor algebra.
/// </summary>
public interface IInstantNgpBackend
{
    /// <summary>Encodes one hash-grid level into its columns of a shared output buffer.</summary>
    void HashGridEncodeLevel(
        IGpuBuffer positions,
        IGpuBuffer hashTable,
        IGpuBuffer output,
        int numPoints,
        int resolution,
        int tableSize,
        int featuresPerLevel,
        int levelOffset,
        int outputStride);

    /// <summary>
    /// Computes one level's table gradient. Each output cell is owned by one thread, avoiding
    /// nondeterministic floating-point atomics when hash collisions occur.
    /// </summary>
    void HashGridEncodeLevelBackward(
        IGpuBuffer positions,
        IGpuBuffer outputGradient,
        IGpuBuffer tableGradient,
        int numPoints,
        int resolution,
        int tableSize,
        int featuresPerLevel,
        int levelOffset,
        int outputStride);
}
