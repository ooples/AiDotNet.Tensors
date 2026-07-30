namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Stable device-side nonzero compaction with a scalar count result.</summary>
public interface INonzeroBackend
{
    /// <summary>
    /// Writes row-major coordinates into <paramref name="outputCapacity"/> and the number of
    /// nonzero input elements as one exactly representable float into <paramref name="outputCount"/>.
    /// </summary>
    void Nonzero(
        IGpuBuffer input,
        IGpuBuffer strides,
        IGpuBuffer outputCapacity,
        IGpuBuffer outputCount,
        int length,
        int rank);
}
