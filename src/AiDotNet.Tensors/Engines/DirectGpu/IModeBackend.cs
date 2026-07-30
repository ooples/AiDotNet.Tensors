namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Deterministic device-side flattened mode reduction.</summary>
public interface IModeBackend
{
    /// <summary>
    /// Writes the most frequent value to output[0] and its count to output[1]. Equal-count
    /// ties use the CPU contract: smallest value wins, while NaN comparisons preserve input order.
    /// </summary>
    void Mode(IGpuBuffer input, IGpuBuffer output, int length);
}
