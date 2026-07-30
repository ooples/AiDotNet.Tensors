namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Deterministic device-resident greedy non-maximum suppression.</summary>
public interface INmsBackend
{
    /// <summary>
    /// Writes kept indices into an N-element capacity buffer and the kept count into one scalar.
    /// Class IDs are consulted only when <paramref name="batched"/> is nonzero.
    /// </summary>
    void Nms(
        IGpuBuffer boxes,
        IGpuBuffer scores,
        IGpuBuffer classIds,
        IGpuBuffer suppressed,
        IGpuBuffer outputCapacity,
        IGpuBuffer outputCount,
        int length,
        float iouThreshold,
        int batched);
}
