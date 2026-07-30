namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>GPU primitive for fixed-shape inverse-CDF sampling used by NeRF rendering.</summary>
public interface IImportanceSamplingBackend
{
    /// <summary>
    /// Produces <paramref name="numFineSamples"/> stratified samples per ray. Each output element
    /// is owned by one invocation, so the operation requires no host-visible indices or atomics.
    /// </summary>
    void ImportanceSampling(
        IGpuBuffer tValuesCoarse,
        IGpuBuffer weightsCoarse,
        IGpuBuffer fineTValues,
        int numRays,
        int numCoarseSamples,
        int numFineSamples,
        uint seed);
}
