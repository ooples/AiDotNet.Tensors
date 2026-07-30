namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Native GPU primitive for resident CTC forward dynamic programming.</summary>
public interface ICtcLossBackend
{
    /// <summary>
    /// Computes one loss per batch item. Integer metadata is represented as exactly-valued floats
    /// because direct GPU buffers use a uniform float storage contract.
    /// </summary>
    void CtcLoss(
        IGpuBuffer logProbs,
        IGpuBuffer targets,
        IGpuBuffer inputLengths,
        IGpuBuffer targetLengths,
        IGpuBuffer workspace,
        IGpuBuffer losses,
        int maxTime,
        int batchSize,
        int numClasses,
        int maxTargetLength,
        int blank);
}
