namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Optional resident one-hot categorical sampling capability.</summary>
internal interface ICategoricalSamplingBackend
{
    bool CanCategoricalSample(int rows, int classes);

    bool TryCategoricalSample(
        IGpuBuffer probabilities,
        IGpuBuffer oneHot,
        int rows,
        int classes,
        ulong seed);
}
