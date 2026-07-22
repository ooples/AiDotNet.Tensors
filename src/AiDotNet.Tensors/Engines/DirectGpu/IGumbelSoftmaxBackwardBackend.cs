namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Optional resident direct Gumbel-softmax backward capability.</summary>
internal interface IGumbelSoftmaxBackwardBackend
{
    bool CanGumbelSoftmaxBackward(int rows, int classes);

    bool TryGumbelSoftmaxBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer softOutput,
        IGpuBuffer gradInput,
        int rows,
        int classes,
        float temperature);
}
