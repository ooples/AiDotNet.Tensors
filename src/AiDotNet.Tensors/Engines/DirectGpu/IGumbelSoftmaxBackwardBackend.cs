namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Optional resident direct Gumbel-softmax backward capability.</summary>
[BackendSpecificRoute("CUDA direct-PTX fused Gumbel-softmax backward; every backend computes it as a softmax backward then a scale",
    nameof(IDirectGpuBackend.SoftmaxBackward), nameof(IDirectGpuBackend.Scale))]
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
