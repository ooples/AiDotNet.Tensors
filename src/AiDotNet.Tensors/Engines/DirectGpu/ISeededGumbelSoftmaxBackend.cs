namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability for a resident, explicitly seeded Gumbel-softmax launch.
/// </summary>
internal interface ISeededGumbelSoftmaxBackend
{
    void GumbelSoftmax(
        IGpuBuffer logits,
        IGpuBuffer output,
        int outerSize,
        int innerSize,
        float temperature,
        ulong seed);
}
