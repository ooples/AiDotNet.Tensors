namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// The forward runs one invocation per batch row and the backward one invocation in all, so no work-group holds a
// whole row and the hidden size is bounded only to match the other backends.
public sealed partial class VulkanBackend : IFusedLstmSequenceTraining
{
    int IFusedLstmSequenceTraining.MaxFusedLstmHidden => 1024;
}