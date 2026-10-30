namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// The forward and backward dispatch per timestep with an invocation per (batch row, hidden unit), so no work-group
// holds a whole row and the hidden size is bounded only to match the other backends. The backward writes every gradient.
public sealed partial class VulkanBackend : IFusedLstmSequenceTraining
{
    PortableGenericFft.Launcher IFusedLstmSequenceTraining.StatefulLstmLauncher => LaunchSolver;
    int IFusedLstmSequenceTraining.MaxFusedLstmHidden => 1024;
    bool IFusedLstmSequenceTraining.LstmBackwardOverwritesGradients => true;
}