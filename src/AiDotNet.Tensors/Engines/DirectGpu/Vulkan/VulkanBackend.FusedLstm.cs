namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// The sequence runs the shared lstm_p_ kernels (PortableLstmSequence), one launch per step with a thread per unit, so no
// work-group holds a whole row and the hidden size is bounded only to match the other backends.
public sealed partial class VulkanBackend : IFusedLstmSequenceTraining
{
    PortableGenericFft.Launcher IFusedLstmSequenceTraining.StatefulLstmLauncher => LaunchSolver;
    int IFusedLstmSequenceTraining.MaxFusedLstmHidden => 1024;
    bool IFusedLstmSequenceTraining.LstmBackwardOverwritesGradients => false;
}