// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// #1107: LstmForwardSequence / LstmBackwardSequence run the shared lstm_p_ kernels (PortableLstmSequence), whose
// source LstmSequenceBackendParityTests checks against the CPU forward + BPTT on OpenCL, Vulkan and HIP. One thread per
// output element with no work-group coupling, so there is no hidden-size limit.
public sealed partial class MetalBackend : IFusedLstmSequenceTraining
{
    int IFusedLstmSequenceTraining.MaxFusedLstmHidden => int.MaxValue;
    bool IFusedLstmSequenceTraining.LstmBackwardOverwritesGradients => false;
}
