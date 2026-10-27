namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// One block per batch row holds every hidden unit, so the sequence kernels need H <= MaxRnnBlockSize.
public sealed partial class HipBackend : IFusedLstmSequenceTraining
{
    PortableGenericFft.Launcher IFusedLstmSequenceTraining.StatefulLstmLauncher => LaunchBatch;
    int IFusedLstmSequenceTraining.MaxFusedLstmHidden => MaxRnnBlockSize;
    bool IFusedLstmSequenceTraining.LstmBackwardOverwritesGradients => false;
}