namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// One block per batch row holds every hidden unit, so the sequence kernels need H <= MaxRnnBlockSize.
public sealed partial class CudaBackend : IFusedLstmSequenceTraining
{
    int IFusedLstmSequenceTraining.MaxFusedLstmHidden => MaxRnnBlockSize;
}