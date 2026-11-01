// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// The GRU sequence on the portable gru_p_ kernels (see IFusedGruSequence; GruSequenceBackendParityTests checks them
// against the CPU reference on every backend).
public sealed partial class MetalBackend : IFusedGruSequence
{
    void IFusedGruSequence.GruForwardSequence(IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh,
        IGpuBuffer biasIh, IGpuBuffer biasHh, IGpuBuffer output, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        int seqLen, int batch, int inputSize, int hiddenSize, IGpuBuffer? hFinal)
        => PortableGruSequence.ForwardResetAfter(this, input, hInit, weightsIh, weightsHh, biasIh, biasHh, output, allH, gates, ghn, seqLen, batch, inputSize, hiddenSize, hFinal);

    void IFusedGruSequence.GruBackwardSequence(IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input, IGpuBuffer gradInput, IGpuBuffer gradHInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize, bool lastStepOnly)
        => PortableGruSequence.BackwardResetAfter(this, gradOutput, allH, gates, ghn, weightsIh, weightsHh, input, gradInput, gradHInit, gradWeightsIh, gradWeightsHh, gradBiasIh, gradBiasHh, seqLen, batch, inputSize, hiddenSize, lastStepOnly);
}
