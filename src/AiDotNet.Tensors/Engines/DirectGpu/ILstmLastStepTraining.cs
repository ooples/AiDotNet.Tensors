namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: the BPTT of a fused LSTM whose output is only its final hidden state (a model reading h_T, as
/// <c>LstmSequenceForward(returnSequences: false)</c> does). The backward takes that state's gradient [batch, hidden]
/// directly. Composed, the compiled step sliced the last step out of the [batch, seq, hidden] output in the forward and,
/// in the backward, built a zeroed full-sequence gradient plus a slice copy for the sequence backward to read.
/// </summary>
[BackendSpecificRoute("CUDA register-resident LSTM backward recurrence reading only the final hidden state's gradient; every backend runs LstmBackwardSequence over a full-sequence gradient",
    nameof(IDirectGpuBackend.LstmBackwardSequence))]
internal interface ILstmLastStepTraining
{
    /// <summary>True when <see cref="LstmBackwardLastStep"/> serves this hidden size.</summary>
    bool SupportsLstmLastStepBackward(int hiddenSize);

    /// <summary>
    /// <see cref="IDirectGpuBackend.LstmBackwardSequence"/> with <paramref name="gradLastHidden"/> [batch, hidden] as the
    /// gradient of the final hidden state and no gradient on the earlier outputs. Writes every gradient output.
    /// </summary>
    void LstmBackwardLastStep(IGpuBuffer gradLastHidden, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        IGpuBuffer hInit, IGpuBuffer cInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit, IGpuBuffer gradWeightsIh,
        IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh, int seqLen, int batch, int inputSize,
        int hiddenSize);
}
