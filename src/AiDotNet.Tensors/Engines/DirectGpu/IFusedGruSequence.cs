namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The GRU sequence (reset gate after the recurrent sum, PyTorch's nn.GRU) as a backend operation over the caches of
/// <see cref="PortableGruSequence"/>: allH [(T + 1), B, H] with h0 at 0, gates [T, B, 3H] (r, z, n) and ghn [T, B, H],
/// batch-major input [B, T, In] and output [B, T, H]. CUDA runs its fused kernels (the input projection as one GEMM, one
/// register-resident recurrence launch per direction, the backward's dense products as GEMMs) where they cover the
/// shape; HIP, Metal, OpenCL, Vulkan and WebGPU run the portable gru_p_ kernels on the device, as does CUDA otherwise.
/// The caches are the same either way, so any forward pairs with any backward.
/// </summary>
internal interface IFusedGruSequence
{
    /// <summary>The forward, writing the output and the three caches; and the final hidden state [B, H] into
    /// <paramref name="hFinal"/> when given.</summary>
    void GruForwardSequence(IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh,
        IGpuBuffer biasIh, IGpuBuffer biasHh, IGpuBuffer output, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        int seqLen, int batch, int inputSize, int hiddenSize, IGpuBuffer? hFinal);

    /// <summary>BPTT over the caches; every gradient output is written, not accumulated into. With
    /// <paramref name="lastStepOnly"/>, <paramref name="gradOutput"/> is the final hidden state's gradient [B, H] alone (a
    /// model that reads h_T only); otherwise every step's output gradient [B, T, H].</summary>
    void GruBackwardSequence(IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input, IGpuBuffer gradInput, IGpuBuffer gradHInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize, bool lastStepOnly);
}
