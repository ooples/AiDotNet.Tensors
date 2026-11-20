namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Sparse multi-head graph attention (CpuEngine.GraphAttentionCsr's device form): one thread per (graph, node, head)
/// row walks the node's CSR neighbourhood; the backward gathers into each node through the transposed (CSC) pattern,
/// so nothing uses atomics and the sums are deterministic. The dropout draw is CpuEngine.GraphAttentionUniform.
/// </summary>
internal static class CudaGraphAttentionKernels
{
    public static string[] GetKernelNames() =>
        ["gat_csr_scores", "gat_csr_forward", "gat_csr_backward_rows", "gat_csr_backward_cols", "gat_csr_attention_grad"];

    // The source lives in KernelSources/CudaGraphAttention.c (an embedded resource) to keep it out of the
    // assembly's user-string heap, which sits at the CS8103 cap.
    public static string GetSource() => KernelSourceResources.Read("CudaGraphAttention.c");
}
