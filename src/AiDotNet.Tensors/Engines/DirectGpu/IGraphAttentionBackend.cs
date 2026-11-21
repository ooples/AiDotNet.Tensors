namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Sparse (CSR) multi-head graph attention kernels behind <see cref="IEngine.GraphAttentionCsr{T}"/>: one thread per
/// (graph, node, head) row over its neighbourhood, and a backward that gathers through the transposed pattern. A
/// backend without it reports the unsupported operation.
/// </summary>
internal interface IGraphAttentionBackend
{
    /// <summary>Self and neighbour attention scores per (row, head).</summary>
    void GraphAttentionScores(IGpuBuffer wh, IGpuBuffer attention, IGpuBuffer selfScore, IGpuBuffer neighScore,
        int rows, int heads, int features);

    /// <summary>The softmax-weighted (and dropped-out) aggregation over each row's neighbours.</summary>
    void GraphAttentionForward(IGpuBuffer wh, IGpuBuffer selfScore, IGpuBuffer neighScore, IGpuBuffer rowPtr,
        IGpuBuffer col, IGpuBuffer output, int batch, int nodes, int heads, int features, int edges,
        double slope, uint cutoff, float keep, uint seed);

    /// <summary>Per-edge score gradients, the dropped coefficients, and the self-score gradient per row.</summary>
    void GraphAttentionBackwardRows(IGpuBuffer wh, IGpuBuffer gradOutput, IGpuBuffer selfScore, IGpuBuffer neighScore,
        IGpuBuffer rowPtr, IGpuBuffer col, IGpuBuffer gz, IGpuBuffer cp, IGpuBuffer gSelf,
        int batch, int nodes, int heads, int features, int edges, double slope, uint cutoff, float keep, uint seed);

    /// <summary>The transformed-feature gradient, gathered per node through the transposed pattern.</summary>
    void GraphAttentionBackwardCols(IGpuBuffer gradOutput, IGpuBuffer attention, IGpuBuffer gz, IGpuBuffer cp,
        IGpuBuffer gSelf, IGpuBuffer colPtr, IGpuBuffer tRow, IGpuBuffer tEdge, IGpuBuffer gNeigh, IGpuBuffer gradWh,
        int batch, int nodes, int heads, int features, int edges);

    /// <summary>The attention-vector gradient: score gradients times the transformed features, summed over rows.</summary>
    void GraphAttentionAttentionGrad(IGpuBuffer wh, IGpuBuffer gSelf, IGpuBuffer gNeigh, IGpuBuffer gradAttention,
        int rows, int heads, int features);
}
