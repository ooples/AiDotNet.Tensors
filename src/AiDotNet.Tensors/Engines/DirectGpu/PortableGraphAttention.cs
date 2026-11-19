// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Dispatches CSR attention through the backend's model-op launcher. Splitting the
/// two backward gathers keeps every pass within WebGPU's eight-storage-buffer guarantee.</summary>
internal sealed class PortableGraphAttention(PortableGenericFft.Launcher launch)
{
    public void Scores(IGpuBuffer wh, IGpuBuffer attention, IGpuBuffer selfScore, IGpuBuffer neighScore,
        int rows, int heads, int features)
    {
        int count = checked(rows * heads);
        if (count > 0) launch("gat_scores", count, wh, attention, selfScore, neighScore, count, heads, features);
    }

    public void Forward(IGpuBuffer wh, IGpuBuffer selfScore, IGpuBuffer neighScore, IGpuBuffer rowPtr,
        IGpuBuffer col, IGpuBuffer output, int batch, int nodes, int heads, int features, int edges,
        double slope, uint cutoff, float keep, uint seed)
    {
        int count = checked(batch * nodes * heads);
        if (count == 0) return;
        long bits = BitConverter.DoubleToInt64Bits(slope);
        launch("gat_forward", count, wh, selfScore, neighScore, rowPtr, col, output, count, nodes, heads,
            features, edges, unchecked((int)bits), (int)(bits >> 32), unchecked((int)cutoff), keep, unchecked((int)seed));
    }

    public void BackwardRows(IGpuBuffer wh, IGpuBuffer gradOutput, IGpuBuffer selfScore, IGpuBuffer neighScore,
        IGpuBuffer rowPtr, IGpuBuffer col, IGpuBuffer gz, IGpuBuffer cp, IGpuBuffer gSelf,
        int batch, int nodes, int heads, int features, int edges, double slope, uint cutoff, float keep, uint seed)
    {
        int count = checked(batch * nodes * heads);
        if (count == 0) return;
        long bits = BitConverter.DoubleToInt64Bits(slope);
        launch("gat_coefficients", count, selfScore, neighScore, rowPtr, col, cp, count, nodes, heads,
            edges, unchecked((int)bits), (int)(bits >> 32));
        launch("gat_row_grad", count, wh, gradOutput, rowPtr, col, cp, gz, gSelf, count, nodes, heads,
            features, edges, unchecked((int)bits), (int)(bits >> 32), unchecked((int)cutoff), keep, unchecked((int)seed));
    }

    public void BackwardCols(IGpuBuffer gradOutput, IGpuBuffer attention, IGpuBuffer gz, IGpuBuffer cp,
        IGpuBuffer gSelf, IGpuBuffer colPtr, IGpuBuffer tRow, IGpuBuffer tEdge, IGpuBuffer gNeigh, IGpuBuffer gradWh,
        int batch, int nodes, int heads, int features, int edges)
    {
        int count = checked(batch * nodes * heads);
        if (count == 0) return;
        launch("gat_column_grad", count, gradOutput, gz, cp, colPtr, tRow, tEdge, gNeigh, gradWh,
            count, nodes, heads, features, edges);
        int elements = checked(count * features);
        launch("gat_feature_grad", elements, attention, gSelf, gNeigh, gradWh, elements, heads, features);
    }

    public void AttentionGrad(IGpuBuffer wh, IGpuBuffer gSelf, IGpuBuffer gNeigh, IGpuBuffer gradAttention,
        int rows, int heads, int features)
    {
        int count = checked(2 * heads * features);
        if (count > 0) launch("gat_attention_grad", count, wh, gSelf, gNeigh, gradAttention, count, rows, heads, features);
    }
}
