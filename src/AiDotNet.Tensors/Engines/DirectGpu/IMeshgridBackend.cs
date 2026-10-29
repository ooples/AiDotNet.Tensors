// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Fused two-vector meshgrid in one launch (#922, #1142): both grids are written from the two source vectors without
/// materializing the reshape-and-broadcast intermediates.
/// </summary>
public interface IMeshgridBackend
{
    /// <summary>
    /// <c>'ij'</c> (<paramref name="xy"/> false): <paramref name="output0"/>[i, j] = <paramref name="source0"/>[i] and
    /// <paramref name="output1"/>[i, j] = <paramref name="source1"/>[j] over an [n0, n1] grid. <c>'xy'</c>: the grids
    /// are [n1, n0], with <paramref name="source0"/> along the columns and <paramref name="source1"/> along the rows.
    /// </summary>
    void Meshgrid2DPair(IGpuBuffer source0, IGpuBuffer source1, IGpuBuffer output0, IGpuBuffer output1,
        int n0, int n1, bool xy);
}
