// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Computes the chunked fused cross-entropy's target positions from device-resident int32 ids, so the fused linear +
/// cross-entropy forward and backward need no host read of the ids: what a captured training step requires.
/// </summary>
internal interface IFusedCrossEntropyPositionsBackend
{
    /// <summary>
    /// Writes, for every row <c>r</c> in <c>[0, rows)</c>, the int bits of <c>(r % chunk) * vocab + id_r</c> into
    /// <paramref name="positionBits"/> (a float buffer read as int bits by Gather and ScatterAdd). An id outside
    /// <c>[0, vocab)</c> is clamped into range so no position leaves its row.
    /// </summary>
    void FusedCrossEntropyTargetPositions(IGpuBuffer int32Ids, IGpuBuffer positionBits, int rows, int chunk, int vocab);
}
