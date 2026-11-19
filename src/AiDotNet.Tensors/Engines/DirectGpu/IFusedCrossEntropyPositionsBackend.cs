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
    /// <paramref name="positionBits"/>[r] (a float buffer read as int bits by Gather and ScatterAdd), and a row flag
    /// into <paramref name="positionBits"/>[rows + r]: 0 for an id in <c>[0, vocab)</c>, NaN otherwise. The buffer holds
    /// <c>2 * rows</c> floats. An out-of-range id's position is clamped into its row (no read leaves the row), and the
    /// caller adds the flags into the loss and the row's gradient, so a bad id yields NaN instead of training on a
    /// clamped class: a captured step cannot read the ids on the host to throw as an eager call does.
    /// </summary>
    void FusedCrossEntropyTargetPositions(IGpuBuffer int32Ids, IGpuBuffer positionBits, int rows, int chunk, int vocab);
}
