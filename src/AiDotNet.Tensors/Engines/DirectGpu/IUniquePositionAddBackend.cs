// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Adds one value per position into a destination whose positions are known to be distinct, such as the chunked fused
/// cross-entropy's one target logit per row. Every cell receives at most one contribution, so the result does not
/// depend on the order of the adds and needs no sort to be deterministic.
/// </summary>
internal interface IUniquePositionAddBackend
{
    /// <summary>
    /// <c>destination[positionBits[i]] += values[i]</c> for i in <c>[0, count)</c>, where <paramref name="positionBits"/>
    /// holds int32 bits (as Gather and ScatterAdd read them) and no two of its first <paramref name="count"/> entries
    /// are equal. An entry outside <c>[0, destinationSize)</c> contributes nowhere.
    /// </summary>
    void AddAtUniquePositions(IGpuBuffer values, IGpuBuffer positionBits, IGpuBuffer destination, int count,
        int destinationSize);
}
