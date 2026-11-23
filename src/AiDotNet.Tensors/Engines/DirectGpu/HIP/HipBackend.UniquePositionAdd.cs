// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IUniquePositionAddBackend
{
    // Distinct positions give every cell at most one contribution, so this backend's scatter-add (each route of which
    // sums a cell's contributions in a fixed order, or adds a single one) computes it exactly, with no scratch sized to
    // the destination.
    void IUniquePositionAddBackend.AddAtUniquePositions(IGpuBuffer values, IGpuBuffer positionBits, IGpuBuffer destination,
        int count, int destinationSize)
    {
        if (count <= 0) return;
        ScatterAdd(values, positionBits, destination, count, destinationSize);
    }
}
