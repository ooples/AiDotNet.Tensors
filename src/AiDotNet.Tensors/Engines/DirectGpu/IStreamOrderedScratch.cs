// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Scratch whose allocation and release are ordered on the compute stream (#1154), so a temporary needs no host sync.
/// Backends whose allocations are not stream-ordered return an ordinary buffer.
/// </summary>
internal interface IStreamOrderedScratch
{
    /// <summary>A float scratch buffer of <paramref name="size"/> elements.</summary>
    IGpuBuffer AllocateStreamOrderedScratchBuffer(int size);
}
