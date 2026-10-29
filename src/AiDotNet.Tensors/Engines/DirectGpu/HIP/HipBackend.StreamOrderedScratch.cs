// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IStreamOrderedScratch
{
    /// <remarks>Allocation here is not stream-ordered, so an ordinary buffer serves.</remarks>
    IGpuBuffer IStreamOrderedScratch.AllocateStreamOrderedScratchBuffer(int size) => AllocateBuffer(size);
}
