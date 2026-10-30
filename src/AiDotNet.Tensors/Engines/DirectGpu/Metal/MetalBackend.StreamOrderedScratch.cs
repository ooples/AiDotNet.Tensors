// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IStreamOrderedScratch
{
    /// <remarks>Allocation here is not stream-ordered, so an ordinary buffer serves.</remarks>
    IGpuBuffer IStreamOrderedScratch.AllocateStreamOrderedScratchBuffer(int size) => AllocateBuffer(size);
}
