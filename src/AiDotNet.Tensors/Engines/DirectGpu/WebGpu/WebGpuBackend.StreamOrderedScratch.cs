#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IStreamOrderedScratch
{
    /// <remarks>Allocation here is not stream-ordered, so an ordinary buffer serves.</remarks>
    IGpuBuffer IStreamOrderedScratch.AllocateStreamOrderedScratchBuffer(int size) => AllocateBuffer(size);
}
#endif
