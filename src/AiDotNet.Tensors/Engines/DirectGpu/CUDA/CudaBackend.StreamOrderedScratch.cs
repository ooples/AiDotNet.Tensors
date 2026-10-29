// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IStreamOrderedScratch
{
    /// <remarks>cuMemAllocAsync from the stream-ordered pool (CudaBackend.FftGeneric.cs).</remarks>
    IGpuBuffer IStreamOrderedScratch.AllocateStreamOrderedScratchBuffer(int size) => AllocateStreamOrderedScratchBuffer(size);
}
