// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

// No fused class-gather or float-id embedding kernels here: the engine composes them from the generic device ops,
// and PortableFusedIndexKernels implements the contract with the CUDA kernels' exact semantics.
// The where-select op is part of this backend's core kernel set.
public sealed partial class OpenClBackend : IFusedIndexKernels
{
    bool IFusedIndexKernels.HasClassGatherKernels => false;

    bool IFusedIndexKernels.HasFloatIdEmbeddingKernels => false;

    bool IFusedIndexKernels.HasWhereSelectKernel => true;

    void IFusedIndexKernels.GatherClassValues(IGpuBuffer values, IGpuBuffer classIndices, IGpuBuffer output, int rows, int numClasses)
        => PortableFusedIndexKernels.GatherClassValues(this, values, classIndices, output, rows, numClasses);

    void IFusedIndexKernels.ScatterClassGrad(IGpuBuffer gradOutput, IGpuBuffer classIndices, IGpuBuffer grad, int rows, int numClasses)
        => PortableFusedIndexKernels.ScatterClassGrad(this, gradOutput, classIndices, grad, rows, numClasses);

    void IFusedIndexKernels.EmbeddingFromFloatIds(IGpuBuffer ids, IGpuBuffer table, IGpuBuffer output, int numIndices, int embeddingDim, int vocabSize)
        => PortableFusedIndexKernels.EmbeddingFromFloatIds(this, ids, table, output, numIndices, embeddingDim, vocabSize);

    void IFusedIndexKernels.EmbeddingBackwardFromFloatIds(IGpuBuffer gradOutput, IGpuBuffer ids, IGpuBuffer gradTable, int numIndices, int embeddingDim, int vocabSize)
        => PortableFusedIndexKernels.EmbeddingBackwardFromFloatIds(this, gradOutput, ids, gradTable, numIndices, embeddingDim, vocabSize);

    void IFusedIndexKernels.EmbeddingBackwardFromFloatIdsDeterministic(IGpuBuffer gradOutput, IGpuBuffer ids, IGpuBuffer links, IGpuBuffer gradTable, int numIndices, int embeddingDim, int vocabSize)
        => PortableFusedIndexKernels.EmbeddingBackwardFromFloatIds(this, gradOutput, ids, gradTable, numIndices, embeddingDim, vocabSize);
}
