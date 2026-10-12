// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The fused index kernels live in CudaBackend.cs; this maps the contract onto them.
public sealed partial class CudaBackend : IFusedIndexKernels
{
    bool IFusedIndexKernels.HasClassGatherKernels => HasClassGatherKernels;

    bool IFusedIndexKernels.HasFloatIdEmbeddingKernels => HasFloatIdEmbeddingKernels;

    bool IFusedIndexKernels.HasWhereSelectKernel => HasWhereSelectKernel;

    void IFusedIndexKernels.GatherClassValues(IGpuBuffer values, IGpuBuffer classIndices, IGpuBuffer output, int rows, int numClasses)
        => GatherClassValues(values, classIndices, output, rows, numClasses);

    void IFusedIndexKernels.ScatterClassGrad(IGpuBuffer gradOutput, IGpuBuffer classIndices, IGpuBuffer grad, int rows, int numClasses)
        => ScatterClassGrad(gradOutput, classIndices, grad, rows, numClasses);

    void IFusedIndexKernels.EmbeddingFromFloatIds(IGpuBuffer ids, IGpuBuffer table, IGpuBuffer output, int numIndices, int embeddingDim, int vocabSize)
        => EmbeddingFromFloatIds(ids, table, output, numIndices, embeddingDim, vocabSize);

    void IFusedIndexKernels.EmbeddingBackwardFromFloatIds(IGpuBuffer gradOutput, IGpuBuffer ids, IGpuBuffer gradTable, int numIndices, int embeddingDim, int vocabSize)
        => EmbeddingBackwardFromFloatIds(gradOutput, ids, gradTable, numIndices, embeddingDim, vocabSize);

    void IFusedIndexKernels.EmbeddingBackwardFromFloatIdsDeterministic(IGpuBuffer gradOutput, IGpuBuffer ids, IGpuBuffer links, IGpuBuffer gradTable, int numIndices, int embeddingDim, int vocabSize)
        => EmbeddingBackwardFromFloatIdsDeterministic(gradOutput, ids, links, gradTable, numIndices, embeddingDim, vocabSize);
}
