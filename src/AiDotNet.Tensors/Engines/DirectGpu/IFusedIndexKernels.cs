// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Fused index kernels (#1154): class gather / scatter and float-id embedding in one launch each. Where a backend lacks
/// them its flag is false and the engine composes the same result from the generic device ops; the operations are
/// called only after their flag reported true. <see cref="HasWhereSelectKernel"/> is false only where the where-select
/// kernel failed to compile.
/// </summary>
internal interface IFusedIndexKernels
{
    bool HasClassGatherKernels { get; }

    bool HasFloatIdEmbeddingKernels { get; }

    bool HasWhereSelectKernel { get; }

    void GatherClassValues(IGpuBuffer values, IGpuBuffer classIndices, IGpuBuffer output, int rows, int numClasses);

    void ScatterClassGrad(IGpuBuffer gradOutput, IGpuBuffer classIndices, IGpuBuffer grad, int rows, int numClasses);

    void EmbeddingFromFloatIds(IGpuBuffer ids, IGpuBuffer table, IGpuBuffer output, int numIndices, int embeddingDim, int vocabSize);

    void EmbeddingBackwardFromFloatIds(IGpuBuffer gradOutput, IGpuBuffer ids, IGpuBuffer gradTable, int numIndices, int embeddingDim, int vocabSize);

    void EmbeddingBackwardFromFloatIdsDeterministic(IGpuBuffer gradOutput, IGpuBuffer ids, IGpuBuffer links, IGpuBuffer gradTable, int numIndices, int embeddingDim, int vocabSize);
}
