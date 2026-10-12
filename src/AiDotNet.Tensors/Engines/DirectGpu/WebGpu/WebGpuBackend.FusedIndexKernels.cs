#if NET7_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// No fused class-gather or float-id embedding kernels here: the engine composes them from the generic device ops.
// The where-select op is part of this backend's core kernel set.
public sealed partial class WebGpuBackend : IFusedIndexKernels
{
    private static NotSupportedException NoFusedIndexKernel(string kernel)
        => new($"{kernel}: this backend has no fused index kernel; check its flag first.");

    bool IFusedIndexKernels.HasClassGatherKernels => false;

    bool IFusedIndexKernels.HasFloatIdEmbeddingKernels => false;

    bool IFusedIndexKernels.HasWhereSelectKernel => true;

    void IFusedIndexKernels.GatherClassValues(IGpuBuffer values, IGpuBuffer classIndices, IGpuBuffer output, int rows, int numClasses)
        => throw NoFusedIndexKernel(nameof(IFusedIndexKernels.GatherClassValues));

    void IFusedIndexKernels.ScatterClassGrad(IGpuBuffer gradOutput, IGpuBuffer classIndices, IGpuBuffer grad, int rows, int numClasses)
        => throw NoFusedIndexKernel(nameof(IFusedIndexKernels.ScatterClassGrad));

    void IFusedIndexKernels.EmbeddingFromFloatIds(IGpuBuffer ids, IGpuBuffer table, IGpuBuffer output, int numIndices, int embeddingDim, int vocabSize)
        => throw NoFusedIndexKernel(nameof(IFusedIndexKernels.EmbeddingFromFloatIds));

    void IFusedIndexKernels.EmbeddingBackwardFromFloatIds(IGpuBuffer gradOutput, IGpuBuffer ids, IGpuBuffer gradTable, int numIndices, int embeddingDim, int vocabSize)
        => throw NoFusedIndexKernel(nameof(IFusedIndexKernels.EmbeddingBackwardFromFloatIds));

    void IFusedIndexKernels.EmbeddingBackwardFromFloatIdsDeterministic(IGpuBuffer gradOutput, IGpuBuffer ids, IGpuBuffer links, IGpuBuffer gradTable, int numIndices, int embeddingDim, int vocabSize)
        => throw NoFusedIndexKernel(nameof(IFusedIndexKernels.EmbeddingBackwardFromFloatIdsDeterministic));
}
#endif
