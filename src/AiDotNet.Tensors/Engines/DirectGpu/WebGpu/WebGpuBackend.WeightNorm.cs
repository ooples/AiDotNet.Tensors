#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private const int WeightNormReductionWidth = 256;

    public void WeightNorm(IGpuBuffer direction, IGpuBuffer gain, IGpuBuffer output, int rows, int columns, float epsilon)
        => LaunchBatch(BatchExecutionCudaKernels.WeightNorm, checked(rows * WeightNormReductionWidth), direction, gain, output, rows, columns, epsilon);

    public void WeightNormStats(IGpuBuffer gradOutput, IGpuBuffer direction, IGpuBuffer stats, int rows, int columns, float epsilon)
        => LaunchBatch(BatchExecutionCudaKernels.WeightNormStats, checked(rows * WeightNormReductionWidth), gradOutput, direction, stats, rows, columns, epsilon);

    public void WeightNormDirectionGradient(IGpuBuffer gradOutput, IGpuBuffer direction, IGpuBuffer gain, IGpuBuffer stats,
        IGpuBuffer gradDirection, int rows, int columns, bool accumulate)
        => LaunchBatch(BatchExecutionCudaKernels.WeightNormDirectionGrad, checked(rows * columns),
            gradOutput, direction, gain, stats, gradDirection, rows, columns, accumulate ? 1 : 0);

    public void WeightNormGainGradient(IGpuBuffer stats, IGpuBuffer gradGain, int rows, bool accumulate)
        => LaunchBatch(BatchExecutionCudaKernels.WeightNormGainGrad, rows, stats, gradGain, rows, accumulate ? 1 : 0);
}
#endif

