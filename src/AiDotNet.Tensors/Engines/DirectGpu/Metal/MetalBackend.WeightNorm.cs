// Copyright (c) CrowdTrainer LLC. All rights reserved.
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    private const int WeightNormReductionWidth = 256;

    public void WeightNorm(IGpuBuffer direction, IGpuBuffer gain, IGpuBuffer output, int rows, int columns, float epsilon)
        => LaunchWeightNormRows(BatchExecutionCudaKernels.WeightNorm, rows, direction, gain, output, rows, columns, epsilon);

    public void WeightNormStats(IGpuBuffer gradOutput, IGpuBuffer direction, IGpuBuffer stats, int rows, int columns, float epsilon)
        => LaunchWeightNormRows(BatchExecutionCudaKernels.WeightNormStats, rows, gradOutput, direction, stats, rows, columns, epsilon);

    // Unlike a flat elementwise dispatch, the shared-memory tree reduction needs
    // a power-of-two group width. A pipeline maximum aligned to SIMD width (for
    // example 192) does not provide that guarantee.
    private void LaunchWeightNormRows(string name, int rows, params object[] args)
    {
        if (rows <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetBatchPipeline(name);
        int maximum = Math.Min(WeightNormReductionWidth, pipeline.MaxTotalThreadsPerThreadgroup);
        if (maximum < 1) throw new InvalidOperationException("Metal pipeline has no valid weight-norm workgroup width.");
        int local = 1;
        while (local * 2 <= maximum) local *= 2;
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        BindBatchArguments(in encoder, name, args);
        encoder.DispatchThreadgroups(new MTLSize((ulong)rows, 1, 1), new MTLSize((ulong)local, 1, 1));
    }

    public void WeightNormDirectionGradient(IGpuBuffer gradOutput, IGpuBuffer direction, IGpuBuffer gain, IGpuBuffer stats,
        IGpuBuffer gradDirection, int rows, int columns, bool accumulate)
        => LaunchBatch(BatchExecutionCudaKernels.WeightNormDirectionGrad, checked(rows * columns),
            gradOutput, direction, gain, stats, gradDirection, rows, columns, accumulate ? 1 : 0);

    public void WeightNormGainGradient(IGpuBuffer stats, IGpuBuffer gradGain, int rows, bool accumulate)
        => LaunchBatch(BatchExecutionCudaKernels.WeightNormGainGrad, rows, stats, gradGain, rows, accumulate ? 1 : 0);
}

