// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend
{
    // The cooperative row reductions require a power-of-two local group. Some
    // OpenCL devices support fewer than 256 threads, unlike a flat elementwise call.
    private void LaunchWeightNormRows(string name, int rows, params object[] args)
    {
        if (rows <= 0) return;
        int maximum = CalculateOptimalWorkGroupSize1D(256);
        int local = 1;
        while (local * 2 <= maximum && local < 256) local *= 2;
        var kernel = GetBatchKernel(name);
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case IGpuBuffer buffer: kernel.SetArg((uint)i, BatchHandle(buffer)); break;
                case int value: kernel.SetArg((uint)i, value); break;
                case float value: kernel.SetArg((uint)i, value); break;
                default: throw new ArgumentException("Invalid weight-norm kernel argument.", nameof(args));
            }
        }
        kernel.Execute1D(checked(rows * local), local);
    }

    public void WeightNorm(IGpuBuffer direction, IGpuBuffer gain, IGpuBuffer output, int rows, int columns, float epsilon)
        => LaunchWeightNormRows(BatchExecutionCudaKernels.WeightNorm, rows, direction, gain, output, rows, columns, epsilon);
    public void WeightNormStats(IGpuBuffer gradOutput, IGpuBuffer direction, IGpuBuffer stats, int rows, int columns, float epsilon)
        => LaunchWeightNormRows(BatchExecutionCudaKernels.WeightNormStats, rows, gradOutput, direction, stats, rows, columns, epsilon);
    public void WeightNormDirectionGradient(IGpuBuffer gradOutput, IGpuBuffer direction, IGpuBuffer gain, IGpuBuffer stats,
        IGpuBuffer gradDirection, int rows, int columns, bool accumulate)
        => LaunchBatch(BatchExecutionCudaKernels.WeightNormDirectionGrad, checked(rows * columns),
            gradOutput, direction, gain, stats, gradDirection, rows, columns, accumulate ? 1 : 0);
    public void WeightNormGainGradient(IGpuBuffer stats, IGpuBuffer gradGain, int rows, bool accumulate)
        => LaunchBatch(BatchExecutionCudaKernels.WeightNormGainGrad, rows, stats, gradGain, rows, accumulate ? 1 : 0);
}
