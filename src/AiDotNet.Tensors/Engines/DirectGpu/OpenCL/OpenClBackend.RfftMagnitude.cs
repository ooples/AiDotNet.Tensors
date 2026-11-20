// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IRfftMagnitudeKernels
{
    IRfftNativePlan? IRfftMagnitudeKernels.GetOptimizedRealFftPlan(int n, int rows, bool inverse) => null;
    void IRfftMagnitudeKernels.RfftMagnitude(IGpuBuffer spectrum, IGpuBuffer output, int rows, int n, int spectrumStride, float epsilon)
        => LaunchBatch(BatchExecutionCudaKernels.RfftMagnitude, checked(rows * (n / 2 + 1)), spectrum, output, rows, n, spectrumStride, epsilon);
    void IRfftMagnitudeKernels.RfftMagnitudeSeed(IGpuBuffer spectrum, IGpuBuffer upstream, IGpuBuffer seed, int rows, int n, int spectrumStride, float epsilon)
        => LaunchBatch(BatchExecutionCudaKernels.RfftMagnitudeSeed, checked(rows * n), spectrum, upstream, seed, rows, n, spectrumStride, epsilon);
    void IRfftMagnitudeKernels.RfftRealGradient(IGpuBuffer complex, IGpuBuffer output, int count, float scale, bool accumulate)
        => LaunchBatch(BatchExecutionCudaKernels.RfftRealGradient, count, complex, output, count, scale, accumulate ? 1 : 0);
}

