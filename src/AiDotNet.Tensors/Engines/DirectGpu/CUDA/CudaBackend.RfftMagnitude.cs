// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IRfftMagnitudeKernels
{
    IRfftNativePlan? IRfftMagnitudeKernels.GetOptimizedRealFftPlan(int n, int rows, bool inverse)
        => GetRealFftPlan(n, rows, inverse);
    void IRfftMagnitudeKernels.RfftMagnitude(IGpuBuffer spectrum, IGpuBuffer output, int rows, int n, int spectrumStride, float epsilon)
        => LaunchBatchKernel(BatchExecutionCudaKernels.RfftMagnitude, checked(rows * (n / 2 + 1)), spectrum, output, rows, n, spectrumStride, epsilon);
    void IRfftMagnitudeKernels.RfftMagnitudeSeed(IGpuBuffer spectrum, IGpuBuffer upstream, IGpuBuffer seed, int rows, int n, int spectrumStride, float epsilon)
        => LaunchBatchKernel(BatchExecutionCudaKernels.RfftMagnitudeSeed, checked(rows * n), spectrum, upstream, seed, rows, n, spectrumStride, epsilon);
    void IRfftMagnitudeKernels.RfftRealGradient(IGpuBuffer complex, IGpuBuffer output, int count, float scale, bool accumulate)
        => LaunchBatchKernel(BatchExecutionCudaKernels.RfftRealGradient, count, complex, output, count, scale, accumulate ? 1 : 0);
}

