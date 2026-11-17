// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

// Internal capability keeps the low-level layout contract out of the public
// tensor API. Every production DirectGpu backend implements these native calls.
internal interface IRfftMagnitudeKernels
{
    // Optional vendor optimization. Null requests the complete portable native
    // BatchedFFT route; it never requests a host transform or reports a fake plan.
    IRfftNativePlan? GetOptimizedRealFftPlan(int n, int rows, bool inverse);
    void RfftMagnitude(IGpuBuffer spectrum, IGpuBuffer output, int rows, int n, int spectrumStride, float epsilon);
    void RfftMagnitudeSeed(IGpuBuffer spectrum, IGpuBuffer upstream, IGpuBuffer seed, int rows, int n, int spectrumStride, float epsilon);
    void RfftRealGradient(IGpuBuffer complex, IGpuBuffer output, int count, float scale, bool accumulate);
}

internal interface IRfftNativePlan
{
    int WorkspaceFloats { get; }
    void Execute(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? workspace);
}
