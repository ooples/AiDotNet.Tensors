// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

// Internal capability keeps the low-level layout contract out of the public
// tensor API. Every production DirectGpu backend implements these native calls.
internal interface IRfftMagnitudeKernels
{
    void RfftMagnitude(IGpuBuffer spectrum, IGpuBuffer output, int rows, int n, int spectrumStride, float epsilon);
    void RfftMagnitudeSeed(IGpuBuffer spectrum, IGpuBuffer upstream, IGpuBuffer seed, int rows, int n, int spectrumStride, float epsilon);
    void RfftRealGradient(IGpuBuffer complex, IGpuBuffer output, int count, float scale, bool accumulate);
}
