// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="EmptySliceTests"/> on the GPU engine. It lives in this namespace so the CUDA lane's
/// Engines.DirectGpu filter runs it; it sets the process-wide fallback switch, so it runs with the other DirectGpu
/// tests, one at a time.
/// </summary>
[Collection("DirectGpuSerial")]
public class EmptySliceGpuTests
{
    [SkippableFact]
    public void ConcatenateBackward_WithAZeroWidthInput_OnTheGpuEngine()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        // A CPU fallback would pass the value checks while hiding a missing GPU path, so fallbacks throw here and
        // the probe must record none.
        bool savedThrowOnFallback = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        try
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
            GpuLaunchProbe.Reset();
            EmptySliceTests.AssertConcatenateBackward(gpu);
            Assert.Empty(GpuLaunchProbe.Fallbacks);
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = savedThrowOnFallback;
        }
    }
}
