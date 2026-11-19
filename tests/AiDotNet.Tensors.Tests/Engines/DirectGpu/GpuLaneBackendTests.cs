// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A GPU lane states the backend its suites must run on (AIDOTNET_EXPECT_GPU_BACKEND, e.g. CUDA in
/// gpu-cuda-selfhosted.yml). Automatic selection falls through to the next backend when one fails to initialize, so
/// without this check a CUDA lane whose CUDA backend broke would pass every backend-agnostic test on OpenCL (#1569).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuLaneBackendTests
{
    public const string ExpectedBackendVariable = "AIDOTNET_EXPECT_GPU_BACKEND";

    [SkippableFact]
    public void TheEngine_ResolvesTheBackendTheLaneRequires()
    {
        string? expected = Environment.GetEnvironmentVariable(ExpectedBackendVariable);
        Skip.If(string.IsNullOrWhiteSpace(expected), $"{ExpectedBackendVariable} is not set: this run names no required backend.");

        using var gpu = new DirectGpuTensorEngine();
        Assert.True(gpu.IsGpuAvailable, $"The lane requires {expected}, but no DirectGpu backend initialized.");
        var backend = Assert.IsAssignableFrom<AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend>(gpu.GetBackend());
        Assert.True(string.Equals(backend.BackendName, expected, StringComparison.OrdinalIgnoreCase),
            $"The lane requires {expected}, but the engine resolved {backend.BackendName} ({backend.DeviceName}).");
    }
}