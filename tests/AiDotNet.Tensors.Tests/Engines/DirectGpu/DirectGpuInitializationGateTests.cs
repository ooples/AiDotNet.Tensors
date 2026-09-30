// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The gate the GPU-parity workflow runs before its shard. With AIDOTNET_REQUIRE_GPU_TESTS=1, a DirectGpu engine
/// that does not come up is a failure, not a skip. Most GPU tests skip when no device is found, so a backend that
/// failed to initialize (an OpenCL program that did not build) read as all-skipped and green for weeks.
/// Without the variable this passes, so ordinary machines without a GPU are unaffected.
/// </summary>
public sealed class DirectGpuInitializationGateTests
{
    [Fact]
    public void Engine_initializes_when_gpu_tests_are_required()
    {
        bool required = string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal);
        if (!required) return;

        Exception? failure = null;
        bool available = false;
        try
        {
            using var engine = new DirectGpuTensorEngine();
            available = engine.IsGpuAvailable;
        }
        catch (Exception ex)
        {
            failure = ex;
        }

        Assert.True(available,
            "AIDOTNET_REQUIRE_GPU_TESTS=1 but the DirectGpu engine did not initialize a GPU backend" +
            (failure is null ? "" : $" ({failure.GetType().Name}: {failure.Message})") +
            ". Run with AIDOTNET_GPU_VERBOSE=1 to see which backend or kernel program failed.");
    }
}
