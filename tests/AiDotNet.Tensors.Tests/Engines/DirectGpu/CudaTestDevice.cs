using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

internal static class CudaTestDevice
{
    internal static void Require(CudaBackend backend)
    {
        if (backend.IsAvailable) return;
        Assert.False(string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_EXPECT_GPU_BACKEND"), "CUDA",
            StringComparison.OrdinalIgnoreCase), "The CUDA lane requires a working CUDA backend.");
        Skip.If(true, "CUDA is unavailable on this machine.");
    }
}
