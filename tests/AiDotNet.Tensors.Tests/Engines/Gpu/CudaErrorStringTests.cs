// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

public sealed class CudaErrorStringTests
{
    [Fact]
    public void InvalidPtx_ReportsJitCompilationFailure()
    {
        string message = CuBlasNative.GetCudaErrorString(CudaResult.InvalidPtx);

        Assert.Contains("Invalid PTX", message, StringComparison.Ordinal);
        Assert.Contains("JIT compilation failed", message, StringComparison.Ordinal);
    }
}
