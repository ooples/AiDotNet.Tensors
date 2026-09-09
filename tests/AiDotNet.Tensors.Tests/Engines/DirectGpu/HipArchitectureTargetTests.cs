// Copyright (c) AiDotNet. All rights reserved.

#nullable enable

using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class HipArchitectureTargetTests
{
    [Theory]
    [InlineData("gfx1012:xnack-", "gfx1012", AmdGpuArchitecture.RDNA)]
    [InlineData("GFX1031", "gfx1031", AmdGpuArchitecture.RDNA2)]
    [InlineData("gfx1102:sramecc+:xnack-", "gfx1102", AmdGpuArchitecture.RDNA3)]
    [InlineData("gfx90a", "gfx90a", AmdGpuArchitecture.MI200)]
    [InlineData("gfx942", "gfx942", AmdGpuArchitecture.MI300)]
    [InlineData("gfx1201", "gfx1201", AmdGpuArchitecture.GCN)]
    public void TryParseArchitectureTarget_PreservesExactDeviceTarget(
        string reported,
        string expectedTarget,
        AmdGpuArchitecture expectedArchitecture)
    {
        bool parsed = HipBackend.TryParseArchitectureTarget(
            reported,
            out string target,
            out AmdGpuArchitecture architecture);

        Assert.True(parsed);
        Assert.Equal(expectedTarget, target);
        Assert.Equal(expectedArchitecture, architecture);
        Assert.Equal($"--offload-arch={expectedTarget}", HipMfmaKernel.GetCompileFlags(target));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("1012")]
    [InlineData("gfx10-12")]
    [InlineData("gfx1012;--evil")]
    public void TryParseArchitectureTarget_RejectsMissingOrUnsafeTargets(string? reported)
    {
        bool parsed = HipBackend.TryParseArchitectureTarget(
            reported,
            out string target,
            out AmdGpuArchitecture architecture);

        Assert.False(parsed);
        Assert.Empty(target);
        Assert.Equal(AmdGpuArchitecture.Unknown, architecture);
    }
}
