// Copyright (c) CrowdTrainer LLC. All rights reserved.

#nullable enable

using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Host-side checks of the HIP GEMM sources. Device compilation for every AMD target is checked offline by
/// scripts/ci/check-hip-kernels-offline.py (hip-offline-compile.yml); these pin what the host relies on.
/// </summary>
public sealed class HipMfmaKernelSourceTests
{
    [Fact]
    public void LaunchTile_MatchesTheKernelWorkgroupTile()
    {
        string source = HipMfmaKernel.GetMfmaSource();
        Assert.Equal(HipMfmaKernel.MfmaTile, Define(source, "MFMA_WG_M"));
        Assert.Equal(HipMfmaKernel.MfmaTile, Define(source, "MFMA_WG_N"));
    }

    [Fact]
    public void EachModule_DefinesExactlyItsListedKernels()
    {
        Assert.Equal(HipMfmaKernel.GetPortableKernelNames(), Kernels(HipMfmaKernel.GetPortableSource()));
        Assert.Equal(HipMfmaKernel.GetMfmaKernelNames(), Kernels(HipMfmaKernel.GetMfmaSource()));
    }

    [Fact]
    public void MfmaSource_UsesOnlyMatrixInstructionsThatExist()
    {
        // __builtin_amdgcn_mfma_f32_32x32x8f32 does not exist; using it failed the whole module on every CDNA GPU.
        string source = HipMfmaKernel.GetMfmaSource();
        Assert.DoesNotContain("mfma_f32_32x32x8f32", source);
        Assert.Contains("__builtin_amdgcn_mfma_f32_32x32x2f32", source);
        Assert.Contains("__builtin_amdgcn_mfma_f32_32x32x8f16", source);
        Assert.Contains("defined(__gfx950__)", source);
    }

    [Theory]
    [InlineData(AmdGpuArchitecture.MI100, true)]
    [InlineData(AmdGpuArchitecture.MI200, true)]
    [InlineData(AmdGpuArchitecture.MI300, true)]
    [InlineData(AmdGpuArchitecture.MI350, true)]
    [InlineData(AmdGpuArchitecture.RDNA3, false)]
    [InlineData(AmdGpuArchitecture.GCN, false)]
    public void MfmaSupport_FollowsTheArchitecture(AmdGpuArchitecture architecture, bool expected)
    {
        Assert.Equal(expected, HipMfmaKernel.SupportsMfma(architecture));
        Assert.Equal(expected, HipMfmaKernel.GetRecommendedKernel(architecture) == "mfma_gemm_f32");
    }

    private static int Define(string source, string name)
    {
        Match m = Regex.Match(source, @"#define\s+" + name + @"\s+(\d+)");
        Assert.True(m.Success, name + " is not defined");
        return int.Parse(m.Groups[1].Value);
    }

    private static string[] Kernels(string source)
    {
        var names = new System.Collections.Generic.List<string>();
        foreach (Match m in Regex.Matches(source, @"__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?(\w+)\s*\("))
        {
            names.Add(m.Groups[1].Value);
        }

        return names.ToArray();
    }
}
