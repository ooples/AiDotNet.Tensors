using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The generic FFT capability gate WebGPU and Metal share (#963, #965): the capability holds only once every kernel
/// compiled, a failure is remembered (the device compiler is not asked again), and an explicit request for an
/// unavailable transform raises a NotSupportedException that names the failing kernel.
/// </summary>
public sealed class GenericFftReadinessTests
{
    [Fact]
    public void Ready_OnlyAfterEveryKernelCompiled_AndAskedOnce()
    {
        var asked = new List<string>();
        var readiness = new GenericFftReadiness(name => { asked.Add(name); return null; });
        Assert.True(readiness.IsReady(FftElementType.Float32));
        Assert.Equal(PortableGenericFft.KernelNames, asked);
        Assert.True(readiness.IsReady(FftElementType.Float16));
        Assert.True(readiness.IsReady(FftElementType.BFloat16));
        Assert.Equal(PortableGenericFft.KernelNames.Length, asked.Count);
        readiness.Require(FftElementType.Float32, "Test");
    }

    [Fact]
    public void ACompileFailure_WithholdsTheCapability_AndExplainsTheRequest()
    {
        int calls = 0;
        var readiness = new GenericFftReadiness(name => { calls++; return name == "bx_fft_cmul" ? "unsupported builtin" : null; });
        Assert.False(readiness.IsReady(FftElementType.Float32));
        int afterFirst = calls;
        Assert.False(readiness.IsReady(FftElementType.BFloat16));
        Assert.Equal(afterFirst, calls);
        var ex = Assert.Throws<GpuCapabilityUnavailableException>(() => readiness.Require(FftElementType.Float16, "Test"));
        Assert.IsAssignableFrom<NotSupportedException>(ex);
        Assert.Contains("bx_fft_cmul: unsupported builtin", ex.Message);
    }

    [Fact]
    public void KernelNames_CoverEveryLaunchInPortableGenericFft()
    {
        string source = System.IO.File.ReadAllText(System.IO.Path.Combine(RepoRoot(), "src", "AiDotNet.Tensors", "Engines",
            "DirectGpu", "PortableGenericFft.cs"));
        var launched = new HashSet<string>(StringComparer.Ordinal);
        foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(source,
                     @"launch\(""(\w+)""", System.Text.RegularExpressions.RegexOptions.None, TimeSpan.FromSeconds(1)))
            launched.Add(m.Groups[1].Value);
        Assert.NotEmpty(launched);
        // Both ways: a launch missing from the list would let the capability report a kernel that never compiled, and a
        // listed kernel that is never launched would withhold the capability for nothing.
        Assert.Equal(launched.OrderBy(n => n, StringComparer.Ordinal), PortableGenericFft.KernelNames.OrderBy(n => n, StringComparer.Ordinal));
    }

    private static string RepoRoot()
    {
        var dir = new System.IO.DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !System.IO.File.Exists(System.IO.Path.Combine(dir.FullName, "src", "AiDotNet.Tensors", "AiDotNet.Tensors.csproj")))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("Repository root not found.");
    }
}