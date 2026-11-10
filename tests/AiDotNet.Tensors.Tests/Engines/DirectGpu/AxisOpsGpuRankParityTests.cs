// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System.Linq;
using AiDotNet.Tensors.Engines;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The GPU half of the rank conformance gate: every axis op the CPU sweep probes, on ranks 1 to 4 and every axis,
/// must give the CPU engine's result on the GPU engine. GPU overrides are a second implementation of each op, often
/// written for one rank with a fallback for the rest; this finds a fast path that is taken at a rank or axis it
/// computes wrong, for every op at once (see <see cref="AxisOpsRankConformanceTests"/>).
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class AxisOpsGpuRankParityTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly ITestOutputHelper _out;

    public AxisOpsGpuRankParityTests(DirectGpuTensorEngineTestFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _out = output;
    }

    [SkippableFact]
    public void Every_axis_op_on_every_rank_matches_the_cpu_engine()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var (failures, unprobed, probed) = AxisOpsRankConformanceTests.Sweep(_fixture.Engine!, new CpuEngine());
        _out.WriteLine($"probed {probed} ops; {failures.Count} failures; {unprobed.Count} unprobed");
        foreach (var f in failures) _out.WriteLine("FAIL " + f);
        Assert.True(failures.Count == 0 && unprobed.Count == 0,
            $"{failures.Count} GPU/CPU mismatches, {unprobed.Count} unprobed:\n" + string.Join("\n", failures.Concat(unprobed)));
    }
}
