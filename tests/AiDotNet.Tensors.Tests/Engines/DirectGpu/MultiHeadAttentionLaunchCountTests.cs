using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// AiDotNet #2384: an unmasked GPU <c>MultiHeadAttentionForward</c> looped over every (batch, head) pair with about
/// seven launches each, so Transformer and ViT inference time grew linearly with batch (85x PyTorch at b128). It now
/// attends over all heads at once (batched GEMMs, or the fused kernel for long sequences), so its launch count must
/// not depend on the batch.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class MultiHeadAttentionLaunchCountTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private const int Seq = 32;
    private const int Model = 64;
    private const int Heads = 4;

    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly ITestOutputHelper _output;

    public MultiHeadAttentionLaunchCountTests(DirectGpuTensorEngineTestFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    [SkippableFact]
    public void Unmasked_forward_launch_count_does_not_grow_with_batch_and_matches_the_cpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        IEngine engine = gpu;
        var cpu = new CpuEngine();

        var qW = Rand([Model, Model], 11, 0.2f);
        var kW = Rand([Model, Model], 12, 0.2f);
        var vW = Rand([Model, Model], 13, 0.2f);
        var oW = Rand([Model, Model], 14, 0.2f);

        long Launches(Tensor<float> input)
        {
            engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, Heads);   // warm: kernels and buffer pools
            long before = GpuLaunchProbe.TotalEver;
            engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, Heads);
            return GpuLaunchProbe.TotalEver - before;
        }

        var small = Rand([1, Seq, Model], 1, 1f);
        var large = Rand([16, Seq, Model], 2, 1f);
        // A device route that threw and fell back to the CPU would pass both the launch bound and the parity check.
        var fallbacksBefore = GpuLaunchProbe.Fallbacks;
        long smallLaunches = Launches(small);
        long largeLaunches = Launches(large);
        _output.WriteLine($"b1: {smallLaunches} launches, b16: {largeLaunches} launches");

        // The per-(batch, head) loop took about 7 launches per pair: 28 at b1 and 448 at b16.
        Assert.True(largeLaunches <= smallLaunches,
            $"b16 took {largeLaunches} launches against {smallLaunches} at b1; the per-head loop is back.");

        var actual = engine.MultiHeadAttentionForward(large, qW, kW, vW, oW, Heads).ToArray();
        Assert.Equal(fallbacksBefore, GpuLaunchProbe.Fallbacks);
        var expected = cpu.MultiHeadAttentionForward(large, qW, kW, vW, oW, Heads).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(expected[i])),
                $"output[{i}]: gpu {actual[i]} vs cpu {expected[i]}");
    }

    [SkippableFact]
    public void Past_the_score_budget_the_fused_kernel_runs_and_matches_the_cpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        IEngine engine = gpu!;
        var cpu = new CpuEngine();
        var qW = Rand([Model, Model], 21, 0.2f);
        var kW = Rand([Model, Model], 22, 0.2f);
        var vW = Rand([Model, Model], 23, 0.2f);
        var oW = Rand([Model, Model], 24, 0.2f);
        var input = Rand([4, Seq, Model], 3, 1f);

        long priorBudget = DirectGpuTensorEngine.MhaBatchedScoreElementBudget;
        DirectGpuTensorEngine.MhaBatchedScoreElementBudget = 0;   // every shape is past it: the fused branch
        bool priorShort = DirectGpuTensorEngine.ShortAttentionEnabled;
        DirectGpuTensorEngine.ShortAttentionEnabled = false;      // and the short-sequence kernel would take this shape first
        try
        {
            var fallbacksBefore = GpuLaunchProbe.Fallbacks;
            engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, Heads);   // warm
            long before = GpuLaunchProbe.TotalEver;
            var actual = engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, Heads).ToArray();
            long launches = GpuLaunchProbe.TotalEver - before;
            _output.WriteLine($"fused branch, b4: {launches} launches");
            Assert.Equal(fallbacksBefore, GpuLaunchProbe.Fallbacks);
            // The per-head loop would take ~7 per (batch, head) pair: 112 here.
            Assert.True(launches < 4 * Heads * 7, $"The fused branch took {launches} launches; it fell back to the loop.");

            var expected = cpu.MultiHeadAttentionForward(input, qW, kW, vW, oW, Heads).ToArray();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(expected[i])),
                    $"output[{i}]: gpu {actual[i]} vs cpu {expected[i]}");
        }
        finally
        {
            DirectGpuTensorEngine.MhaBatchedScoreElementBudget = priorBudget;
            DirectGpuTensorEngine.ShortAttentionEnabled = priorShort;
        }
    }
}