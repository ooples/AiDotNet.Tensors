using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
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

        // CUDA's supported fused route projects [Q|K|V] in one GEMM and the output in another.
        // Measure their shape-dependent dispatch costs independently instead of granting fixed slack.
        // Other routes retain the original whole-call invariant; their projection topology can differ.
        bool packedCuda = gpu!.GetBackend() is CudaBackend backend
            && backend is IFusedAttentionBackend fused && fused.SupportsFusedAttention(Model / Heads)
            && backend is IGpuBufferViews views && views.ViewsAreFreeAtAnyOffset
            && backend is IFusedAttentionPreference preference
            && preference.PrefersFusedAttention(1, Heads, Seq, Seq, Model / Heads)
            && preference.PrefersFusedAttention(16, Heads, Seq, Seq, Model / Heads)
            && !GraphMode.IsActive && !AiDotNet.Tensors.Engines.Gpu.AutocastScope.IsEnabled
            && !AiDotNet.Tensors.Engines.Autodiff.DifferentiableOps.IsRecording<float>()
            && AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>.Current is null
            && !AiDotNet.Tensors.Engines.Autodiff.AnomalyModeScope.IsActive;
        long smallProjection = 0, largeProjection = 0;
        if (packedCuda)
        {
            long ProjectionLaunches(int batch)
            {
                var cuda = (CudaBackend)gpu.GetBackend()!;
                int rows = batch * Seq;
                using var input = cuda.AllocateBuffer(new float[rows * Model]);
                using var qkvWeight = cuda.AllocateBuffer(new float[Model * 3 * Model]);
                using var qkv = cuda.AllocateBuffer(new float[rows * 3 * Model]);
                using var heads = cuda.AllocateBuffer(new float[rows * Model]);
                using var outWeight = cuda.AllocateBuffer(new float[Model * Model]);
                using var output = cuda.AllocateBuffer(new float[rows * Model]);
                void Project()
                {
                    cuda.Gemm(input, qkvWeight, qkv, rows, 3 * Model, Model);
                    cuda.Gemm(heads, outWeight, output, rows, Model, Model);
                }
                Project();
                long before = GpuLaunchProbe.TotalEver;
                Project();
                return GpuLaunchProbe.TotalEver - before;
            }
            smallProjection = ProjectionLaunches(1);
            largeProjection = ProjectionLaunches(16);
            Assert.True(smallProjection > 0 && largeProjection > 0);
            Assert.True(smallLaunches > smallProjection && largeLaunches > largeProjection,
                "The attention must launch device work beyond its two projections.");
            _output.WriteLine($"projection launches: b1 {smallProjection}, b16 {largeProjection}");
        }
        Assert.True(largeLaunches - largeProjection <= smallLaunches - smallProjection,
            $"b16 took {largeLaunches} launches ({largeProjection} projections) against " +
            $"{smallLaunches} at b1 ({smallProjection} projections); attention launches grew with batch.");

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