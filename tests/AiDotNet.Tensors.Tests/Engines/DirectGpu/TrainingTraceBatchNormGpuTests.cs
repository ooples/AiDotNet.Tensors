using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Inside a training trace a 4D float BatchNorm on a GPU engine is recorded without being computed, as on the CPU. The
/// trace used to normalize on the device, then realize the recorded node (normalizing again, and realizing every
/// unrealized producer above it) to copy the result in; the pooled eager result kept its device buffer for the run:
/// 1.1 GB of trace-time BatchNorm outputs on DenseNet-121. A read realizes the node, and its output and batch
/// statistics must equal eager BatchNorm on the same engine bit for bit.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class TrainingTraceBatchNormGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public TrainingTraceBatchNormGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    [SkippableFact]
    public void TracedBatchNorm_ComputesNothingUntilRead_ThenEqualsEager()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        IEngine engine = gpu;

        var x = Filled(new[] { 2, 8, 6, 5 }, 21, 3f);
        var scale = Filled(new[] { 2, 8, 6, 5 }, 22, 1f);
        var gamma = Filled(new[] { 8 }, 23, 1f);
        var beta = Filled(new[] { 8 }, 24, 1f);
        var eager = engine.BatchNorm(engine.TensorMultiply(x, scale), gamma, beta, 1e-5, out var eagerMean, out var eagerVar);
        float[] expected = eager.ToArray(), expectedMean = eagerMean.ToArray(), expectedVar = eagerVar.ToArray();

        using var scope = GraphMode.EnableTraining(new[] { scale, gamma, beta });
        var input = engine.TensorMultiply(x, scale);
        var y = engine.BatchNorm(input, gamma, beta, 1e-5, out var mean, out var variance);
        Assert.False(y.LazySource?.IsRealized ?? true, "the trace normalized the BatchNorm");
        Assert.False(input.LazySource?.IsRealized ?? true, "the trace computed the BatchNorm's input");
        Assert.False(mean.LazySource?.IsRealized ?? true, "the trace realized the batch mean");
        Assert.False(variance.LazySource?.IsRealized ?? true, "the trace realized the batch variance");

        // The statistics are read first: they must realize the BatchNorm, not copy its unset statistics.
        Assert.Equal(expectedMean, mean.ToArray());
        Assert.Equal(expectedVar, variance.ToArray());
        Assert.Equal(expected, y.ToArray());
    }
}
