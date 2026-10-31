using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Inside a training trace a 4D float BatchNorm on the CPU is recorded like any other op: the trace computes nothing (it
/// used to normalize eagerly, which realized the whole forward and paged in every recorded output, including those the
/// compiled plan drops). A read realizes it, normalizing straight into its recorded output; its output, batch mean and
/// batch variance must equal eager BatchNorm's bit for bit, including when the input is itself a recorded (lazy) tensor.
/// The trace's channel-bias add likewise writes its recorded output directly and must equal the eager op.
/// </summary>
[Collection("CompilationGlobalState")]
public class TrainingTraceBatchNormTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    [Fact]
    public void TracedBatchNorm_EqualsEager_BitForBit()
    {
        var engine = new CpuEngine();
        var x = Filled(new[] { 3, 5, 4, 6 }, 1, 3f);
        var scale = Filled(new[] { 3, 5, 4, 6 }, 2, 1f);
        var gamma = Filled(new[] { 5 }, 3, 1f);
        var beta = Filled(new[] { 5 }, 4, 1f);

        var eagerInput = engine.TensorMultiply(x, scale);
        var eager = engine.BatchNorm(eagerInput, gamma, beta, 1e-5, out var eagerMean, out var eagerVar);

        float[] traced, tracedMean, tracedVar;
        using (var scope = GraphMode.EnableTraining(new[] { scale, gamma, beta }))
        {
            var input = engine.TensorMultiply(x, scale);   // recorded: the BatchNorm reads a lazy input
            var y = engine.BatchNorm(input, gamma, beta, 1e-5, out var mean, out var variance);
            traced = y.ToArray();
            tracedMean = mean.ToArray();
            tracedVar = variance.ToArray();
        }

        Assert.Equal(eager.ToArray(), traced);
        Assert.Equal(eagerMean.ToArray(), tracedMean);
        Assert.Equal(eagerVar.ToArray(), tracedVar);
    }

    [Fact]
    public void TracedBatchNorm_ComputesNothingUntilRead_ThenStatisticsEqualEager()
    {
        var engine = new CpuEngine();
        var x = Filled(new[] { 2, 4, 5, 3 }, 5, 3f);
        var scale = Filled(new[] { 2, 4, 5, 3 }, 6, 1f);
        var gamma = Filled(new[] { 4 }, 7, 1f);
        var beta = Filled(new[] { 4 }, 8, 1f);
        engine.BatchNorm(engine.TensorMultiply(x, scale), gamma, beta, 1e-5, out var eagerMean, out var eagerVar);

        using var scope = GraphMode.EnableTraining(new[] { scale, gamma, beta });
        var input = engine.TensorMultiply(x, scale);
        var y = engine.BatchNorm(input, gamma, beta, 1e-5, out var mean, out var variance);
        Assert.False(y.LazySource?.IsRealized ?? true, "the trace normalized the BatchNorm");
        Assert.False(input.LazySource?.IsRealized ?? true, "the trace computed the BatchNorm's input");
        Assert.False(mean.LazySource?.IsRealized ?? true, "the trace realized the batch mean");
        Assert.False(variance.LazySource?.IsRealized ?? true, "the trace realized the batch variance");

        // The statistics are read first: they must realize the BatchNorm, not copy its unset statistics.
        Assert.Equal(eagerMean.ToArray(), mean.ToArray());
        Assert.Equal(eagerVar.ToArray(), variance.ToArray());
    }

    [Fact]
    public void TracedChannelBiasAdd_EqualsEager_BitForBit()
    {
        var engine = new CpuEngine();
        var x = Filled(new[] { 3, 5, 4, 6 }, 9, 3f);
        var scale = Filled(new[] { 3, 5, 4, 6 }, 10, 1f);
        var bias = Filled(new[] { 5 }, 11, 1f);
        var eager = engine.TensorChannelBiasAdd(engine.TensorMultiply(x, scale), bias).ToArray();

        float[] traced;
        using (var scope = GraphMode.EnableTraining(new[] { scale, bias }))
        {
            var y = engine.TensorChannelBiasAdd(engine.TensorMultiply(x, scale), bias);
            traced = y.ToArray();
        }
        Assert.Equal(eager, traced);
    }
}
