using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Inside a training trace a 4D float BatchNorm on the CPU normalizes straight into its recorded output (no eager result
/// copied in, no copy of the input). Its output, batch mean and batch variance must equal eager BatchNorm's bit for bit,
/// including when the input is itself a recorded (lazy) tensor.
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
}
