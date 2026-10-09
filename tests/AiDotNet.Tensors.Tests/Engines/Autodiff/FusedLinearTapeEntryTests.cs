using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A fused linear layer under a tape records ONE entry, whose backward covers the matmul, the bias and the activation.
/// The activation used to be applied while recording, adding an entry that the fused op's cleanup then removed in place
/// of the matmul's, which stayed on the tape.
/// </summary>
public class FusedLinearTapeEntryTests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = (float)(rng.NextDouble() - 0.5);
        return tensor;
    }

    [Theory]
    [InlineData(FusedActivationType.ReLU, true)]
    [InlineData(FusedActivationType.Sigmoid, true)]
    [InlineData(FusedActivationType.ReLU, false)]
    [InlineData(FusedActivationType.None, true)]
    public void FusedLinear_RecordsExactlyOneTapeEntry(FusedActivationType activation, bool withBias)
    {
        var engine = new CpuEngine();
        var x = Rand(new[] { 4, 6 }, 1);
        var w = Rand(new[] { 6, 5 }, 2);
        var b = withBias ? Rand(new[] { 5 }, 3) : null;

        using var tape = new GradientTape<float>();
        engine.FusedLinear(x, w, b, activation);

        Assert.Equal(1, tape.EntryCount);
    }

    [Fact]
    public void FusedLinearRelu_GradientsMatchTheUnfusedOps()
    {
        var engine = new CpuEngine();
        var x = Rand(new[] { 4, 6 }, 4);
        var w = Rand(new[] { 6, 5 }, 5);
        var b = Rand(new[] { 5 }, 6);

        float[][] Grads(bool fused)
        {
            using var tape = new GradientTape<float>();
            var y = fused
                ? engine.FusedLinear(x, w, b, FusedActivationType.ReLU)
                : engine.ReLU(engine.TensorBroadcastAdd(engine.TensorMatMul(x, w), b));
            var loss = engine.ReduceSum(engine.TensorMultiply(y, y), null);
            var g = tape.ComputeGradients(loss, new[] { x, w, b });
            return new[] { g[x].ToArray(), g[w].ToArray(), g[b].ToArray() };
        }

        var fusedGrads = Grads(fused: true);
        var unfusedGrads = Grads(fused: false);
        for (int t = 0; t < 3; t++)
            for (int i = 0; i < fusedGrads[t].Length; i++)
                Assert.True(Math.Abs(fusedGrads[t][i] - unfusedGrads[t][i]) <= 1e-5f * (1 + Math.Abs(unfusedGrads[t][i])),
                    $"gradient {t}[{i}]: fused {fusedGrads[t][i]} unfused {unfusedGrads[t][i]}");
    }
}
