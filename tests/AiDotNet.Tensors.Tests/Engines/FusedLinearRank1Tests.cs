using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Any-rank support: a rank-1 input [K] is a single unbatched sample. FusedLinear
/// (and MlpForward, which chains it) must accept it — promoting to [1, K], computing,
/// and squeezing back to [N] — rather than throwing "TensorMatMul requires rank >= 2".
/// Regression guard: this broke FeedForwardNeuralNetwork.Predict on unbatched inputs.
/// </summary>
public class FusedLinearRank1Tests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    [Fact]
    public void FusedLinear_Rank1Input_MatchesRank2Squeezed()
    {
        var engine = new CpuEngine();
        const int K = 12, N = 5;
        var weights = Rand(new[] { K, N }, 1);
        var bias = Rand(new[] { N }, 2);
        var x1 = Rand(new[] { K }, 3);                       // rank-1 [K]
        var x2 = engine.Reshape(x1, new[] { 1, K });          // [1, K]

        var r1 = engine.FusedLinear(x1, weights, bias, FusedActivationType.ReLU);
        var r2 = engine.FusedLinear(x2, weights, bias, FusedActivationType.ReLU);

        Assert.Equal(1, r1.Rank);                             // squeezed back to [N]
        Assert.Equal(N, r1.Shape[0]);
        Assert.Equal(new[] { 1, N }, r2.Shape);

        var a = r1.ToArray(); var b = r2.ToArray();
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(a[i] - b[i]) <= 1e-6f, $"[{i}] {a[i]} vs {b[i]}");
    }

    [Fact]
    public void MlpForward_Rank1Input_MatchesRank2Squeezed()
    {
        var engine = new CpuEngine();
        int[] dims = { 12, 8, 4 };
        var w = new System.Collections.Generic.List<Tensor<float>>
        {
            Rand(new[] { dims[0], dims[1] }, 10),
            Rand(new[] { dims[1], dims[2] }, 11),
        };
        var biases = new System.Collections.Generic.List<Tensor<float>?>
        {
            Rand(new[] { dims[1] }, 12),
            Rand(new[] { dims[2] }, 13),
        };
        var x1 = Rand(new[] { dims[0] }, 14);
        var x2 = engine.Reshape(x1, new[] { 1, dims[0] });

        var r1 = engine.MlpForward(x1, w, biases, FusedActivationType.ReLU, FusedActivationType.None);
        var r2 = engine.MlpForward(x2, w, biases, FusedActivationType.ReLU, FusedActivationType.None);

        Assert.Equal(dims[2], r1.ToArray().Length);
        var a = r1.ToArray(); var b = r2.ToArray();
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(a[i] - b[i]) <= 1e-6f, $"[{i}] {a[i]} vs {b[i]}");
    }
}
