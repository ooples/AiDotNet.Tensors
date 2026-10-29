using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// BatchNormInference with a ReLU or ReLU6 clamps inside the batch-norm kernel on the CPU's float path (one pass over
/// the activation). It must equal batch norm followed by the activation, element for element, keep NaN, and do so on
/// both the NCHW and NCHWc8 layouts.
/// </summary>
public sealed class BatchNormInferenceFusedActivationTests
{
    private static Tensor<float> Rand(int seed, float scale, params int[] shape)
    {
        var rng = new Random(seed);
        int n = 1; foreach (var s in shape) n *= s;
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<float>(d, shape);
    }

    private static (Tensor<float> G, Tensor<float> B, Tensor<float> M, Tensor<float> V) Stats(int c)
    {
        var g = Rand(2, 2f, c); var b = Rand(3, 3f, c); var m = Rand(4, 1f, c);
        var v = Rand(5, 1f, c);
        var vd = v.ToArray(); for (int i = 0; i < vd.Length; i++) vd[i] = Math.Abs(vd[i]) + 0.1f;
        return (g, b, m, new Tensor<float>(vd, new[] { c }));
    }

    [Theory]
    [InlineData(FusedActivationType.ReLU, 2, 16, 9, 7)]     // spatial 63: a vector body and a scalar tail per plane
    [InlineData(FusedActivationType.ReLU6, 2, 16, 9, 7)]
    [InlineData(FusedActivationType.ReLU6, 4, 32, 16, 16)]  // past the parallel threshold
    [InlineData(FusedActivationType.ReLU, 1, 3, 2, 2)]      // smaller than one vector
    public void Nchw_EqualsBatchNormThenActivation(FusedActivationType act, int n, int c, int h, int w)
    {
        var e = new CpuEngine();
        var x = Rand(1, 8f, n, c, h, w);
        var (g, b, m, v) = Stats(c);
        var fused = e.BatchNormInference(x, g, b, m, v, 1e-5, act).ToArray();
        var plain = e.BatchNormInference(x, g, b, m, v, 1e-5).ToArray();
        float high = act == FusedActivationType.ReLU6 ? 6f : float.PositiveInfinity;
        int clampedLow = 0, clampedHigh = 0;
        for (int i = 0; i < plain.Length; i++)
        {
            float expected = Math.Min(Math.Max(plain[i], 0f), high);
            if (plain[i] < 0) clampedLow++;
            if (plain[i] > high) clampedHigh++;
            Assert.Equal(expected, fused[i]);
        }
        // The inputs exercise the clamp on both sides (a ReLU has no upper bound to reach).
        Assert.True(clampedLow > 0, "no value was below zero");
        if (act == FusedActivationType.ReLU6) Assert.True(clampedHigh > 0, "no value was above six");
    }

    [Fact]
    public void Nchwc8_EqualsTheNchwResult()
    {
        var e = new CpuEngine();
        var x = Rand(11, 8f, 2, 16, 5, 6);
        var (g, b, m, v) = Stats(16);
        var expected = e.BatchNormInference(x, g, b, m, v, 1e-5, FusedActivationType.ReLU6).ToArray();
        var packed = e.BatchNormInference(e.ReorderToNchwc(x, TensorLayout.Nchwc8), g, b, m, v, 1e-5,
            FusedActivationType.ReLU6);
        Assert.Equal(TensorLayout.Nchwc8, packed.Layout);
        Assert.Equal(expected, e.ReorderToNchw(packed).ToArray());
    }

    [Theory]
    [InlineData(FusedActivationType.ReLU)]
    [InlineData(FusedActivationType.ReLU6)]
    [InlineData(FusedActivationType.None)]
    public void NaN_Propagates(FusedActivationType act)
    {
        // PyTorch's relu(NaN) and hardtanh(NaN) are NaN; a max/min with the bound as the second operand returned the
        // bound instead. Both the vector body and the scalar tail are covered (spatial 11).
        var e = new CpuEngine();
        var x = Rand(21, 4f, 1, 2, 1, 11);
        var xd = x.ToArray(); xd[3] = float.NaN; xd[10] = float.NaN; xd[14] = float.NaN;
        var (g, b, m, v) = Stats(2);
        var y = e.BatchNormInference(new Tensor<float>(xd, x.Shape.ToArray()), g, b, m, v, 1e-5, act).ToArray();
        Assert.True(float.IsNaN(y[3]) && float.IsNaN(y[10]) && float.IsNaN(y[14]), "NaN did not propagate");
    }
}