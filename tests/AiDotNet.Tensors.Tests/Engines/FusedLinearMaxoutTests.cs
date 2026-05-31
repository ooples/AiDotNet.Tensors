using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #499: FusedLinearMaxout — GEMM + bias + grouped-max. Maxout (Goodfellow et al.
/// 2013) reduces the feature dimension N to N/numPieces by taking the max over
/// consecutive groups, so it's a shape-changing op, not an activation epilogue.
/// </summary>
public class FusedLinearMaxoutTests
{
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void FusedLinearMaxout_MatchesLinearThenGroupedMax(int numPieces)
    {
        var engine = new CpuEngine();
        const int batch = 5, inF = 7;
        int units = 6, n = units * numPieces;
        var rng = new Random(20260610);

        var wData = new float[inF * n];
        for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var w = new Tensor<float>(wData, new[] { inF, n });
        var bData = new float[n];
        for (int i = 0; i < n; i++) bData[i] = (float)(rng.NextDouble() - 0.5);
        var bias = new Tensor<float>(bData, new[] { n });
        var xData = new float[batch * inF];
        for (int i = 0; i < xData.Length; i++) xData[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
        var x = new Tensor<float>(xData, new[] { batch, inF });

        // Reference: linear (no activation) then grouped max.
        var pre = engine.FusedLinear(x, w, bias, FusedActivationType.None); // [batch, n]
        var actual = engine.FusedLinearMaxout(x, w, bias, numPieces);        // [batch, units]

        Assert.Equal(units, actual._shape[actual._shape.Length - 1]);
        Assert.Equal(batch * units, actual.Length);
        for (int r = 0; r < batch; r++)
            for (int u = 0; u < units; u++)
            {
                float mx = float.NegativeInfinity;
                for (int j = 0; j < numPieces; j++) mx = Math.Max(mx, pre[r * n + u * numPieces + j]);
                float got = actual[r * units + u];
                Assert.True(Math.Abs(mx - got) < 1e-5, $"maxout[{r},{u}] {got} != {mx}");
            }
    }

    [Fact]
    public void FusedLinearMaxout_RejectsIndivisibleFeatureDim()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 1, 4 });
        var w = new Tensor<float>(new float[12], new[] { 4, 3 }); // N=3, not divisible by 2
        Assert.Throws<ArgumentException>(() => engine.FusedLinearMaxout(x, w, null, 2));
    }
}
