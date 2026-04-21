using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// C4 parity: inference BatchNorm on NCHWc must bit-match (within float
/// summation tolerance) the NCHW reference. If lane packing or per-channel
/// scale/bias alignment is off by one, downstream conv outputs corrupt.
/// </summary>
public class NchwcBatchNormParity
{
    [Theory]
    [InlineData(1, 8, 7, 7)]    // 1 channel-group
    [InlineData(2, 16, 14, 14)] // ResNet block-shape sample
    [InlineData(1, 32, 28, 28)] // mid-network spatial
    [InlineData(2, 64, 4, 4)]   // 8 channel-groups, small spatial
    public void BatchNormInference_Nchwc8_BitExact_vs_Nchw(int N, int C, int H, int W)
    {
        var engine = new CpuEngine();
        var rng = new Random(0xB00B);

        var x = new Tensor<float>(new[] { N, C, H, W });
        var gamma = new Tensor<float>(new[] { C });
        var beta = new Tensor<float>(new[] { C });
        var mean = new Tensor<float>(new[] { C });
        var varT = new Tensor<float>(new[] { C });
        RandomFill(x.AsWritableSpan(), rng, -1f, 1f);
        RandomFill(gamma.AsWritableSpan(), rng, 0.5f, 1.5f);
        RandomFill(beta.AsWritableSpan(), rng, -0.5f, 0.5f);
        RandomFill(mean.AsWritableSpan(), rng, -0.2f, 0.2f);
        // Variance must be positive — sample from [0.1, 1.1].
        RandomFill(varT.AsWritableSpan(), rng, 0.1f, 1.1f);

        const float epsilon = 1e-5f;

        // Reference: NCHW fused path.
        var nchwOut = engine.BatchNormInference(x, gamma, beta, mean, varT, epsilon);
        Assert.Equal(TensorLayout.Nchw, nchwOut.Layout);

        // NCHWc8 path: pack input, run inference BN, reorder back.
        var xPacked = engine.ReorderToNchwc(x, TensorLayout.Nchwc8);
        var nchwcOut = engine.BatchNormInference(xPacked, gamma, beta, mean, varT, epsilon);
        Assert.Equal(TensorLayout.Nchwc8, nchwcOut.Layout);
        var nchwcBack = engine.ReorderToNchw(nchwcOut);

        var refArr = nchwOut.AsSpan().ToArray();
        var ourArr = nchwcBack.AsSpan().ToArray();
        Assert.Equal(refArr.Length, ourArr.Length);
        for (int i = 0; i < refArr.Length; i++)
        {
            float d = Math.Abs(refArr[i] - ourArr[i]);
            float scale = Math.Max(Math.Abs(refArr[i]), 1f);
            Assert.True(d <= 1e-4f * scale,
                $"Element {i}: NCHW {refArr[i]} vs NCHWc {ourArr[i]}, diff {d}");
        }
    }

    [Fact]
    public void BatchNormInference_MatchesManualFormula()
    {
        // Smoke test: for a known scale/bias/mean/var, the fused path must
        // equal gamma * (x - mean) / sqrt(var + eps) + beta.
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 1, 4, 2, 2 });
        var s = x.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = i;
        var gamma = new Tensor<float>(new[] { 4 });
        var beta = new Tensor<float>(new[] { 4 });
        var mean = new Tensor<float>(new[] { 4 });
        var varT = new Tensor<float>(new[] { 4 });
        var g = gamma.AsWritableSpan(); g[0] = 1; g[1] = 2; g[2] = 0.5f; g[3] = -1;
        var b = beta.AsWritableSpan(); b[0] = 0; b[1] = 0.1f; b[2] = -0.5f; b[3] = 0.25f;
        var m = mean.AsWritableSpan(); m[0] = 0.5f; m[1] = 1; m[2] = 2; m[3] = 3;
        var v = varT.AsWritableSpan(); v[0] = 0.25f; v[1] = 1; v[2] = 4; v[3] = 0.5f;
        const float eps = 1e-5f;

        var y = engine.BatchNormInference(x, gamma, beta, mean, varT, eps);
        var yArr = y.AsSpan().ToArray();
        var xArr = x.AsSpan().ToArray();
        for (int c = 0; c < 4; c++)
        {
            float scaleC = (float)(g[c] / Math.Sqrt(v[c] + eps));
            float biasC = b[c] - scaleC * m[c];
            for (int sp = 0; sp < 4; sp++)
            {
                int idx = c * 4 + sp;
                float expected = xArr[idx] * scaleC + biasC;
                Assert.Equal(expected, yArr[idx], 3);
            }
        }
    }

    private static void RandomFill(Span<float> span, Random rng, float lo, float hi)
    {
        float range = hi - lo;
        for (int i = 0; i < span.Length; i++)
            span[i] = lo + (float)rng.NextDouble() * range;
    }
}
