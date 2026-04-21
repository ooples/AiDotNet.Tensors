using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// C6 parity: MaxPool / AvgPool / GlobalAvgPool on NCHWc8 must bit-match
/// (within float tolerance) the NCHW reference path. Each output cell
/// reduces per-channel-lane, so a single off-by-one on lane indexing would
/// silently cross-contaminate channels.
/// </summary>
public class NchwcPoolParity
{
    [Theory]
    [InlineData(1, 8, 14, 14, 3, 2, 1)]   // 3x3 stride2 pad1
    [InlineData(2, 16, 16, 16, 2, 2, 0)]  // 2x2 stride2 nopad — classic downsample
    [InlineData(1, 32, 7, 7, 3, 1, 1)]    // 3x3 stride1 pad1
    public void MaxPool_Nchwc8_BitExact(int N, int C, int H, int W, int k, int s, int p)
    {
        var engine = new CpuEngine();
        var x = RandomTensor(N, C, H, W, 0x1234);

        var refOut = engine.MaxPool2D(x, k, s, p);
        Assert.Equal(TensorLayout.Nchw, refOut.Layout);

        var xPacked = engine.ReorderToNchwc(x, TensorLayout.Nchwc8);
        var packedOut = engine.MaxPool2D(xPacked, k, s, p);
        Assert.Equal(TensorLayout.Nchwc8, packedOut.Layout);
        var back = engine.ReorderToNchw(packedOut);

        AssertArraysMatch(refOut.AsSpan().ToArray(), back.AsSpan().ToArray(), 1e-5f);
    }

    [Theory]
    [InlineData(1, 8, 14, 14, 3, 2, 1)]
    [InlineData(2, 16, 16, 16, 2, 2, 0)]
    [InlineData(1, 32, 7, 7, 3, 1, 1)]
    public void AvgPool_Nchwc8_BitExact(int N, int C, int H, int W, int k, int s, int p)
    {
        var engine = new CpuEngine();
        var x = RandomTensor(N, C, H, W, 0x5678);

        var refOut = engine.AvgPool2D(x, k, s, p);
        Assert.Equal(TensorLayout.Nchw, refOut.Layout);

        var xPacked = engine.ReorderToNchwc(x, TensorLayout.Nchwc8);
        var packedOut = engine.AvgPool2D(xPacked, k, s, p);
        Assert.Equal(TensorLayout.Nchwc8, packedOut.Layout);
        var back = engine.ReorderToNchw(packedOut);

        // AvgPool divides: ~5e-4 relative tolerance after float summation.
        AssertArraysMatch(refOut.AsSpan().ToArray(), back.AsSpan().ToArray(), 5e-4f);
    }

    [Theory]
    [InlineData(1, 16, 7, 7)]
    [InlineData(2, 32, 4, 4)]
    [InlineData(1, 64, 14, 14)]
    public void GlobalAvgPool_Nchwc8_BitExact(int N, int C, int H, int W)
    {
        var engine = new CpuEngine();
        var x = RandomTensor(N, C, H, W, 0xABCD);

        var refOut = engine.GlobalAvgPool2D(x);
        Assert.Equal(new[] { N, C, 1, 1 }, refOut._shape);

        var xPacked = engine.ReorderToNchwc(x, TensorLayout.Nchwc8);
        var packedOut = engine.GlobalAvgPool2D(xPacked);
        Assert.Equal(new[] { N, C, 1, 1 }, packedOut._shape);
        // GlobalAvgPool collapses spatial → output is NCHW (1x1 spatial, no
        // meaningful packed layout).
        Assert.Equal(TensorLayout.Nchw, packedOut.Layout);

        AssertArraysMatch(refOut.AsSpan().ToArray(), packedOut.AsSpan().ToArray(), 5e-4f);
    }

    private static Tensor<float> RandomTensor(int N, int C, int H, int W, int seed)
    {
        var t = new Tensor<float>(new[] { N, C, H, W });
        var rng = new Random(seed);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertArraysMatch(float[] refArr, float[] ourArr, float relTol)
    {
        Assert.Equal(refArr.Length, ourArr.Length);
        for (int i = 0; i < refArr.Length; i++)
        {
            float d = Math.Abs(refArr[i] - ourArr[i]);
            float scale = Math.Max(Math.Abs(refArr[i]), 1f);
            Assert.True(d <= relTol * scale,
                $"[{i}] ref {refArr[i]} vs ours {ourArr[i]} diff {d}");
        }
    }
}
