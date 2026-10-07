using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Tensor.BroadcastDivide / BroadcastMultiply through the stride-coalescing kernel: a per-row scalar ([B,S,H,D] op
/// [B,S,H,1], large enough for the parallel outer loop), a small serial case, and equal shapes on offset views.
/// </summary>
public class BroadcastDivideKernelTests
{
    private static float[] Rand(Random rng, int n, double lo) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(lo + rng.NextDouble()); return a; }

    [Theory]
    [InlineData(8, 128, 8, 64)]
    [InlineData(2, 3, 2, 5)]
    public void RowScalar_MatchesReference(int b, int s, int h, int d)
    {
        var rng = new Random(b * 31 + d);
        var xa = Rand(rng, b * s * h * d, -1); var ya = Rand(rng, b * s * h, 0.5);
        var x = new Tensor<float>(xa, new[] { b, s, h, d });
        var y = new Tensor<float>(ya, new[] { b, s, h, 1 });
        var q = x.BroadcastDivide(y).ToArray();
        var p = x.BroadcastMultiply(y).ToArray();
        var e = new CpuEngine().TensorDivide(x, y).ToArray();
        for (int i = 0; i < xa.Length; i++)
        {
            Assert.Equal(xa[i] / ya[i / d], q[i], 5);
            Assert.Equal(xa[i] * ya[i / d], p[i], 5);
            Assert.Equal(q[i], e[i]);
        }
    }

    [Fact]
    public void EqualShapes_OnOffsetViews_ReadTheirOwnElements()
    {
        var rng = new Random(7);
        var xa = Rand(rng, 40, -1); var ya = Rand(rng, 40, 0.5);
        // Row 1 of a [2,20] tensor: a contiguous view with storage offset 20.
        var x = new Tensor<float>(xa, new[] { 2, 20 }).Slice(0, 1, 2).Reshape(20);
        var y = new Tensor<float>(ya, new[] { 2, 20 }).Slice(0, 1, 2).Reshape(20);
        Assert.True(x._storageOffset == 20 && y._storageOffset == 20, "the slices must be offset views");
        var q = x.BroadcastDivide(y).ToArray();
        var p = x.BroadcastMultiply(y).ToArray();
        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(xa[20 + i] / ya[20 + i], q[i], 5);
            Assert.Equal(xa[20 + i] * ya[20 + i], p[i], 5);
        }
    }
}
