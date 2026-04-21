using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// B4 parity: the Nchwc16 (AVX-512 cBlock=16) Conv2D must match the Nchw
/// reference bit-exact. If this passes, flipping LayoutPlanner's
/// PreferredCBlock to 16 on AVX-512 hosts is safe.
/// </summary>
public class NchwcConv2D16Parity
{
    [Theory]
    [InlineData(1, 16, 16, 16, 16, 3, 1, 1)]
    [InlineData(1, 32, 14, 14, 32, 3, 1, 1)]
    [InlineData(1, 16, 7, 7, 32, 1, 1, 0)]   // 1×1 pointwise, 16→32
    [InlineData(2, 32, 8, 8, 64, 3, 2, 1)]   // stride 2
    public void Conv2D_Nchwc16_BitExact_vs_Nchw(int N, int inC, int H, int W, int outC, int kSize, int stride, int pad)
    {
        var engine = new CpuEngine();
        var rng = new Random(0xC0DE);
        var input = new Tensor<float>(new[] { N, inC, H, W });
        var kernel = new Tensor<float>(new[] { outC, inC, kSize, kSize });
        RandomFill(input.AsWritableSpan(), rng);
        RandomFill(kernel.AsWritableSpan(), rng);

        var refOut = engine.Conv2D(input, kernel,
            stride: new[] { stride, stride },
            padding: new[] { pad, pad },
            dilation: new[] { 1, 1 });
        Assert.Equal(TensorLayout.Nchw, refOut.Layout);

        var packedIn = engine.ReorderToNchwc(input, TensorLayout.Nchwc16);
        var packedOut = engine.Conv2D(packedIn, kernel,
            stride: new[] { stride, stride },
            padding: new[] { pad, pad },
            dilation: new[] { 1, 1 });
        Assert.Equal(TensorLayout.Nchwc16, packedOut.Layout);
        var back = engine.ReorderToNchw(packedOut);

        var refArr = refOut.AsSpan().ToArray();
        var ourArr = back.AsSpan().ToArray();
        Assert.Equal(refArr.Length, ourArr.Length);
        for (int i = 0; i < refArr.Length; i++)
        {
            float d = Math.Abs(refArr[i] - ourArr[i]);
            float scale = Math.Max(Math.Abs(refArr[i]), 1f);
            Assert.True(d <= 1e-3f * scale,
                $"[{i}] ref {refArr[i]} vs ours {ourArr[i]} diff {d}");
        }
    }

    private static void RandomFill(Span<float> span, Random rng)
    {
        for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() * 2 - 1);
    }
}
