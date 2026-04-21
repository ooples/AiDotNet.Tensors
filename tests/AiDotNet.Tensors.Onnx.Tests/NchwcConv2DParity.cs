using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// C3 parity: NCHWc Conv2D must produce bit-exact (within float-summation
/// tolerance) output vs the NCHW Conv2D path. Any divergence here means
/// the layout-aware inner loop's index math is wrong.
/// </summary>
public class NchwcConv2DParity
{
    [Theory]
    [InlineData(1, 8, 16, 16, 16, 3, 1, 1)]    // tiny 3x3 stride1 pad1
    [InlineData(1, 16, 32, 32, 32, 3, 1, 1)]   // ResNet block-shape sample
    [InlineData(1, 8, 8, 8, 16, 1, 1, 0)]      // 1x1 pointwise (kH=kW=1)
    [InlineData(2, 16, 14, 14, 32, 3, 2, 1)]   // stride 2
    [InlineData(1, 32, 7, 7, 64, 3, 1, 1)]     // 32→64 channels
    public void NchwcConv2D_BitExact_vs_Nchw(int N, int inC, int H, int W, int outC, int kSize, int stride, int pad)
    {
        var engine = new CpuEngine();
        var rng = new Random(0xDEAD);
        var input = new Tensor<float>(new[] { N, inC, H, W });
        var kernel = new Tensor<float>(new[] { outC, inC, kSize, kSize });
        RandomFill(input.AsWritableSpan(), rng);
        RandomFill(kernel.AsWritableSpan(), rng);

        // Reference: NCHW Conv2D (routes through im2col + SimdGemm fast path
        // for float, which is already well-tested bit-exact vs ORT).
        var nchwOut = engine.Conv2D(input, kernel,
            stride: new[] { stride, stride },
            padding: new[] { pad, pad },
            dilation: new[] { 1, 1 });
        Assert.Equal(TensorLayout.Nchw, nchwOut.Layout);

        // NCHWc path: reorder input + kernel in, run conv, reorder output back.
        var inputNchwc = engine.ReorderToNchwc(input, TensorLayout.Nchwc8);
        var nchwcOut = engine.Conv2D(inputNchwc, kernel,
            stride: new[] { stride, stride },
            padding: new[] { pad, pad },
            dilation: new[] { 1, 1 });
        Assert.Equal(TensorLayout.Nchwc8, nchwcOut.Layout);
        var nchwcReordered = engine.ReorderToNchw(nchwcOut);

        var refArr = nchwOut.AsSpan().ToArray();
        var ourArr = nchwcReordered.AsSpan().ToArray();
        Assert.Equal(refArr.Length, ourArr.Length);
        // FMA vs non-FMA summation ordering differs at LSB — allow 1e-4
        // relative tolerance (matches our ORT parity tolerance).
        float maxDiff = 0f;
        for (int i = 0; i < refArr.Length; i++)
        {
            float d = Math.Abs(refArr[i] - ourArr[i]);
            float scale = Math.Max(Math.Abs(refArr[i]), 1f);
            Assert.True(d <= 1e-3f * scale,
                $"Element {i}: NCHW {refArr[i]} vs NCHWc {ourArr[i]}, diff {d}");
            if (d > maxDiff) maxDiff = d;
        }
    }

    private static void RandomFill(Span<float> span, Random rng)
    {
        for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() * 2 - 1);
    }
}
