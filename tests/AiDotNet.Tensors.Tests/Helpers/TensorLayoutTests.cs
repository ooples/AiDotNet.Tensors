using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using TensorLayout = AiDotNet.Tensors.Helpers.TensorLayout;

namespace AiDotNet.Tensors.Tests.Helpers;

public class TensorLayoutTests
{
    [Fact]
    public void ToNHWC_CorrectShape()
    {
        var nchw = new Tensor<float>(new[] { 2, 3, 4, 5 });
        var nhwc = TensorLayout.ToNHWC(nchw);
        Assert.Equal(new[] { 2, 4, 5, 3 }, nhwc.Shape.ToArray());
    }

    [Fact]
    public void ToNCHW_CorrectShape()
    {
        var nhwc = new Tensor<float>(new[] { 2, 4, 5, 3 });
        var nchw = TensorLayout.ToNCHW(nhwc);
        Assert.Equal(new[] { 2, 3, 4, 5 }, nchw.Shape.ToArray());
    }

    [Fact]
    public void RoundTrip_NCHW_NHWC_NCHW_Preserves()
    {
        var original = new Tensor<float>(new[] { 1, 3, 4, 4 });
        for (int i = 0; i < original.Length; i++)
            original.AsWritableSpan()[i] = i + 1f;

        var nhwc = TensorLayout.ToNHWC(original);
        var back = TensorLayout.ToNCHW(nhwc);

        Assert.Equal(original.Shape, back.Shape);
        for (int i = 0; i < original.Length; i++)
            Assert.Equal(original.AsSpan()[i], back.AsSpan()[i]);
    }

    [Fact]
    public void ToNHWC_PixelLayout_ChannelsContiguous()
    {
        // [1, 3, 2, 2] — 3 channels, 2x2 spatial
        // NCHW: [R00,R01,R10,R11, G00,G01,G10,G11, B00,B01,B10,B11]
        var nchw = new Tensor<float>(new[] { 1, 3, 2, 2 });
        var span = nchw.AsWritableSpan();
        // Channel 0 (R)
        span[0] = 1f; span[1] = 2f; span[2] = 3f; span[3] = 4f;
        // Channel 1 (G)
        span[4] = 5f; span[5] = 6f; span[6] = 7f; span[7] = 8f;
        // Channel 2 (B)
        span[8] = 9f; span[9] = 10f; span[10] = 11f; span[11] = 12f;

        var nhwc = TensorLayout.ToNHWC(nchw);
        var dst = nhwc.AsSpan();

        // NHWC: pixel (0,0) = [R00,G00,B00] = [1,5,9]
        Assert.Equal(1f, dst[0]);   // R at (0,0)
        Assert.Equal(5f, dst[1]);   // G at (0,0)
        Assert.Equal(9f, dst[2]);   // B at (0,0)
        // pixel (0,1) = [R01,G01,B01] = [2,6,10]
        Assert.Equal(2f, dst[3]);
        Assert.Equal(6f, dst[4]);
        Assert.Equal(10f, dst[5]);
    }

    [Fact]
    public void ToNHWCInto_ZeroAlloc()
    {
        var nchw = new Tensor<float>(new[] { 1, 3, 4, 4 });
        for (int i = 0; i < nchw.Length; i++)
            nchw.AsWritableSpan()[i] = i;

        var nhwc = new Tensor<float>(new[] { 1, 4, 4, 3 });
        TensorLayout.ToNHWCInto(nhwc, nchw);

        // Verify round-trip
        var back = TensorLayout.ToNCHW(nhwc);
        for (int i = 0; i < nchw.Length; i++)
            Assert.Equal(nchw.AsSpan()[i], back.AsSpan()[i]);
    }

    [Fact]
    public void ToNCHWInto_ZeroAlloc()
    {
        var nhwc = new Tensor<float>(new[] { 1, 4, 4, 3 });
        for (int i = 0; i < nhwc.Length; i++)
            nhwc.AsWritableSpan()[i] = i;

        var nchw = new Tensor<float>(new[] { 1, 3, 4, 4 });
        TensorLayout.ToNCHWInto(nchw, nhwc);

        // Verify round-trip
        var back = TensorLayout.ToNHWC(nchw);
        for (int i = 0; i < nhwc.Length; i++)
            Assert.Equal(nhwc.AsSpan()[i], back.AsSpan()[i]);
    }

    [Fact]
    public void ToNHWC_WrongRank_Throws()
    {
        var t = new Tensor<float>(new[] { 4, 4 }); // 2D
        Assert.Throws<ArgumentException>(() => TensorLayout.ToNHWC(t));
    }

    [Fact]
    public void MultiBatch_PreservesPerBatch()
    {
        var nchw = new Tensor<float>(new[] { 2, 2, 3, 3 });
        var span = nchw.AsWritableSpan();
        // Batch 0: fill with 1s, Batch 1: fill with 2s
        int half = 2 * 3 * 3;
        for (int i = 0; i < half; i++) span[i] = 1f;
        for (int i = half; i < nchw.Length; i++) span[i] = 2f;

        var nhwc = TensorLayout.ToNHWC(nchw);
        var dst = nhwc.AsSpan();

        // First half (batch 0) should all be 1s
        for (int i = 0; i < half; i++)
            Assert.Equal(1f, dst[i]);
        // Second half (batch 1) should all be 2s
        for (int i = half; i < nchw.Length; i++)
            Assert.Equal(2f, dst[i]);
    }
}
