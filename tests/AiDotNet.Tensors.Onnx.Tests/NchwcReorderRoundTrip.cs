using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// C2 sanity: NCHW → NCHWc → NCHW round-trips bit-exact. These reorder
/// primitives are the foundation of every downstream NCHWc op kernel, so
/// any off-by-one here would corrupt every conv/BN/pool output.
/// </summary>
public class NchwcReorderRoundTrip
{
    [Fact] public void Nchw_ToNchwc8_ToNchw_BitExact()
    {
        var engine = new CpuEngine();
        const int N = 2, C = 16, H = 4, W = 5;
        var rng = new Random(101);
        var src = new Tensor<float>(new[] { N, C, H, W });
        var srcSpan = src.AsWritableSpan();
        for (int i = 0; i < srcSpan.Length; i++)
            srcSpan[i] = (float)(rng.NextDouble() * 2 - 1);

        var packed = engine.ReorderToNchwc(src, TensorLayout.Nchwc8);
        Assert.Equal(TensorLayout.Nchwc8, packed.Layout);
        Assert.Equal(src.AsSpan().Length, packed.AsSpan().Length);

        var back = engine.ReorderToNchw(packed);
        Assert.Equal(TensorLayout.Nchw, back.Layout);
        var srcArr = src.AsSpan().ToArray();
        var backArr = back.AsSpan().ToArray();
        Assert.Equal(srcArr.Length, backArr.Length);
        for (int i = 0; i < srcArr.Length; i++)
            Assert.Equal(srcArr[i], backArr[i]);
    }

    [Fact] public void Nchw_ToNchwc16_ToNchw_BitExact()
    {
        var engine = new CpuEngine();
        const int N = 1, C = 32, H = 3, W = 3;
        var rng = new Random(102);
        var src = new Tensor<float>(new[] { N, C, H, W });
        var srcSpan = src.AsWritableSpan();
        for (int i = 0; i < srcSpan.Length; i++)
            srcSpan[i] = (float)(rng.NextDouble() * 2 - 1);

        var packed = engine.ReorderToNchwc(src, TensorLayout.Nchwc16);
        Assert.Equal(TensorLayout.Nchwc16, packed.Layout);

        var back = engine.ReorderToNchw(packed);
        var srcArr = src.AsSpan().ToArray();
        var backArr = back.AsSpan().ToArray();
        for (int i = 0; i < srcArr.Length; i++)
            Assert.Equal(srcArr[i], backArr[i]);
    }

    [Fact] public void NchwcPacking_HasExpectedLayout()
    {
        // Verify the cBlock=4 style lane grouping is right: after ToNchwc
        // with cBlock=8 on a [1, 8, 1, 1] input of values 0..7, the packed
        // output should be [0, 1, 2, 3, 4, 5, 6, 7] — the 8 channels at
        // spatial position (0,0) land contiguously.
        var engine = new CpuEngine();
        var src = new Tensor<float>(new[] { 1, 8, 1, 1 });
        var s = src.AsWritableSpan();
        for (int i = 0; i < 8; i++) s[i] = i;
        var packed = engine.ReorderToNchwc(src, TensorLayout.Nchwc8);
        var p = packed.AsSpan();
        for (int i = 0; i < 8; i++)
            Assert.Equal((float)i, p[i]);
    }

    [Fact] public void NchwcReorder_ChannelsNotDivisible_Throws()
    {
        var engine = new CpuEngine();
        var src = new Tensor<float>(new[] { 1, 6, 2, 2 }); // C=6 not divisible by 8
        Assert.Throws<ArgumentException>(() =>
            engine.ReorderToNchwc(src, TensorLayout.Nchwc8));
    }

    [Fact] public void NchwcReorder_SameLayout_ReturnsInput()
    {
        var engine = new CpuEngine();
        var src = new Tensor<float>(new[] { 1, 16, 2, 2 });
        src.Layout = TensorLayout.Nchwc8;
        var result = engine.ReorderToNchwc(src, TensorLayout.Nchwc8);
        Assert.Same(src, result);
    }

    [Fact] public void NchwReorder_AlreadyNchw_ReturnsInput()
    {
        var engine = new CpuEngine();
        var src = new Tensor<float>(new[] { 1, 8, 2, 2 });
        Assert.Equal(TensorLayout.Nchw, src.Layout);
        var result = engine.ReorderToNchw(src);
        Assert.Same(src, result);
    }
}
