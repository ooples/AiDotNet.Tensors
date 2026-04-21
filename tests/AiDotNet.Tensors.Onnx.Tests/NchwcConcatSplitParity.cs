using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// C7 parity: Concat/Split along the channel axis with NCHWc8 inputs must
/// round-trip to the same bytes as the NCHW-reference result. If the
/// channel-group index math is wrong, one output slab silently contains
/// another slab's channels.
/// </summary>
public class NchwcConcatSplitParity
{
    [Fact]
    public void Concat_AxisC_Nchwc8_BitExact_vs_Nchw()
    {
        var engine = new CpuEngine();
        var a = RandomTensor(1, 8, 4, 4, 11);
        var b = RandomTensor(1, 16, 4, 4, 22);
        var c = RandomTensor(1, 8, 4, 4, 33);

        // Reference: NCHW concat along axis=1.
        var refOut = engine.Concat(new[] { a, b, c }, axis: 1);
        Assert.Equal(new[] { 1, 32, 4, 4 }, refOut._shape);

        // Packed: concat on NCHWc8 inputs stays NCHWc8, reorder back to NCHW.
        var aP = engine.ReorderToNchwc(a, TensorLayout.Nchwc8);
        var bP = engine.ReorderToNchwc(b, TensorLayout.Nchwc8);
        var cP = engine.ReorderToNchwc(c, TensorLayout.Nchwc8);
        var packedOut = engine.Concat(new[] { aP, bP, cP }, axis: 1);
        Assert.Equal(TensorLayout.Nchwc8, packedOut.Layout);
        var back = engine.ReorderToNchw(packedOut);

        AssertArraysBitExact(refOut.AsSpan().ToArray(), back.AsSpan().ToArray());
    }

    [Fact]
    public void Split_AxisC_Nchwc8_BitExact_vs_Nchw()
    {
        var engine = new CpuEngine();
        var x = RandomTensor(2, 24, 3, 3, 77);

        // Reference: split NCHW into 3 chunks of 8 along axis=1.
        var refChunks = engine.TensorSplit(x, 3, axis: 1);
        Assert.Equal(3, refChunks.Length);
        foreach (var r in refChunks)
            Assert.Equal(new[] { 2, 8, 3, 3 }, r._shape);

        // Packed: same split on NCHWc8.
        var xP = engine.ReorderToNchwc(x, TensorLayout.Nchwc8);
        var packedChunks = engine.TensorSplit(xP, 3, axis: 1);
        Assert.Equal(3, packedChunks.Length);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(TensorLayout.Nchwc8, packedChunks[i].Layout);
            var back = engine.ReorderToNchw(packedChunks[i]);
            // NCHW split returns views (non-contiguous); force contiguous for byte comparison.
            var refContig = refChunks[i].Contiguous();
            AssertArraysBitExact(refContig.AsSpan().ToArray(), back.AsSpan().ToArray());
        }
    }

    private static Tensor<float> RandomTensor(int N, int C, int H, int W, int seed)
    {
        var t = new Tensor<float>(new[] { N, C, H, W });
        var rng = new Random(seed);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertArraysBitExact(float[] refArr, float[] ourArr)
    {
        Assert.Equal(refArr.Length, ourArr.Length);
        for (int i = 0; i < refArr.Length; i++)
            Assert.Equal(refArr[i], ourArr[i]);
    }
}
