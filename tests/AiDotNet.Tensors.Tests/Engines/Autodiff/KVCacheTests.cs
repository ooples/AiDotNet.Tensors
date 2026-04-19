using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class KVCacheTests
{
    [Fact]
    public void AppendAndSlice_ReturnsExactlyAppendedTokens()
    {
        var cache = new KVCache<float>(maxBatch: 2, maxSeq: 8, heads: 1, headDim: 3);
        var k = TensorFromPattern(new[] { 4, 1, 3 }, i => i + 1f);
        var v = TensorFromPattern(new[] { 4, 1, 3 }, i => (i + 1f) * 10f);
        cache.Append(0, k, v);
        Assert.Equal(4, cache.GetLength(0));
        var (kSlice, vSlice) = cache.Slice(0);
        Assert.Equal(new[] { 4, 1, 3 }, kSlice._shape);
        Assert.Equal(k.AsSpan().ToArray(), kSlice.AsSpan().ToArray());
        Assert.Equal(v.AsSpan().ToArray(), vSlice.AsSpan().ToArray());
    }

    [Fact]
    public void MultipleAppends_AccumulateCorrectly()
    {
        var cache = new KVCache<float>(2, 16, 2, 4);
        var k1 = TensorFromPattern(new[] { 3, 2, 4 }, i => i);
        var k2 = TensorFromPattern(new[] { 2, 2, 4 }, i => i + 100);
        var v1 = TensorFromPattern(new[] { 3, 2, 4 }, i => -i);
        var v2 = TensorFromPattern(new[] { 2, 2, 4 }, i => -(i + 100));
        cache.Append(1, k1, v1);
        cache.Append(1, k2, v2);
        Assert.Equal(5, cache.GetLength(1));
        var (kSlice, _) = cache.Slice(1);
        Assert.Equal(new[] { 5, 2, 4 }, kSlice._shape);
        // First 3 tokens match k1, last 2 match k2.
        var data = kSlice.AsSpan().ToArray();
        for (int i = 0; i < 3 * 2 * 4; i++) Assert.Equal(k1.AsSpan()[i], data[i]);
        for (int i = 0; i < 2 * 2 * 4; i++) Assert.Equal(k2.AsSpan()[i], data[3 * 2 * 4 + i]);
    }

    [Fact]
    public void IsolatedBatchRows_DoNotCrossContaminate()
    {
        var cache = new KVCache<float>(2, 4, 1, 2);
        cache.Append(0, Ones(new[] { 2, 1, 2 }), Ones(new[] { 2, 1, 2 }));
        cache.Append(1, Fives(new[] { 3, 1, 2 }), Fives(new[] { 3, 1, 2 }));
        Assert.Equal(2, cache.GetLength(0));
        Assert.Equal(3, cache.GetLength(1));
        var (k0, _) = cache.Slice(0);
        foreach (var f in k0.AsSpan().ToArray()) Assert.Equal(1f, f);
        var (k1, _) = cache.Slice(1);
        foreach (var f in k1.AsSpan().ToArray()) Assert.Equal(5f, f);
    }

    [Fact]
    public void Overflow_Throws()
    {
        var cache = new KVCache<float>(1, 4, 1, 1);
        cache.Append(0, new Tensor<float>(new[] { 3, 1, 1 }), new Tensor<float>(new[] { 3, 1, 1 }));
        Assert.Throws<InvalidOperationException>(() =>
            cache.Append(0, new Tensor<float>(new[] { 2, 1, 1 }), new Tensor<float>(new[] { 2, 1, 1 })));
    }

    [Fact]
    public void ResetClearsLengthButKeepsBufferAllocated()
    {
        var cache = new KVCache<float>(1, 4, 1, 1);
        cache.Append(0, new Tensor<float>(new[] { 2, 1, 1 }), new Tensor<float>(new[] { 2, 1, 1 }));
        cache.Reset(0);
        Assert.Equal(0, cache.GetLength(0));
    }

    [Fact]
    public void ShapeMismatch_Throws()
    {
        var cache = new KVCache<float>(1, 4, 1, 1);
        Assert.Throws<ArgumentException>(() =>
            cache.Append(0, new Tensor<float>(new[] { 1, 2, 1 }), new Tensor<float>(new[] { 1, 2, 1 })));
    }

    private static Tensor<float> TensorFromPattern(int[] shape, Func<int, float> gen)
    {
        var t = new Tensor<float>(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = gen(i);
        return t;
    }

    private static Tensor<float> Ones(int[] shape) => TensorFromPattern(shape, _ => 1f);
    private static Tensor<float> Fives(int[] shape) => TensorFromPattern(shape, _ => 5f);
}
