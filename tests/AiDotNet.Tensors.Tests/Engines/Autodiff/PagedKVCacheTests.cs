using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class PagedKVCacheTests
{
    [Fact]
    public void Append_SingleToken_AllocatesOneBlock()
    {
        var cache = new PagedKVCache<float>(maxBlocks: 4, blockSize: 16, heads: 1, headDim: 1);
        cache.Append(42, Seq(1), Seq(1));
        Assert.Equal(1, cache.GetLength(42));
        Assert.Equal(1, cache.AllocatedBlocks);
    }

    [Fact]
    public void Append_SpansBlockBoundary_AllocatesNewBlock()
    {
        var cache = new PagedKVCache<float>(maxBlocks: 4, blockSize: 4, heads: 1, headDim: 1);
        cache.Append(1, Seq(6), Seq(6)); // 6 tokens, 4 per block → 2 blocks
        Assert.Equal(6, cache.GetLength(1));
        Assert.Equal(2, cache.AllocatedBlocks);
        Assert.Equal(2, cache.GetBlockTable(1).Count);
    }

    [Fact]
    public void Materialize_RoundTripsAppendedTokens()
    {
        var cache = new PagedKVCache<float>(maxBlocks: 8, blockSize: 4, heads: 2, headDim: 3);
        var k = SeqWithShape(5, 2, 3);
        var v = SeqWithShape(5, 2, 3, offset: 100f);
        cache.Append(7, k, v);
        var (kOut, vOut) = cache.Materialize(7);
        Assert.Equal(k.AsSpan().ToArray(), kOut.AsSpan().ToArray());
        Assert.Equal(v.AsSpan().ToArray(), vOut.AsSpan().ToArray());
    }

    [Fact]
    public void Free_ReleasesBlocksBackToPool()
    {
        var cache = new PagedKVCache<float>(maxBlocks: 2, blockSize: 2, heads: 1, headDim: 1);
        cache.Append(1, new Tensor<float>(new[] { 4, 1, 1 }), new Tensor<float>(new[] { 4, 1, 1 }));
        Assert.Equal(2, cache.AllocatedBlocks);
        cache.Free(1);
        Assert.Equal(0, cache.AllocatedBlocks);
        Assert.Equal(0, cache.GetLength(1));
    }

    [Fact]
    public void ShareBlocks_PrefixDeduplicatesStorage()
    {
        // Two sequences with the same 4-token prompt share one block.
        var cache = new PagedKVCache<float>(maxBlocks: 4, blockSize: 4, heads: 1, headDim: 1);
        cache.Append(1, Seq(4), Seq(4));
        Assert.Equal(1, cache.AllocatedBlocks);
        cache.ShareBlocks(sourceSeqId: 1, targetSeqId: 2, prefixLen: 4);
        // Still only one physical block despite two sequences.
        Assert.Equal(1, cache.AllocatedBlocks);
        Assert.Equal(4, cache.GetLength(2));
        var (k2, _) = cache.Materialize(2);
        Assert.Equal(cache.Materialize(1).K.AsSpan().ToArray(), k2.AsSpan().ToArray());
    }

    [Fact]
    public void PoolExhaustion_Throws()
    {
        var cache = new PagedKVCache<float>(maxBlocks: 1, blockSize: 2, heads: 1, headDim: 1);
        // Need 2 blocks for 3 tokens but pool has 1.
        Assert.Throws<InvalidOperationException>(() =>
            cache.Append(1, new Tensor<float>(new[] { 3, 1, 1 }), new Tensor<float>(new[] { 3, 1, 1 })));
    }

    private static Tensor<float> Seq(int n)
    {
        var t = new Tensor<float>(new[] { n, 1, 1 });
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = i + 1f;
        return t;
    }

    private static Tensor<float> SeqWithShape(int seq, int heads, int headDim, float offset = 0f)
    {
        var t = new Tensor<float>(new[] { seq, heads, headDim });
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = offset + i;
        return t;
    }
}
