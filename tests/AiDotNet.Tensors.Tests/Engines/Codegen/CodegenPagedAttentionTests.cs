// Copyright (c) AiDotNet. All rights reserved.
// Paged attention as a declared program (#890 phase 4d). The reference here is the online-softmax loop that
// paged_attention_decode runs -- read off that kernel, not off the declared program -- so the page indirection and
// the k = page*blockSize + pos arithmetic have to agree with the kernel they replace rather than with themselves.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenPagedAttentionTests
{
    private static double[] Fill(long count, double scale, int salt)
    {
        var v = new double[count];
        for (long i = 0; i < count; i++) v[i] = scale * ((((i * 41 + salt * 97) % 89) - 44) / 44.0);
        return v;
    }

    /// <summary>paged_attention_decode's loop: one thread per head, streaming max and sum over the sequence.</summary>
    private static double[] DecodeReference(double[] q, double[] kcache, double[] vcache, int[] blockTable,
        int heads, int kvHeads, int headDim, int blockSize, int seqLen, double scale)
    {
        int group = heads / kvHeads;
        var outbuf = new double[heads * headDim];
        for (int h = 0; h < heads; h++)
        {
            var acc = new double[headDim];
            double m = double.NegativeInfinity, l = 0.0;
            for (int t = 0; t < seqLen; t++)
            {
                int blk = blockTable[t / blockSize], pos = t % blockSize;
                long at = (((long)blk * blockSize + pos) * kvHeads + h / group) * headDim;
                double dot = 0.0;
                for (int d = 0; d < headDim; d++) dot += q[h * headDim + d] * kcache[at + d];
                double logit = dot * scale;
                double next = Math.Max(m, logit);
                double correction = Math.Exp(m - next), p = Math.Exp(logit - next);
                l = l * correction + p;
                for (int d = 0; d < headDim; d++) acc[d] = acc[d] * correction + p * vcache[at + d];
                m = next;
            }
            double inverse = l > 0.0 ? 1.0 / l : 0.0;
            for (int d = 0; d < headDim; d++) outbuf[h * headDim + d] = acc[d] * inverse;
        }
        return outbuf;
    }

    /// <summary>
    /// The declared program against that loop, at a sequence that does NOT fill its last page (so the key bias has
    /// to do real work) and with the pages deliberately out of order in the block table, so a program that ignored
    /// the indirection and read the cache sequentially would disagree.
    /// </summary>
    [Theory]
    [InlineData(4, 4, 8, 4, 3, 10)]    // 3 pages of 4, sequence 10: the last page is half used
    [InlineData(6, 2, 4, 2, 4, 8)]     // grouped-query: 6 heads over 2 kv heads, sequence fills its pages
    [InlineData(2, 1, 16, 8, 2, 9)]    // one kv head, a wide head dim, one key in the last page
    public void PagedDecode_MatchesTheKernelsOnlineSoftmax(int heads, int kvHeads, int headDim, int blockSize,
        int pages, int seqLen)
    {
        const double Scale = 0.125;
        const int Blocks = 7;                       // more physical blocks than the sequence uses
        int[] blockTable = new int[pages];
        for (int i = 0; i < pages; i++) blockTable[i] = (Blocks - 1) - i * 2 >= 0 ? (Blocks - 1) - i * 2 : i;
        int seqK = pages * blockSize;
        double[] q = Fill((long)heads * headDim, 1.0, 3);
        double[] kcache = Fill((long)Blocks * blockSize * kvHeads * headDim, 1.0, 5);
        double[] vcache = Fill((long)Blocks * blockSize * kvHeads * headDim, 1.0, 7);

        var keyBias = new double[seqK];
        for (int k = 0; k < seqK; k++) keyBias[k] = k < seqLen ? 0.0 : CodegenBackendPrograms.AttentionMaskedLogit;
        var table = new double[pages];
        for (int i = 0; i < pages; i++) table[i] = blockTable[i];

        var outputs = CodegenBackendPrograms
            .PagedAttentionDecode(heads, kvHeads, headDim, blockSize, pages, Blocks, Scale)
            .Interpret(new Dictionary<string, double[]>
            {
                ["block_table"] = table, ["q"] = q, ["kcache"] = kcache, ["vcache"] = vcache, ["key_bias"] = keyBias,
            });

        double[] expected = DecodeReference(q, kcache, vcache, blockTable, heads, kvHeads, headDim, blockSize,
            seqLen, Scale);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], outputs["out"][i], 9);
    }

    /// <summary>The program's sources are the backend's five buffers, with no intermediate among them.</summary>
    [Fact]
    public void PagedDecode_TakesTheBackendsBuffers()
    {
        var names = new List<string>();
        foreach (var source in CodegenBackendPrograms.PagedAttentionDecode(4, 4, 8, 4, 3, 7, 0.125).Sources())
            names.Add(source.Name);
        names.Sort();
        Assert.Equal(new[] { "block_table", "kcache", "key_bias", "q", "vcache" }, names);
    }

    [Fact]
    public void PagedDecode_RefusesAHeadCountThatIsNotAMultipleOfTheKvHeads()
    {
        Assert.Throws<ArgumentException>(() => CodegenBackendPrograms.PagedAttentionDecode(6, 4, 8, 4, 2, 4, 0.125));
    }
}
