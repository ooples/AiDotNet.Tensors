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

        // The backend's own builder, not a copy of its rule: the route is only correct if the buffer CudaBackend
        // hands the program masks exactly the positions the program treats as outside the sequence.
        float[] built = AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.PagedKeyBias(seqK, seqLen);
        var keyBias = new double[seqK];
        for (int k = 0; k < seqK; k++) keyBias[k] = built[k];
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

    /// <summary>
    /// paged_attention_prefill's loop: one thread per (query, head), the same online softmax as decode but stopping
    /// at the query's own logical position, which is what makes the pass causal.
    /// </summary>
    private static double[] PrefillReference(double[] q, double[] kcache, double[] vcache, int[] blockTable,
        int heads, int kvHeads, int headDim, int blockSize, int numQueries, int startPos, double scale)
    {
        int group = heads / kvHeads;
        var outbuf = new double[(long)numQueries * heads * headDim];
        for (int qi = 0; qi < numQueries; qi++)
            for (int h = 0; h < heads; h++)
            {
                var acc = new double[headDim];
                double m = double.NegativeInfinity, l = 0.0;
                long qAt = ((long)qi * heads + h) * headDim;
                for (int t = 0; t <= startPos + qi; t++)
                {
                    int blk = blockTable[t / blockSize], pos = t % blockSize;
                    long at = (((long)blk * blockSize + pos) * kvHeads + h / group) * headDim;
                    double dot = 0.0;
                    for (int d = 0; d < headDim; d++) dot += q[qAt + d] * kcache[at + d];
                    double logit = dot * scale;
                    double next = Math.Max(m, logit);
                    double correction = Math.Exp(m - next), p = Math.Exp(logit - next);
                    l = l * correction + p;
                    for (int d = 0; d < headDim; d++) acc[d] = acc[d] * correction + p * vcache[at + d];
                    m = next;
                }
                double inverse = l > 0.0 ? 1.0 / l : 0.0;
                for (int d = 0; d < headDim; d++) outbuf[qAt + d] = acc[d] * inverse;
            }
        return outbuf;
    }

    /// <summary>
    /// The declared prefill against that loop. Every row has a different key count -- that is the causal rule the
    /// bias carries -- and the block table is out of order again, so neither the causality nor the indirection can
    /// be satisfied by reading the cache straight through.
    /// </summary>
    [Theory]
    [InlineData(4, 4, 8, 4, 3, 0)]     // 3 queries from position 0: rows see 1, 2 and 3 keys
    [InlineData(6, 2, 4, 2, 2, 5)]     // grouped-query, resuming at position 5 mid-page
    [InlineData(2, 1, 16, 8, 4, 3)]    // one kv head, a wide head dim, the last page part used
    public void PagedPrefill_MatchesTheKernelsCausalOnlineSoftmax(int heads, int kvHeads, int headDim, int blockSize,
        int numQueries, int startPos)
    {
        const double Scale = 0.125;
        const int Blocks = 7;
        int seqLen = startPos + numQueries;
        int pages = (seqLen + blockSize - 1) / blockSize;
        int[] blockTable = new int[pages];
        for (int i = 0; i < pages; i++) blockTable[i] = (Blocks - 1) - i * 2 >= 0 ? (Blocks - 1) - i * 2 : i;
        int length = pages * blockSize;
        double[] q = Fill((long)numQueries * heads * headDim, 1.0, 11);
        double[] kcache = Fill((long)Blocks * blockSize * kvHeads * headDim, 1.0, 5);
        double[] vcache = Fill((long)Blocks * blockSize * kvHeads * headDim, 1.0, 7);

        // Again the backend's builder, so the causal rule under test is the one the route actually uploads.
        float[] built = AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.PagedCausalBias(numQueries, length, startPos);
        var bias = new double[(long)numQueries * length];
        for (int i = 0; i < bias.Length; i++) bias[i] = built[i];
        var table = new double[pages];
        for (int i = 0; i < pages; i++) table[i] = blockTable[i];

        var outputs = CodegenBackendPrograms
            .PagedAttentionPrefill(heads, kvHeads, headDim, blockSize, pages, Blocks, numQueries, Scale)
            .Interpret(new Dictionary<string, double[]>
            {
                ["block_table"] = table, ["q"] = q, ["kcache"] = kcache, ["vcache"] = vcache, ["causal_bias"] = bias,
            });

        double[] expected = PrefillReference(q, kcache, vcache, blockTable, heads, kvHeads, headDim, blockSize,
            numQueries, startPos, Scale);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], outputs["out"][i], 9);

        // The control for the mask itself: the same program with an all-zero bias attends to every page position, so
        // it must NOT match the causal reference. Without this, a program that dropped the bias input would pass the
        // comparison above whenever causality happened to be vacuous.
        var unmasked = CodegenBackendPrograms
            .PagedAttentionPrefill(heads, kvHeads, headDim, blockSize, pages, Blocks, numQueries, Scale)
            .Interpret(new Dictionary<string, double[]>
            {
                ["block_table"] = table, ["q"] = q, ["kcache"] = kcache, ["vcache"] = vcache,
                ["causal_bias"] = new double[(long)numQueries * length],
            });
        bool differs = false;
        for (int i = 0; i < expected.Length && !differs; i++)
            differs = Math.Abs(expected[i] - unmasked["out"][i]) > 1e-9;
        Assert.True(differs, "an unmasked prefill must differ from the causal one: the bias is the causal rule");
    }

    /// <summary>The bias the backend builds is the one the program reads: same shape, same rule.</summary>
    [Fact]
    public void PagedPrefill_TakesTheBackendsBuffers()
    {
        var names = new List<string>();
        foreach (var source in CodegenBackendPrograms.PagedAttentionPrefill(4, 4, 8, 4, 3, 7, 3, 0.125).Sources())
            names.Add(source.Name);
        names.Sort();
        Assert.Equal(new[] { "block_table", "causal_bias", "kcache", "q", "vcache" }, names);
    }

    [Fact]
    public void PagedPrefill_RefusesAHeadCountThatIsNotAMultipleOfTheKvHeads()
    {
        Assert.Throws<ArgumentException>(() =>
            CodegenBackendPrograms.PagedAttentionPrefill(6, 4, 8, 4, 2, 4, 2, 0.125));
    }
}
