// Copyright (c) AiDotNet. All rights reserved.
// Gather and deterministic scatter (#890 phase 4e): Embedding / Gather run the declared lookup
// (CodegenBackendPrograms.Embedding), and the deterministic EmbeddingBackward and ScatterAdd run the sorted kernels
// (CudaSortedScatterKernels). A gather copies and the sorted scatter makes the same additions in the same order, so
// both are checked bit for bit against the hand-written kernels, and against a reference written out here.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

[Collection("DirectGpuSerial")]
public class DeclaredGatherRouteTests
{
    private static CudaBackend OpenBackendOrSkip()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("CUDA driver was detected, but the CUDA backend failed to initialize.");
        }
        Skip.IfNot(backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ codegen conveyor.");
        return backend;
    }

    private static float[] Fill(int count, int salt)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((((i * 37 + salt * 101) % 97) - 48) / 97.0 + 0.003 * (i % 7));
        return values;
    }

    // Hashed ids (so they repeat), with two out of range in either direction.
    private static int[] Ids(int count, int vocabulary)
    {
        var ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = (int)((i * 2654435761L) % vocabulary);
        if (count > 4) { ids[1] = -1; ids[3] = vocabulary + 2; }
        return ids;
    }

    private static void AssertBitIdentical(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected[i]) == TestHelpers.MathCompat.SingleToInt32Bits(actual[i]),
                $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    /// <summary>
    /// The declared lookup serves Embedding: every in-range row equals the table row (the hand-written kernel's result
    /// too), and an out-of-range id yields a zero row instead of reading outside the table.
    /// </summary>
    [SkippableTheory]
    [InlineData(4096, 1000, 64)]
    [InlineData(300, 77, 130)]
    [InlineData(1024, 50257, 16)]
    public void TheDeclaredEmbeddingCopiesRowsAndZeroesOutOfRangeIds(int tokens, int vocabulary, int width)
    {
        using var backend = OpenBackendOrSkip();
        int[] ids = Ids(tokens, vocabulary);
        float[] table = Fill(vocabulary * width, 1);
        using var bids = backend.AllocateIntBuffer(ids);
        using var btable = backend.AllocateBuffer(table);
        using var bout = backend.AllocateBuffer(Fill(tokens * width, 2));
        long before = backend.DeclaredProgramLaunches;
        backend.Embedding(bids, btable, bout, tokens, width);
        backend.Synchronize();
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        float[] got = backend.DownloadBuffer(bout);
        for (int t = 0; t < tokens; t++)
            for (int e = 0; e < width; e++)
            {
                bool valid = ids[t] >= 0 && ids[t] < vocabulary;
                float expected = valid ? table[ids[t] * width + e] : 0f;
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected) ==
                    TestHelpers.MathCompat.SingleToInt32Bits(got[t * width + e]), $"[{t},{e}]: {expected} vs {got[t * width + e]}");
            }
    }

    /// <summary>
    /// The deterministic EmbeddingBackward (overwrite) and ScatterAdd (accumulate onto a seed) through the sorted
    /// kernels are bit-identical to the scanning kernels and to a reference summing in increasing token order -- with
    /// repeated and out-of-range ids, and vocabularies needing 1 to 4 radix passes.
    /// </summary>
    [SkippableTheory]
    [InlineData(4096, 1000, 32)]
    [InlineData(900, 32000, 8)]
    [InlineData(4096, 200, 64)]
    [InlineData(20000, 15, 300)]
    [InlineData(9000, 50257, 4)]
    public void TheSortedScatterIsBitIdenticalToTheScanningKernels(int tokens, int vocabulary, int width)
    {
        using var backend = OpenBackendOrSkip();
        bool previous = AiDotNetEngine.DeterministicMode;
        AiDotNetEngine.SetDeterministicMode(true);
        try
        {
            int[] ids = Ids(tokens, vocabulary);
            float[] grad = Fill(tokens * width, 3), seed = Fill(vocabulary, 4);
            float[] source = Fill(tokens, 5);
            // The reference: increasing token order, fp32, as the scanning kernels add.
            var embedding = new float[vocabulary * width];
            var scatter = (float[])seed.Clone();
            var sums = new float[vocabulary];
            for (int t = 0; t < tokens; t++)
            {
                if (ids[t] < 0 || ids[t] >= vocabulary) continue;
                for (int e = 0; e < width; e++) embedding[ids[t] * width + e] += grad[t * width + e];
                sums[ids[t]] += source[t];
            }
            for (int v = 0; v < vocabulary; v++) scatter[v] += sums[v];

            var results = new float[2][][];
            foreach (bool sorted in new[] { false, true })
            {
                backend.SortedScatterSwitch = sorted;
                try
                {
                    using var bids = backend.AllocateIntBuffer(ids);
                    using var bgrad = backend.AllocateBuffer(grad);
                    using var bsource = backend.AllocateBuffer(source);
                    using var btable = backend.AllocateBuffer(Fill(vocabulary * width, 6));
                    using var bseed = backend.AllocateBuffer(seed);
                    long before = backend.SortedScatterLaunches;
                    backend.EmbeddingBackward(bgrad, bids, btable, tokens, width, vocabulary);
                    backend.ScatterAdd(bsource, bids, bseed, tokens, vocabulary);
                    backend.Synchronize();
                    if (sorted) Assert.Equal(before + 2, backend.SortedScatterLaunches);
                    else Assert.Equal(before, backend.SortedScatterLaunches);
                    results[sorted ? 1 : 0] = new[] { backend.DownloadBuffer(btable), backend.DownloadBuffer(bseed) };
                }
                finally
                {
                    backend.SortedScatterSwitch = null;
                }
            }
            AssertBitIdentical(embedding, results[1][0], "sorted embedding gradient vs reference");
            AssertBitIdentical(scatter, results[1][1], "sorted scatter vs reference");
            AssertBitIdentical(results[0][0], results[1][0], "sorted vs scanning embedding gradient");
            AssertBitIdentical(results[0][1], results[1][1], "sorted vs scanning scatter");
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(previous);
        }
    }

    /// <summary>
    /// The graph edge scatter (target-keyed, reading each edge's source row, optionally weighted) and ScatterMean (sums
    /// and per-row counts, then the divide) through the sorted kernels are bit-identical to the scanning kernels.
    /// </summary>
    [SkippableTheory]
    [InlineData(20000, 3000, 32, true)]
    [InlineData(20000, 3000, 32, false)]
    [InlineData(6000, 40, 128, true)]
    public void TheSortedEdgeScatterAndScatterMeanAreBitIdenticalToTheScanningKernels(int edges, int nodes, int features,
        bool weighted)
    {
        using var backend = OpenBackendOrSkip();
        bool previous = AiDotNetEngine.DeterministicMode;
        AiDotNetEngine.SetDeterministicMode(true);
        try
        {
            int[] targets = Ids(edges, nodes);
            var sources = new int[edges];
            for (int e = 0; e < edges; e++) sources[e] = (int)((e * 40503L + 7) % nodes);
            float[] input = Fill(nodes * features, 7), edgeValues = Fill(edges, 8), rows = Fill(edges * features, 9);
            var results = new float[2][][];
            foreach (bool sorted in new[] { false, true })
            {
                backend.SortedScatterSwitch = sorted;
                try
                {
                    using var bsrc = backend.AllocateIntBuffer(sources);
                    using var btgt = backend.AllocateIntBuffer(targets);
                    using var binput = backend.AllocateBuffer(input);
                    using var bweights = backend.AllocateBuffer(edgeValues);
                    using var bout = backend.AllocateBuffer(Fill(nodes * features, 10));
                    using var brows = backend.AllocateBuffer(rows);
                    using var bmean = backend.AllocateBuffer(nodes * features);
                    using var bcounts = backend.AllocateBuffer(nodes);
                    long before = backend.SortedScatterLaunches;
                    backend.ScatterAddEdges(binput, bsrc, btgt, weighted ? bweights : null, bout, nodes, edges, features);
                    backend.ScatterMean(brows, btgt, bmean, bcounts, edges * features, nodes, features);
                    backend.Synchronize();
                    Assert.Equal(sorted ? before + 2 : before, backend.SortedScatterLaunches);
                    results[sorted ? 1 : 0] = new[] { backend.DownloadBuffer(bout), backend.DownloadBuffer(bmean), backend.DownloadBuffer(bcounts) };
                }
                finally
                {
                    backend.SortedScatterSwitch = null;
                }
            }
            AssertBitIdentical(results[0][0], results[1][0], "edge scatter");
            AssertBitIdentical(results[0][1], results[1][1], "scatter mean");
            AssertBitIdentical(results[0][2], results[1][2], "scatter mean counts");
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(previous);
        }
    }

    /// <summary>
    /// grid_sample's deterministic input gradient through the sorted scatter is bit-identical to the scanning kernel,
    /// with zero and border padding, both corner alignments, and a grid that samples outside the image.
    /// </summary>
    [SkippableTheory]
    [InlineData(2, 3, 40, 48, 36, 44, 0, false)]
    [InlineData(2, 3, 40, 48, 36, 44, 1, true)]
    [InlineData(1, 8, 32, 32, 40, 40, 1, false)]
    public void TheSortedGridSampleGradientIsBitIdenticalToTheScanningKernel(int batch, int channels, int inHeight,
        int inWidth, int outHeight, int outWidth, int paddingMode, bool alignCorners)
    {
        using var backend = OpenBackendOrSkip();
        bool previous = AiDotNetEngine.DeterministicMode;
        AiDotNetEngine.SetDeterministicMode(true);
        try
        {
            var grid = new float[batch * outHeight * outWidth * 2];
            // Coordinates in [-1.2, 1.2): some samples fall outside the image.
            for (int i = 0; i < grid.Length; i++) grid[i] = (float)(((i * 7919L) % 1000) / 1000.0 * 2.4 - 1.2);
            float[] gradOut = Fill(batch * channels * outHeight * outWidth, 11), input = Fill(batch * channels * inHeight * inWidth, 12);
            var results = new float[2][];
            foreach (bool sorted in new[] { false, true })
            {
                backend.SortedScatterSwitch = sorted;
                try
                {
                    using var bgo = backend.AllocateBuffer(gradOut);
                    using var binput = backend.AllocateBuffer(input);
                    using var bgrid = backend.AllocateBuffer(grid);
                    using var bgi = backend.AllocateBuffer(Fill(input.Length, 13));
                    using var bgg = backend.AllocateBuffer(grid.Length);
                    long before = backend.SortedScatterLaunches;
                    backend.GridSampleBackward(bgo, binput, bgrid, bgi, bgg, batch, channels, inHeight, inWidth, outHeight,
                        outWidth, paddingMode, alignCorners);
                    backend.Synchronize();
                    Assert.Equal(sorted ? before + 1 : before, backend.SortedScatterLaunches);
                    results[sorted ? 1 : 0] = backend.DownloadBuffer(bgi);
                }
                finally
                {
                    backend.SortedScatterSwitch = null;
                }
            }
            AssertBitIdentical(results[0], results[1], "grid sample input gradient");
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(previous);
        }
    }

    /// <summary>
    /// The sorted scatter captures into a graph once its scratch exists, and a replay is bit-identical to the eager
    /// call; first met inside a capture (no scratch yet), it leaves the call to the scanning kernel.
    /// </summary>
    [SkippableFact]
    public void TheSortedScatterCapturesAndReplaysBitIdentically()
    {
        using var backend = OpenBackendOrSkip();
        bool previous = AiDotNetEngine.DeterministicMode;
        AiDotNetEngine.SetDeterministicMode(true);
        try
        {
            const int tokens = 2048, vocabulary = 5000, width = 16;
            using var bids = backend.AllocateIntBuffer(Ids(tokens, vocabulary));
            using var bgrad = backend.AllocateBuffer(Fill(tokens * width, 3));
            using var btable = backend.AllocateBuffer(vocabulary * width);
            void Call() => backend.EmbeddingBackward(bgrad, bids, btable, tokens, width, vocabulary);

            long before = backend.SortedScatterLaunches;
            IntPtr cold = backend.CaptureGraph(Call);
            backend.DestroyCapturedGraph(cold);
            Assert.Equal(before, backend.SortedScatterLaunches);

            Call();
            backend.Synchronize();
            Assert.Equal(before + 1, backend.SortedScatterLaunches);
            float[] eager = backend.DownloadBuffer(btable);
            backend.Fill(btable, float.NaN, vocabulary * width);
            IntPtr graph = backend.CaptureGraph(Call);
            try
            {
                backend.LaunchCapturedGraph(graph);
                backend.Synchronize();
                AssertBitIdentical(eager, backend.DownloadBuffer(btable), "replay");
            }
            finally
            {
                backend.DestroyCapturedGraph(graph);
            }
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(previous);
        }
    }

    /// <summary>An embedding lookup then a second lookup of its rows, as one resident launch, bit-identical.</summary>
    [SkippableFact]
    public void EmbeddingStagesRunAsOneResidentLaunchBitIdenticalToSeparateLaunches()
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.SupportsResidentProgram, "A declared resident program needs a cooperative launch.");
        const int tokens = 512, vocabulary = 900, width = 64;
        using var barrier = backend.AllocateResidentBarrier();
        var results = new float[2][][];
        for (int mode = 0; mode < 2; mode++)
        {
            using var bids = backend.AllocateIntBuffer(Ids(tokens, vocabulary));
            using var table = backend.AllocateBuffer(Fill(vocabulary * width, 1));
            using var first = backend.AllocateBuffer(tokens * width);
            using var second = backend.AllocateBuffer(tokens * width);
            var stages = new[]
            {
                backend.EmbeddingStage(bids, table, first, tokens, width),
                // The first lookup's output as a table of `tokens` rows, read by the same ids (those past it give zero rows).
                backend.EmbeddingStage(bids, first, second, tokens, width),
            };
            long before = backend.DeclaredProgramLaunches;
            if (mode == 0) backend.RunDeclaredStagesSeparately(stages);
            else Assert.True(backend.RunDeclaredResidentProgram(stages, barrier) >= 1);
            backend.Synchronize();
            Assert.Equal(mode == 0 ? 2 : 1, backend.DeclaredProgramLaunches - before);
            results[mode] = new[] { backend.DownloadBuffer(first), backend.DownloadBuffer(second) };
        }
        AssertBitIdentical(results[0][0], results[1][0], "first");
        AssertBitIdentical(results[0][1], results[1][1], "second");
    }
}
