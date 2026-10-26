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
        // Disposed before the skip: Skip throws, and the caller's `using` only takes ownership once this returns, so
        // skipping here would leak the retained primary context and the main stream for the rest of the run.
        if (!backend.DeclaredKernelsEnabled)
        {
            backend.Dispose();
            Skip.If(true, "The declared route is not released on this architecture (see AIDOTNET_DECLARED_KERNELS_ARCH).");
        }
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
    /// IndexSelect through the declared gather: every row equals the source row the index names -- bit for bit what the
    /// hand-written kernel copies (declared route off) -- and an out-of-range index yields zeros. Below the size where
    /// the declared call pays, the hand-written kernel still serves it.
    /// </summary>
    [SkippableTheory]
    [InlineData(4, 300, 512, 32, true)]
    [InlineData(1, 50, 4096, 8, true)]
    [InlineData(2, 10, 6, 5, false)]
    public void TheDeclaredIndexSelectCopiesTheIndexedRows(int outer, int sourceAxis, int indexAxis, int inner,
        bool declared)
    {
        using var backend = OpenBackendOrSkip();
        int[] indices = Ids(indexAxis, sourceAxis);
        // Below the declared size the hand-written kernel serves the call, and it reads outside the source for an
        // out-of-range index: only in-range indices have a defined result there.
        if (!declared)
            for (int j = 0; j < indices.Length; j++) indices[j] = ((indices[j] % sourceAxis) + sourceAxis) % sourceAxis;
        float[] source = Fill(outer * sourceAxis * inner, 3);
        using var bindices = backend.AllocateIntBuffer(indices);
        using var bsource = backend.AllocateBuffer(source);
        using var bout = backend.AllocateBuffer(Fill(outer * indexAxis * inner, 4));
        long before = backend.DeclaredProgramLaunches;
        backend.IndexSelect(bsource, bindices, bout, outer, sourceAxis, indexAxis, inner);
        backend.Synchronize();
        Assert.Equal(before + (declared ? 1 : 0), backend.DeclaredProgramLaunches);
        float[] got = backend.DownloadBuffer(bout);

        // The hand-written kernel on the in-range indices (it reads outside the source for the others).
        var inRange = (int[])indices.Clone();
        for (int j = 0; j < inRange.Length; j++) if (inRange[j] < 0 || inRange[j] >= sourceAxis) inRange[j] = 0;
        using var bInRange = backend.AllocateIntBuffer(inRange);
        using var bHand = backend.AllocateBuffer(outer * indexAxis * inner);
        backend.DeclaredKernelsSwitch = false;
        try
        {
            backend.IndexSelect(bsource, bInRange, bHand, outer, sourceAxis, indexAxis, inner);
            backend.Synchronize();
        }
        finally { backend.DeclaredKernelsSwitch = null; }
        float[] hand = backend.DownloadBuffer(bHand);

        for (int o = 0; o < outer; o++)
            for (int j = 0; j < indexAxis; j++)
            {
                bool valid = indices[j] >= 0 && indices[j] < sourceAxis;
                for (int i = 0; i < inner; i++)
                {
                    int at = (o * indexAxis + j) * inner + i;
                    float expected = valid ? source[(o * sourceAxis + indices[j]) * inner + i] : 0f;
                    Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected) ==
                        TestHelpers.MathCompat.SingleToInt32Bits(got[at]), $"[{o},{j},{i}]: {expected} vs {got[at]}");
                    if (valid)
                        Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(hand[at]) ==
                            TestHelpers.MathCompat.SingleToInt32Bits(got[at]), $"hand-written [{o},{j},{i}]: {hand[at]} vs {got[at]}");
                }
            }
    }

    /// <summary>
    /// TakeAlongDim through the declared gather, whose float indices are converted as the hand-written kernel's
    /// <c>(int)</c>: bit-identical to that kernel for every index -- fractional (truncated toward zero, so -0.5 is 0),
    /// negative, past the axis, and NaN (0) -- and to a reference written out here.
    /// </summary>
    [SkippableTheory]
    [InlineData(8, 64, 64, 50, true)]
    [InlineData(2, 3000, 4, 700, true)]
    [InlineData(3, 4, 5, 6, false)]
    public void TheDeclaredTakeAlongDimMatchesTheHandWrittenKernel(int outer, int axisOut, int inner, int axisIn,
        bool declared)
    {
        using var backend = OpenBackendOrSkip();
        int count = outer * axisOut * inner;
        var indices = new float[count];
        for (int i = 0; i < count; i++) indices[i] = (float)((i * 2654435761L) % (axisIn + 4) - 2) + (i % 3) * 0.5f;
        indices[0] = float.NaN;
        indices[1] = -0.5f;
        indices[2] = axisIn - 0.25f;
        float[] input = Fill(outer * axisIn * inner, 5);
        using var bindices = backend.AllocateBuffer(indices);
        using var binput = backend.AllocateBuffer(input);
        using var bout = backend.AllocateBuffer(Fill(count, 6));
        using var bhand = backend.AllocateBuffer(count);
        long before = backend.DeclaredProgramLaunches;
        backend.TakeAlongDim(binput, bindices, bout, outer, axisOut, inner, axisIn);
        backend.Synchronize();
        Assert.Equal(before + (declared ? 1 : 0), backend.DeclaredProgramLaunches);
        backend.DeclaredKernelsSwitch = false;
        try
        {
            backend.TakeAlongDim(binput, bindices, bhand, outer, axisOut, inner, axisIn);
            backend.Synchronize();
        }
        finally { backend.DeclaredKernelsSwitch = null; }
        float[] got = backend.DownloadBuffer(bout), hand = backend.DownloadBuffer(bhand);
        AssertBitIdentical(hand, got, "take_along_dim vs the hand-written kernel");
        for (int at = 0; at < count; at++)
        {
            int o = at / (axisOut * inner), i = at % inner;
            int j = float.IsNaN(indices[at]) ? 0 : (int)indices[at];
            float expected = j >= 0 && j < axisIn ? input[(o * axisIn + j) * inner + i] : 0f;
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected) == TestHelpers.MathCompat.SingleToInt32Bits(got[at]),
                $"[{at}] index {indices[at]}: {expected} vs {got[at]}");
        }
    }

    /// <summary>
    /// The float-index rule in the IR itself (no GPU): the spec interprets as the device converts, and an index tensor
    /// whose element type contradicts the declaration is refused rather than read as a reinterpreted bit pattern.
    /// </summary>
    [Fact]
    public void FloatIndicesAreDeclaredAndConvertedAsTheDeviceDoes()
    {
        var spec = AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenBackendPrograms.TakeAlongDim(1, 5, 1, 4).Passes[0];
        var result = spec.Interpret(new[]
        {
            new[] { double.NaN, -0.5, 3.75, 4.0, -1.0 },
            new[] { 10.0, 20.0, 30.0, 40.0 },
        });
        Assert.Equal(new[] { 10.0, 10.0, 40.0, 0.0, 0.0 }, result);

        var position = AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenAffineExpr.Axis(0);
        var intIndices = new AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenTensorBinding(0, "indices", new[] { 3 },
            new[] { position }, elementType: AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenElementType.Int32);
        var table = new AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenTensorBinding(1, "table", new[] { 4 },
            new[] { AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenAffineExpr.Const(0) },
            indirect: new AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenIndirectIndex?[]
            {
                new AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenIndirectIndex(0, position, 4, floatIndices: true),
            });
        var output = new AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenTensorBinding(2, "out", new[] { 3 },
            new[] { position }, isOutput: true);
        Assert.Throws<ArgumentException>(() => new AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenKernelSpec("bad",
            new AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenIterationSpace(
                AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenAxis.Parallel("t", 3)),
            new[] { intIndices, table }, output, new[] { 1 },
            AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenReduceKind.None));
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
        // Below the crossover the declared scatter-add takes these calls ahead of both kernels compared here (it is
        // checked against the ordered sum in DeclaredScatterAddRouteTests); off, the sorted and scanning kernels serve.
        backend.DeclaredKernelsSwitch = false;
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

    /// <summary>An embedding lookup, a second lookup of its rows, then an IndexSelect of those, as one resident launch, bit-identical.</summary>
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
            using var third = backend.AllocateBuffer(64 * width);
            var stages = new[]
            {
                backend.EmbeddingStage(bids, table, first, tokens, width),
                // The first lookup's output as a table of `tokens` rows, read by the same ids (those past it give zero rows).
                backend.EmbeddingStage(bids, first, second, tokens, width),
                // IndexSelect of the rows the first 64 ids name, reading the second lookup's output as [1, tokens, width].
                backend.IndexSelectStage(second, bids, third, 1, tokens, 64, width),
            };
            long before = backend.DeclaredProgramLaunches;
            if (mode == 0) backend.RunDeclaredStagesSeparately(stages);
            else Assert.True(backend.RunDeclaredResidentProgram(stages, barrier) >= 1);
            backend.Synchronize();
            Assert.Equal(mode == 0 ? 3 : 1, backend.DeclaredProgramLaunches - before);
            results[mode] = new[] { backend.DownloadBuffer(first), backend.DownloadBuffer(second), backend.DownloadBuffer(third) };
        }
        AssertBitIdentical(results[0][0], results[1][0], "first");
        AssertBitIdentical(results[0][1], results[1][1], "second");
        AssertBitIdentical(results[0][2], results[1][2], "index-selected");
    }
}
