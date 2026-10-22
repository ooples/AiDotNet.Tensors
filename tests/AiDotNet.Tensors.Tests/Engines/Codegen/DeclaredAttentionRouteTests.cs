// Copyright (c) AiDotNet. All rights reserved.
// The declared attention route (#890 phase 4d): CudaBackend.ScaledDotProductAttention runs a declared program -- the
// scores as an implicit GEMM with the scale and mask bias in its epilogue, a row softmax, the weighted values as a
// second implicit GEMM -- in the hand-written kernel's contract. Both routes are checked against an fp64 reference
// written out here; the program under graph capture; and the call as a resident-program stage.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

[Collection("DirectGpuSerial")]
public class DeclaredAttentionRouteTests
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

    // Every 5th key masked, and query row 3 masked throughout (a row with nothing to attend to); NaN keeps its key, as
    // in the hand-written kernel's `mask == 0` test.
    private static float[] Mask(int count, int seqQ, int seqK)
    {
        var mask = new float[count];
        for (int i = 0; i < count; i++)
            mask[i] = i % 5 == 0 || (i / seqK) % seqQ == 3 ? 0f : i % 11 == 0 ? float.NaN : 1f;
        return mask;
    }

    // The hand-written kernel's contract in fp64.
    private static (double[] Output, double[] Weights) Reference(float[] q, float[] k, float[] v, float[]? mask,
        bool fullMask, int batch, int heads, int kvHeads, int seqQ, int seqK, int headDim, float scale, bool causal)
    {
        int group = heads / kvHeads;
        var weights = new double[batch * heads * seqQ * seqK];
        var output = new double[batch * heads * seqQ * headDim];
        for (int row = 0; row < batch * heads * seqQ; row++)
        {
            int qi = row % seqQ, bh = row / seqQ, b = bh / heads, h = bh % heads, kvh = b * kvHeads + h / group;
            var logit = new double[seqK];
            var keep = new bool[seqK];
            double max = double.NegativeInfinity;
            for (int ki = 0; ki < seqK; ki++)
            {
                int mi = (fullMask ? bh * seqQ * seqK : 0) + qi * seqK + ki;
                keep[ki] = !(causal && ki > qi + seqK - seqQ) && (mask is null || mask[mi] != 0f);
                if (!keep[ki]) continue;
                double s = 0;
                for (int e = 0; e < headDim; e++) s += (double)q[row * headDim + e] * k[(kvh * seqK + ki) * headDim + e];
                logit[ki] = s * scale;
                max = Math.Max(max, logit[ki]);
            }
            double sum = 0;
            for (int ki = 0; ki < seqK; ki++) if (keep[ki]) sum += Math.Exp(logit[ki] - max);
            for (int ki = 0; ki < seqK; ki++)
                weights[row * seqK + ki] = keep[ki] && sum > 0 ? Math.Exp(logit[ki] - max) / sum : 0;
            for (int e = 0; e < headDim; e++)
            {
                double acc = 0;
                for (int ki = 0; ki < seqK; ki++) acc += weights[row * seqK + ki] * v[(kvh * seqK + ki) * headDim + e];
                output[row * headDim + e] = acc;
            }
        }
        return (output, weights);
    }

    private static double MaxError(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]), "NaN at " + i);
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]) / Math.Max(1.0, Math.Abs(expected[i])));
        }
        return worst;
    }

    /// <summary>
    /// The declared program serves the call (one declared launch), and its output and weights are within 2e-5 of fp64
    /// or within twice the hand-written kernel's own error -- causal, shared and full masks, grouped-query heads, more
    /// queries than keys (causal rows with no key), with and without the weights stored.
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 2, 2, 16, 24, 32, 0, false, true)]
    [InlineData(1, 2, 2, 16, 24, 32, 0, true, true)]
    [InlineData(2, 4, 2, 20, 20, 16, 1, false, true)]
    [InlineData(1, 2, 2, 12, 18, 8, 2, true, false)]
    [InlineData(1, 2, 2, 9, 4, 16, 0, true, true)]
    [InlineData(1, 1, 1, 5, 7, 6, 1, false, true)]
    public void TheDeclaredAttentionMatchesTheReference(int batch, int heads, int kvHeads, int seqQ, int seqK,
        int headDim, int maskMode, bool causal, bool storeWeights)
    {
        using var backend = OpenBackendOrSkip();
        float scale = (float)(1.0 / Math.Sqrt(headDim));
        float[] q = Fill(batch * heads * seqQ * headDim, 1), k = Fill(batch * kvHeads * seqK * headDim, 2);
        float[] v = Fill(batch * kvHeads * seqK * headDim, 3);
        bool full = maskMode == 2;
        float[]? mask = maskMode == 0 ? null : Mask(full ? batch * heads * seqQ * seqK : seqQ * seqK, seqQ, seqK);
        var (expectedOut, expectedWeights) =
            Reference(q, k, v, mask, full, batch, heads, kvHeads, seqQ, seqK, headDim, scale, causal);

        var errors = new double[2, 2];
        foreach (bool declared in new[] { false, true })
        {
            backend.DeclaredKernelsSwitch = declared;
            try
            {
                using var bq = backend.AllocateBuffer(q);
                using var bk = backend.AllocateBuffer(k);
                using var bv = backend.AllocateBuffer(v);
                using var bmask = mask is null ? null : backend.AllocateBuffer(mask);
                // Pre-filled, so an element a route fails to write cannot pass as a zero.
                using var bout = backend.AllocateBuffer(Fill(expectedOut.Length, 4));
                using var bweights = storeWeights ? backend.AllocateBuffer(Fill(expectedWeights.Length, 5)) : null;
                void Call() => backend.ScaledDotProductAttention(bq, bk, bv, bout, bweights, bmask, batch, heads, seqQ, seqK,
                    headDim, scale, causal, numKVHeads: kvHeads);
                Call();   // the first call builds the program (and compiles it)
                long before = backend.DeclaredProgramLaunches;
                Call();
                backend.Synchronize();
                Assert.Equal(declared ? 1 : 0, backend.DeclaredProgramLaunches - before);
                int route = declared ? 1 : 0;
                errors[route, 0] = MaxError(expectedOut, backend.DownloadBuffer(bout));
                if (bweights is not null) errors[route, 1] = MaxError(expectedWeights, backend.DownloadBuffer(bweights));
            }
            finally
            {
                backend.DeclaredKernelsSwitch = null;
            }
        }
        for (int o = 0; o < 2; o++)
            Assert.True(errors[1, o] <= Math.Max(2e-5, 2 * errors[0, o]),
                $"{(o == 0 ? "out" : "weights")}: declared error {errors[1, o]:E3}, hand-written {errors[0, o]:E3}");
    }

    /// <summary>
    /// The backward through its declared program (one launch) against fp64 gradients computed from the same stored
    /// weights -- masked keys (zero weight) included -- within 2e-5 or twice the composite path's own error.
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 2, 16, 24, 32)]
    [InlineData(2, 3, 20, 12, 16)]
    [InlineData(1, 1, 5, 7, 6)]
    public void TheDeclaredAttentionBackwardMatchesTheReference(int batch, int heads, int seqQ, int seqK, int headDim)
    {
        using var backend = OpenBackendOrSkip();
        float scale = (float)(1.0 / Math.Sqrt(headDim));
        float[] q = Fill(batch * heads * seqQ * headDim, 1), k = Fill(batch * heads * seqK * headDim, 2);
        float[] v = Fill(batch * heads * seqK * headDim, 3), dout = Fill(batch * heads * seqQ * headDim, 6);
        var mask = Mask(seqQ * seqK, seqQ, seqK);
        var (_, p64) = Reference(q, k, v, mask, false, batch, heads, heads, seqQ, seqK, headDim, scale, false);
        var p = new float[p64.Length];
        for (int i = 0; i < p.Length; i++) p[i] = (float)p64[i];

        var dq = new double[q.Length];
        var dk = new double[k.Length];
        var dv = new double[v.Length];
        for (int bh = 0; bh < batch * heads; bh++)
            for (int qi = 0; qi < seqQ; qi++)
            {
                int row = bh * seqQ + qi;
                var dp = new double[seqK];
                double dot = 0;
                for (int ki = 0; ki < seqK; ki++)
                {
                    for (int e = 0; e < headDim; e++) dp[ki] += (double)dout[row * headDim + e] * v[(bh * seqK + ki) * headDim + e];
                    dot += p[row * seqK + ki] * dp[ki];
                }
                for (int ki = 0; ki < seqK; ki++)
                {
                    double pk = p[row * seqK + ki], ds = scale * pk * (dp[ki] - dot);
                    for (int e = 0; e < headDim; e++)
                    {
                        dv[(bh * seqK + ki) * headDim + e] += pk * dout[row * headDim + e];
                        dq[row * headDim + e] += ds * k[(bh * seqK + ki) * headDim + e];
                        dk[(bh * seqK + ki) * headDim + e] += ds * q[row * headDim + e];
                    }
                }
            }

        var errors = new double[2, 3];
        foreach (bool declared in new[] { false, true })
        {
            backend.DeclaredKernelsSwitch = declared;
            try
            {
                using var bdout = backend.AllocateBuffer(dout);
                using var bq = backend.AllocateBuffer(q);
                using var bk = backend.AllocateBuffer(k);
                using var bv = backend.AllocateBuffer(v);
                using var bp = backend.AllocateBuffer(p);
                using var bdq = backend.AllocateBuffer(Fill(dq.Length, 7));
                using var bdk = backend.AllocateBuffer(Fill(dk.Length, 8));
                using var bdv = backend.AllocateBuffer(Fill(dv.Length, 9));
                void Call() => backend.ScaledDotProductAttentionBackward(bdout, bq, bk, bv, bp, bdq, bdk, bdv, batch, heads,
                    seqQ, seqK, headDim, scale, isCausal: false);
                Call();   // compiled once the shape recurs
                long before = backend.DeclaredProgramLaunches;
                Call();
                backend.Synchronize();
                Assert.True(!declared || backend.DeclaredProgramLaunches - before == 1, "the declared program did not serve the call");
                int route = declared ? 1 : 0;
                errors[route, 0] = MaxError(dq, backend.DownloadBuffer(bdq));
                errors[route, 1] = MaxError(dk, backend.DownloadBuffer(bdk));
                errors[route, 2] = MaxError(dv, backend.DownloadBuffer(bdv));
            }
            finally
            {
                backend.DeclaredKernelsSwitch = null;
            }
        }
        string[] what = { "dq", "dk", "dv" };
        for (int o = 0; o < 3; o++)
            Assert.True(errors[1, o] <= Math.Max(2e-5, 2 * errors[0, o]),
                $"{what[o]}: declared error {errors[1, o]:E3}, composite {errors[0, o]:E3}");
    }

    /// <summary>
    /// Decoding, whose seqK grows every token, compiles a program for its first length only and keeps the hand-written
    /// kernel at the new lengths after it, instead of compiling a program per token; a length that recurs is served
    /// declared from its second call (#1529).
    /// </summary>
    [SkippableFact]
    public void ADecodeLoopCompilesOnlyItsFirstLengthAndALengthThatRecurs()
    {
        using var backend = OpenBackendOrSkip();
        const int heads = 2, dim = 16, capacity = 12;
        using var q = backend.AllocateBuffer(Fill(heads * dim, 1));
        using var k = backend.AllocateBuffer(Fill(heads * capacity * dim, 2));
        using var v = backend.AllocateBuffer(Fill(heads * capacity * dim, 3));
        using var o = backend.AllocateBuffer(heads * dim);
        long before = backend.DeclaredProgramLaunches;
        // The geometry's first length is built at once (#1529); every later length is new, so none is built.
        for (int seqK = 1; seqK <= 8; seqK++)
            backend.ScaledDotProductAttention(q, k, v, o, null, null, 1, heads, 1, seqK, dim, 0.25f, true);
        backend.Synchronize();
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        backend.ScaledDotProductAttention(q, k, v, o, null, null, 1, heads, 1, 8, dim, 0.25f, true);
        backend.Synchronize();
        Assert.Equal(before + 2, backend.DeclaredProgramLaunches);
    }

    /// <summary>
    /// #1529: a shape is declared from its first call, so two identical calls take the same route and agree bit for
    /// bit. Built only from the second sighting, the first ran the hand-written kernel and the second the program.
    /// </summary>
    [SkippableFact]
    public void ANewShapeIsDeclaredFromItsFirstCall_SoIdenticalCallsAgreeBitForBit()
    {
        using var backend = OpenBackendOrSkip();
        const int batch = 1, heads = 2, seq = 24, dim = 16;
        using var q = backend.AllocateBuffer(Fill(batch * heads * seq * dim, 1));
        using var k = backend.AllocateBuffer(Fill(batch * heads * seq * dim, 2));
        using var v = backend.AllocateBuffer(Fill(batch * heads * seq * dim, 3));
        using var o = backend.AllocateBuffer(batch * heads * seq * dim);
        float[] Call()
        {
            backend.ScaledDotProductAttention(q, k, v, o, null, null, batch, heads, seq, seq, dim, 0.25f, true);
            backend.Synchronize();
            return backend.DownloadBuffer(o);
        }
        long before = backend.DeclaredProgramLaunches;
        float[] first = Call();
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        float[] second = Call();
        Assert.Equal(before + 2, backend.DeclaredProgramLaunches);
        for (int i = 0; i < first.Length; i++)
            Assert.Equal(TestHelpers.MathCompat.SingleToInt32Bits(first[i]), TestHelpers.MathCompat.SingleToInt32Bits(second[i]));
    }

    /// <summary>
    /// FlashAttentionV2 through the declared program: the output against fp64, softmaxStats (the row log-sum-exp)
    /// within 2e-5 of the fused kernel's, and FlashAttentionBackward fed each route's forward gives the same gradients
    /// to 2e-5 -- the stats contract its backward reads.
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 2, 32, 32, false)]
    [InlineData(2, 2, 40, 16, true)]
    [InlineData(1, 1, 7, 6, true)]
    public void FlashAttentionTakesTheDeclaredProgramInItsStatsContract(int batch, int heads, int seq, int headDim,
        bool causal)
    {
        using var backend = OpenBackendOrSkip();
        float scale = (float)(1.0 / Math.Sqrt(headDim));
        int n = batch * heads * seq * headDim, rows = batch * heads * seq;
        float[] q = Fill(n, 1), k = Fill(n, 2), v = Fill(n, 3), dout = Fill(n, 4);
        var (expected, _) = Reference(q, k, v, null, false, batch, heads, heads, seq, seq, headDim, scale, causal);
        var results = new float[2][][];
        foreach (bool declared in new[] { false, true })
        {
            backend.DeclaredKernelsSwitch = declared;
            try
            {
                using var bq = backend.AllocateBuffer(q);
                using var bk = backend.AllocateBuffer(k);
                using var bv = backend.AllocateBuffer(v);
                using var bdout = backend.AllocateBuffer(dout);
                using var bo = backend.AllocateBuffer(Fill(n, 5));
                using var stats = backend.AllocateBuffer(Fill(rows, 6));
                using var dq = backend.AllocateBuffer(n);
                using var dk = backend.AllocateBuffer(n);
                using var dv = backend.AllocateBuffer(n);
                void Forward() => backend.FlashAttentionV2(bq, bk, bv, bo, stats, batch, heads, seq, seq, headDim, scale, causal);
                Forward();   // compiled once the shape recurs
                long before = backend.DeclaredProgramLaunches;
                Forward();
                backend.Synchronize();
                Assert.Equal(declared ? 1 : 0, backend.DeclaredProgramLaunches - before);
                backend.DeclaredKernelsSwitch = false;   // the same backward for both forwards
                backend.FlashAttentionBackward(bdout, bq, bk, bv, bo, stats, dq, dk, dv, batch, heads, seq, seq, headDim,
                    scale, causal);
                backend.Synchronize();
                results[declared ? 1 : 0] = new[]
                {
                    backend.DownloadBuffer(bo), backend.DownloadBuffer(stats), backend.DownloadBuffer(dq),
                    backend.DownloadBuffer(dk), backend.DownloadBuffer(dv),
                };
            }
            finally
            {
                backend.DeclaredKernelsSwitch = null;
            }
        }
        Assert.True(MaxError(expected, results[1][0]) <= 2e-5, "declared output");
        string[] what = { "out", "stats", "dq", "dk", "dv" };
        for (int o = 1; o < what.Length; o++)
        {
            var reference = new double[results[0][o].Length];
            for (int i = 0; i < reference.Length; i++) reference[i] = results[0][o][i];
            Assert.True(MaxError(reference, results[1][o]) <= 2e-5, what[o] + ": declared vs fused");
        }
    }

    /// <summary>
    /// FlashAttentionBackward through its declared program (weights recomputed from softmaxStats) gives the hand-written
    /// kernel's gradients to 2e-5 from the same forward, causal and not.
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 2, 32, 32, false)]
    [InlineData(2, 2, 40, 16, true)]
    [InlineData(1, 1, 7, 6, true)]
    public void FlashAttentionBackwardTakesTheDeclaredProgram(int batch, int heads, int seq, int headDim, bool causal)
    {
        using var backend = OpenBackendOrSkip();
        float scale = (float)(1.0 / Math.Sqrt(headDim));
        int n = batch * heads * seq * headDim, rows = batch * heads * seq;
        using var bq = backend.AllocateBuffer(Fill(n, 1));
        using var bk = backend.AllocateBuffer(Fill(n, 2));
        using var bv = backend.AllocateBuffer(Fill(n, 3));
        using var bdout = backend.AllocateBuffer(Fill(n, 4));
        using var bo = backend.AllocateBuffer(n);
        using var stats = backend.AllocateBuffer(rows);
        backend.DeclaredKernelsSwitch = false;   // one forward for both backwards
        backend.FlashAttentionV2(bq, bk, bv, bo, stats, batch, heads, seq, seq, headDim, scale, causal);
        backend.DeclaredKernelsSwitch = null;
        var results = new float[2][][];
        foreach (bool declared in new[] { false, true })
        {
            backend.DeclaredKernelsSwitch = declared;
            try
            {
                using var dq = backend.AllocateBuffer(Fill(n, 5));
                using var dk = backend.AllocateBuffer(Fill(n, 6));
                using var dv = backend.AllocateBuffer(Fill(n, 7));
                void Call() => backend.FlashAttentionBackward(bdout, bq, bk, bv, bo, stats, dq, dk, dv, batch, heads, seq,
                    seq, headDim, scale, causal);
                Call();   // compiled once the shape recurs
                long before = backend.DeclaredProgramLaunches;
                Call();
                backend.Synchronize();
                Assert.Equal(declared ? 1 : 0, backend.DeclaredProgramLaunches - before);
                results[declared ? 1 : 0] = new[] { backend.DownloadBuffer(dq), backend.DownloadBuffer(dk), backend.DownloadBuffer(dv) };
            }
            finally
            {
                backend.DeclaredKernelsSwitch = null;
            }
        }
        string[] what = { "dq", "dk", "dv" };
        for (int o = 0; o < 3; o++)
        {
            var reference = new double[results[0][o].Length];
            for (int i = 0; i < reference.Length; i++) reference[i] = results[0][o][i];
            Assert.True(MaxError(reference, results[1][o]) <= 2e-5, what[o] + ": declared vs hand-written");
        }
    }

    /// <summary>A soft-capped call keeps the hand-written kernel (its bias would have to follow the tanh cap).</summary>
    [SkippableFact]
    public void ASoftCappedCallKeepsTheHandWrittenKernel()
    {
        using var backend = OpenBackendOrSkip();
        const int heads = 2, seq = 8, dim = 8;
        using var q = backend.AllocateBuffer(Fill(heads * seq * dim, 1));
        using var k = backend.AllocateBuffer(Fill(heads * seq * dim, 2));
        using var v = backend.AllocateBuffer(Fill(heads * seq * dim, 3));
        using var o = backend.AllocateBuffer(heads * seq * dim);
        long before = backend.DeclaredProgramLaunches;
        for (int i = 0; i < 2; i++)
            backend.ScaledDotProductAttention(q, k, v, o, null, null, 1, heads, seq, seq, dim, 0.35f, false, softcap: 30f);
        backend.Synchronize();
        Assert.Equal(before, backend.DeclaredProgramLaunches);
        for (int i = 0; i < 2; i++)
            backend.ScaledDotProductAttention(q, k, v, o, null, null, 1, heads, seq, seq, dim, 0.35f, false);
        backend.Synchronize();
        Assert.Equal(before + 2, backend.DeclaredProgramLaunches);   // both uncapped calls (#1529: built at first sighting)
    }

    /// <summary>
    /// First met inside a capture, the call keeps the hand-written kernel; once built outside, the declared program
    /// captures, and the replay is bit-identical to the eager launch.
    /// </summary>
    [SkippableFact]
    public void TheDeclaredAttentionCapturesAndReplaysBitIdentically()
    {
        using var backend = OpenBackendOrSkip();
        const int batch = 1, heads = 4, seqQ = 32, seqK = 48, dim = 32;
        float scale = 0.17f;
        using var q = backend.AllocateBuffer(Fill(batch * heads * seqQ * dim, 1));
        using var k = backend.AllocateBuffer(Fill(batch * heads * seqK * dim, 2));
        using var v = backend.AllocateBuffer(Fill(batch * heads * seqK * dim, 3));
        using var mask = backend.AllocateBuffer(Mask(seqQ * seqK, seqQ, seqK));
        using var o = backend.AllocateBuffer(batch * heads * seqQ * dim);
        using var w = backend.AllocateBuffer(batch * heads * seqQ * seqK);
        void Call() => backend.ScaledDotProductAttention(q, k, v, o, w, mask, batch, heads, seqQ, seqK, dim, scale, true);

        long before = backend.DeclaredProgramLaunches;
        IntPtr unbuilt = backend.CaptureGraph(Call);
        backend.DestroyCapturedGraph(unbuilt);
        Assert.Equal(before, backend.DeclaredProgramLaunches);

        Call();   // the first sighting keeps the hand-written kernel
        before = backend.DeclaredProgramLaunches;
        Call();   // the shape recurs: built and run declared
        backend.Synchronize();
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        float[] eagerOut = backend.DownloadBuffer(o), eagerWeights = backend.DownloadBuffer(w);
        backend.Fill(o, float.NaN, batch * heads * seqQ * dim);
        backend.Fill(w, float.NaN, batch * heads * seqQ * seqK);
        IntPtr graph = backend.CaptureGraph(Call);
        try
        {
            backend.LaunchCapturedGraph(graph);
            backend.Synchronize();
            AssertBitIdentical(eagerOut, backend.DownloadBuffer(o), "out");
            AssertBitIdentical(eagerWeights, backend.DownloadBuffer(w), "weights");
        }
        finally
        {
            backend.DestroyCapturedGraph(graph);
        }
    }

    /// <summary>
    /// Two attention calls (the second attends from the first's output) as one resident program are bit-identical to
    /// the same stages launched separately, in one cooperative launch.
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 2, 8, 8, 16, false)]
    [InlineData(1, 2, 16, 24, 32, true)]
    public void AttentionStagesRunAsOneResidentLaunchBitIdenticalToSeparateLaunches(int batch, int heads, int seqQ,
        int seqK, int dim, bool causal)
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.SupportsResidentProgram, "A declared resident program needs a cooperative launch.");
        float scale = (float)(1.0 / Math.Sqrt(dim));
        float[] q = Fill(batch * heads * seqQ * dim, 1), k = Fill(batch * heads * seqK * dim, 2);
        float[] v = Fill(batch * heads * seqK * dim, 3), mask = Mask(seqQ * seqK, seqQ, seqK);
        using var barrier = backend.AllocateResidentBarrier();
        var results = new float[2][][];
        for (int mode = 0; mode < 2; mode++)
        {
            using var bq = backend.AllocateBuffer(q);
            using var bk = backend.AllocateBuffer(k);
            using var bv = backend.AllocateBuffer(v);
            using var bmask = backend.AllocateBuffer(mask);
            using var first = backend.AllocateBuffer(batch * heads * seqQ * dim);
            using var second = backend.AllocateBuffer(batch * heads * seqQ * dim);
            using var weights = backend.AllocateBuffer(batch * heads * seqQ * seqK);
            var stages = new[]
            {
                backend.AttentionStage(bq, bk, bv, first, weights, bmask, batch, heads, heads, seqQ, seqK, dim, scale, causal),
                backend.AttentionStage(first, bk, bv, second, null, null, batch, heads, heads, seqQ, seqK, dim, scale, causal),
            };
            long before = backend.DeclaredProgramLaunches;
            if (mode == 0) backend.RunDeclaredStagesSeparately(stages);
            else Assert.True(backend.RunDeclaredResidentProgram(stages, barrier) >= 1);
            backend.Synchronize();
            Assert.Equal(mode == 0 ? 2 : 1, backend.DeclaredProgramLaunches - before);
            results[mode] = new[] { backend.DownloadBuffer(first), backend.DownloadBuffer(weights), backend.DownloadBuffer(second) };
        }
        var (expected, _) = Reference(q, k, v, mask, false, batch, heads, heads, seqQ, seqK, dim, scale, causal);
        Assert.True(MaxError(expected, results[0][0]) <= 2e-5);
        for (int o = 0; o < results[0].Length; o++) AssertBitIdentical(results[0][o], results[1][o], "buffer " + o);
    }

    private static void AssertBitIdentical(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected[i]) == TestHelpers.MathCompat.SingleToInt32Bits(actual[i]),
                $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }
}
