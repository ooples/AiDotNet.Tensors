using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The fused float LSTM training kernel splits the batch into fixed chunks (8+ rows each, at most 16 chunks) and runs
/// each chunk's whole forward recurrence and BPTT on one worker, summing the chunks' partial weight and bias gradients
/// in chunk order. The other fused-LSTM tests use batches of 2-3 rows, which is a single chunk; these use batches that
/// split into several chunks -- unevenly (17 = 8 + 9), into many (40 = five of 8) and at the 16-chunk cap (136, chunks
/// of 8 and 9) -- and check the output and every gradient against the decomposed differentiable graph (the
/// state-returning overload records ordinary primitives), plus that the results do not depend on the thread count.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class LstmFusedBatchChunkTests
{
    private static Tensor<float> Rand(int[] shape, Random rng, float scale = 0.5f)
    {
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    private sealed class Case
    {
        public Case(int batch, int seq, int inF, int hidden, bool returnSequences, int seed)
        {
            var rng = new Random(seed);
            int g = 4 * hidden;
            Input = Rand(new[] { batch, seq, inF }, rng);
            WIh = Rand(new[] { g, inF }, rng);
            WHh = Rand(new[] { g, hidden }, rng);
            BIh = Rand(new[] { g }, rng);
            BHh = Rand(new[] { g }, rng);
            H0 = Rand(new[] { batch, hidden }, rng);
            C0 = Rand(new[] { batch, hidden }, rng);
            Coeff = Rand(returnSequences ? new[] { batch, seq, hidden } : new[] { batch, hidden }, rng);
            ReturnSequences = returnSequences;
        }

        public Tensor<float> Input, WIh, WHh, BIh, BHh, H0, C0, Coeff;
        public bool ReturnSequences;
        public Tensor<float>[] Sources => new[] { Input, WIh, WHh, BIh, BHh, H0, C0 };

        public (float[] output, float[][] grads) Fused(CpuEngine eng)
        {
            using var tape = new GradientTape<float>();
            var y = eng.LstmSequenceForward(Input, H0, C0, WIh, WHh, BIh, BHh, ReturnSequences);
            var grads = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, Coeff), null), Sources);
            return (y.ToArray(), Collect(grads));
        }

        public (float[] output, float[][] grads) Decomposed(CpuEngine eng)
        {
            using var tape = new GradientTape<float>();
            var y = eng.LstmSequenceForward(Input, H0, C0, WIh, WHh, BIh, BHh, out _, out _, ReturnSequences);
            var grads = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, Coeff), null), Sources);
            return (y.ToArray(), Collect(grads));
        }

        private float[][] Collect(Dictionary<Tensor<float>, Tensor<float>> grads)
        {
            var src = Sources;
            var result = new float[src.Length][];
            for (int i = 0; i < src.Length; i++) result[i] = grads[src[i]].ToArray();
            return result;
        }
    }

    private static readonly string[] Names = { "input", "wIh", "wHh", "bIh", "bHh", "h0", "c0" };

    [Theory]
    [InlineData(17, 64, true)]    // two uneven chunks (8 + 9)
    [InlineData(17, 19, false)]   // vector body + scalar tail in every cell row
    [InlineData(40, 64, false)]   // five chunks of 8
    [InlineData(136, 32, true)]   // the 16-chunk cap: chunks of 8 and 9 rows
    public void MultiChunkFused_MatchesDecomposedGraph(int batch, int hidden, bool returnSequences)
    {
        var eng = new CpuEngine();
        var c = new Case(batch, seq: 5, inF: 7, hidden, returnSequences, seed: batch * 131 + hidden);
        var (fusedOut, fused) = c.Fused(eng);
        var (refOut, reference) = c.Decomposed(eng);

        AssertClose(refOut, fusedOut, "output");
        for (int i = 0; i < Names.Length; i++)
            AssertClose(reference[i], fused[i], Names[i]);
    }

    [Fact]
    public void MultiChunkFused_IsIndependentOfThreadCount_BitExact()
    {
        var eng = new CpuEngine();
        var c = new Case(batch: 72, seq: 6, inF: 12, hidden: 24, returnSequences: true, seed: 77);
        int prior = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            var (serialOut, serial) = c.Fused(eng);
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount);
            var (parallelOut, parallel) = c.Fused(eng);

            AssertBitEqual(serialOut, parallelOut, "output");
            for (int i = 0; i < Names.Length; i++)
                AssertBitEqual(serial[i], parallel[i], Names[i]);
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }
    }

    /// <summary>
    /// With only the weights requested, the relevance filter marks the data input's gradient (a whole GEMM per chunk)
    /// as unread and the kernel skips it, along with the h0/c0 gradients and the last dh carry GEMM. The weight and
    /// bias gradients must come out bit-identical to the all-sources run.
    /// </summary>
    [Fact]
    public void MultiChunkFused_WeightsOnlyRequest_WeightGradientsBitIdentical()
    {
        var eng = new CpuEngine();
        var c = new Case(batch: 24, seq: 4, inF: 6, hidden: 16, returnSequences: false, seed: 5);
        var (_, all) = c.Fused(eng);

        using var tape = new GradientTape<float>();
        var y = eng.LstmSequenceForward(c.Input, c.H0, c.C0, c.WIh, c.WHh, c.BIh, c.BHh, c.ReturnSequences);
        var weights = new[] { c.WIh, c.WHh, c.BIh, c.BHh };
        var grads = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, c.Coeff), null), weights);

        for (int i = 0; i < weights.Length; i++)
            AssertBitEqual(all[i + 1], grads[weights[i]].ToArray(), Names[i + 1]);
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            if (Math.Abs(e - a) > 2e-5f * (1f + Math.Abs(e)))
                Assert.Fail($"{what}: element {i} decomposed={e:R} fused={a:R}");
        }
    }

    private static int Bits(float v) => BitConverter.ToInt32(BitConverter.GetBytes(v), 0);

    private static void AssertBitEqual(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (Bits(expected[i]) != Bits(actual[i]))
                Assert.Fail($"{what}: element {i} expected={expected[i]:R} actual={actual[i]:R}");
    }
}
