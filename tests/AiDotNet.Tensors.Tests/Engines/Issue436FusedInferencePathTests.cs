using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #436: the single-dispatch inference paths. The MLP runs row stripes through every layer, MHA runs whole batch
/// elements per worker, and the LSTM fan-out projects each chunk's own rows. Each is checked against the path it
/// replaces (or a float64 reference) at shapes that split unevenly, with masks, biases and initial states, and for
/// independence from the thread count.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class Issue436FusedInferencePathTests
{
    private static Tensor<float> Rand(Random rng, float scale, params int[] shape)
    {
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    private static double MaxRelative(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0, big = 1e-30;
        for (int i = 0; i < expected.Length; i++)
        {
            big = Math.Max(big, Math.Abs(expected[i]));
            worst = Math.Max(worst, Math.Abs((double)actual[i] - expected[i]));
        }
        return worst / big;
    }

    private static T WithDop<T>(int dop, Func<T> body)
    {
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = dop;
        try { return body(); }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    [Theory]
    [InlineData(128, new[] { 784, 512, 128, 10 }, true, false)]
    [InlineData(37, new[] { 33, 50, 17 }, true, true)]     // uneven stripes, column tails (17, 50 = 3x16 + 2)
    [InlineData(9, new[] { 8, 24 }, false, true)]          // one layer, ReLU output, two stripes
    [InlineData(64, new[] { 16, 40, 40, 3 }, true, false)]
    public void MlpRowStripes_MatchThePerLayerPath(int rows, int[] dims, bool withBiases, bool reluOutput)
    {
        var rng = new Random(rows * 31 + dims.Length);
        var engine = new CpuEngine();
        var x = Rand(rng, 1f, rows, dims[0]);
        var weights = new List<Tensor<float>>();
        var biases = new List<Tensor<float>?>();
        for (int i = 0; i + 1 < dims.Length; i++)
        {
            weights.Add(Rand(rng, 1f / (float)Math.Sqrt(dims[i]), dims[i], dims[i + 1]));
            biases.Add(withBiases && i % 2 == 0 ? Rand(rng, 0.2f, dims[i + 1]) : null);
        }
        var outputActivation = reluOutput ? FusedActivationType.ReLU : FusedActivationType.None;
        bool saved = CpuEngine.MlpRowStripesEnabled;
        try
        {
            CpuEngine.MlpRowStripesEnabled = false;
            float[] expected = engine.MlpForward(x, weights, biases, FusedActivationType.ReLU, outputActivation).ToArray();
            CpuEngine.MlpRowStripesEnabled = true;
            foreach (int dop in new[] { 1, 3, 16, Environment.ProcessorCount })
            {
                float[] actual = WithDop(dop, () =>
                    engine.MlpForward(x, weights, biases, FusedActivationType.ReLU, outputActivation).ToArray());
                Assert.True(MaxRelative(expected, actual) <= 1e-5, $"dop {dop}: {MaxRelative(expected, actual):G3}");
            }
        }
        finally
        {
            CpuEngine.MlpRowStripesEnabled = saved;
        }
    }

    [Fact]
    public void MlpRowStripes_LeaveOtherActivationsToThePerLayerPath()
    {
        var rng = new Random(7);
        var engine = new CpuEngine();
        var x = Rand(rng, 1f, 32, 12);
        var w = new List<Tensor<float>> { Rand(rng, 0.3f, 12, 20), Rand(rng, 0.3f, 20, 5) };
        var b = new List<Tensor<float>?> { Rand(rng, 0.1f, 20), null };
        bool saved = CpuEngine.MlpRowStripesEnabled;
        try
        {
            CpuEngine.MlpRowStripesEnabled = false;
            float[] expected = engine.MlpForward(x, w, b, FusedActivationType.GELU, FusedActivationType.None).ToArray();
            CpuEngine.MlpRowStripesEnabled = true;
            float[] actual = engine.MlpForward(x, w, b, FusedActivationType.GELU, FusedActivationType.None).ToArray();
            Assert.Equal(expected, actual);
        }
        finally
        {
            CpuEngine.MlpRowStripesEnabled = saved;
        }
    }

    [Theory]
    [InlineData(128, 32, 64, 4, false)]
    [InlineData(3, 5, 12, 3, false)]
    [InlineData(7, 9, 24, 2, true)]
    [InlineData(2, 16, 32, 8, true)]
    public void MhaBatchFused_MatchesTheThreePhasePath(int batch, int seq, int dModel, int heads, bool masked)
    {
        var rng = new Random(batch * 97 + seq);
        var engine = new CpuEngine();
        var x = Rand(rng, 1f, batch, seq, dModel);
        var q = Rand(rng, 0.3f, dModel, dModel);
        var k = Rand(rng, 0.3f, dModel, dModel);
        var v = Rand(rng, 0.3f, dModel, dModel);
        var o = Rand(rng, 0.3f, dModel, dModel);
        Tensor<bool>? mask = null;
        if (masked)
        {
            // Per-batch padding [batch, 1, seq, seq]; the last batch element masks every key of its first query row.
            var data = new bool[batch * seq * seq];
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < seq; i++)
                    for (int j = 0; j < seq; j++)
                        data[(b * seq + i) * seq + j] = j < seq - (b % 3) && !(b == batch - 1 && i == 0);
            mask = new Tensor<bool>(data, new[] { batch, 1, seq, seq });
        }
        bool saved = CpuEngine.MhaBatchFusedEnabled;
        try
        {
            CpuEngine.MhaBatchFusedEnabled = false;
            float[] expected = engine.MultiHeadAttentionForward(x, q, k, v, o, heads, mask).ToArray();
            CpuEngine.MhaBatchFusedEnabled = true;
            foreach (int dop in new[] { 1, 2, Environment.ProcessorCount })
            {
                float[] actual = WithDop(dop, () => engine.MultiHeadAttentionForward(x, q, k, v, o, heads, mask).ToArray());
                Assert.True(MaxRelative(expected, actual) <= 1e-5, $"dop {dop}: {MaxRelative(expected, actual):G3}");
            }
        }
        finally
        {
            CpuEngine.MhaBatchFusedEnabled = saved;
        }
    }

    // float64 LSTM, PyTorch gate order i, f, g, o.
    private static (double[] sequence, double[] lastH, double[] lastC) LstmReference(
        float[] x, float[] wIh, float[] wHh, float[]? bIh, float[]? bHh, float[]? h0, float[]? c0,
        int batch, int seq, int inF, int hidden)
    {
        var seqOut = new double[batch * seq * hidden];
        var lastH = new double[batch * hidden];
        var lastC = new double[batch * hidden];
        static double Sig(double z) => 1 / (1 + Math.Exp(-z));
        for (int b = 0; b < batch; b++)
        {
            var h = new double[hidden];
            var c = new double[hidden];
            for (int j = 0; j < hidden; j++) { h[j] = h0?[b * hidden + j] ?? 0; c[j] = c0?[b * hidden + j] ?? 0; }
            for (int t = 0; t < seq; t++)
            {
                var gates = new double[4 * hidden];
                for (int g = 0; g < 4 * hidden; g++)
                {
                    double s = (bIh?[g] ?? 0) + (bHh?[g] ?? 0);
                    for (int f = 0; f < inF; f++) s += x[(b * seq + t) * inF + f] * (double)wIh[g * inF + f];
                    for (int f = 0; f < hidden; f++) s += h[f] * wHh[g * hidden + f];
                    gates[g] = s;
                }
                for (int j = 0; j < hidden; j++)
                {
                    double i = Sig(gates[j]), fg = Sig(gates[hidden + j]), gg = Math.Tanh(gates[2 * hidden + j]), o = Sig(gates[3 * hidden + j]);
                    c[j] = fg * c[j] + i * gg;
                    h[j] = o * Math.Tanh(c[j]);
                    seqOut[(b * seq + t) * hidden + j] = h[j];
                }
            }
            Array.Copy(h, 0, lastH, b * hidden, hidden);
            Array.Copy(c, 0, lastC, b * hidden, hidden);
        }
        return (seqOut, lastH, lastC);
    }

    private static void AssertClose(double[] expected, float[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
        Assert.True(worst <= tolerance, $"{what}: max abs difference {worst:G3}");
    }

    [Theory]
    [InlineData(128, 32, 32, 64, false, false)]
    [InlineData(33, 7, 5, 16, true, true)]     // uneven chunks, odd input width, biases and initial states
    [InlineData(17, 4, 3, 12, true, false)]    // hidden not a multiple of 8: the scalar cell
    public void LstmPerChunkProjection_MatchesTheReference(int batch, int seq, int inF, int hidden, bool biasAndState, bool returnSequences)
    {
        var rng = new Random(batch * 13 + hidden);
        var engine = new CpuEngine();
        var x = Rand(rng, 1f, batch, seq, inF);
        var wIh = Rand(rng, 0.4f, 4 * hidden, inF);
        var wHh = Rand(rng, 0.4f, 4 * hidden, hidden);
        var bIh = biasAndState ? Rand(rng, 0.2f, 4 * hidden) : null;
        var bHh = biasAndState ? Rand(rng, 0.2f, 4 * hidden) : null;
        var h0 = biasAndState ? Rand(rng, 0.5f, batch, hidden) : null;
        var c0 = biasAndState ? Rand(rng, 0.5f, batch, hidden) : null;
        var (seqRef, hRef, cRef) = LstmReference(x.ToArray(), wIh.ToArray(), wHh.ToArray(), bIh?.ToArray(), bHh?.ToArray(),
            h0?.ToArray(), c0?.ToArray(), batch, seq, inF, hidden);

        float[]? first = null;
        foreach (int dop in new[] { 1, 4, Environment.ProcessorCount })
        {
            var (output, finalH, finalC) = WithDop(dop, () =>
            {
                var y = engine.LstmSequenceForward(x, h0, c0, wIh, wHh, bIh, bHh, out var fh, out var fc,
                    returnSequences);
                return (y.ToArray(), fh.ToArray(), fc.ToArray());
            });
            AssertClose(returnSequences ? seqRef : hRef, output, 2e-5, $"output, dop {dop}");
            AssertClose(hRef, finalH, 2e-5, $"final hidden, dop {dop}");
            AssertClose(cRef, finalC, 2e-5, $"final cell, dop {dop}");
            // The chunk split depends on the thread count; each row's arithmetic must not.
            if (first is null) first = output;
            else Assert.Equal(first, output);
        }
    }
}
