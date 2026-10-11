using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #653 phase 3: LayerNorm and GroupNorm backward fan out across cores (gamma/beta accumulation over feature-column
/// ranges or groups, GroupNorm's input gradient over (batch, group)). The gate is bit-identical output at one thread
/// and at many; a double-precision reference checks the values themselves.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class NormBackwardFanOutTests
{
    private static Tensor<T> Random<T>(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<T>(shape);
        var ops = MathHelper.GetNumericOperations<T>();
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = ops.FromDouble(rng.NextDouble() * 4 - 1.5);
        return t;
    }

    private static T[] Run<T>(int dop, Func<T[]> body)
    {
        int prior = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = dop;
            return body();
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }
    }

    private static T[] Concat<T>(params Tensor<T>[] parts)
    {
        int n = 0;
        foreach (var p in parts) n += p.Length;
        var all = new T[n];
        int at = 0;
        foreach (var p in parts) { p.AsSpan().CopyTo(new Span<T>(all, at, p.Length)); at += p.Length; }
        return all;
    }

    private static void AssertBitEqual<T>(T[] serial, T[] parallel)
    {
        Assert.Equal(serial.Length, parallel.Length);
        for (int i = 0; i < serial.Length; i++)
            Assert.True(serial[i]!.Equals(parallel[i]), $"element {i}: serial {serial[i]} vs parallel {parallel[i]}");
    }

    private static void AssertClose(double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * (1 + Math.Abs(expected[i])),
                $"element {i}: expected {expected[i]}, got {actual[i]}");
    }

    private static double[] ToDouble<T>(T[] values)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var d = new double[values.Length];
        for (int i = 0; i < d.Length; i++) d[i] = ops.ToDouble(values[i]);
        return d;
    }

    // ---------------------------------------------------------------- LayerNorm

    private static T[] LayerNormBackward<T>(int rows, int features)
    {
        var engine = new CpuEngine();
        var x = Random<T>(1, rows, features);
        var gamma = Random<T>(2, features);
        var beta = Random<T>(3, features);
        var gradOut = Random<T>(4, rows, features);
        engine.LayerNorm(x, gamma, beta, 1e-5, out var mean, out var variance);
        var gradInput = engine.LayerNormBackward(gradOut, x, gamma, mean, variance, 1e-5, out var gradGamma, out var gradBeta);
        return Concat(gradInput, gradGamma, gradBeta);
    }

    // Reference in double: gradGamma[f] = sum_b go * xhat, gradBeta[f] = sum_b go,
    // gradInput = invStd / F * (F * gamma * go - sum(gamma * go) - xhat * sum(gamma * go * xhat)).
    private static double[] LayerNormBackwardReference<T>(int rows, int features)
    {
        var x = ToDouble(Random<T>(1, rows, features).ToArray());
        var gamma = ToDouble(Random<T>(2, features).ToArray());
        var go = ToDouble(Random<T>(4, rows, features).ToArray());
        var gi = new double[rows * features];
        var gg = new double[features];
        var gb = new double[features];
        for (int r = 0; r < rows; r++)
        {
            double m = 0, v = 0;
            for (int f = 0; f < features; f++) m += x[r * features + f];
            m /= features;
            for (int f = 0; f < features; f++) v += (x[r * features + f] - m) * (x[r * features + f] - m);
            v /= features;
            double invStd = 1 / Math.Sqrt(v + 1e-5);
            double sumG = 0, sumGX = 0;
            for (int f = 0; f < features; f++)
            {
                int i = r * features + f;
                double xhat = (x[i] - m) * invStd;
                gg[f] += go[i] * xhat;
                gb[f] += go[i];
                sumG += gamma[f] * go[i];
                sumGX += gamma[f] * go[i] * xhat;
            }
            for (int f = 0; f < features; f++)
            {
                int i = r * features + f;
                double xhat = (x[i] - m) * invStd;
                gi[i] = invStd / features * (features * gamma[f] * go[i] - sumG - xhat * sumGX);
            }
        }
        var all = new double[gi.Length + 2 * features];
        gi.CopyTo(all, 0); gg.CopyTo(all, gi.Length); gb.CopyTo(all, gi.Length + features);
        return all;
    }

    // 1000 features: three column ranges of 384, 384 and 232 (the last holding the SIMD tail); 4096: four of 1024.
    [Theory]
    [InlineData(600, 1000)]
    [InlineData(64, 4096)]
    public void LayerNormBackward_Float_SameAtAnyThreadCount_AndCorrect(int rows, int features)
    {
        var serial = Run(1, () => LayerNormBackward<float>(rows, features));
        var parallel = Run(Math.Max(4, Environment.ProcessorCount), () => LayerNormBackward<float>(rows, features));
        AssertBitEqual(serial, parallel);
        AssertClose(LayerNormBackwardReference<float>(rows, features), ToDouble(parallel), 2e-3);
    }

    [Theory]
    [InlineData(600, 1000)]
    [InlineData(64, 4096)]
    public void LayerNormBackward_Double_SameAtAnyThreadCount_AndCorrect(int rows, int features)
    {
        var serial = Run(1, () => LayerNormBackward<double>(rows, features));
        var parallel = Run(Math.Max(4, Environment.ProcessorCount), () => LayerNormBackward<double>(rows, features));
        AssertBitEqual(serial, parallel);
        AssertClose(LayerNormBackwardReference<double>(rows, features), ToDouble(parallel), 1e-9);
    }

    // ---------------------------------------------------------------- RMSNorm

    private static T[] RmsNormBackward<T>(int rows, int features)
    {
        var engine = new CpuEngine();
        var x = Random<T>(9, rows, features);
        var gamma = Random<T>(10, features);
        var gradOut = Random<T>(11, rows, features);
        engine.RMSNorm(x, gamma, 1e-6, out var rms);
        var gradInput = engine.RMSNormBackward(gradOut, x, gamma, rms, 1e-6, out var gradGamma);
        return Concat(gradInput, gradGamma);
    }

    // gradGamma[f] = sum_b go * x / rms; gradInput = (gamma * go - xhat * mean_f(gamma * go * xhat)) / rms.
    private static double[] RmsNormBackwardReference<T>(int rows, int features)
    {
        var x = ToDouble(Random<T>(9, rows, features).ToArray());
        var gamma = ToDouble(Random<T>(10, features).ToArray());
        var go = ToDouble(Random<T>(11, rows, features).ToArray());
        var gi = new double[rows * features];
        var gg = new double[features];
        for (int r = 0; r < rows; r++)
        {
            double ms = 0;
            for (int f = 0; f < features; f++) ms += x[r * features + f] * x[r * features + f];
            double rms = Math.Sqrt(ms / features + 1e-6);
            double corr = 0;
            for (int f = 0; f < features; f++)
            {
                int i = r * features + f;
                gg[f] += go[i] * x[i] / rms;
                corr += gamma[f] * go[i] * x[i] / rms;
            }
            corr /= features;
            for (int f = 0; f < features; f++)
            {
                int i = r * features + f;
                gi[i] = (gamma[f] * go[i] - x[i] / rms * corr) / rms;
            }
        }
        var all = new double[gi.Length + features];
        gi.CopyTo(all, 0); gg.CopyTo(all, gi.Length);
        return all;
    }

    [Theory]
    [InlineData(600, 1000)]
    [InlineData(64, 4096)]
    public void RmsNormBackward_Float_SameAtAnyThreadCount_AndCorrect(int rows, int features)
    {
        var serial = Run(1, () => RmsNormBackward<float>(rows, features));
        var parallel = Run(Math.Max(4, Environment.ProcessorCount), () => RmsNormBackward<float>(rows, features));
        AssertBitEqual(serial, parallel);
        AssertClose(RmsNormBackwardReference<float>(rows, features), ToDouble(parallel), 2e-3);
    }

    [Theory]
    [InlineData(600, 1000)]
    [InlineData(64, 4096)]
    public void RmsNormBackward_Double_SameAtAnyThreadCount_AndCorrect(int rows, int features)
    {
        var serial = Run(1, () => RmsNormBackward<double>(rows, features));
        var parallel = Run(Math.Max(4, Environment.ProcessorCount), () => RmsNormBackward<double>(rows, features));
        AssertBitEqual(serial, parallel);
        AssertClose(RmsNormBackwardReference<double>(rows, features), ToDouble(parallel), 1e-9);
    }

    // ---------------------------------------------------------------- GroupNorm

    private static T[] GroupNormBackward<T>(int batch, int channels, int side, int groups)
    {
        var engine = new CpuEngine();
        var x = Random<T>(5, batch, channels, side, side);
        var gamma = Random<T>(6, channels);
        var beta = Random<T>(7, channels);
        var gradOut = Random<T>(8, batch, channels, side, side);
        engine.GroupNorm(x, groups, gamma, beta, 1e-5, out var mean, out var variance);
        var gradInput = engine.GroupNormBackward(gradOut, x, groups, gamma, mean, variance, 1e-5, out var gradGamma, out var gradBeta);
        return Concat(gradInput, gradGamma, gradBeta);
    }

    private static double[] GroupNormBackwardReference<T>(int batch, int channels, int side, int groups)
    {
        int spatial = side * side, cpg = channels / groups, n = cpg * spatial;
        var x = ToDouble(Random<T>(5, batch, channels, side, side).ToArray());
        var gamma = ToDouble(Random<T>(6, channels).ToArray());
        var go = ToDouble(Random<T>(8, batch, channels, side, side).ToArray());
        var gi = new double[x.Length];
        var gg = new double[channels];
        var gb = new double[channels];
        for (int b = 0; b < batch; b++)
            for (int g = 0; g < groups; g++)
            {
                int start = (b * channels + g * cpg) * spatial;
                double m = 0, v = 0;
                for (int i = 0; i < n; i++) m += x[start + i];
                m /= n;
                for (int i = 0; i < n; i++) v += (x[start + i] - m) * (x[start + i] - m);
                v /= n;
                double invStd = 1 / Math.Sqrt(v + 1e-5);
                double sumG = 0, sumGX = 0;
                for (int i = 0; i < n; i++)
                {
                    int c = g * cpg + i / spatial;
                    double xhat = (x[start + i] - m) * invStd;
                    gg[c] += go[start + i] * xhat;
                    gb[c] += go[start + i];
                    sumG += gamma[c] * go[start + i];
                    sumGX += gamma[c] * go[start + i] * xhat;
                }
                for (int i = 0; i < n; i++)
                {
                    int c = g * cpg + i / spatial;
                    double xhat = (x[start + i] - m) * invStd;
                    gi[start + i] = invStd / n * (n * gamma[c] * go[start + i] - sumG - xhat * sumGX);
                }
            }
        var all = new double[gi.Length + 2 * channels];
        gi.CopyTo(all, 0); gg.CopyTo(all, gi.Length); gb.CopyTo(all, gi.Length + channels);
        return all;
    }

    // Batch 1 (diffusion: the old per-batch split ran on one core), and an odd spatial size for the scalar tails.
    [Theory]
    [InlineData(1, 64, 32, 8)]
    [InlineData(3, 32, 17, 4)]
    public void GroupNormBackward_Float_SameAtAnyThreadCount_AndCorrect(int batch, int channels, int side, int groups)
    {
        var serial = Run(1, () => GroupNormBackward<float>(batch, channels, side, groups));
        var parallel = Run(Math.Max(4, Environment.ProcessorCount), () => GroupNormBackward<float>(batch, channels, side, groups));
        AssertBitEqual(serial, parallel);
        AssertClose(GroupNormBackwardReference<float>(batch, channels, side, groups), ToDouble(parallel), 2e-3);
    }

    [Theory]
    [InlineData(1, 64, 32, 8)]
    [InlineData(3, 32, 17, 4)]
    public void GroupNormBackward_Double_SameAtAnyThreadCount_AndCorrect(int batch, int channels, int side, int groups)
    {
        var serial = Run(1, () => GroupNormBackward<double>(batch, channels, side, groups));
        var parallel = Run(Math.Max(4, Environment.ProcessorCount), () => GroupNormBackward<double>(batch, channels, side, groups));
        AssertBitEqual(serial, parallel);
        AssertClose(GroupNormBackwardReference<double>(batch, channels, side, groups), ToDouble(parallel), 1e-9);
    }
}

/// <summary>
/// #653 phase 3: float CPU attention fans out over (head, 64-query-row block). Same output bits at one thread and
/// many, and values matching a double-precision reference -- including a broadcast mask, a fully masked row (zero
/// weights, zero output) and a query length that is not a multiple of the block.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class AttentionFanOutTests
{
    private static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static (float[] Output, float[] Weights) Run(int dop, Tensor<float> q, Tensor<float> k, Tensor<float> v, Tensor<bool>? mask)
    {
        int prior = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = dop;
            var o = new CpuEngine().ScaledDotProductAttention(q, k, v, mask, null, out var w);
            return (o.ToArray(), w.ToArray());
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = prior; }
    }

    [Theory]
    [InlineData(1, 3, 150, 40, 16, false)]
    [InlineData(2, 2, 64, 200, 32, true)]
    public void Sdpa_Float_SameAtAnyThreadCount_AndCorrect(int batch, int heads, int seqQ, int seqK, int d, bool masked)
    {
        var q = Random(1, batch, heads, seqQ, d);
        var k = Random(2, batch, heads, seqK, d);
        var v = Random(3, batch, heads, seqK, d);
        Tensor<bool>? mask = null;
        if (masked)
        {
            // [1, 1, seqQ, seqK]: banded, with query row 5 masked everywhere.
            mask = new Tensor<bool>(new[] { 1, 1, seqQ, seqK });
            for (int i = 0; i < seqQ; i++)
                for (int j = 0; j < seqK; j++)
                    mask[0, 0, i, j] = i != 5 && Math.Abs(i * seqK / seqQ - j) < 40;
        }
        var serial = Run(1, q, k, v, mask);
        var parallel = Run(Math.Max(4, Environment.ProcessorCount), q, k, v, mask);
        Assert.Equal(serial.Output, parallel.Output);
        Assert.Equal(serial.Weights, parallel.Weights);

        double scale = 1 / Math.Sqrt(d);
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < seqQ; i++)
                {
                    var s = new double[seqK];
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < seqK; j++)
                    {
                        double dot = 0;
                        for (int c = 0; c < d; c++) dot += q[b, h, i, c] * (double)k[b, h, j, c];
                        s[j] = mask is null || mask[0, 0, i, j] ? dot * scale : double.NegativeInfinity;
                        max = Math.Max(max, s[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j < seqK; j++) { s[j] = double.IsNegativeInfinity(max) ? 0 : Math.Exp(s[j] - max); sum += s[j]; }
                    for (int j = 0; j < seqK; j++)
                    {
                        double w = sum > 0 ? s[j] / sum : 0;
                        Assert.True(Math.Abs(w - parallel.Weights[((b * heads + h) * seqQ + i) * seqK + j]) < 1e-5, $"weight ({b},{h},{i},{j})");
                    }
                    for (int c = 0; c < d; c++)
                    {
                        double o = 0;
                        for (int j = 0; j < seqK; j++) o += (sum > 0 ? s[j] / sum : 0) * v[b, h, j, c];
                        Assert.True(Math.Abs(o - parallel.Output[((b * heads + h) * seqQ + i) * d + c]) < 1e-4, $"output ({b},{h},{i},{c})");
                    }
                }
    }
}
