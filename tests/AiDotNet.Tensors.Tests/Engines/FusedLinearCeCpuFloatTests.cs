using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The CPU float path of <see cref="CpuEngine.FusedLinearCrossEntropyWithLogits{T}(Tensor{T},Tensor{T},Tensor{T},Tensor{int})"/>
/// (row-parallel SIMD log-softmax / softmax-minus-onehot, SIMD bias add and column sums) must agree with the double
/// path on the same inputs - loss and all three gradients - both below the row-parallel threshold (serial) and above it
/// (chunked across the pool), with a vocabulary that is not a multiple of the SIMD width so the row tails run.
/// </summary>
public class FusedLinearCeCpuFloatTests
{
    private static double[] Gen(int len, int seed, double scale)
    {
        var a = new double[len];
        for (int i = 0; i < len; i++) a[i] = Math.Sin(0.37 * (i + 1) + 1.9 * seed) * scale + 0.1 * Math.Cos(0.011 * i * seed);
        return a;
    }

    private static (Tensor<T> loss, Dictionary<Tensor<T>, Tensor<T>> grads, Tensor<T>[] p) Run<T>(
        double[] h, double[] w, double[] b, int[] ids, int n, int d, int vocab, Func<double, T> cast)
    {
        T[] C(double[] x) { var r = new T[x.Length]; for (int i = 0; i < x.Length; i++) r[i] = cast(x[i]); return r; }
        var engine = new CpuEngine();
        var p = new[]
        {
            new Tensor<T>(C(h), new[] { n, d }),
            new Tensor<T>(C(w), new[] { d, vocab }),
            new Tensor<T>(C(b), new[] { vocab }),
        };
        using var tape = new GradientTape<T>();
        var loss = engine.FusedLinearCrossEntropyWithLogits(p[0], p[1], p[2], new Tensor<int>(ids, new[] { n }));
        var grads = tape.ComputeGradients(loss, p);
        return (loss, grads, p);
    }

    [Theory]
    [InlineData(2, 3, 5)]        // serial path (n * vocab below the row-parallel threshold)
    [InlineData(64, 16, 4099)]   // row-parallel path; vocab % 8 != 0 exercises the SIMD tails
    [InlineData(96, 8, 49152)]   // LM-head vocabulary
    public void FloatPath_MatchesDoublePath(int n, int d, int vocab)
    {
        var h = Gen(n * d, 1, 0.8);
        var w = Gen(d * vocab, 2, 0.6);
        var b = Gen(vocab, 3, 0.3);
        var ids = new int[n];
        for (int r = 0; r < n; r++) ids[r] = (r * 7919 + 3) % vocab;

        var (lf, gf, pf) = Run<float>(h, w, b, ids, n, d, vocab, x => (float)x);
        var (ld, gd, pd) = Run<double>(h, w, b, ids, n, d, vocab, x => x);

        double lossF = lf[0], lossD = ld[0];
        Assert.True(Math.Abs(lossF - lossD) <= 1e-5 * Math.Max(1.0, Math.Abs(lossD)), $"loss float={lossF} double={lossD}");

        string[] names = { "hidden", "weight", "bias" };
        for (int k = 0; k < 3; k++)
        {
            var f = gf[pf[k]].ToArray();
            var dd = gd[pd[k]].ToArray();
            Assert.Equal(dd.Length, f.Length);
            double maxAbs = 0, maxDiff = 0;
            for (int i = 0; i < f.Length; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(dd[i]));
                maxDiff = Math.Max(maxDiff, Math.Abs(f[i] - dd[i]));
            }
            Assert.True(maxAbs > 0, $"{names[k]} gradient is all zero");
            Assert.True(maxDiff <= 1e-4 * maxAbs, $"{names[k]} gradient max diff {maxDiff:E3} vs max |g| {maxAbs:E3}");
        }
    }
}
