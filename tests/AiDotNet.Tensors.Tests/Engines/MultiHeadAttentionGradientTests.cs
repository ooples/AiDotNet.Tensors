using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// <see cref="CpuEngine.MultiHeadAttentionForward{T}"/> under an eager <see cref="GradientTape{T}"/> (#436). The fused
/// float / double kernels record nothing on the tape, so before the fix every gradient through the attention (the input
/// and all four projection weights) was missing. Checked against central finite differences of the same tape-path loss,
/// with a random-weighted loss so the upstream gradient is not all ones.
/// </summary>
public class MultiHeadAttentionGradientTests
{
    private const int Batch = 2, Seq = 4, DModel = 6, Heads = 2;

    public static IEnumerable<object?[]> Masks()
    {
        yield return new object?[] { "none" };
        yield return new object?[] { "causal [1,1,S,S]" };
        yield return new object?[] { "padding [B,1,1,S]" };
    }

    private static Tensor<bool>? MakeMask(string kind)
    {
        if (kind.StartsWith("causal", StringComparison.Ordinal))
        {
            var m = new Tensor<bool>(new[] { 1, 1, Seq, Seq });
            for (int i = 0; i < Seq; i++)
                for (int j = 0; j < Seq; j++) m[0, 0, i, j] = j <= i;
            return m;
        }
        if (kind.StartsWith("padding", StringComparison.Ordinal))
        {
            // The second sample's last key is padding.
            var m = new Tensor<bool>(new[] { Batch, 1, 1, Seq });
            for (int b = 0; b < Batch; b++)
                for (int j = 0; j < Seq; j++) m[b, 0, 0, j] = !(b == 1 && j == Seq - 1);
            return m;
        }
        return null;
    }

    [Theory]
    [MemberData(nameof(Masks))]
    public void Float_EveryGradientExists_AndMatchesFiniteDifference(string maskKind)
    {
        var eng = new CpuEngine();
        var rng = new Random(436);
        Tensor<float> R(params int[] s)
        {
            var t = new Tensor<float>(s);
            var sp = t.AsWritableSpan();
            for (int i = 0; i < sp.Length; i++) sp[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            return t;
        }
        var x = R(Batch, Seq, DModel);
        var wq = R(DModel, DModel); var wk = R(DModel, DModel); var wv = R(DModel, DModel); var wo = R(DModel, DModel);
        var coeff = R(Batch, Seq, DModel);
        var mask = MakeMask(maskKind);

        float Loss()
        {
            using var tape = new GradientTape<float>();
            var y = eng.MultiHeadAttentionForward(x, wq, wk, wv, wo, Heads, mask);
            return eng.ReduceSum(eng.TensorMultiply(y, coeff), null).AsSpan()[0];
        }

        Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            var y = eng.MultiHeadAttentionForward(x, wq, wk, wv, wo, Heads, mask);
            var loss = eng.ReduceSum(eng.TensorMultiply(y, coeff), null);
            grads = tape.ComputeGradients(loss, new[] { x, wq, wk, wv, wo });
        }

        foreach (var (name, param) in new[] { ("input", x), ("qWeight", wq), ("kWeight", wk), ("vWeight", wv), ("outWeight", wo) })
        {
            Assert.True(grads.TryGetValue(param, out var analytic), $"{name}: no gradient recorded ({maskKind}).");
            const float eps = 1e-2f;
            var p = param.AsWritableSpan();
            var a = analytic.AsSpan();
            float worstAbs = 0f, worstRel = 0f;
            for (int i = 0; i < p.Length; i++)
            {
                float orig = p[i];
                p[i] = orig + eps; float lp = Loss();
                p[i] = orig - eps; float lm = Loss();
                p[i] = orig;
                float num = (lp - lm) / (2 * eps);
                float abs = Math.Abs(num - a[i]);
                worstAbs = Math.Max(worstAbs, abs);
                worstRel = Math.Max(worstRel, abs / (1e-3f + Math.Abs(num) + Math.Abs(a[i])));
            }
            Assert.True(worstAbs < 2e-2f || worstRel < 2e-2f,
                $"{name} ({maskKind}): gradient vs finite difference, maxAbs={worstAbs:E3}, maxRel={worstRel:E3}.");
        }
    }

    [Theory]
    [MemberData(nameof(Masks))]
    public void Double_EveryGradientExists_AndMatchesFiniteDifference(string maskKind)
    {
        var eng = new CpuEngine();
        var rng = new Random(1436);
        Tensor<double> R(params int[] s)
        {
            var t = new Tensor<double>(s);
            var sp = t.AsWritableSpan();
            for (int i = 0; i < sp.Length; i++) sp[i] = (rng.NextDouble() * 2 - 1) * 0.5;
            return t;
        }
        var x = R(Batch, Seq, DModel);
        var wq = R(DModel, DModel); var wk = R(DModel, DModel); var wv = R(DModel, DModel); var wo = R(DModel, DModel);
        var coeff = R(Batch, Seq, DModel);
        var mask = MakeMask(maskKind);

        double Loss()
        {
            using var tape = new GradientTape<double>();
            var y = eng.MultiHeadAttentionForward(x, wq, wk, wv, wo, Heads, mask);
            return eng.ReduceSum(eng.TensorMultiply(y, coeff), null).AsSpan()[0];
        }

        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var y = eng.MultiHeadAttentionForward(x, wq, wk, wv, wo, Heads, mask);
            var loss = eng.ReduceSum(eng.TensorMultiply(y, coeff), null);
            grads = tape.ComputeGradients(loss, new[] { x, wq, wk, wv, wo });
        }

        foreach (var (name, param) in new[] { ("input", x), ("qWeight", wq), ("kWeight", wk), ("vWeight", wv), ("outWeight", wo) })
        {
            Assert.True(grads.TryGetValue(param, out var analytic), $"{name}: no gradient recorded ({maskKind}).");
            const double eps = 1e-5;
            var p = param.AsWritableSpan();
            var a = analytic.AsSpan();
            double worst = 0;
            for (int i = 0; i < p.Length; i++)
            {
                double orig = p[i];
                p[i] = orig + eps; double lp = Loss();
                p[i] = orig - eps; double lm = Loss();
                p[i] = orig;
                double num = (lp - lm) / (2 * eps);
                worst = Math.Max(worst, Math.Abs(num - a[i]) / (1e-6 + Math.Abs(num) + Math.Abs(a[i])));
            }
            Assert.True(worst < 1e-6, $"{name} ({maskKind}): gradient vs finite difference, maxRel={worst:E3}.");
        }
    }

    [Theory]
    [MemberData(nameof(Masks))]
    public void Float_TapePathOutput_MatchesTheFusedInferenceOutput(string maskKind)
    {
        var eng = new CpuEngine();
        var rng = new Random(77);
        var x = new Tensor<float>(new[] { Batch, Seq, DModel });
        var ws = new Tensor<float>[4];
        foreach (var t in new[] { x })
        {
            var sp = t.AsWritableSpan();
            for (int i = 0; i < sp.Length; i++) sp[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        for (int w = 0; w < 4; w++)
        {
            ws[w] = new Tensor<float>(new[] { DModel, DModel });
            var sp = ws[w].AsWritableSpan();
            for (int i = 0; i < sp.Length; i++) sp[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        }
        var mask = MakeMask(maskKind);

        var fused = eng.MultiHeadAttentionForward(x, ws[0], ws[1], ws[2], ws[3], Heads, mask).AsSpan().ToArray();
        float[] taped;
        using (new GradientTape<float>())
            taped = eng.MultiHeadAttentionForward(x, ws[0], ws[1], ws[2], ws[3], Heads, mask).AsSpan().ToArray();

        for (int i = 0; i < fused.Length; i++)
            Assert.True(Math.Abs(fused[i] - taped[i]) <= 1e-4f * (1f + Math.Abs(fused[i])),
                $"output {i} ({maskKind}): fused {fused[i]} vs tape path {taped[i]}.");
    }

    [Theory]
    [MemberData(nameof(Masks))]
    public void FusedFloatGradients_MatchTheDoubleDecomposedChain(string maskKind)
    {
        // Float takes the fused tape node; double records the decomposed chain. Same values in both, so the fused
        // backward is checked against an independent derivation at far tighter tolerance than float finite differences.
        var eng = new CpuEngine();
        var rng = new Random(2436);
        const int batch = 3, seq = 7, dModel = 12, heads = 3;
        double[] Values(int n) { var a = new double[n]; for (int i = 0; i < n; i++) a[i] = (rng.NextDouble() * 2 - 1) * 0.6; return a; }
        var shapes = new[] { new[] { batch, seq, dModel }, new[] { dModel, dModel }, new[] { dModel, dModel }, new[] { dModel, dModel }, new[] { dModel, dModel }, new[] { batch, seq, dModel } };
        var values = new double[shapes.Length][];
        for (int i = 0; i < shapes.Length; i++) { int n = 1; foreach (var s in shapes[i]) n *= s; values[i] = Values(n); }
        Tensor<bool>? mask = null;
        if (maskKind.StartsWith("causal", StringComparison.Ordinal))
        {
            mask = new Tensor<bool>(new[] { 1, 1, seq, seq });
            for (int i = 0; i < seq; i++) for (int j = 0; j < seq; j++) mask[0, 0, i, j] = j <= i;
        }
        else if (maskKind.StartsWith("padding", StringComparison.Ordinal))
        {
            mask = new Tensor<bool>(new[] { batch, 1, 1, seq });
            for (int b = 0; b < batch; b++) for (int j = 0; j < seq; j++) mask[b, 0, 0, j] = j < seq - b;
        }

        var f = new Tensor<float>[shapes.Length];
        var d = new Tensor<double>[shapes.Length];
        for (int i = 0; i < shapes.Length; i++)
        {
            f[i] = new Tensor<float>(shapes[i]); d[i] = new Tensor<double>(shapes[i]);
            var fs = f[i].AsWritableSpan(); var ds = d[i].AsWritableSpan();
            for (int j = 0; j < values[i].Length; j++) { fs[j] = (float)values[i][j]; ds[j] = values[i][j]; }
        }

        Dictionary<Tensor<float>, Tensor<float>> gf;
        using (var tape = new GradientTape<float>())
        {
            int before = tape.EntryCount;
            var y = eng.MultiHeadAttentionForward(f[0], f[1], f[2], f[3], f[4], heads, mask);
            Assert.Equal(before + 1, tape.EntryCount);   // the fused node, not the decomposed chain
            gf = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, f[5]), null), new[] { f[0], f[1], f[2], f[3], f[4] });
        }
        Dictionary<Tensor<double>, Tensor<double>> gd;
        using (var tape = new GradientTape<double>())
        {
            int before = tape.EntryCount;
            var y = eng.MultiHeadAttentionForward(d[0], d[1], d[2], d[3], d[4], heads, mask);
            Assert.True(tape.EntryCount > before + 1);    // the decomposed chain: an independent derivation
            gd = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, d[5]), null), new[] { d[0], d[1], d[2], d[3], d[4] });
        }

        string[] names = { "input", "qWeight", "kWeight", "vWeight", "outWeight" };
        for (int i = 0; i < 5; i++)
        {
            var a = gf[f[i]].AsSpan(); var r = gd[d[i]].AsSpan();
            double scaleRef = 0; for (int j = 0; j < r.Length; j++) scaleRef = Math.Max(scaleRef, Math.Abs(r[j]));
            for (int j = 0; j < r.Length; j++)
                Assert.True(Math.Abs(a[j] - r[j]) <= 1e-4 * (1 + scaleRef),
                    $"{names[i]}[{j}] ({maskKind}): fused float {a[j]} vs double chain {r[j]}.");
        }
    }
}