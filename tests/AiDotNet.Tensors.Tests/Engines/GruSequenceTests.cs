// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The fused GRU sequence (#1568) against PyTorch nn.GRU's equations, evaluated step by step in double precision:
/// forward values, gradients by central differences, and the GPU engine's device kernels against the CPU.
/// </summary>
public sealed class GruSequenceTests
{
    private const int B = 3, T = 4, In = 5, H = 6;

    private sealed class Weights<TT>
    {
        public Weights(Tensor<TT> x, Tensor<TT> h0, Tensor<TT> wIh, Tensor<TT> wHh, Tensor<TT> bIh, Tensor<TT> bHh)
        {
            X = x; H0 = h0; WIh = wIh; WHh = wHh; BIh = bIh; BHh = bHh;
        }

        public Tensor<TT> X { get; }
        public Tensor<TT> H0 { get; }
        public Tensor<TT> WIh { get; }
        public Tensor<TT> WHh { get; }
        public Tensor<TT> BIh { get; }
        public Tensor<TT> BHh { get; }
    }

    private static double[] RandArray(int seed, int n, double scale)
    {
        var rng = new Random(seed);
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = (rng.NextDouble() * 2 - 1) * scale;
        return a;
    }

    private static Weights<double> MakeDouble() => new(
        new Tensor<double>(RandArray(1, B * T * In, 1.0), new[] { B, T, In }),
        new Tensor<double>(RandArray(2, B * H, 0.5), new[] { B, H }),
        new Tensor<double>(RandArray(3, 3 * H * In, 0.4), new[] { 3 * H, In }),
        new Tensor<double>(RandArray(4, 3 * H * H, 0.4), new[] { 3 * H, H }),
        new Tensor<double>(RandArray(5, 3 * H, 0.3), new[] { 3 * H }),
        new Tensor<double>(RandArray(6, 3 * H, 0.3), new[] { 3 * H }));

    private static Tensor<float> F(Tensor<double> d) => new(Array.ConvertAll(d.ToArray(), v => (float)v), d.Shape.ToArray());

    private static Weights<float> ToFloat(Weights<double> w) => new(F(w.X), F(w.H0), F(w.WIh), F(w.WHh), F(w.BIh), F(w.BHh));

    // nn.GRU's equations written out: r and z from both sums; with the reset gate after the recurrent sum (PyTorch),
    // n = tanh(Win x + bin + r * (Whn h + bhn)); before it (Cho et al., Keras reset_after=False),
    // n = tanh(Win x + bin + Whn (r * h) + bhn).
    private static double[] Reference(Weights<double> w, bool useH0, bool useBias, bool resetAfter = true)
    {
        var x = w.X.ToArray(); var wih = w.WIh.ToArray(); var whh = w.WHh.ToArray();
        var bih = useBias ? w.BIh.ToArray() : new double[3 * H];
        var bhh = useBias ? w.BHh.ToArray() : new double[3 * H];
        var y = new double[B * T * H];
        for (int b = 0; b < B; b++)
        {
            var h = new double[H];
            if (useH0) Array.Copy(w.H0.ToArray(), b * H, h, 0, H);
            for (int t = 0; t < T; t++)
            {
                double InputSum(int row)
                {
                    double s = bih[row];
                    for (int i = 0; i < In; i++) s += wih[row * In + i] * x[(b * T + t) * In + i];
                    return s;
                }
                double RecurrentSum(int row, double[] v)
                {
                    double s = bhh[row];
                    for (int k = 0; k < H; k++) s += whh[row * H + k] * v[k];
                    return s;
                }
                var r = new double[H]; var z = new double[H];
                for (int j = 0; j < H; j++)
                {
                    r[j] = 1 / (1 + Math.Exp(-(InputSum(j) + RecurrentSum(j, h))));
                    z[j] = 1 / (1 + Math.Exp(-(InputSum(H + j) + RecurrentSum(H + j, h))));
                }
                var rh = new double[H];
                for (int k = 0; k < H; k++) rh[k] = r[k] * h[k];
                var next = new double[H];
                for (int j = 0; j < H; j++)
                {
                    double n = Math.Tanh(InputSum(2 * H + j)
                        + (resetAfter ? r[j] * RecurrentSum(2 * H + j, h) : RecurrentSum(2 * H + j, rh)));
                    next[j] = (1 - z[j]) * n + z[j] * h[j];
                    y[(b * T + t) * H + j] = next[j];
                }
                h = next;
            }
        }
        return y;
    }
    private static Tensor<TT> Run<TT>(IEngine e, Weights<TT> w, bool useH0, bool useBias, bool sequences, out Tensor<TT> final,
        bool resetAfter = true)
        => e.GruSequenceForward(w.X, useH0 ? w.H0 : null, w.WIh, w.WHh, useBias ? w.BIh : null, useBias ? w.BHh : null,
            out final, sequences, resetAfter);

    [Theory]
    [InlineData(true, true, true)]
    [InlineData(false, true, true)]
    [InlineData(true, false, true)]
    [InlineData(true, true, false)]
    [InlineData(false, true, false)]
    [InlineData(true, false, false)]
    public void Cpu_MatchesNnGruEquations(bool useH0, bool useBias, bool resetAfter)
    {
        var w = MakeDouble();
        var expected = Reference(w, useH0, useBias, resetAfter);
        var y = Run(new CpuEngine(), w, useH0, useBias, sequences: true, out var final, resetAfter);
        var actual = y.ToArray();
        Assert.Equal(new[] { B, T, H }, y.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 12);

        var last = Run(new CpuEngine(), w, useH0, useBias, sequences: false, out _, resetAfter).ToArray();
        var finalValues = final.ToArray();
        for (int b = 0; b < B; b++)
            for (int j = 0; j < H; j++)
            {
                Assert.Equal(expected[(b * T + T - 1) * H + j], last[b * H + j], 12);
                Assert.Equal(expected[(b * T + T - 1) * H + j], finalValues[b * H + j], 12);
            }
    }

    // A loss that weights every output differently, so a wrong gate or index cannot cancel out.
    private static Tensor<TT> Loss<TT>(IEngine e, Tensor<TT> y, Tensor<TT> projection)
        => e.ReduceMean(e.TensorMultiply(y, projection), new[] { 0, 1, 2 }, keepDims: false);

    private static Dictionary<string, double[]> TapeGradients<TT>(IEngine e, Weights<TT> w, Tensor<TT> projection,
        bool resetAfter = true)
    {
        var sources = new[] { w.X, w.H0, w.WIh, w.WHh, w.BIh, w.BHh };
        using var tape = new GradientTape<TT>();
        var y = e.GruSequenceForward(w.X, w.H0, w.WIh, w.WHh, w.BIh, w.BHh, returnSequences: true, resetAfter: resetAfter);
        var g = tape.ComputeGradients(Loss(e, y, projection), sources);
        var names = new[] { "x", "h0", "wIh", "wHh", "bIh", "bHh" };
        var result = new Dictionary<string, double[]>();
        for (int i = 0; i < sources.Length; i++)
            result[names[i]] = Array.ConvertAll(g[sources[i]].ToArray(), v => Convert.ToDouble(v));
        return result;
    }

    private static double Cosine(double[] a, double[] b)
    {
        double ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.Length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
        return ab / Math.Sqrt(aa * bb);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Cpu_GradientsMatchCentralDifferences(bool resetAfter)
    {
        var cpu = new CpuEngine();
        var w = MakeDouble();
        var projection = new Tensor<double>(RandArray(7, B * T * H, 1.0), new[] { B, T, H });
        var analytic = TapeGradients(cpu, w, projection, resetAfter);

        double LossAt() => Loss(cpu, cpu.GruSequenceForward(w.X, w.H0, w.WIh, w.WHh, w.BIh, w.BHh, returnSequences: true,
            resetAfter: resetAfter), projection).ToArray()[0];
        var tensors = new Dictionary<string, Tensor<double>>
        {
            ["x"] = w.X, ["h0"] = w.H0, ["wIh"] = w.WIh, ["wHh"] = w.WHh, ["bIh"] = w.BIh, ["bHh"] = w.BHh,
        };
        const double Eps = 1e-6;
        foreach (var entry in tensors)
        {
            string name = entry.Key;
            var tensor = entry.Value;
            var numeric = new double[tensor.Length];
            for (int i = 0; i < tensor.Length; i++)
            {
                double v = tensor.GetFlatIndexValue(i);
                tensor.SetFlatIndexValue(i, v + Eps); double up = LossAt();
                tensor.SetFlatIndexValue(i, v - Eps); double down = LossAt();
                tensor.SetFlatIndexValue(i, v);
                numeric[i] = (up - down) / (2 * Eps);
            }
            Assert.True(Cosine(analytic[name], numeric) >= 0.99999,
                $"{name}: gradient cosine {Cosine(analytic[name], numeric)} against central differences.");
            for (int i = 0; i < numeric.Length; i++)
                Assert.True(Math.Abs(analytic[name][i] - numeric[i]) <= 1e-6 + 1e-4 * Math.Abs(numeric[i]),
                    $"{name}[{i}]: analytic {analytic[name][i]}, numeric {numeric[i]}.");
        }
    }

    // The float path splits a large batch across threads (the shapes above run inline): against the double path, which
    // is separate code, at a batch that takes the parallel route.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Cpu_FloatPathOnALargeBatch_MatchesTheDoublePath(bool resetAfter)
    {
        const int Batch = 64, Steps = 6, Inputs = 8, Hidden = 64;
        Tensor<double> D(int seed, double scale, params int[] shape)
        {
            int n = 1; foreach (var s in shape) n *= s;
            return new Tensor<double>(RandArray(seed, n, scale), shape);
        }
        var wd = new Weights<double>(D(11, 1.0, Batch, Steps, Inputs), D(12, 0.5, Batch, Hidden),
            D(13, 0.2, 3 * Hidden, Inputs), D(14, 0.2, 3 * Hidden, Hidden), D(15, 0.2, 3 * Hidden), D(16, 0.2, 3 * Hidden));
        var projD = D(17, 1.0, Batch, Steps, Hidden);
        var cpu = new CpuEngine();

        var yD = cpu.GruSequenceForward(wd.X, wd.H0, wd.WIh, wd.WHh, wd.BIh, wd.BHh, returnSequences: true, resetAfter: resetAfter).ToArray();
        var wf = ToFloat(wd);
        var yF = cpu.GruSequenceForward(wf.X, wf.H0, wf.WIh, wf.WHh, wf.BIh, wf.BHh, returnSequences: true, resetAfter: resetAfter).ToArray();
        for (int i = 0; i < yD.Length; i++)
            Assert.True(Math.Abs(yD[i] - yF[i]) <= 1e-4, $"y[{i}]: double {yD[i]}, float {yF[i]}.");

        var gD = TapeGradients(cpu, wd, projD, resetAfter);
        var gF = TapeGradients(cpu, wf, F(projD), resetAfter);
        foreach (var name in gD.Keys)
            Assert.True(Cosine(gD[name], gF[name]) >= 0.99999, $"{name}: float-path gradient cosine {Cosine(gD[name], gF[name])}.");
    }

    private static DirectGpuTensorEngine? TryGpu()
    {
        try
        {
            var gpu = new DirectGpuTensorEngine();
            if (gpu.IsGpuAvailable) return gpu;
            gpu.Dispose();
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException) { }
        return null;
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void Gpu_ForwardMatchesTheCpu_OnTheDevice(bool resetAfter)
    {
        using var gpu = TryGpu();
        Skip.If(gpu is null, "No DirectGpu backend.");
        var w = ToFloat(MakeDouble());
        var expected = Run(new CpuEngine(), w, true, true, sequences: true, out _, resetAfter).ToArray();
        AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Reset();
        var actual = Run(gpu, w, true, true, sequences: true, out var final, resetAfter);
        Assert.True(AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Count >= 2 * T,
            "the GPU engine did not launch the GRU kernels (two per step).");
        var values = actual.ToArray();
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], values[i], 4);
        Assert.Equal(new[] { B, H }, final.Shape.ToArray());
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void Gpu_GradientsMatchTheCpu(bool resetAfter)
    {
        using var gpu = TryGpu();
        Skip.If(gpu is null, "No DirectGpu backend.");
        var w = ToFloat(MakeDouble());
        var projection = F(new Tensor<double>(RandArray(7, B * T * H, 1.0), new[] { B, T, H }));
        var cpu = TapeGradients(new CpuEngine(), w, projection, resetAfter);
        // The device path must have run, forward and backward: two launches per step each way, then four for the input,
        // weight and bias gradients. A host fallback would pass the value checks below without launching any of them.
        AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Reset();
        var device = TapeGradients(gpu, w, projection, resetAfter);
        Assert.True(AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Count >= 4 * T + 4,
            $"the GRU device kernels did not run (launches {AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Count}).");
        foreach (var name in cpu.Keys)
        {
            double cosine = Cosine(cpu[name], device[name]);
            Assert.True(cosine >= 0.99999, $"{name}: GPU gradient cosine {cosine} against the CPU.");
            // A cosine cannot see a scale error (a doubled bias gradient, a missing 1/N): compare every element too.
            for (int i = 0; i < cpu[name].Length; i++)
                Assert.True(Math.Abs(cpu[name][i] - device[name][i]) <= 1e-4 + 1e-3 * Math.Abs(cpu[name][i]),
                    $"{name}[{i}]: CPU {cpu[name][i]}, GPU {device[name][i]}.");
        }
    }
}
