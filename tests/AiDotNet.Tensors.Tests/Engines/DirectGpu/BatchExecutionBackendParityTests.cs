#if NET6_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
#if NET7_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
#endif
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Every <see cref="IGpuBatchExecution"/> operation on every backend against a double-precision CPU
/// reference (#1142, #916). The references follow PyTorch semantics: <c>ReduceNormL2</c> is the norm,
/// <c>ReduceLogSumExp</c> is the full log-sum-exp, and every operation overwrites its output, so outputs
/// are pre-filled with a sentinel that a correct kernel never leaves behind. The random kernels share
/// one counter-based hash, so the same seed yields the same values on every backend and the reference
/// reproduces them exactly.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class BatchExecutionBackendParityTests
{
    private const float Sentinel = 123.25f;

    private static float[] Values(int n, int salt, float scale = 0.125f, float offset = 0f)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = offset + (((i + salt) * 7919 + salt * 31) % 61 - 30) * scale;
        return v;
    }

    private static float[] Filled(int n, float value)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = value;
        return v;
    }

    private static uint Hash(uint counter, uint key)
    {
        ulong p = (ulong)counter * 0xCD9E8D57u;
        counter = (uint)(p >> 32) ^ key ^ (uint)p;
        p = (ulong)counter * 0xCD9E8D57u;
        return (uint)(p >> 32) ^ (key + 1u) ^ (uint)p;
    }

    private static uint Key(ulong seed) => (uint)(seed ^ (seed >> 32));

    private sealed class Run
    {
        private readonly IGpuBatchExecution _b;
        private readonly List<IGpuBuffer> _live = new List<IGpuBuffer>();
        public readonly List<string> Failures = new List<string>();

        public Run(IGpuBatchExecution b) { _b = b; }

        public IGpuBuffer Up(float[] data)
        {
            var buffer = _b.AllocateBuffer(data);
            _live.Add(buffer);
            return buffer;
        }

        public IGpuBuffer Out(int n) => Up(Filled(n, Sentinel));

        public void Check(string op, IGpuBuffer buffer, double[] expected, double rel = 2e-4, double abs = 2e-5)
        {
            float[] actual = _b.DownloadBuffer(buffer);
            if (actual.Length < expected.Length)
            {
                Failures.Add($"{op}: {actual.Length} values downloaded, expected {expected.Length}");
                return;
            }
            for (int i = 0; i < expected.Length; i++)
            {
                double e = expected[i], a = actual[i];
                bool ok = double.IsNaN(e) ? double.IsNaN(a)
                    : double.IsInfinity(e) ? e == a
                    : Math.Abs(e - a) <= abs + rel * Math.Abs(e);
                if (!ok)
                {
                    Failures.Add($"{op}[{i}]: expected {e:R}, got {a:R}");
                    return;
                }
            }
        }

        public void Op(string op, Action body)
        {
            try { body(); }
            catch (Exception ex) { Failures.Add($"{op}: threw {ex.GetType().Name}: {ex.Message}"); }
        }

        public void Dispose()
        {
            foreach (var buffer in _live) buffer.Dispose();
            _live.Clear();
        }
    }

    private static readonly (int Outer, int Inner)[] RowShapes = { (3, 7), (300, 5), (2, 600) };

    private static void CheckAll(IDirectGpuBackend backend)
    {
        var b = Assert.IsAssignableFrom<IGpuBatchExecution>(backend);
        var run = new Run(b);
        try
        {
            FullReductions(b, run);
            foreach (var (outer, inner) in RowShapes) RowOps(b, run, outer, inner);
            Elementwise(b, run);
            Gated(b, run);
            Shapes(b, run);
            Losses(b, run);
            Random(b, run);
            Distances(b, run);
            Plumbing(b, run);
        }
        finally
        {
            run.Dispose();
        }
        Assert.True(run.Failures.Count == 0,
            $"{run.Failures.Count} IGpuBatchExecution mismatches on {backend.GetType().Name}:\n"
            + string.Join("\n", run.Failures));
    }

    private static void FullReductions(IGpuBatchExecution b, Run run)
    {
        foreach (int n in new[] { 1, 37, 1000, 70_000 })
        {
            var x = Values(n, n);
            var near1 = Values(n, n + 1, 0.0005f, 1f);
            double sumSq = x.Sum(v => (double)v * v);
            double max = x.Max();
            double lse = max + Math.Log(x.Sum(v => Math.Exp(v - max)));
            string s = $"[{n}]";
            run.Op("ReduceProduct" + s, () =>
            {
                var o = run.Out(1);
                b.ReduceProduct(run.Up(near1), o, n);
                run.Check("ReduceProduct" + s, o, new[] { near1.Aggregate(1.0, (p, v) => p * v) }, 1e-3);
            });
            run.Op("ReduceNormL2" + s, () =>
            {
                var o = run.Out(1);
                b.ReduceNormL2(run.Up(x), o, n);
                run.Check("ReduceNormL2" + s, o, new[] { Math.Sqrt(sumSq) });
            });
            run.Op("ReduceSumOfSquares" + s, () =>
            {
                var o = run.Out(1);
                b.ReduceSumOfSquares(run.Up(x), o, n);
                run.Check("ReduceSumOfSquares" + s, o, new[] { sumSq });
            });
            run.Op("ReduceMaxMagnitude" + s, () =>
            {
                var o = run.Out(1);
                b.ReduceMaxMagnitude(run.Up(x), o, n);
                run.Check("ReduceMaxMagnitude" + s, o, new[] { x.Max(v => (double)Math.Abs(v)) });
            });
            run.Op("ReduceMinMagnitude" + s, () =>
            {
                var o = run.Out(1);
                b.ReduceMinMagnitude(run.Up(x), o, n);
                run.Check("ReduceMinMagnitude" + s, o, new[] { x.Min(v => (double)Math.Abs(v)) });
            });
            run.Op("ReduceLogSumExp" + s, () =>
            {
                var o = run.Out(1);
                b.ReduceLogSumExp(run.Up(x), o, (float)max, n);
                run.Check("ReduceLogSumExp" + s, o, new[] { lse });
            });
        }
    }

    private static double[] PerRow(float[] x, int outer, int inner, Func<double[], double> f)
    {
        var r = new double[outer];
        for (int o = 0; o < outer; o++)
            r[o] = f(Enumerable.Range(0, inner).Select(j => (double)x[o * inner + j]).ToArray());
        return r;
    }

    private static double[] MapRow(float[] x, int outer, int inner, Func<double[], double[]> f)
    {
        var r = new double[outer * inner];
        for (int o = 0; o < outer; o++)
        {
            var row = f(Enumerable.Range(0, inner).Select(j => (double)x[o * inner + j]).ToArray());
            Array.Copy(row, 0, r, o * inner, inner);
        }
        return r;
    }

    private static double Variance(double[] r)
    {
        double m = r.Average();
        return r.Sum(v => (v - m) * (v - m)) / r.Length;
    }

    private static double[] Softmax(double[] z)
    {
        double m = z.Max();
        var e = z.Select(v => Math.Exp(v - m)).ToArray();
        double s = e.Sum();
        return e.Select(v => v / s).ToArray();
    }

    private static double[] SparsemaxRow(double[] z)
    {
        var sorted = z.OrderByDescending(v => v).ToArray();
        double cum = 0, tau = 0;
        for (int k = 0; k < sorted.Length; k++)
        {
            cum += sorted[k];
            double candidate = (cum - 1) / (k + 1);
            if (sorted[k] > candidate) tau = candidate;
        }
        return z.Select(v => Math.Max(v - tau, 0)).ToArray();
    }

    private static void RowOps(IGpuBatchExecution b, Run run, int outer, int inner)
    {
        int n = outer * inner;
        var x = Values(n, inner);
        var near1 = Values(n, inner + 3, 0.002f, 1f);
        string s = $"[{outer}x{inner}]";

        void Row(string name, Action<IGpuBuffer, IGpuBuffer> launch, float[] input, double[] expected, int outN)
            => run.Op(name + s, () =>
            {
                var o = run.Out(outN);
                launch(run.Up(input), o);
                run.Check(name + s, o, expected);
            });

        Row("VarianceAxis", (i, o) => b.VarianceAxis(i, o, outer, inner), x, PerRow(x, outer, inner, Variance), outer);
        Row("StdAxis", (i, o) => b.StdAxis(i, o, outer, inner), x,
            PerRow(x, outer, inner, r => Math.Sqrt(Variance(r))), outer);
        Row("ProductAxis", (i, o) => b.ProductAxis(i, o, outer, inner), near1,
            PerRow(near1, outer, inner, r => r.Aggregate(1.0, (p, v) => p * v)), outer);
        Row("NormAxis", (i, o) => b.NormAxis(i, o, outer, inner), x,
            PerRow(x, outer, inner, r => Math.Sqrt(r.Sum(v => v * v))), outer);
        Row("LogSumExpAxis", (i, o) => b.LogSumExpAxis(i, o, outer, inner), x,
            PerRow(x, outer, inner, r => r.Max() + Math.Log(r.Sum(v => Math.Exp(v - r.Max())))), outer);
        Row("ReduceLogVariance", (i, o) => b.ReduceLogVariance(i, o, outer, inner), x,
            PerRow(x, outer, inner, r => Math.Log(Variance(r) + 1e-8)), outer);
        Row("CumSumAxis", (i, o) => b.CumSumAxis(i, o, outer, inner), x, MapRow(x, outer, inner, r =>
        {
            var c = new double[r.Length];
            double acc = 0;
            for (int j = 0; j < r.Length; j++) c[j] = acc += r[j];
            return c;
        }), n);
        Row("NormalizeL2", (i, o) => b.NormalizeL2(i, o, outer, inner), x, MapRow(x, outer, inner, r =>
        {
            double norm = Math.Sqrt(r.Sum(v => v * v)) + 1e-12;
            return r.Select(v => v / norm).ToArray();
        }), n);
        Row("LogSoftmax", (i, o) => b.LogSoftmax(i, o, outer, inner), x,
            MapRow(x, outer, inner, r => Softmax(r).Select(v => Math.Log(v)).ToArray()), n);
        Row("Sparsemax", (i, o) => b.Sparsemax(i, o, outer, inner), x, MapRow(x, outer, inner, SparsemaxRow), n);
        Row("TaylorSoftmax", (i, o) => b.TaylorSoftmax(i, o, outer, inner), x, MapRow(x, outer, inner, r =>
        {
            var t = r.Select(v => Math.Max(1 + v + 0.5 * v * v, 1e-10)).ToArray();
            double sum = t.Sum();
            return t.Select(v => v / sum).ToArray();
        }), n);
        Row("SphericalSoftmax", (i, o) => b.SphericalSoftmax(i, o, outer, inner), x, MapRow(x, outer, inner, r =>
        {
            double norm = Math.Sqrt(r.Sum(v => v * v) + 1e-12);
            return Softmax(r.Select(v => v / norm).ToArray());
        }), n);

        // Backward reductions: grad [outer], stats [outer], input [outer, inner].
        var grad = Values(outer, 5, 0.25f);
        var means = PerRow(x, outer, inner, r => r.Average()).Select(v => (float)v).ToArray();
        var vars = PerRow(x, outer, inner, Variance).Select(v => (float)v).ToArray();
        var rowMax = PerRow(x, outer, inner, r => r.Max()).Select(v => (float)v).ToArray();
        run.Op("ReduceSumBackward" + s, () =>
        {
            var o = run.Out(n);
            b.ReduceSumBackward(run.Up(grad), o, outer, inner);
            run.Check("ReduceSumBackward" + s, o, Enumerable.Range(0, n).Select(i => (double)grad[i / inner]).ToArray());
        });
        run.Op("ReduceMaxBackward" + s, () =>
        {
            var o = run.Out(n);
            b.ReduceMaxBackward(run.Up(grad), run.Up(x), run.Up(rowMax), o, outer, inner);
            run.Check("ReduceMaxBackward" + s, o, Enumerable.Range(0, n)
                .Select(i => x[i] == rowMax[i / inner] ? (double)grad[i / inner] : 0.0).ToArray());
        });
        run.Op("ReduceVarianceBackward" + s, () =>
        {
            var o = run.Out(n);
            b.ReduceVarianceBackward(run.Up(grad), run.Up(x), run.Up(means), o, outer, inner);
            run.Check("ReduceVarianceBackward" + s, o, Enumerable.Range(0, n)
                .Select(i => grad[i / inner] * 2.0 * (x[i] - means[i / inner]) / inner).ToArray());
        });
        run.Op("ReduceLogVarianceBackward" + s, () =>
        {
            var o = run.Out(n);
            b.ReduceLogVarianceBackward(run.Up(grad), run.Up(x), run.Up(means), run.Up(vars), o, outer, inner);
            run.Check("ReduceLogVarianceBackward" + s, o, Enumerable.Range(0, n)
                .Select(i => grad[i / inner] / (vars[i / inner] + 1e-8) * 2.0 * (x[i] - means[i / inner]) / inner)
                .ToArray(), 1e-3);
        });

        // Broadcast over the leading axis: a [outer], b [outer, inner].
        var a = Values(outer, 9, 0.5f);
        run.Op("BroadcastAddFirst" + s, () =>
        {
            var o = run.Out(n);
            b.BroadcastAddFirst(run.Up(a), run.Up(x), o, outer, inner);
            run.Check("BroadcastAddFirst" + s, o, Enumerable.Range(0, n).Select(i => (double)a[i / inner] + x[i]).ToArray());
        });
        run.Op("BroadcastMulFirst" + s, () =>
        {
            var o = run.Out(n);
            b.BroadcastMulFirst(run.Up(a), run.Up(x), o, outer, inner);
            run.Check("BroadcastMulFirst" + s, o, Enumerable.Range(0, n).Select(i => (double)a[i / inner] * x[i]).ToArray());
        });
    }

    private static void Elementwise(IGpuBatchExecution b, Run run)
    {
        const int n = 1031;
        var x = Values(n, 2);
        var pos = Values(n, 3, 0.05f, 2f);
        var unit = Values(n, 4, 0.03f);

        void Map(string name, Action<IGpuBuffer, IGpuBuffer> launch, float[] input, Func<double, double> f)
            => run.Op(name, () =>
            {
                var o = run.Out(n);
                launch(run.Up(input), o);
                run.Check(name, o, input.Select(v => f(v)).ToArray());
            });

        Map("ScalarMinusTensor", (i, o) => b.ScalarMinusTensor(i, o, 2.5f, n), x, v => 2.5 - v);
        Map("DivScalar", (i, o) => b.DivScalar(i, o, -4f, n), x, v => v / -4.0);
        Map("RsqrtKernel", (i, o) => b.RsqrtKernel(i, o, n), pos, v => 1 / Math.Sqrt(v + 1e-12));
        Map("ReluDerivative", (i, o) => b.ReluDerivative(i, o, n), x, v => v > 0 ? 1 : 0);
        Map("SigmoidDerivative", (i, o) => b.SigmoidDerivative(i, o, n), unit, v => v * (1 - v));
        Map("TanhDerivative", (i, o) => b.TanhDerivative(i, o, n), unit, v => 1 - v * v);
        run.Op("SinCosKernel", () =>
        {
            var so = run.Out(n);
            var co = run.Out(n);
            b.SinCosKernel(run.Up(x), so, co, n);
            run.Check("SinCosKernel.sin", so, x.Select(v => Math.Sin(v)).ToArray());
            run.Check("SinCosKernel.cos", co, x.Select(v => Math.Cos(v)).ToArray());
        });
    }

    private static void Gated(IGpuBatchExecution b, Run run)
    {
        foreach (var (outer, half) in new[] { (4, 3), (300, 2) })
        {
            int full = half * 2;
            var input = Values(outer * full, 6, 0.1f);
            var grad = Values(outer * half, 7, 0.2f);
            string s = $"[{outer}x{full}]";

            void Gate(string name, Action<IGpuBuffer, IGpuBuffer, IGpuBuffer> launch,
                Func<double, double> act, Func<double, double> dAct)
                => run.Op(name + s, () =>
                {
                    var expected = new double[outer * full];
                    for (int o = 0; o < outer; o++)
                        for (int d = 0; d < half; d++)
                        {
                            double v = input[o * full + d], g = input[o * full + half + d], go = grad[o * half + d];
                            expected[o * full + d] = go * act(g);
                            expected[o * full + half + d] = go * v * dAct(g);
                        }
                    var gi = run.Out(outer * full);
                    launch(run.Up(grad), run.Up(input), gi);
                    run.Check(name + s, gi, expected, 1e-3, 1e-5);
                });

            Gate("GluBackward", (go, i, gi) => b.GluBackward(go, i, gi, outer, half), Sig, v => Sig(v) * (1 - Sig(v)));
            Gate("GeGluBackward", (go, i, gi) => b.GeGluBackward(go, i, gi, outer, half), Gelu, DGelu);
            Gate("ReGluBackward", (go, i, gi) => b.ReGluBackward(go, i, gi, outer, half),
                v => Math.Max(v, 0), v => v > 0 ? 1 : 0);
            Gate("SwiGluBackward", (go, i, gi) => b.SwiGluBackward(go, i, gi, outer, half),
                v => v * Sig(v), v => Sig(v) + v * Sig(v) * (1 - Sig(v)));
        }
    }

    private const double GeluC = 0.7978845608;

    private static double Sig(double v) => 1 / (1 + Math.Exp(-v));

    private static double Gelu(double v) => 0.5 * v * (1 + Math.Tanh(GeluC * (v + 0.044715 * v * v * v)));

    private static double DGelu(double v)
    {
        double t = Math.Tanh(GeluC * (v + 0.044715 * v * v * v));
        return 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * GeluC * (1 + 0.134145 * v * v);
    }

    private static void Shapes(IGpuBatchExecution b, Run run)
    {
        const int outer = 37, aIn = 5, bIn = 9;
        var a = Values(outer * aIn, 11);
        var bb = Values(outer * bIn, 12);
        run.Op("ConcatAxis", () =>
        {
            var o = run.Out(outer * (aIn + bIn));
            b.ConcatAxis(run.Up(a), run.Up(bb), o, outer, aIn, bIn);
            var e = new double[outer * (aIn + bIn)];
            for (int r = 0; r < outer; r++)
                for (int j = 0; j < aIn + bIn; j++)
                    e[r * (aIn + bIn) + j] = j < aIn ? a[r * aIn + j] : bb[r * bIn + j - aIn];
            run.Check("ConcatAxis", o, e);
        });
        run.Op("SliceLastAxis", () =>
        {
            var o = run.Out(outer * 4);
            b.SliceLastAxis(run.Up(bb), o, outer, bIn, 3, 4);
            run.Check("SliceLastAxis", o, Enumerable.Range(0, outer * 4).Select(i => (double)bb[i / 4 * bIn + 3 + i % 4]).ToArray());
        });
        run.Op("SetSliceLastAxis", () =>
        {
            var o = run.Up(bb.ToArray());
            b.SetSliceLastAxis(o, run.Up(a), outer, bIn, 2, aIn);
            var e = bb.Select(v => (double)v).ToArray();
            for (int r = 0; r < outer; r++)
                for (int j = 0; j < aIn; j++) e[r * bIn + 2 + j] = a[r * aIn + j];
            run.Check("SetSliceLastAxis", o, e);
        });
        const int axis = 6, stride = 7;
        var cube = Values(outer * axis * stride, 13);
        run.Op("SliceAxis", () =>
        {
            var o = run.Out(outer * stride);
            b.SliceAxis(run.Up(cube), o, outer, axis, stride, 4);
            run.Check("SliceAxis", o, Enumerable.Range(0, outer * stride)
                .Select(i => (double)cube[i / stride * axis * stride + 4 * stride + i % stride]).ToArray());
        });
        run.Op("SetSliceAxis", () =>
        {
            var vals = Values(outer * stride, 14);
            var o = run.Up(cube.ToArray());
            b.SetSliceAxis(o, run.Up(vals), outer, axis, stride, 1);
            var e = cube.Select(v => (double)v).ToArray();
            for (int i = 0; i < outer * stride; i++) e[i / stride * axis * stride + stride + i % stride] = vals[i];
            run.Check("SetSliceAxis", o, e);
        });
        run.Op("Stack2", () =>
        {
            var o = run.Out(2 * outer * aIn);
            var a2 = Values(outer * aIn, 15);
            b.Stack2(run.Up(a), run.Up(a2), o, outer * aIn);
            run.Check("Stack2", o, Enumerable.Range(0, 2 * outer * aIn).Select(i => (double)(i % 2 == 0 ? a[i / 2] : a2[i / 2])).ToArray());
        });

        const int bt = 2, ch = 3, inH = 5, inW = 6, outH = 9, outW = 8, top = 3, left = 1;
        var img = Values(bt * ch * inH * inW, 16);
        double PadAt(int i)
        {
            int w = i % outW, h = i / outW % outH, bc = i / (outW * outH);
            int sh = h - top, sw = w - left;
            return sh >= 0 && sh < inH && sw >= 0 && sw < inW ? img[(bc * inH + sh) * inW + sw] : -2.0;
        }
        run.Op("Pad2D", () =>
        {
            var o = run.Out(bt * ch * outH * outW);
            b.Pad2D(run.Up(img), o, bt, ch, inH, inW, outH, outW, top, left, -2f);
            run.Check("Pad2D", o, Enumerable.Range(0, bt * ch * outH * outW).Select(PadAt).ToArray());
        });
        var gPad = Values(bt * ch * outH * outW, 17);
        run.Op("Pad2DBackward", () =>
        {
            var o = run.Out(bt * ch * inH * inW);
            b.Pad2DBackward(run.Up(gPad), o, bt, ch, inH, inW, outH, outW, top, left);
            run.Check("Pad2DBackward", o, Enumerable.Range(0, bt * ch * inH * inW).Select(i =>
            {
                int w = i % inW, h = i / inW % inH, bc = i / (inW * inH);
                return (double)gPad[(bc * outH + h + top) * outW + w + left];
            }).ToArray());
        });
        const int cH = 3, cW = 4, offH = 1, offW = 2;
        run.Op("Crop2D", () =>
        {
            var o = run.Out(bt * ch * cH * cW);
            b.Crop2D(run.Up(img), o, bt, ch, inH, inW, cH, cW, offH, offW);
            run.Check("Crop2D", o, Enumerable.Range(0, bt * ch * cH * cW).Select(i =>
            {
                int w = i % cW, h = i / cW % cH, bc = i / (cW * cH);
                return (double)img[(bc * inH + h + offH) * inW + w + offW];
            }).ToArray());
        });
        run.Op("Crop2DBackward", () =>
        {
            var gc = Values(bt * ch * cH * cW, 18);
            var o = run.Out(bt * ch * inH * inW);
            b.Crop2DBackward(run.Up(gc), o, bt, ch, inH, inW, cH, cW, offH, offW);
            run.Check("Crop2DBackward", o, Enumerable.Range(0, bt * ch * inH * inW).Select(i =>
            {
                int w = i % inW - offW, h = i / inW % inH - offH, bc = i / (inW * inH);
                return h >= 0 && h < cH && w >= 0 && w < cW ? gc[(bc * cH + h) * cW + w] : 0.0;
            }).ToArray());
        });

        run.Op("TileLastAxis", () =>
        {
            var o = run.Out(outer * aIn * 3);
            b.TileLastAxis(run.Up(a), o, outer, aIn, 3);
            run.Check("TileLastAxis", o, Enumerable.Range(0, outer * aIn * 3)
                .Select(i => (double)a[i / (aIn * 3) * aIn + i % (aIn * 3) % aIn]).ToArray());
        });
        run.Op("RepeatElements", () =>
        {
            var o = run.Out(outer * aIn * 3);
            b.RepeatElements(run.Up(a), o, outer, aIn, 3);
            run.Check("RepeatElements", o, Enumerable.Range(0, outer * aIn * 3)
                .Select(i => (double)a[i / (aIn * 3) * aIn + i % (aIn * 3) / 3]).ToArray());
        });
        run.Op("LinspaceKernel", () =>
        {
            var o = run.Out(513);
            b.LinspaceKernel(o, -3f, 0.0125f, 513);
            run.Check("LinspaceKernel", o, Enumerable.Range(0, 513).Select(i => -3.0 + 0.0125 * i).ToArray());
        });
        const int dn = 19;
        var diag = Values(dn, 19);
        run.Op("DiagKernel", () =>
        {
            var o = run.Out(dn * dn);
            b.DiagKernel(run.Up(diag), o, dn);
            run.Check("DiagKernel", o, Enumerable.Range(0, dn * dn).Select(i => i / dn == i % dn ? (double)diag[i / dn] : 0.0).ToArray());
        });
        run.Op("ExtractDiagKernel", () =>
        {
            var m = Values(dn * 23, 20);
            var o = run.Out(dn);
            b.ExtractDiagKernel(run.Up(m), o, dn, 23);
            run.Check("ExtractDiagKernel", o, Enumerable.Range(0, dn).Select(i => (double)m[i * 23 + i]).ToArray());
        });
        run.Op("TriangularMask", () =>
        {
            const int rows = 17, cols = 21;
            var o = run.Out(rows * cols);
            b.TriangularMask(o, rows, cols, 2, -1e9f);
            run.Check("TriangularMask", o, Enumerable.Range(0, rows * cols)
                .Select(i => i % cols > i / cols + 2 ? -1e9 : 0.0).ToArray());
        });
        run.Op("IndexSelect", () =>
        {
            var table = Values(11 * bIn, 21);
            var idx = new float[] { 3, 0, 10, 3, 7 };
            var o = run.Out(idx.Length * bIn);
            b.IndexSelect(run.Up(table), run.Up(idx), o, idx.Length, bIn);
            run.Check("IndexSelect", o, Enumerable.Range(0, idx.Length * bIn)
                .Select(i => (double)table[(int)idx[i / bIn] * bIn + i % bIn]).ToArray());
        });
    }

    private static void Losses(IGpuBatchExecution b, Run run)
    {
        foreach (var (batch, classes) in new[] { (5, 7), (300, 3) })
        {
            var logits = Values(batch * classes, 22, 0.3f);
            var targets = new float[batch * classes];
            for (int r = 0; r < batch; r++)
            {
                targets[r * classes + r % classes] = 0.75f;
                targets[r * classes + (r + 1) % classes] += 0.25f;
            }
            string s = $"[{batch}x{classes}]";
            run.Op("CrossEntropyLoss" + s, () =>
            {
                var o = run.Out(batch);
                b.CrossEntropyLoss(run.Up(logits), run.Up(targets), o, batch, classes);
                run.Check("CrossEntropyLoss" + s, o, Enumerable.Range(0, batch).Select(r =>
                {
                    var row = Enumerable.Range(0, classes).Select(c => (double)logits[r * classes + c]).ToArray();
                    double m = row.Max(), lse = m + Math.Log(row.Sum(v => Math.Exp(v - m)));
                    return -Enumerable.Range(0, classes).Sum(c => targets[r * classes + c] * (row[c] - lse));
                }).ToArray());
            });
            run.Op("MseLoss" + s, () =>
            {
                var o = run.Out(batch);
                var pred = Values(batch * classes, 23);
                b.MseLoss(run.Up(pred), run.Up(logits), o, batch, classes);
                run.Check("MseLoss" + s, o, Enumerable.Range(0, batch).Select(r => Enumerable.Range(0, classes)
                    .Average(c => Math.Pow(pred[r * classes + c] - logits[r * classes + c], 2))).ToArray());
            });
        }
    }

    private static void Random(IGpuBatchExecution b, Run run)
    {
        const int n = 4099;
        const ulong seed = 0x1234_5678_9ABC_DEF1UL;
        uint key = Key(seed);
        run.Op("DropoutMask", () =>
        {
            var o = run.Out(n);
            b.DropoutMask(o, n, 0.7f, seed);
            run.Check("DropoutMask", o, Enumerable.Range(0, n).Select(i =>
            {
                float u = (Hash((uint)i, key) >> 8) * (1.0f / 16777216.0f);
                return u < 0.7f ? 1.0 / 0.7f : 0.0;
            }).ToArray());
        });
        run.Op("GaussianNoise", () =>
        {
            var o = run.Out(n);
            b.GaussianNoise(o, n, 0.5f, 2f, seed);
            run.Check("GaussianNoise", o, Enumerable.Range(0, n).Select(i =>
            {
                double u1 = (Hash((uint)(2 * i), key) >> 8) / 16777216.0 + 0.5 / 16777216.0;
                double u2 = (Hash((uint)(2 * i + 1), key + 2u) >> 8) / 16777216.0;
                return 0.5 + 2.0 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            }).ToArray(), 1e-3, 1e-3);
        });
        foreach (var (outer, inner) in RowShapes)
        {
            var x = Values(outer * inner, 24);
            string s = $"[{outer}x{inner}]";
            var expected = new double[outer * inner];
            for (int r = 0; r < outer; r++)
            {
                var perturbed = new double[inner];
                for (int j = 0; j < inner; j++)
                {
                    int i = r * inner + j;
                    double u = (Hash((uint)i, key) >> 8) / 16777216.0 + 0.5 / 16777216.0;
                    perturbed[j] = (x[i] - Math.Log(-Math.Log(u))) / 0.8;
                }
                Array.Copy(Softmax(perturbed), 0, expected, r * inner, inner);
            }
            run.Op("GumbelSoftmax" + s, () =>
            {
                var o = run.Out(outer * inner);
                b.GumbelSoftmax(run.Up(x), o, outer, inner, 0.8f, seed);
                run.Check("GumbelSoftmax" + s, o, expected, 1e-3, 1e-6);
            });
        }
    }

    private static void Distances(IGpuBatchExecution b, Run run)
    {
        const int batch = 33, m = 4, nn = 6, dim = 13;
        var a = Values(batch * m, 25);
        var c = Values(batch * nn, 26);
        run.Op("BatchOuterProduct", () =>
        {
            var o = run.Out(batch * m * nn);
            b.BatchOuterProduct(run.Up(a), run.Up(c), o, batch, m, nn);
            run.Check("BatchOuterProduct", o, Enumerable.Range(0, batch * m * nn).Select(i =>
                (double)a[i / (m * nn) * m + i % (m * nn) / nn] * c[i / (m * nn) * nn + i % nn]).ToArray());
        });
        var p = Values(batch * dim, 27);
        var q = Values(batch * dim, 28);
        run.Op("CosineSimilarity", () =>
        {
            var o = run.Out(batch);
            b.CosineSimilarity(run.Up(p), run.Up(q), o, batch, dim);
            run.Check("CosineSimilarity", o, Enumerable.Range(0, batch).Select(r =>
            {
                double dot = 0, na = 0, nb = 0;
                for (int d = 0; d < dim; d++)
                {
                    dot += p[r * dim + d] * q[r * dim + d];
                    na += p[r * dim + d] * p[r * dim + d];
                    nb += q[r * dim + d] * q[r * dim + d];
                }
                return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-8);
            }).ToArray());
        });
        const int rowsA = 21, rowsB = 17;
        var pa = Values(rowsA * dim, 29);
        var pb = Values(rowsB * dim, 30);
        double Sq(int i, int j) => Enumerable.Range(0, dim).Sum(d => Math.Pow(pa[i * dim + d] - pb[j * dim + d], 2));
        run.Op("PairwiseDistance", () =>
        {
            var o = run.Out(rowsA * rowsB);
            b.PairwiseDistance(run.Up(pa), run.Up(pb), o, rowsA, rowsB, dim);
            run.Check("PairwiseDistance", o, Enumerable.Range(0, rowsA * rowsB).Select(i => Math.Sqrt(Sq(i / rowsB, i % rowsB))).ToArray());
        });
        run.Op("PairwiseDistanceSquared", () =>
        {
            var o = run.Out(rowsA * rowsB);
            b.PairwiseDistanceSquared(run.Up(pa), run.Up(pb), o, rowsA, rowsB, dim);
            run.Check("PairwiseDistanceSquared", o, Enumerable.Range(0, rowsA * rowsB).Select(i => Sq(i / rowsB, i % rowsB)).ToArray());
        });
    }

    private static void Plumbing(IGpuBatchExecution b, Run run)
    {
        const int n = 777;
        var x = Values(n, 31);
        run.Op("AllocateWorkspaceBuffer", () =>
        {
            using var ws = b.AllocateWorkspaceBuffer(n);
            if (ws.Size < n) run.Failures.Add($"AllocateWorkspaceBuffer: size {ws.Size} < {n}");
        });
        run.Op("BeginBatch/EndBatch", () =>
        {
            var input = run.Up(x);
            var mid = run.Out(n);
            var o = run.Out(n);
            var last = run.Out(n);
            // The same kernel twice on different buffers: a batch must not let the later binding leak into the
            // earlier recorded dispatch.
            if (b.SupportsBatchExecution) b.BeginBatch();
            b.ScalarMinusTensor(input, mid, 1f, n);
            b.InsertBarrier(mid);
            b.DivScalar(mid, o, 2f, n);
            b.InsertBarrier(o);
            b.ScalarMinusTensor(o, last, 5f, n);
            if (b.SupportsBatchExecution) b.EndBatch();
            run.Check("BeginBatch/EndBatch.mid", mid, x.Select(v => 1.0 - v).ToArray());
            run.Check("BeginBatch/EndBatch", o, x.Select(v => (1.0 - v) / 2.0).ToArray());
            run.Check("BeginBatch/EndBatch.last", last, x.Select(v => 5.0 - (1.0 - v) / 2.0).ToArray());
        });
        run.Op("BeginSecondaryStream/EndSecondaryStream", () =>
        {
            var input = run.Up(x);
            var o = run.Out(n);
            b.BeginSecondaryStream();
            b.ScalarMinusTensor(input, o, 3f, n);
            b.EndSecondaryStream();
            run.Check("BeginSecondaryStream/EndSecondaryStream", o, x.Select(v => 3.0 - v).ToArray());
        });
    }

    [SkippableFact]
    public void OpenCl_BatchExecution_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Vulkan_BatchExecution_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable,
            "Vulkan with a GLSL compiler is not available.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Cuda_BatchExecution_MatchesCpu()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Hip_BatchExecution_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Metal_BatchExecution_MatchesCpu()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        CheckAll(backend);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_BatchExecution_MatchesCpu()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();   // the real backend, executing on Dawn; shared, so not disposed
        CheckAll(backend);
    }
#endif

    /// <summary>
    /// WebGPU's WGSL ports on Dawn, driven through the dispatch WebGPU ships (<see cref="DawnWebGpuBatchBackend"/>).
    /// Needs AIDOTNET_WEBGPU_NODE_DIR and an OpenCL host for the buffers.
    /// </summary>
    private static DawnWebGpuBatchBackend? DawnOrSkip(out IDisposable owner)
    {
        owner = new NoOp();
        var dawn = DawnWgslLauncher.TryCreate();
        Skip.If(dawn is null, "Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).");
        var host = new AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend();
        if (!host.IsAvailable)
        {
            host.Dispose();
            dawn?.Dispose();
            Skip.If(true, "OpenCL (the buffer host) is not available.");
        }
        owner = new Owned(host, dawn);
        return dawn is null ? null : new DawnWebGpuBatchBackend(host, dawn);
    }

    private sealed class NoOp : IDisposable { public void Dispose() { } }

    private sealed class Owned : IDisposable
    {
        private readonly IDisposable _host;
        private readonly IDisposable? _dawn;
        public Owned(IDisposable host, IDisposable? dawn) { _host = host; _dawn = dawn; }
        public void Dispose() { _dawn?.Dispose(); _host.Dispose(); }
    }

    [SkippableFact]
    public void WebGpuWgsl_BatchExecution_MatchesCpu_OnDawn()
    {
        var backend = DawnOrSkip(out var owner);
        using (owner)
        {
            if (backend is null) return;
            CheckAll(backend);
        }
    }

}
#endif
