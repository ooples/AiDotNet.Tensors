#if NET6_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The Metal ops the first run on an Apple GPU (the macOS CI lane) and EveryKernelLookup_NamesAKernelDefinedInThatLibrary
/// found broken: kernels asked of a library that did not define them, scalars (clip bounds, exponent, fill value) never
/// bound, one size passed where a kernel reads two, and the tiled GEMM's threadgroup tiles never sized. Each op runs on
/// the device and is checked against a CPU reference. Skips off macOS.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class MetalFusedOpsParityTests
{
    private static float[] Values(int n, int salt, float scale = 2f)
    {
        var rng = new Random(salt);
        return Enumerable.Range(0, n).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
    }

    private static void Close(double[] expected, float[] actual, double tol, string what)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            double err = Math.Abs(actual[i] - expected[i]);
            Assert.True(err <= tol * Math.Max(1.0, Math.Abs(expected[i])),
                $"{what}[{i}]: got {actual[i]:R}, expected {expected[i]:R}");
        }
    }

    private static float[] Run(MetalBackend b, int outLength, Action<IGpuBuffer> launch)
    {
        using var o = b.AllocateBuffer(outLength);
        launch(o);
        return b.DownloadBuffer(o);
    }

    [SkippableFact]
    public void Metal_FusedOps_MatchCpu()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var b = new MetalBackend();
        Skip.IfNot(b.IsAvailable, "Metal is not available.");

        const int n = 517;   // not a multiple of any threadgroup width
        var x = Values(n, 1, 3f);
        var y = Values(n, 2, 3f);
        using var dx = b.AllocateBuffer(x);
        using var dy = b.AllocateBuffer(y);

        Close(new[] { x.Average(v => (double)v) }, Run(b, 1, o => b.ReduceMean(dx, o, n)), 1e-5, "ReduceMean");
        Close(x.Select(v => (double)Math.Min(1.5f, Math.Max(-0.5f, v))).ToArray(), Run(b, n, o => b.ClipKernel(dx, o, -0.5f, 1.5f, n)), 0, "ClipKernel");
        var pos = x.Select(Math.Abs).ToArray();
        using (var dpos = b.AllocateBuffer(pos))
            Close(pos.Select(v => Math.Pow(v, 1.7)).ToArray(), Run(b, n, o => b.PowScalar(dpos, o, 1.7f, n)), 1e-5, "PowScalar");
        Close(x.Select(v => (double)(v - (float)Math.Floor(v))).ToArray(), Run(b, n, o => b.FracKernel(dx, o, n)), 1e-6, "FracKernel");
        Close(x.Select(v => (double)(v - 0.75f)).ToArray(), Run(b, n, o => b.SubScalar(dx, o, 0.75f, n)), 1e-6, "SubScalar");

        Close(Enumerable.Range(0, 49).Select(i => i / 7 == i % 7 ? 1.0 : 0.0).ToArray(), Run(b, 49, o => b.EyeKernel(o, 7)), 0, "EyeKernel");
        var idx = new float[] { 2, 0, 4, 1 };
        using (var didx = b.AllocateBuffer(idx))
            Close(Enumerable.Range(0, 20).Select(i => (int)idx[i / 5] == i % 5 ? 1.0 : 0.0).ToArray(), Run(b, 20, o => b.OneHotKernel(didx, o, 4, 5)), 0, "OneHotKernel");

        var mask = x.Select((v, i) => i % 3 == 0 ? 1f : 0f).ToArray();
        using (var dmask = b.AllocateBuffer(mask))
            Close(x.Select((v, i) => mask[i] != 0 ? -9.0 : v).ToArray(), Run(b, n, o => b.MaskedFillKernel(dx, dmask, o, -9f, n)), 0, "MaskedFillKernel");

        var ea = new[] { 1f, float.NaN, 0f, -0f, 2f, float.PositiveInfinity };
        var eb = new[] { 1f, float.NaN, -0f, 0f, 3f, float.PositiveInfinity };
        using (var dea = b.AllocateBuffer(ea))
        using (var deb = b.AllocateBuffer(eb))
        {
            var eq = new double[] { 1, 0, 1, 1, 0, 1 };
            Close(eq, Run(b, 6, o => b.EqualsKernel(dea, deb, o, 6)), 0, "EqualsKernel");
            Close(eq.Select(v => 1 - v).ToArray(), Run(b, 6, o => b.NotEqualsKernel(dea, deb, o, 6)), 0, "NotEqualsKernel");
        }

        const int m = 11, k = 13;
        var a1 = Values(m, 3); var b1 = Values(k, 4);
        using (var da1 = b.AllocateBuffer(a1))
        using (var db1 = b.AllocateBuffer(b1))
            Close(Enumerable.Range(0, m * k).Select(i => (double)a1[i / k] * b1[i % k]).ToArray(), Run(b, m * k, o => b.OuterProduct(da1, db1, o, m, k)), 1e-6, "OuterProduct");

        const int batch = 9, dim = 57;
        Close(Enumerable.Range(0, batch).Select(r => Enumerable.Range(0, dim).Sum(c => (double)x[r * dim + c] * y[r * dim + c])).ToArray(),
            Run(b, batch, o => b.BatchDotProduct(dx, dy, o, batch, dim)), 1e-5, "BatchDotProduct");

        // Gated activations over [outer, 2 * half]: value half then gate half.
        const int outer = 7, half = 36;
        double Value(int i) => x[(i / half) * 2 * half + i % half];
        double Gate(int i) => x[(i / half) * 2 * half + half + i % half];
        static double Sigmoid(double v) => 1 / (1 + Math.Exp(-v));
        static double Gelu(double v) => 0.5 * v * (1 + Math.Tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
        var gi = Enumerable.Range(0, outer * half).ToArray();
        Close(gi.Select(i => Value(i) * Sigmoid(Gate(i))).ToArray(), Run(b, outer * half, o => b.GluForward(dx, o, outer, half)), 1e-5, "GluForward");
        Close(gi.Select(i => Value(i) * Gelu(Gate(i))).ToArray(), Run(b, outer * half, o => b.GeGluForward(dx, o, outer, half)), 1e-4, "GeGluForward");
        Close(gi.Select(i => Value(i) * Math.Max(0, Gate(i))).ToArray(), Run(b, outer * half, o => b.ReGluForward(dx, o, outer, half)), 1e-6, "ReGluForward");
        Close(gi.Select(i => Value(i) * Gate(i) * Sigmoid(Gate(i))).ToArray(), Run(b, outer * half, o => b.SwiGluForward(dx, o, outer, half)), 1e-5, "SwiGluForward");

        // Last-axis broadcasts: A [rows, cols], B [cols].
        const int rows = 11, cols = 47;
        var bb = Values(cols, 5).Select(v => v + 3f).ToArray();
        using (var dbb = b.AllocateBuffer(bb))
        {
            double A(int i) => x[i];
            var bi = Enumerable.Range(0, rows * cols).ToArray();
            Close(bi.Select(i => A(i) + bb[i % cols]).ToArray(), Run(b, rows * cols, o => b.BroadcastAddLast(dx, dbb, o, rows, cols)), 1e-6, "BroadcastAddLast");
            Close(bi.Select(i => A(i) - bb[i % cols]).ToArray(), Run(b, rows * cols, o => b.BroadcastSubLast(dx, dbb, o, rows, cols)), 1e-6, "BroadcastSubLast");
            Close(bi.Select(i => A(i) * bb[i % cols]).ToArray(), Run(b, rows * cols, o => b.BroadcastMulLast(dx, dbb, o, rows, cols)), 1e-6, "BroadcastMulLast");
            Close(bi.Select(i => A(i) / (bb[i % cols] + 1e-12)).ToArray(), Run(b, rows * cols, o => b.BroadcastDivLast(dx, dbb, o, rows, cols)), 1e-5, "BroadcastDivLast");
        }
    }

    [SkippableFact]
    public void Metal_ActivationAndLossBackwards_MatchCpu()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var b = new MetalBackend();
        Skip.IfNot(b.IsAvailable, "Metal is not available.");

        const int n = 301;
        var x = Values(n, 11, 4f);
        var g = Values(n, 12);
        using var dx = b.AllocateBuffer(x);
        using var dg = b.AllocateBuffer(g);

        const float alpha = 0.7f;
        var eluOut = x.Select(v => v > 0 ? v : alpha * (float)(Math.Exp(v) - 1)).ToArray();
        using (var deo = b.AllocateBuffer(eluOut))
            Close(x.Select((v, i) => v > 0 ? g[i] : g[i] * (eluOut[i] + (double)alpha)).ToArray(),
                Run(b, n, o => b.EluBackward(dg, dx, deo, o, alpha, n)), 1e-5, "EluBackward");
        static double Sig(double v) => 1 / (1 + Math.Exp(-v));
        Close(x.Select((v, i) => g[i] * (v * Sig(v) + Sig(v) * (1 - v * Sig(v)))).ToArray(), Run(b, n, o => b.SwishBackward(dg, dx, o, n)), 1e-4, "SwishBackward");
        Close(x.Select((v, i) =>
        {
            double sp = Math.Log(1 + Math.Exp(v)), t = Math.Tanh(sp);
            return g[i] * (t + v * (1 - t * t) * Sig(v));
        }).ToArray(), Run(b, n, o => b.MishBackward(dg, dx, o, n)), 1e-4, "MishBackward");
        Close(x.Select((v, i) => g[i] * (v <= -3 ? 0 : v >= 3 ? 1 : (2 * v + 3) / 6.0)).ToArray(), Run(b, n, o => b.HardswishBackward(dg, dx, o, n)), 1e-6, "HardswishBackward");

        var t01 = Values(n, 13).Select(v => (v + 2f) / 4f).ToArray();   // targets in (0, 1)
        using var dt = b.AllocateBuffer(t01);
        Close(x.Select((v, i) => Math.Max(v, 0) - v * t01[i] + Math.Log(1 + Math.Exp(-Math.Abs(v)))).ToArray(),
            Run(b, n, o => b.BceWithLogitsLoss(dx, dt, o, n)), 1e-5, "BceWithLogitsLoss");
        Close(x.Select((v, i) => t01[i] * (Math.Log(t01[i]) - v)).ToArray(), Run(b, n, o => b.KlDivLoss(dx, dt, o, n)), 1e-4, "KlDivLoss");
        const float invN = 1f / n;
        using var dscalar = b.AllocateBuffer(new[] { 1.3f });
        Close(x.Select((v, i) => 1.3 * Math.Sign(v - t01[i]) * invN).ToArray(), Run(b, n, o => b.L1LossBackward(dscalar, dx, dt, o, n, invN)), 1e-6, "L1LossBackward");
        Close(x.Select((v, i) => 1.3 * (Sig(v) - t01[i]) * invN).ToArray(), Run(b, n, o => b.BceWithLogitsBackward(dscalar, dx, dt, o, n, invN)), 1e-5, "BceWithLogitsBackward");
    }

    [SkippableFact]
    public void Metal_SoftmaxBackwardAndTiledGemm_MatchCpu()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var b = new MetalBackend();
        Skip.IfNot(b.IsAvailable, "Metal is not available.");

        // Softmax backward over rows of a non-power-of-two width (the reference candidate and the dispatched op).
        const int rows = 13, features = 37;
        var logits = Values(rows * features, 21);
        var y = new float[rows * features];
        for (int r = 0; r < rows; r++)
        {
            double max = Enumerable.Range(0, features).Max(c => logits[r * features + c]);
            double sum = Enumerable.Range(0, features).Sum(c => Math.Exp(logits[r * features + c] - max));
            for (int c = 0; c < features; c++) y[r * features + c] = (float)(Math.Exp(logits[r * features + c] - max) / sum);
        }
        var go = Values(rows * features, 22);
        var expected = new double[rows * features];
        for (int r = 0; r < rows; r++)
        {
            double dot = Enumerable.Range(0, features).Sum(c => (double)go[r * features + c] * y[r * features + c]);
            for (int c = 0; c < features; c++) expected[r * features + c] = y[r * features + c] * (go[r * features + c] - dot);
        }
        using (var dgo = b.AllocateBuffer(go))
        using (var dy = b.AllocateBuffer(y))
            Close(expected, Run(b, rows * features, o => b.SoftmaxBackward(dgo, dy, o, rows, features)), 1e-5, "SoftmaxBackward");

        // GEMM and A·Bᵀ at shapes that are not multiples of the 16 x 16 tile.
        const int M = 37, N = 21, K = 45;
        var a = Values(M * K, 23); var bm = Values(K * N, 24);
        var product = new double[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                product[i * N + j] = Enumerable.Range(0, K).Sum(p => (double)a[i * K + p] * bm[p * N + j]);
        var bt = new float[N * K];
        for (int p = 0; p < K; p++)
            for (int j = 0; j < N; j++) bt[j * K + p] = bm[p * N + j];
        using var da = b.AllocateBuffer(a);
        using var db = b.AllocateBuffer(bm);
        using var dbt = b.AllocateBuffer(bt);
        Close(product, Run(b, M * N, o => b.Gemm(da, db, o, M, N, K)), 1e-4, "Gemm");
        Close(product, Run(b, M * N, o => b.MatMulTransposed(da, dbt, o, M, N, K)), 1e-4, "MatMulTransposed");
    }
}
#endif
