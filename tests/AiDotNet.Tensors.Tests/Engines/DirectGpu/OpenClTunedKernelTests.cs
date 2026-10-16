#if !NET462
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using AiDotNet.Tensors.Tests.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Every OpenCL tuned-kernel candidate (#1114: the established reference and each generated variant) against a CPU
/// reference computed in double, over shapes that straddle the lane and tile boundaries; plus the registry seam end
/// to end through the public backend API. The OpenCL port of CudaTunedKernelTests.
/// </summary>
[Collection(TunedKernelRegistryCollection.Name)]
public sealed class OpenClTunedKernelTests
{
    private const double Tolerance = 2e-5;

    // rows x columns: degenerate, ragged, exactly a lane count, one past it, multi-pass rows, and the parity shapes.
    private static readonly (int Rows, int N)[] Shapes =
    {
        (1, 1), (3, 5), (7, 8), (9, 16), (17, 31), (33, 32), (31, 33), (5, 64), (65, 65), (11, 100),
        (129, 257), (3, 511), (2, 1024), (2048, 64), (8192, 32),
    };

    private static float[] Values(int count, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return a;
    }

    // termScale: the magnitude of the terms an output is a difference of. An output that cancels to (near) zero, such as
    // LayerNorm's dx on a one-column row, is only accurate relative to those terms, not to its own value.
    private static void AssertClose(float[] expected, float[] actual, int count, string what, double termScale = 1e-30)
    {
        double scale = termScale;
        for (int i = 0; i < count; i++) scale = Math.Max(scale, Math.Abs(expected[i]));
        for (int i = 0; i < count; i++)
        {
            double err = Math.Abs((double)actual[i] - expected[i]) / scale;
            Assert.True(err <= Tolerance, $"{what}: index {i} expected {expected[i]} actual {actual[i]} (rel {err:E2})");
        }
    }

    private static OpenClBackend RequireBackend()
    {
        Skip.IfNot(OpenClBackend.IsGpuAvailable, "Requires an OpenCL GPU.");
        var backend = new OpenClBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            Skip.If(true, "Requires an OpenCL GPU.");
        }
        return backend;
    }

    private static IEnumerable<ITunedKernelCandidate<TArgs>> All<TArgs>(TunedKernelSlot<TArgs> slot) where TArgs : struct =>
        slot.CandidateIds.Select(id => slot.Candidate(id)).OfType<ITunedKernelCandidate<TArgs>>();

    // The candidates a loop checks for this shape. Never empty, and always including the reference (the slot's first
    // candidate): a regression that made nothing applicable would otherwise leave the loop body unrun and the test green.
    private static List<ITunedKernelCandidate<TArgs>> Applicable<TArgs>(TunedKernelSlot<TArgs> slot, TunedShape shape) where TArgs : struct
    {
        var list = All(slot).Where(c => c.IsApplicable(shape)).ToList();
        Assert.True(list.Count > 0, $"no {slot.Op} candidate is applicable to {shape}");
        Assert.Contains(list, c => string.Equals(c.Id, slot.CandidateIds[0], StringComparison.Ordinal));
        return list;
    }

    [SkippableFact]
    public void Softmax_EveryCandidate_MatchesCpu()
    {
        using var backend = RequireBackend();
        foreach (var (rows, n) in Shapes)
        {
            float[] x = Values(rows * n, rows * 31 + n, 6f);
            var expected = new float[x.Length];
            for (int r = 0; r < rows; r++)
            {
                double m = double.NegativeInfinity, s = 0;
                for (int i = 0; i < n; i++) m = Math.Max(m, x[r * n + i]);
                for (int i = 0; i < n; i++) s += Math.Exp(x[r * n + i] - m);
                for (int i = 0; i < n; i++) expected[r * n + i] = (float)(Math.Exp(x[r * n + i] - m) / s);
            }
            using var input = backend.AllocateBuffer(x);
            using var output = backend.AllocateBuffer(x.Length);
            var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);
            var args = new TunedSoftmaxArgs(input, output, rows, n);
            foreach (var c in Applicable(backend.SoftmaxSlot, shape))
            {
                c.Execute(args);
                backend.Synchronize();
                AssertClose(expected, backend.DownloadBuffer(output), x.Length, $"{c.Id} {rows}x{n}");
            }
        }
    }

    [SkippableFact]
    public void SoftmaxBackward_EveryCandidate_MatchesCpu()
    {
        using var backend = RequireBackend();
        foreach (var (rows, n) in Shapes)
        {
            float[] y = Values(rows * n, 5 + rows, 1f).Select(Math.Abs).ToArray();
            float[] dy = Values(rows * n, 7 + n, 1f);
            var expected = new float[y.Length];
            for (int r = 0; r < rows; r++)
            {
                double dot = 0;
                for (int i = 0; i < n; i++) dot += (double)dy[r * n + i] * y[r * n + i];
                for (int i = 0; i < n; i++) expected[r * n + i] = (float)(y[r * n + i] * (dy[r * n + i] - dot));
            }
            using var yb = backend.AllocateBuffer(y);
            using var dyb = backend.AllocateBuffer(dy);
            using var dx = backend.AllocateBuffer(y.Length);
            var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);
            var args = new TunedSoftmaxBackwardArgs(dyb, yb, dx, rows, n);
            foreach (var c in Applicable(backend.SoftmaxBackwardSlot, shape))
            {
                c.Execute(args);
                backend.Synchronize();
                AssertClose(expected, backend.DownloadBuffer(dx), y.Length, $"{c.Id} {rows}x{n}");
            }
        }
    }

    [SkippableFact]
    public void LayerNormForwardAndBackward_EveryCandidate_MatchesCpu()
    {
        using var backend = RequireBackend();
        const float eps = 1e-5f;
        foreach (var (rows, n) in Shapes)
        {
            float[] x = Values(rows * n, 11 + rows, 3f);
            float[] gamma = Values(n, 13 + n, 1f);
            float[] beta = Values(n, 17 + n, 1f);
            float[] dy = Values(rows * n, 19 + rows, 1f);
            var mean = new float[rows];
            var inv = new float[rows];
            var y = new float[rows * n];
            var dx = new float[rows * n];
            var dGamma = new float[n];
            var dBeta = new float[n];
            var dGammaD = new double[n];
            var dBetaD = new double[n];
            double dxTermScale = 1e-30; // max over rows of invStd * |dy * gamma|: dx is a difference of such terms
            for (int r = 0; r < rows; r++)
            {
                double m = 0, v = 0;
                for (int i = 0; i < n; i++) m += x[r * n + i];
                m /= n;
                for (int i = 0; i < n; i++) v += (x[r * n + i] - m) * (x[r * n + i] - m);
                double iv = 1.0 / Math.Sqrt(v / n + eps);
                mean[r] = (float)m;
                inv[r] = (float)iv;
                double sumDy = 0, sumDyXmu = 0;
                for (int i = 0; i < n; i++)
                {
                    double xhat = (x[r * n + i] - m) * iv;
                    y[r * n + i] = (float)(gamma[i] * xhat + beta[i]);
                    double g = (double)dy[r * n + i] * gamma[i];
                    dxTermScale = Math.Max(dxTermScale, Math.Abs(g) * iv);
                    sumDy += g;
                    sumDyXmu += g * (x[r * n + i] - m);
                    dGammaD[i] += dy[r * n + i] * xhat;
                    dBetaD[i] += dy[r * n + i];
                }
                for (int i = 0; i < n; i++)
                {
                    double xmu = x[r * n + i] - m;
                    double g = (double)dy[r * n + i] * gamma[i];
                    dx[r * n + i] = (float)(iv * (g - (sumDy + xmu * iv * iv * sumDyXmu) / n));
                }
            }
            for (int i = 0; i < n; i++) { dGamma[i] = (float)dGammaD[i]; dBeta[i] = (float)dBetaD[i]; }

            using var xb = backend.AllocateBuffer(x);
            using var gb = backend.AllocateBuffer(gamma);
            using var bb = backend.AllocateBuffer(beta);
            using var dyb = backend.AllocateBuffer(dy);
            using var yb = backend.AllocateBuffer(rows * n);
            using var mb = backend.AllocateBuffer(rows);
            using var ib = backend.AllocateBuffer(rows);
            using var dxb = backend.AllocateBuffer(rows * n);
            using var dgb = backend.AllocateBuffer(n);
            using var dbb = backend.AllocateBuffer(n);
            var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);

            var fwd = new TunedLayerNormArgs(xb, yb, gb, bb, mb, ib, rows, n, eps);
            foreach (var c in Applicable(backend.LayerNormSlot, shape))
            {
                c.Execute(fwd);
                backend.Synchronize();
                AssertClose(y, backend.DownloadBuffer(yb), y.Length, $"{c.Id} y {rows}x{n}");
                AssertClose(mean, backend.DownloadBuffer(mb), rows, $"{c.Id} mean {rows}x{n}");
                AssertClose(inv, backend.DownloadBuffer(ib), rows, $"{c.Id} invStd {rows}x{n}");
            }

            // Backward uses the CPU statistics so each kernel is checked in isolation.
            using var mcpu = backend.AllocateBuffer(mean);
            using var icpu = backend.AllocateBuffer(inv);
            var bwd = new TunedNormBackwardArgs(dyb, xb, gb, mcpu, icpu, dxb, null, rows, n);
            foreach (var c in Applicable(backend.LayerNormBackwardSlot, shape))
            {
                c.Execute(bwd);
                backend.Synchronize();
                AssertClose(dx, backend.DownloadBuffer(dxb), dx.Length, $"{c.Id} dx {rows}x{n}", dxTermScale);
            }
            var par = new TunedNormBackwardArgs(dyb, xb, null, mcpu, icpu, dgb, dbb, rows, n);
            foreach (var c in Applicable(backend.LayerNormGradParametersSlot, shape))
            {
                c.Execute(par);
                backend.Synchronize();
                AssertClose(dGamma, backend.DownloadBuffer(dgb), n, $"{c.Id} dGamma {rows}x{n}");
                AssertClose(dBeta, backend.DownloadBuffer(dbb), n, $"{c.Id} dBeta {rows}x{n}");
            }
        }
    }

    [SkippableFact]
    public void RmsNormGradGamma_EveryCandidate_MatchesCpu()
    {
        using var backend = RequireBackend();
        foreach (var (rows, n) in Shapes)
        {
            float[] x = Values(rows * n, 23 + rows, 2f);
            float[] dy = Values(rows * n, 29 + n, 1f);
            float[] rms = Values(rows, 31, 1f).Select(v => Math.Abs(v) + 0.5f).ToArray();
            var expected = new float[n];
            for (int i = 0; i < n; i++)
            {
                double s = 0;
                for (int r = 0; r < rows; r++) s += (double)dy[r * n + i] * x[r * n + i] / rms[r];
                expected[i] = (float)s;
            }
            using var xb = backend.AllocateBuffer(x);
            using var dyb = backend.AllocateBuffer(dy);
            using var rb = backend.AllocateBuffer(rms);
            using var dgb = backend.AllocateBuffer(n);
            var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);
            var args = new TunedNormBackwardArgs(dyb, xb, null, rb, null, dgb, null, rows, n);
            foreach (var c in Applicable(backend.RmsNormGradGammaSlot, shape))
            {
                c.Execute(args);
                backend.Synchronize();
                AssertClose(expected, backend.DownloadBuffer(dgb), n, $"{c.Id} {rows}x{n}");
            }
        }
    }

    [SkippableFact]
    public void GeneratedVariants_AreCompiled_AndDeterministic()
    {
        using var backend = RequireBackend();
        foreach (string id in backend.SoftmaxSlot.CandidateIds.Skip(1))
            Assert.StartsWith("generated.softmax.lanes", id);
        Assert.Equal(1 + OpenClTunedRowKernels.RowLanes.Length, backend.SoftmaxSlot.CandidateIds.Count);
        // Reference and one generated variant per column-row lane count (OpenCL has no direct-PTX candidate).
        Assert.Equal(1 + OpenClTunedRowKernels.ColumnRowLanes.Length, backend.LayerNormGradParametersSlot.CandidateIds.Count);

        const int rows = 2048, n = 64;
        float[] x = Values(rows * n, 3, 4f);
        using var input = backend.AllocateBuffer(x);
        using var a = backend.AllocateBuffer(x.Length);
        using var b = backend.AllocateBuffer(x.Length);
        var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);
        // Every variant applies at d = 64, so a generated kernel that failed to compile (inapplicable) fails here.
        Assert.Equal(backend.SoftmaxSlot.CandidateIds.Count, Applicable(backend.SoftmaxSlot, shape).Count);
        Assert.Equal(backend.LayerNormGradParametersSlot.CandidateIds.Count, Applicable(backend.LayerNormGradParametersSlot, shape).Count);
        foreach (var c in Applicable(backend.SoftmaxSlot, shape))
        {
            Assert.True(c.IsDeterministic);
            c.Execute(new TunedSoftmaxArgs(input, a, rows, n));
            c.Execute(new TunedSoftmaxArgs(input, b, rows, n));
            backend.Synchronize();
            Assert.Equal(backend.DownloadBuffer(a), backend.DownloadBuffer(b));
        }
    }

    [SkippableFact]
    public void PublicSoftmax_ThroughTheRegistry_MatchesTheReferenceAndCachesADecision()
    {
        using var backend = RequireBackend();
        TunedKernelMode? previousMode = TunedKernelPolicy.ModeOverride;
        bool previousPersist = TunedKernelPolicy.PersistDecisions;
        double previousBudget = TunedKernelPolicy.BudgetMilliseconds;
        TunedKernelPolicy.ModeOverride = TunedKernelMode.Tune;
        TunedKernelPolicy.PersistDecisions = false;
        TunedKernelPolicy.BudgetMilliseconds = 60_000;
        try
        {
            // A shape no shipped profile covers: the registry must measure it, and the in-place call must not.
            const int rows = 8192, n = 48;
            float[] x = Values(rows * n, 99, 5f);
            using var input = backend.AllocateBuffer(x);
            using var viaRegistry = backend.AllocateBuffer(x.Length);
            using var viaReference = backend.AllocateBuffer(x.Length);
            backend.Softmax(input, viaRegistry, rows, n);
            (backend.SoftmaxSlot.Candidate("opencl.softmax.group") ?? throw new InvalidOperationException("reference missing")).Execute(new TunedSoftmaxArgs(input, viaReference, rows, n));
            backend.Synchronize();
            AssertClose(backend.DownloadBuffer(viaReference), backend.DownloadBuffer(viaRegistry), x.Length, "registry softmax");
            var decision = Assert.Single(backend.SoftmaxSlot.CachedDecisions);
            Assert.Equal(TunedKernelOp.Softmax, decision.Op);

            // In place: the registry must not measure (a second in-place run would corrupt the data) and the result
            // must equal one reference application.
            using var inPlace = backend.AllocateBuffer(x);
            backend.SoftmaxSlot.InvalidateDecisions();
            backend.Softmax(inPlace, inPlace, rows, n);
            backend.Synchronize();
            AssertClose(backend.DownloadBuffer(viaReference), backend.DownloadBuffer(inPlace), x.Length, "in-place softmax");
            Assert.Empty(backend.SoftmaxSlot.CachedDecisions);
        }
        finally
        {
            TunedKernelPolicy.ModeOverride = previousMode;
            TunedKernelPolicy.PersistDecisions = previousPersist;
            TunedKernelPolicy.BudgetMilliseconds = previousBudget;
        }
    }


    [SkippableFact]
    public void Tuning_ReadbacksAreAttributedToTheRegistry_NotTheOp()
    {
        var backend = RequireBackend();
        TunedKernelMode? previousMode = TunedKernelPolicy.ModeOverride;
        bool previousPersist = TunedKernelPolicy.PersistDecisions;
        TunedKernelPolicy.ModeOverride = TunedKernelMode.Tune;
        TunedKernelPolicy.PersistDecisions = false;
        try
        {
            // A shape no profile covers, so the slot measures it (and snapshots candidate outputs to the host).
            const int rows = 777, n = 40;
            float[] x = Values(rows * n, 5, 3f);
            using var input = backend.AllocateBuffer(x);
            using var output = backend.AllocateBuffer(x.Length);
            backend.SoftmaxSlot.InvalidateDecisions();
            long tuningBefore = GpuLaunchProbe.TuningReadbacks;
            // Thread-scoped: transfers other test classes make in parallel do not land in it.
            using var scope = AiDotNet.Tensors.Engines.Diagnostics.GpuResidencyScope.Begin();
            backend.Softmax(input, output, rows, n);
            var reason = Assert.Single(backend.SoftmaxSlot.CachedDecisions).Reason;
            Assert.True(reason is TunedKernelDecisionReason.Tuned or TunedKernelDecisionReason.ReferenceWon or TunedKernelDecisionReason.Inconclusive, $"measured decision expected, got {reason}");
            Assert.True(GpuLaunchProbe.TuningReadbacks > tuningBefore, "the gate snapshots candidate outputs, so tuning must read back");
            Assert.Equal(0, scope.Downloads);
        }
        finally
        {
            TunedKernelPolicy.ModeOverride = previousMode;
            TunedKernelPolicy.PersistDecisions = previousPersist;
        }
    }
#if NET6_0_OR_GREATER // the shared check lives in MetalTunedKernelTests
    [SkippableFact]
    public void PublicLayerNormAndRmsNormBackward_MatchTheCpu()
    {
        using var backend = RequireBackend();
        MetalTunedKernelTests.CheckPublicNormBackward(backend);
    }
#endif

    // (batch, Cin, H, W, Cout, k, stride, pad, dilation): Winograd's 3x3/s1, 1x1, 5x5, strided, dilated, batched.
    private static readonly (int B, int C, int H, int W, int O, int K, int S, int P, int D)[] ConvShapes =
    {
        (1, 3, 8, 8, 4, 3, 1, 1, 1), (2, 4, 9, 7, 5, 3, 1, 0, 1), (1, 8, 6, 6, 3, 1, 1, 0, 1),
        (2, 3, 11, 10, 2, 5, 2, 2, 1), (1, 2, 12, 12, 3, 3, 1, 2, 2), (3, 5, 7, 9, 6, 3, 2, 1, 1),
    };

    [SkippableFact]
    public void Conv2DForward_EveryCandidate_MatchesCpu()
    {
        var backend = RequireBackend();
        foreach (var (bt, c, h, w, o, k, s, p, d) in ConvShapes)
        {
            int oh = (h + 2 * p - d * (k - 1) - 1) / s + 1, ow = (w + 2 * p - d * (k - 1) - 1) / s + 1;
            float[] x = Values(bt * c * h * w, bt * 13 + k, 2f), wt = Values(o * c * k * k, o * 7 + s, 1f);
            var expected = new float[bt * o * oh * ow];
            for (int b = 0; b < bt; b++)
                for (int co = 0; co < o; co++)
                    for (int y = 0; y < oh; y++)
                        for (int z = 0; z < ow; z++)
                        {
                            double acc = 0;
                            for (int ci = 0; ci < c; ci++)
                                for (int kh = 0; kh < k; kh++)
                                    for (int kw = 0; kw < k; kw++)
                                    {
                                        int ih = y * s - p + kh * d, iw = z * s - p + kw * d;
                                        if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                        acc += (double)x[((b * c + ci) * h + ih) * w + iw] * wt[((co * c + ci) * k + kh) * k + kw];
                                    }
                            expected[((b * o + co) * oh + y) * ow + z] = (float)acc;
                        }
            using var xb = backend.AllocateBuffer(x);
            using var wb = backend.AllocateBuffer(wt);
            using var yb = backend.AllocateBuffer(expected.Length);
            var args = new TunedConv2DArgs(xb, wb, yb, bt, c, h, w, o, oh, ow, k, k, s, s, p, p, d, d);
            foreach (var cand in Applicable(backend.Conv2DForwardSlot, args.Shape()))
            {
                cand.Execute(args);
                backend.Synchronize();
                AssertClose(expected, backend.DownloadBuffer(yb), expected.Length, $"{cand.Id} conv {bt}x{c}x{h}x{w} k{k} s{s} p{p} d{d}", termScale: 1.0);
            }
            // Every candidate applies to these small shapes, so a variant that silently dropped out fails here.
            Assert.Equal(backend.Conv2DForwardSlot.CandidateIds.Count, Applicable(backend.Conv2DForwardSlot, args.Shape()).Count
                + (k > 7 ? 1 : 0));
        }
    }
}
#endif
