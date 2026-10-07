using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using AiDotNet.Tensors.Tests.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Every CUDA tuned-kernel candidate (the established reference and each generated variant) against a CPU
/// reference computed in double, over shapes that straddle the lane and tile boundaries; plus the registry seam end
/// to end through the public backend API, and the windowed deterministic max-pool backward.
/// </summary>
[Collection(TunedKernelRegistryCollection.Name)]
public sealed class CudaTunedKernelTests
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

    private static void AssertClose(float[] expected, float[] actual, int count, string what)
    {
        double scale = 1e-30;
        for (int i = 0; i < count; i++) scale = Math.Max(scale, Math.Abs(expected[i]));
        for (int i = 0; i < count; i++)
        {
            double err = Math.Abs((double)actual[i] - expected[i]) / scale;
            Assert.True(err <= Tolerance, $"{what}: index {i} expected {expected[i]} actual {actual[i]} (rel {err:E2})");
        }
    }

    private static CudaBackend RequireBackend()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            Skip.If(true, "Requires an NVIDIA CUDA GPU.");
        }
        return backend;
    }

    private static IEnumerable<ITunedKernelCandidate<TArgs>> All<TArgs>(TunedKernelSlot<TArgs> slot) where TArgs : struct =>
        slot.CandidateIds.Select(id => slot.Candidate(id)).OfType<ITunedKernelCandidate<TArgs>>();

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
            var args = new CudaSoftmaxArgs(input, output, rows, n);
            foreach (var c in All(backend.SoftmaxSlot).Where(c => c.IsApplicable(shape)))
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
            var args = new CudaSoftmaxBackwardArgs(dyb, yb, dx, rows, n);
            foreach (var c in All(backend.SoftmaxBackwardSlot).Where(c => c.IsApplicable(shape)))
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

            var fwd = new CudaLayerNormArgs(xb, yb, gb, bb, mb, ib, rows, n, eps);
            foreach (var c in All(backend.LayerNormSlot).Where(c => c.IsApplicable(shape)))
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
            var bwd = new CudaNormBackwardArgs(dyb, xb, gb, mcpu, icpu, dxb, null, rows, n);
            foreach (var c in All(backend.LayerNormBackwardSlot).Where(c => c.IsApplicable(shape)))
            {
                c.Execute(bwd);
                backend.Synchronize();
                AssertClose(dx, backend.DownloadBuffer(dxb), dx.Length, $"{c.Id} dx {rows}x{n}");
            }
            var par = new CudaNormBackwardArgs(dyb, xb, null, mcpu, icpu, dgb, dbb, rows, n);
            foreach (var c in All(backend.LayerNormGradParametersSlot).Where(c => c.IsApplicable(shape)))
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
            var args = new CudaNormBackwardArgs(dyb, xb, null, rb, null, dgb, null, rows, n);
            foreach (var c in All(backend.RmsNormGradGammaSlot).Where(c => c.IsApplicable(shape)))
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
        Assert.Equal(1 + CudaTunedRowKernels.RowLanes.Length, backend.SoftmaxSlot.CandidateIds.Count);
        Assert.Equal(1 + CudaTunedRowKernels.ColumnRowLanes.Length, backend.LayerNormGradParametersSlot.CandidateIds.Count);

        const int rows = 2048, n = 64;
        float[] x = Values(rows * n, 3, 4f);
        using var input = backend.AllocateBuffer(x);
        using var a = backend.AllocateBuffer(x.Length);
        using var b = backend.AllocateBuffer(x.Length);
        var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);
        foreach (var c in All(backend.SoftmaxSlot).Where(c => c.IsApplicable(shape)))
        {
            Assert.True(c.IsDeterministic);
            c.Execute(new CudaSoftmaxArgs(input, a, rows, n));
            c.Execute(new CudaSoftmaxArgs(input, b, rows, n));
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
            const int rows = 8192, n = 32;
            float[] x = Values(rows * n, 99, 5f);
            using var input = backend.AllocateBuffer(x);
            using var viaRegistry = backend.AllocateBuffer(x.Length);
            using var viaReference = backend.AllocateBuffer(x.Length);
            backend.Softmax(input, viaRegistry, rows, n);
            (backend.SoftmaxSlot.Candidate("nvrtc.softmax.block256") ?? throw new InvalidOperationException("reference missing")).Execute(new CudaSoftmaxArgs(input, viaReference, rows, n));
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

    [SkippableTheory]
    [InlineData(2, 2, 2, 2, 0, 0, 28, 28)]
    [InlineData(3, 3, 1, 1, 1, 1, 13, 17)]
    [InlineData(3, 3, 2, 2, 1, 1, 15, 15)]
    [InlineData(3, 2, 3, 1, 0, 1, 10, 9)]
    [InlineData(1, 1, 2, 2, 0, 0, 7, 7)]
    [InlineData(5, 5, 1, 1, 2, 2, 6, 6)]
    public void MaxPoolBackwardDeterministic_Windowed_MatchesTheFullScanOnCpu(
        int kh, int kw, int sh, int sw, int ph, int pw, int h, int w)
    {
        using var backend = RequireBackend();
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.GpuDeterminism.IsActive, "Deterministic mode is required.");
        const int batch = 3, channels = 4;
        int oh = (h + 2 * ph - kh) / sh + 1, ow = (w + 2 * pw - kw) / sw + 1;
        float[] x = Values(batch * channels * h * w, kh * 100 + sh, 1f);
        // Ties (equal values) and an all -inf plane exercise the forward's argmax order and its index-0 fallback.
        for (int i = 0; i < x.Length; i += 7) x[i] = 0.25f;
        for (int i = 0; i < h * w; i++) x[i] = float.NegativeInfinity;
        float[] dy = Values(batch * channels * oh * ow, 77, 1f);

        using var xb = backend.AllocateBuffer(x);
        using var yb = backend.AllocateBuffer(batch * channels * oh * ow);
        using var idx = backend.AllocateIntBuffer(batch * channels * oh * ow);
        using var dyb = backend.AllocateBuffer(dy);
        using var dxb = backend.AllocateBuffer(x.Length);
        backend.MaxPool2D(xb, yb, idx, batch, channels, h, w, oh, ow, kh, kw, sh, sw, ph, pw);
        backend.MaxPool2DBackward(dyb, idx, dxb, batch, channels, h, w, oh, ow, kh, kw, sh, sw, ph, pw);
        backend.Synchronize();
        var indices = new int[batch * channels * oh * ow];
        backend.DownloadIntBuffer(idx, indices);
        float[] actual = backend.DownloadBuffer(dxb);

        // The pre-windowing kernel's definition: a full ascending scan of the plane, float accumulation.
        for (int plane = 0; plane < batch * channels; plane++)
            for (int cell = 0; cell < h * w; cell++)
            {
                float sum = 0f;
                for (int o = 0; o < oh * ow; o++)
                    if (indices[plane * oh * ow + o] == cell) sum += dy[plane * oh * ow + o];
                Assert.Equal(sum, actual[plane * h * w + cell]);
            }
    }
}
