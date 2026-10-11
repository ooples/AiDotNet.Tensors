using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using AiDotNet.Tensors.Tests.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1115: every Vulkan tuned-kernel candidate (the established reference and each generated variant) against a CPU
/// reference computed in double, over shapes that straddle the lane and tile boundaries; the registry seam end to end
/// through the public backend API with each candidate forced in turn; the device timestamp clock; and the shipped
/// profile.
/// </summary>
[Collection(TunedKernelRegistryCollection.Name)]
public sealed class VulkanTunedKernelTests
{
    private const double Tolerance = 2e-5;

    // rows x columns: degenerate, ragged, exactly a lane count, one past it, multi-pass rows, a row count that leaves
    // a workgroup partly empty, and the parity shapes.
    private static readonly (int Rows, int N)[] Shapes =
    {
        (1, 1), (3, 5), (7, 8), (9, 16), (17, 31), (33, 32), (31, 33), (5, 64), (65, 65), (11, 100),
        (129, 257), (3, 511), (2, 1024), (5, 2049), (2048, 64), (8192, 32),
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

    // The backend is a process-wide singleton other test classes share, so nothing here disposes it.
    private static VulkanBackend RequireBackend()
    {
        VulkanBackend? backend = null;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize() || !backend.IsGlslCompilerAvailable) backend = null;
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            backend = null;
        }
        Skip.If(backend is null, "No Vulkan device with the runtime GLSL compiler.");
        return backend!;
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

    // Every generated candidate must have been exercised by some shape, or the shape list no longer covers it.
    private static void AssertEveryCandidateRan<TArgs>(TunedKernelSlot<TArgs> slot, HashSet<string> ran) where TArgs : struct
    {
        foreach (string id in slot.CandidateIds)
            Assert.True(ran.Contains(id), $"{slot.Op} candidate {id} was applicable to none of the test shapes");
    }

    private static float[] SoftmaxOnCpu(float[] x, int rows, int n)
    {
        var expected = new float[x.Length];
        for (int r = 0; r < rows; r++)
        {
            double m = double.NegativeInfinity, s = 0;
            for (int i = 0; i < n; i++) m = Math.Max(m, x[r * n + i]);
            for (int i = 0; i < n; i++) s += Math.Exp(x[r * n + i] - m);
            for (int i = 0; i < n; i++) expected[r * n + i] = (float)(Math.Exp(x[r * n + i] - m) / s);
        }
        return expected;
    }

    [SkippableFact]
    public void Softmax_EveryCandidate_MatchesCpu()
    {
        var backend = RequireBackend();
        var ran = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (rows, n) in Shapes)
        {
            float[] x = Values(rows * n, rows * 31 + n, 6f);
            float[] expected = SoftmaxOnCpu(x, rows, n);
            using var input = backend.AllocateBuffer(x);
            using var output = backend.AllocateBuffer(x.Length);
            var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);
            var args = new VulkanSoftmaxArgs(input, output, rows, n);
            foreach (var c in Applicable(backend.SoftmaxSlot, shape))
            {
                backend.Fill(output, float.NaN, x.Length);
                c.Execute(args);
                AssertClose(expected, backend.DownloadBuffer(output), x.Length, $"{c.Id} {rows}x{n}");
                ran.Add(c.Id);
            }
        }
        AssertEveryCandidateRan(backend.SoftmaxSlot, ran);
    }

    [SkippableFact]
    public void SoftmaxBackward_EveryCandidate_MatchesCpu()
    {
        var backend = RequireBackend();
        var ran = new HashSet<string>(StringComparer.Ordinal);
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
            var args = new VulkanSoftmaxBackwardArgs(dyb, yb, dx, rows, n);
            foreach (var c in Applicable(backend.SoftmaxBackwardSlot, shape))
            {
                backend.Fill(dx, float.NaN, y.Length);
                c.Execute(args);
                AssertClose(expected, backend.DownloadBuffer(dx), y.Length, $"{c.Id} {rows}x{n}");
                ran.Add(c.Id);
            }
        }
        AssertEveryCandidateRan(backend.SoftmaxBackwardSlot, ran);
    }

    private sealed class LayerNormCase
    {
        public int Rows, N;
        public float[] X = Array.Empty<float>(), Gamma = Array.Empty<float>(), Beta = Array.Empty<float>(), Dy = Array.Empty<float>();
        public float[] Mean = Array.Empty<float>(), Inv = Array.Empty<float>(), Y = Array.Empty<float>(), Dx = Array.Empty<float>();
        public float[] DGamma = Array.Empty<float>(), DBeta = Array.Empty<float>();
        public double DxTermScale = 1e-30; // max over rows of invStd * |dy * gamma|: dx is a difference of such terms
    }

    private static LayerNormCase LayerNormOnCpu(int rows, int n, float eps)
    {
        var c = new LayerNormCase
        {
            Rows = rows, N = n,
            X = Values(rows * n, 11 + rows, 3f), Gamma = Values(n, 13 + n, 1f), Beta = Values(n, 17 + n, 1f),
            Dy = Values(rows * n, 19 + rows, 1f),
            Mean = new float[rows], Inv = new float[rows], Y = new float[rows * n], Dx = new float[rows * n],
            DGamma = new float[n], DBeta = new float[n],
        };
        var dGammaD = new double[n];
        var dBetaD = new double[n];
        for (int r = 0; r < rows; r++)
        {
            double m = 0, v = 0;
            for (int i = 0; i < n; i++) m += c.X[r * n + i];
            m /= n;
            for (int i = 0; i < n; i++) v += (c.X[r * n + i] - m) * (c.X[r * n + i] - m);
            double iv = 1.0 / Math.Sqrt(v / n + eps);
            c.Mean[r] = (float)m;
            c.Inv[r] = (float)iv;
            double sumDy = 0, sumDyXmu = 0;
            for (int i = 0; i < n; i++)
            {
                double xhat = (c.X[r * n + i] - m) * iv;
                c.Y[r * n + i] = (float)(c.Gamma[i] * xhat + c.Beta[i]);
                double g = (double)c.Dy[r * n + i] * c.Gamma[i];
                c.DxTermScale = Math.Max(c.DxTermScale, Math.Abs(g) * iv);
                sumDy += g;
                sumDyXmu += g * (c.X[r * n + i] - m);
                dGammaD[i] += c.Dy[r * n + i] * xhat;
                dBetaD[i] += c.Dy[r * n + i];
            }
            for (int i = 0; i < n; i++)
            {
                double xmu = c.X[r * n + i] - m;
                double g = (double)c.Dy[r * n + i] * c.Gamma[i];
                c.Dx[r * n + i] = (float)(iv * (g - (sumDy + xmu * iv * iv * sumDyXmu) / n));
            }
        }
        for (int i = 0; i < n; i++) { c.DGamma[i] = (float)dGammaD[i]; c.DBeta[i] = (float)dBetaD[i]; }
        return c;
    }

    [SkippableFact]
    public void LayerNormForwardAndBackward_EveryCandidate_MatchesCpu()
    {
        var backend = RequireBackend();
        const float eps = 1e-5f;
        var ranForward = new HashSet<string>(StringComparer.Ordinal);
        var ranBackward = new HashSet<string>(StringComparer.Ordinal);
        var ranParams = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (rows, n) in Shapes)
        {
            var k = LayerNormOnCpu(rows, n, eps);
            using var xb = backend.AllocateBuffer(k.X);
            using var gb = backend.AllocateBuffer(k.Gamma);
            using var bb = backend.AllocateBuffer(k.Beta);
            using var dyb = backend.AllocateBuffer(k.Dy);
            using var yb = backend.AllocateBuffer(rows * n);
            using var mb = backend.AllocateBuffer(rows);
            using var ib = backend.AllocateBuffer(rows);
            using var dxb = backend.AllocateBuffer(rows * n);
            using var dgb = backend.AllocateBuffer(n);
            using var dbb = backend.AllocateBuffer(n);
            var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);

            var fwd = new VulkanLayerNormArgs(xb, yb, gb, bb, mb, ib, rows, n, eps);
            foreach (var c in Applicable(backend.LayerNormSlot, shape))
            {
                backend.Fill(yb, float.NaN, rows * n);
                backend.Fill(mb, float.NaN, rows);
                backend.Fill(ib, float.NaN, rows);
                c.Execute(fwd);
                AssertClose(k.Y, backend.DownloadBuffer(yb), k.Y.Length, $"{c.Id} y {rows}x{n}");
                AssertClose(k.Mean, backend.DownloadBuffer(mb), rows, $"{c.Id} mean {rows}x{n}", 1e-3);
                AssertClose(k.Inv, backend.DownloadBuffer(ib), rows, $"{c.Id} invStd {rows}x{n}");
                ranForward.Add(c.Id);
            }

            // Backward uses the CPU statistics so each kernel is checked in isolation.
            using var mcpu = backend.AllocateBuffer(k.Mean);
            using var icpu = backend.AllocateBuffer(k.Inv);
            var bwd = new VulkanNormBackwardArgs(dyb, xb, gb, mcpu, icpu, dxb, null, rows, n);
            foreach (var c in Applicable(backend.LayerNormBackwardSlot, shape))
            {
                backend.Fill(dxb, float.NaN, rows * n);
                c.Execute(bwd);
                AssertClose(k.Dx, backend.DownloadBuffer(dxb), k.Dx.Length, $"{c.Id} dx {rows}x{n}", k.DxTermScale);
                ranBackward.Add(c.Id);
            }
            var par = new VulkanNormBackwardArgs(dyb, xb, null, mcpu, icpu, dgb, dbb, rows, n);
            foreach (var c in Applicable(backend.LayerNormGradParametersSlot, shape))
            {
                backend.Fill(dgb, float.NaN, n);
                backend.Fill(dbb, float.NaN, n);
                c.Execute(par);
                AssertClose(k.DGamma, backend.DownloadBuffer(dgb), n, $"{c.Id} dGamma {rows}x{n}", 1e-3);
                AssertClose(k.DBeta, backend.DownloadBuffer(dbb), n, $"{c.Id} dBeta {rows}x{n}", 1e-3);
                ranParams.Add(c.Id);
            }
        }
        AssertEveryCandidateRan(backend.LayerNormSlot, ranForward);
        AssertEveryCandidateRan(backend.LayerNormBackwardSlot, ranBackward);
        AssertEveryCandidateRan(backend.LayerNormGradParametersSlot, ranParams);
    }

    private static float[] RmsGradGammaOnCpu(float[] dy, float[] x, float[] rms, int rows, int n)
    {
        var expected = new float[n];
        for (int i = 0; i < n; i++)
        {
            double s = 0;
            for (int r = 0; r < rows; r++) s += (double)dy[r * n + i] * x[r * n + i] / rms[r];
            expected[i] = (float)s;
        }
        return expected;
    }

    [SkippableFact]
    public void RmsNormGradGamma_EveryCandidate_MatchesCpu()
    {
        var backend = RequireBackend();
        var ran = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (rows, n) in Shapes)
        {
            float[] x = Values(rows * n, 23 + rows, 2f);
            float[] dy = Values(rows * n, 29 + n, 1f);
            float[] rms = Values(rows, 31, 1f).Select(v => Math.Abs(v) + 0.5f).ToArray();
            float[] expected = RmsGradGammaOnCpu(dy, x, rms, rows, n);
            using var xb = backend.AllocateBuffer(x);
            using var dyb = backend.AllocateBuffer(dy);
            using var rb = backend.AllocateBuffer(rms);
            using var dgb = backend.AllocateBuffer(n);
            var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);
            var args = new VulkanNormBackwardArgs(dyb, xb, null, rb, null, dgb, null, rows, n);
            foreach (var c in Applicable(backend.RmsNormGradGammaSlot, shape))
            {
                backend.Fill(dgb, float.NaN, n);
                c.Execute(args);
                AssertClose(expected, backend.DownloadBuffer(dgb), n, $"{c.Id} {rows}x{n}", 1e-3);
                ran.Add(c.Id);
            }
        }
        AssertEveryCandidateRan(backend.RmsNormGradGammaSlot, ran);
    }

    private static float[] Conv2DOnCpu(float[] x, float[] w, int batch, int inC, int h, int wd, int outC, int oh, int ow,
        int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw)
    {
        var y = new float[batch * outC * oh * ow];
        for (int b = 0; b < batch; b++)
            for (int oc = 0; oc < outC; oc++)
                for (int oy = 0; oy < oh; oy++)
                    for (int ox = 0; ox < ow; ox++)
                    {
                        double s = 0;
                        for (int ic = 0; ic < inC; ic++)
                            for (int ky = 0; ky < kh; ky++)
                                for (int kx = 0; kx < kw; kx++)
                                {
                                    int iy = oy * sh - ph + ky * dh, ix = ox * sw - pw + kx * dw;
                                    if (iy < 0 || iy >= h || ix < 0 || ix >= wd) continue;
                                    s += (double)x[((b * inC + ic) * h + iy) * wd + ix] * w[((oc * inC + ic) * kh + ky) * kw + kx];
                                }
                        y[((b * outC + oc) * oh + oy) * ow + ox] = (float)s;
                    }
        return y;
    }

    // batch, inC, H, W, outC, kH, kW, sH, sW, pH, pW, dH, dW. Output channel counts sit below, on, between and above
    // the tile widths (4 and 8), so a ragged last tile and an inapplicable tile both occur.
    public static IEnumerable<object[]> ConvCases() => new[]
    {
        new object[] { 2, 3, 9, 11, 1, 3, 3, 1, 1, 1, 1, 1, 1 },
        new object[] { 2, 3, 9, 11, 4, 3, 3, 1, 1, 1, 1, 1, 1 },
        new object[] { 3, 5, 8, 8, 6, 3, 3, 2, 2, 1, 1, 1, 1 },
        new object[] { 1, 4, 13, 7, 8, 1, 1, 1, 1, 0, 0, 1, 1 },
        new object[] { 2, 2, 12, 12, 13, 5, 3, 1, 2, 2, 1, 1, 1 },
        new object[] { 2, 3, 15, 15, 17, 3, 3, 1, 1, 2, 2, 2, 2 },
        new object[] { 4, 16, 14, 14, 32, 3, 3, 1, 1, 1, 1, 1, 1 },
    };

    [SkippableTheory]
    [MemberData(nameof(ConvCases))]
    public void Conv2D_EveryCandidate_MatchesCpu(int batch, int inC, int h, int w, int outC, int kh, int kw,
        int sh, int sw, int ph, int pw, int dh, int dw)
    {
        var backend = RequireBackend();
        int oh = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1, ow = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
        float[] x = Values(batch * inC * h * w, 41 + outC, 1f);
        float[] wt = Values(outC * inC * kh * kw, 43 + inC, 1f);
        float[] expected = Conv2DOnCpu(x, wt, batch, inC, h, w, outC, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw);
        using var xb = backend.AllocateBuffer(x);
        using var wb = backend.AllocateBuffer(wt);
        using var yb = backend.AllocateBuffer(expected.Length);
        var args = new VulkanConv2DArgs(xb, wb, yb, batch, inC, h, w, outC, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw);
        var applicable = Applicable(backend.Conv2DForwardSlot, args.Shape());
        Assert.Equal(1 + VulkanTunedRowKernels.ConvChannelTiles.Count(t => outC >= t), applicable.Count);
        foreach (var c in applicable)
        {
            backend.Fill(yb, float.NaN, expected.Length);
            c.Execute(args);
            AssertClose(expected, backend.DownloadBuffer(yb), expected.Length, $"{c.Id} outC={outC}");
        }
    }

    [SkippableFact]
    public void GeneratedVariants_AreRegistered_AndDeterministic()
    {
        var backend = RequireBackend();
        foreach (string id in backend.SoftmaxSlot.CandidateIds.Skip(1))
            Assert.StartsWith("generated.softmax.lanes", id);
        Assert.Equal(1 + VulkanTunedRowKernels.RowLanes.Length, backend.SoftmaxSlot.CandidateIds.Count);
        Assert.Equal(1 + VulkanTunedRowKernels.RowLanes.Length, backend.SoftmaxBackwardSlot.CandidateIds.Count);
        Assert.Equal(1 + VulkanTunedRowKernels.RowLanes.Length, backend.LayerNormSlot.CandidateIds.Count);
        Assert.Equal(1 + VulkanTunedRowKernels.RowLanes.Length, backend.LayerNormBackwardSlot.CandidateIds.Count);
        Assert.Equal(1 + VulkanTunedRowKernels.ColumnRowLanes.Length, backend.LayerNormGradParametersSlot.CandidateIds.Count);
        Assert.Equal(1 + VulkanTunedRowKernels.ColumnRowLanes.Length, backend.RmsNormGradGammaSlot.CandidateIds.Count);
        Assert.Equal(1 + VulkanTunedRowKernels.ConvChannelTiles.Length, backend.Conv2DForwardSlot.CandidateIds.Count);
        Assert.StartsWith("vulkan:", backend.SoftmaxSlot.Device);

        const int rows = 2048, n = 64;
        float[] x = Values(rows * n, 3, 4f);
        using var input = backend.AllocateBuffer(x);
        using var a = backend.AllocateBuffer(x.Length);
        using var b = backend.AllocateBuffer(x.Length);
        var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);
        var applicable = Applicable(backend.SoftmaxSlot, shape);
        Assert.Equal(backend.SoftmaxSlot.CandidateIds.Count, applicable.Count);
        foreach (var c in applicable)
        {
            Assert.True(c.IsDeterministic);
            c.Execute(new VulkanSoftmaxArgs(input, a, rows, n));
            c.Execute(new VulkanSoftmaxArgs(input, b, rows, n));
            Assert.Equal(backend.DownloadBuffer(a), backend.DownloadBuffer(b));
        }
    }

    private sealed class PolicyScope : IDisposable
    {
        private readonly TunedKernelMode? _mode = TunedKernelPolicy.ModeOverride;
        private readonly bool _persist = TunedKernelPolicy.PersistDecisions;
        private readonly double _budget = TunedKernelPolicy.BudgetMilliseconds;
        private readonly List<(TunedKernelOp Op, string? Previous)> _overrides = new();

        internal PolicyScope(TunedKernelMode mode)
        {
            TunedKernelPolicy.ModeOverride = mode;
            TunedKernelPolicy.PersistDecisions = false;
            TunedKernelPolicy.BudgetMilliseconds = 60_000;
        }

        internal void Force(TunedKernelOp op, string candidateId)
        {
            _overrides.Add((op, TunedKernelPolicy.OverrideFor(op)));
            TunedKernelPolicy.SetOverride(op, candidateId);
        }

        public void Dispose()
        {
            for (int i = _overrides.Count - 1; i >= 0; i--) TunedKernelPolicy.SetOverride(_overrides[i].Op, _overrides[i].Previous);
            TunedKernelPolicy.ModeOverride = _mode;
            TunedKernelPolicy.PersistDecisions = _persist;
            TunedKernelPolicy.BudgetMilliseconds = _budget;
        }
    }

    [SkippableFact]
    public void PublicSoftmax_ThroughTheRegistry_MatchesTheReferenceAndCachesADecision()
    {
        var backend = RequireBackend();
        using var policy = new PolicyScope(TunedKernelMode.Tune);
        const int rows = 6000, n = 48; // not a shape any shipped profile lists, so the gate itself runs
        var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);
        float[] x = Values(rows * n, 99, 5f);
        using var input = backend.AllocateBuffer(x);
        using var viaRegistry = backend.AllocateBuffer(x.Length);
        using var viaReference = backend.AllocateBuffer(x.Length);
        backend.SoftmaxSlot.InvalidateDecisions();
        backend.Softmax(input, viaRegistry, rows, n);
        (backend.SoftmaxSlot.Candidate("glsl.softmax.invocation_per_row") ?? throw new InvalidOperationException("reference missing"))
            .Execute(new VulkanSoftmaxArgs(input, viaReference, rows, n));
        AssertClose(backend.DownloadBuffer(viaReference), backend.DownloadBuffer(viaRegistry), x.Length, "registry softmax");
        var decision = Assert.Single(backend.SoftmaxSlot.CachedDecisions, d => d.Shape.Equals(shape));
        Assert.Equal(TunedKernelOp.Softmax, decision.Op);
        // Measured, not defaulted: the gate ran on this call.
        Assert.True(decision.Reason is TunedKernelDecisionReason.Tuned or TunedKernelDecisionReason.ReferenceWon,
            $"unexpected reason {decision.Reason}");

        // In place: the registry must not measure (a second in-place run would corrupt the data) and the result
        // must equal one reference application.
        using var inPlace = backend.AllocateBuffer(x);
        backend.SoftmaxSlot.InvalidateDecisions();
        backend.Softmax(inPlace, inPlace, rows, n);
        AssertClose(backend.DownloadBuffer(viaReference), backend.DownloadBuffer(inPlace), x.Length, "in-place softmax");
        Assert.DoesNotContain(backend.SoftmaxSlot.CachedDecisions, d => d.Shape.Equals(shape));
    }

    // Falsifies the wiring: with each candidate forced by override, the PUBLIC backend ops must still give the CPU
    // result and the slot must record that candidate as the one chosen for the shape.
    [SkippableFact]
    public void PublicOps_WithEachCandidateForced_MatchCpu_AndRecordThatCandidate()
    {
        var backend = RequireBackend();
        const int rows = 96, n = 160;
        const float eps = 1e-5f;
        var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);
        var k = LayerNormOnCpu(rows, n, eps);
        float[] sx = Values(rows * n, 61, 5f);
        float[] softmax = SoftmaxOnCpu(sx, rows, n);
        float[] rms = Values(rows, 67, 1f).Select(v => Math.Abs(v) + 0.5f).ToArray();
        float[] rmsGradGamma = RmsGradGammaOnCpu(k.Dy, k.X, rms, rows, n);

        using var xb = backend.AllocateBuffer(k.X);
        using var gb = backend.AllocateBuffer(k.Gamma);
        using var bb = backend.AllocateBuffer(k.Beta);
        using var dyb = backend.AllocateBuffer(k.Dy);
        using var sxb = backend.AllocateBuffer(sx);
        using var rb = backend.AllocateBuffer(rms);
        using var yb = backend.AllocateBuffer(rows * n);
        using var mb = backend.AllocateBuffer(rows);
        using var ib = backend.AllocateBuffer(rows);
        using var dxb = backend.AllocateBuffer(rows * n);
        using var dgb = backend.AllocateBuffer(n);
        using var dbb = backend.AllocateBuffer(n);

        void Forced<TArgs>(TunedKernelSlot<TArgs> slot, Action run) where TArgs : struct
        {
            var applicable = Applicable(slot, shape);
            Assert.True(applicable.Count > 1, $"{slot.Op}: only the reference applies to {shape}");
            foreach (var c in applicable)
            {
                using var policy = new PolicyScope(TunedKernelMode.Pinned);
                policy.Force(slot.Op, c.Id);
                slot.InvalidateDecisions();
                run();
                var decision = Assert.Single(slot.CachedDecisions, d => d.Shape.Equals(shape));
                Assert.Equal(c.Id, decision.CandidateId);
                Assert.Equal(TunedKernelDecisionReason.Override, decision.Reason);
                slot.InvalidateDecisions();
            }
        }

        Forced(backend.SoftmaxSlot, () =>
        {
            backend.Fill(yb, float.NaN, rows * n);
            backend.Softmax(sxb, yb, rows, n);
            AssertClose(softmax, backend.DownloadBuffer(yb), rows * n, "public softmax");
        });
        Forced(backend.SoftmaxBackwardSlot, () =>
        {
            // softmax as y, dy as the upstream gradient
            using var syb = backend.AllocateBuffer(softmax);
            var expected = new float[rows * n];
            for (int r = 0; r < rows; r++)
            {
                double dot = 0;
                for (int i = 0; i < n; i++) dot += (double)k.Dy[r * n + i] * softmax[r * n + i];
                for (int i = 0; i < n; i++) expected[r * n + i] = (float)(softmax[r * n + i] * (k.Dy[r * n + i] - dot));
            }
            backend.Fill(dxb, float.NaN, rows * n);
            backend.SoftmaxBackward(dyb, syb, dxb, rows, n);
            AssertClose(expected, backend.DownloadBuffer(dxb), rows * n, "public softmax backward", 1e-2);
        });
        Forced(backend.LayerNormSlot, () =>
        {
            backend.Fill(yb, float.NaN, rows * n);
            backend.LayerNorm(xb, yb, gb, bb, mb, ib, rows, n, eps);
            AssertClose(k.Y, backend.DownloadBuffer(yb), rows * n, "public layernorm y");
            AssertClose(k.Mean, backend.DownloadBuffer(mb), rows, "public layernorm mean", 1e-3);
            AssertClose(k.Inv, backend.DownloadBuffer(ib), rows, "public layernorm invStd");
        });

        using var mcpu = backend.AllocateBuffer(k.Mean);
        using var icpu = backend.AllocateBuffer(k.Inv);
        void LayerNormBackward()
        {
            backend.Fill(dxb, float.NaN, rows * n);
            backend.Fill(dgb, float.NaN, n);
            backend.Fill(dbb, float.NaN, n);
            backend.LayerNormBackward(dyb, xb, gb, mcpu, icpu, dxb, dgb, dbb, rows, n, eps);
            AssertClose(k.Dx, backend.DownloadBuffer(dxb), rows * n, "public layernorm dx", k.DxTermScale);
            AssertClose(k.DGamma, backend.DownloadBuffer(dgb), n, "public layernorm dGamma", 1e-3);
            AssertClose(k.DBeta, backend.DownloadBuffer(dbb), n, "public layernorm dBeta", 1e-3);
        }
        Forced(backend.LayerNormBackwardSlot, LayerNormBackward);
        Forced(backend.LayerNormGradParametersSlot, LayerNormBackward);
        Forced(backend.RmsNormGradGammaSlot, () =>
        {
            backend.Fill(dgb, float.NaN, n);
            backend.RmsNormBackward(dyb, xb, gb, rb, dxb, dgb, rows, n, eps);
            AssertClose(rmsGradGamma, backend.DownloadBuffer(dgb), n, "public rmsnorm dGamma", 1e-3);
        });
    }

    [SkippableFact]
    public void PublicConv2D_WithEachCandidateForced_MatchesCpu_AndRecordsThatCandidate()
    {
        var backend = RequireBackend();
        const int batch = 2, inC = 6, h = 12, w = 10, outC = 19, kh = 3, kw = 3;
        float[] x = Values(batch * inC * h * w, 71, 1f);
        float[] wt = Values(outC * inC * kh * kw, 73, 1f);
        float[] expected = Conv2DOnCpu(x, wt, batch, inC, h, w, outC, h, w, kh, kw, 1, 1, 1, 1, 1, 1);
        using var xb = backend.AllocateBuffer(x);
        using var wb = backend.AllocateBuffer(wt);
        using var yb = backend.AllocateBuffer(expected.Length);
        var slot = backend.Conv2DForwardSlot;
        var shape = new VulkanConv2DArgs(xb, wb, yb, batch, inC, h, w, outC, h, w, kh, kw, 1, 1, 1, 1, 1, 1).Shape();
        var applicable = Applicable(slot, shape);
        Assert.Equal(slot.CandidateIds.Count, applicable.Count);
        foreach (var c in applicable)
        {
            using var policy = new PolicyScope(TunedKernelMode.Pinned);
            policy.Force(TunedKernelOp.Conv2DForward, c.Id);
            slot.InvalidateDecisions();
            backend.Fill(yb, float.NaN, expected.Length);
            backend.Conv2D(xb, wb, yb, batch, inC, h, w, outC, h, w, kh, kw, 1, 1, 1, 1, 1, 1);
            AssertClose(expected, backend.DownloadBuffer(yb), expected.Length, $"public conv2d via {c.Id}");
            var decision = Assert.Single(slot.CachedDecisions, d => d.Shape.Equals(shape));
            Assert.Equal(c.Id, decision.CandidateId);
            slot.InvalidateDecisions();
        }
    }

    [SkippableFact]
    public void TimingHarness_MeasuresMoreTimeForMoreWork()
    {
        var backend = RequireBackend();
        const int rows = 4096, n = 256;
        float[] x = Values(rows * n, 5, 3f);
        using var input = backend.AllocateBuffer(x);
        using var output = backend.AllocateBuffer(x.Length);
        var reference = backend.SoftmaxSlot.Candidate(backend.SoftmaxSlot.CandidateIds[0]) ?? throw new InvalidOperationException("reference missing");
        var args = new VulkanSoftmaxArgs(input, output, rows, n);
        void Run() => reference.Execute(args);
        backend.MeasureTunedKernelMilliseconds(Run, 2); // warm the pipeline
        // Medians over several samples: one sample of either can be disturbed by another process on the device.
        double Median(int repetitions)
        {
            var samples = new double[7];
            for (int i = 0; i < samples.Length; i++) samples[i] = backend.MeasureTunedKernelMilliseconds(Run, repetitions);
            Array.Sort(samples);
            return samples[samples.Length / 2];
        }
        double one = Median(1), sixteen = Median(16);
        Assert.True(one > 0, $"one run measured {one} ms");
        Assert.True(sixteen > 4 * one, $"16 runs measured {sixteen} ms against {one} ms for one");
        // The RTX-class devices this runs on expose timestamp queries; a device without them times on the host clock.
        if (backend.TunedKernelTimingUsesDeviceClock)
            Assert.True(sixteen < 60_000, "device timestamps are in milliseconds");
    }

    [SkippableFact]
    public void ShippedProfiles_NameOnlyRegisteredCandidates()
    {
        var backend = RequireBackend();
        var registered = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal)
        {
            [TunedKernelOp.Softmax.ToString()] = new(backend.SoftmaxSlot.CandidateIds),
            [TunedKernelOp.SoftmaxBackward.ToString()] = new(backend.SoftmaxBackwardSlot.CandidateIds),
            [TunedKernelOp.LayerNorm.ToString()] = new(backend.LayerNormSlot.CandidateIds),
            [TunedKernelOp.LayerNormBackward.ToString()] = new(backend.LayerNormBackwardSlot.CandidateIds),
            [TunedKernelOp.LayerNormGradParameters.ToString()] = new(backend.LayerNormGradParametersSlot.CandidateIds),
            [TunedKernelOp.RmsNormGradGamma.ToString()] = new(backend.RmsNormGradGammaSlot.CandidateIds),
            [TunedKernelOp.Conv2DForward.ToString()] = new(backend.Conv2DForwardSlot.CandidateIds),
        };
        var asm = typeof(TunedKernelProfiles).Assembly;
        int vulkanProfiles = 0, entries = 0;
        foreach (string name in asm.GetManifestResourceNames().Where(r => r.EndsWith(".tuned-profile.json", StringComparison.OrdinalIgnoreCase)))
        {
            using var stream = asm.GetManifestResourceStream(name) ?? throw new InvalidOperationException(name);
            using var doc = System.Text.Json.JsonDocument.Parse(stream);
            string device = doc.RootElement.GetProperty("device").GetString() ?? string.Empty;
            if (!device.StartsWith("vulkan:", StringComparison.Ordinal)) continue;
            vulkanProfiles++;
            foreach (var e in doc.RootElement.GetProperty("entries").EnumerateArray())
            {
                string op = e.GetProperty("op").GetString() ?? string.Empty;
                string candidate = e.GetProperty("candidate").GetString() ?? string.Empty;
                Assert.True(registered.TryGetValue(op, out var ids), $"{name}: unknown op {op}");
                Assert.True(ids!.Contains(candidate), $"{name}: {op} names unregistered candidate {candidate}");
                Assert.True(TunedShape.TryParse(e.GetProperty("shape").GetString() ?? string.Empty, out _), $"{name}: bad shape");
                entries++;
            }
        }
        Assert.True(vulkanProfiles >= 1, "no shipped Vulkan tuned-kernel profile");
        Assert.True(entries >= 1, "the shipped Vulkan profile is empty");
    }

    // The shipped profile is what a known device runs without measuring: in Pinned mode (no online tuning) the public
    // op must be served by the profiled variant, and still give the CPU result.
    [SkippableFact]
    public void ShippedProfile_IsServedByThePublicOp_OnItsDevice()
    {
        var backend = RequireBackend();
        Skip.IfNot(backend.TunedDeviceKey == "vulkan:NVIDIA GeForce RTX 3080", $"No shipped profile for {backend.TunedDeviceKey}.");
        using var policy = new PolicyScope(TunedKernelMode.Pinned);
        const int rows = 64, n = 4096;
        var shape = TunedShape.Of2(TunedKernelDType.Float32, rows, n);
        float[] x = Values(rows * n, 83, 5f);
        using var input = backend.AllocateBuffer(x);
        using var output = backend.AllocateBuffer(x.Length);
        backend.SoftmaxSlot.InvalidateDecisions();
        backend.Fill(output, float.NaN, x.Length);
        backend.Softmax(input, output, rows, n);
        AssertClose(SoftmaxOnCpu(x, rows, n), backend.DownloadBuffer(output), x.Length, "profiled softmax");
        var decision = Assert.Single(backend.SoftmaxSlot.CachedDecisions, d => d.Shape.Equals(shape));
        Assert.Equal(TunedKernelDecisionReason.Profile, decision.Reason);
        Assert.Equal("generated.softmax.lanes256", decision.CandidateId);
        backend.SoftmaxSlot.InvalidateDecisions();
    }

    // No device needed: libshaderc compiles on the CPU, so this runs wherever the native library loads (CI included).
    [SkippableFact]
    public void EveryGeneratedVariant_CompilesToSpirv()
    {
        VulkanGlslCompiler compiler;
        try { compiler = new VulkanGlslCompiler(); }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException or BadImageFormatException) { compiler = null!; }
        Skip.If(compiler is null || !compiler.IsAvailable, "libshaderc is not available.");
        using (compiler)
        {
            var all = VulkanTunedRowKernels.All();
            Assert.Equal(4 * VulkanTunedRowKernels.RowLanes.Length + 2 * VulkanTunedRowKernels.ColumnRowLanes.Length +
                         VulkanTunedRowKernels.ConvChannelTiles.Length, all.Count);
            foreach (var (name, source, _, _) in all)
            {
                uint[]? spirv = compiler!.CompileToSpirv(source);
                Assert.True(spirv is { Length: > 5 }, $"{name} did not compile: {compiler.LastError}");
            }
        }
    }

    [Fact]
    public void GeneratedVariants_AreDistinctPerLaneCount_AndFitTheWorkgroup()
    {
        var all = VulkanTunedRowKernels.All();
        Assert.Equal(all.Count, all.Select(v => v.Name).Distinct().Count());
        Assert.Equal(all.Count, all.Select(v => v.Source).Distinct().Count());
        foreach (int lanes in VulkanTunedRowKernels.RowLanes)
        {
            Assert.Equal(0, VulkanTunedRowKernels.WorkgroupSize % lanes);
            Assert.True((lanes & (lanes - 1)) == 0, "the fold tree needs a power-of-two lane count");
            Assert.Contains($"const uint L = {lanes}u;", VulkanTunedRowKernels.Softmax(lanes));
        }
        foreach (int rowLanes in VulkanTunedRowKernels.ColumnRowLanes)
        {
            Assert.Equal(0, VulkanTunedRowKernels.WorkgroupSize % rowLanes);
            Assert.True((rowLanes & (rowLanes - 1)) == 0, "the fold tree needs a power-of-two row-lane count");
        }
        // Lane fit: a quarter of the lanes must have work, and short of a whole workgroup no lane walks over 64 columns.
        Assert.False(VulkanTunedRowKernels.RowLanesFit(32, 10, 7));
        Assert.True(VulkanTunedRowKernels.RowLanesFit(32, 10, 8));
        Assert.True(VulkanTunedRowKernels.RowLanesFit(8, 10, 512));
        Assert.False(VulkanTunedRowKernels.RowLanesFit(8, 10, 513));
        Assert.True(VulkanTunedRowKernels.RowLanesFit(256, 10, 1 << 20));
        Assert.False(VulkanTunedRowKernels.RowLanesFit(256, 10, 63));
        Assert.False(VulkanTunedRowKernels.RowLanesFit(256, int.MaxValue / 2, 64));
    }
}
