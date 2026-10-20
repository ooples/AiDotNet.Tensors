// Copyright (c) AiDotNet. All rights reserved.
// The declared-program route (#890 phase 4): CudaBackend.LayerNorm is served by the program
// lowered from its spec, in the hand-written kernel's contract (y, the saved mean and rstd).
//
// The bar is an fp64 reference of that contract, within the fp32 conditioning bound of the
// data: they sit at an offset of 1000, so rounding the mean alone moves y by about
// u * |x| * rstd * |gamma|, and BOTH routes are entitled to that much. Agreement with the
// hand-written kernel would be the wrong bar -- it carries the same rounding, in another order.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

[Collection("DirectGpuSerial")]
public class DeclaredProgramRouteTests
{
    private static CudaBackend OpenBackendOrSkip()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("CUDA driver was detected, but the CUDA backend failed to initialize.");
        }
        return backend;
    }

    /// <summary>
    /// Exact held (64, 1024), ragged held (100, 1000, 7) and ragged streamed (50001, odd, so
    /// scalar): each routed through the declared program and within the bound.
    /// </summary>
    [SkippableTheory]
    [InlineData(256, 1024)]
    [InlineData(33, 96)]
    [InlineData(100, 100)]
    [InlineData(128, 1000)]
    [InlineData(512, 7)]
    [InlineData(16, 50001)]
    public void LayerNormTakesTheDeclaredProgramWithinTheConditioningBound(int rows, int cols)
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ codegen conveyor.");

        var (y, mean, rstd, routed) = RunLayerNorm(backend, rows, cols, declared: true);
        Assert.True(routed, "the call fell through to the hand-written kernel: " +
            backend.DeclaredDeclineReason(CudaBackend.LayerNormKey(rows, cols, Epsilon)));
        AssertWithinBound(rows, cols, y, mean, rstd);
    }

    /// <summary>With the switch off the hand-written kernel serves the call, and no declared launch is made.</summary>
    [SkippableFact]
    public void TheSwitchTurnsTheRouteOff()
    {
        using var backend = OpenBackendOrSkip();
        var (y, mean, rstd, routed) = RunLayerNorm(backend, 64, 1000, declared: false);
        Assert.False(routed);
        AssertWithinBound(64, 1000, y, mean, rstd);
    }

    /// <summary>
    /// Building a program loads modules and allocates scratch, which graph capture forbids: a shape first met under
    /// capture falls through to the hand-written kernel; once built outside, the program captures and replays.
    /// </summary>
    [SkippableFact]
    public void AProgramIsBuiltOutsideCaptureAndReplaysInsideIt()
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ codegen conveyor.");
        const int rows = 48, cols = 200;
        var (x, gamma, beta) = Inputs(rows, cols);
        using var bx = backend.AllocateBuffer(x);
        using var bg = backend.AllocateBuffer(gamma);
        using var bb = backend.AllocateBuffer(beta);
        using var by = backend.AllocateBuffer(rows * cols);
        using var bm = backend.AllocateBuffer(rows);
        using var bv = backend.AllocateBuffer(rows);
        void Call() => backend.LayerNorm(bx, by, bg, bb, bm, bv, rows, cols, Epsilon);

        foreach (bool built in new[] { false, true })
        {
            if (built) Call();   // builds the program outside capture
            long before = backend.DeclaredProgramLaunches;
            IntPtr graph = backend.CaptureGraph(Call);
            try
            {
                Assert.Equal(built, backend.DeclaredProgramLaunches > before);
                backend.LaunchCapturedGraph(graph);
                backend.Synchronize();
                AssertWithinBound(rows, cols, backend.DownloadBuffer(by), backend.DownloadBuffer(bm), backend.DownloadBuffer(bv));
            }
            finally
            {
                backend.DestroyCapturedGraph(graph);
            }
        }
    }

    private const float Epsilon = 1e-5f;

    private static (float[] Y, float[] Mean, float[] Rstd, bool Routed) RunLayerNorm(
        CudaBackend backend, int rows, int cols, bool declared)
    {
        var (x, gamma, beta) = Inputs(rows, cols);
        backend.DeclaredKernelsSwitch = declared;
        try
        {
            using var bx = backend.AllocateBuffer(x);
            using var bg = backend.AllocateBuffer(gamma);
            using var bb = backend.AllocateBuffer(beta);
            using var by = backend.AllocateBuffer(rows * cols);
            using var bm = backend.AllocateBuffer(rows);
            using var bv = backend.AllocateBuffer(rows);
            long before = backend.DeclaredProgramLaunches;
            backend.LayerNorm(bx, by, bg, bb, bm, bv, rows, cols, Epsilon);
            backend.Synchronize();
            return (backend.DownloadBuffer(by), backend.DownloadBuffer(bm), backend.DownloadBuffer(bv),
                backend.DeclaredProgramLaunches > before);
        }
        finally
        {
            backend.DeclaredKernelsSwitch = null;
        }
    }

    private static (float[] X, float[] Gamma, float[] Beta) Inputs(int rows, int cols)
    {
        var rng = new Random(rows * 31 + cols);
        var x = new float[rows * cols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(1000.0 + rng.NextDouble() * 1.5 - 0.75);
        var gamma = new float[cols];
        var beta = new float[cols];
        for (int j = 0; j < cols; j++)
        {
            gamma[j] = (float)(rng.NextDouble() * 1.5 - 0.75);
            beta[j] = (float)(rng.NextDouble() - 0.5);
        }
        return (x, gamma, beta);
    }

    private static void AssertWithinBound(int rows, int cols, float[] y, float[] mean, float[] rstd)
    {
        var (x, gamma, beta) = Inputs(rows, cols);
        double maxRstd = 0;
        var refMean = new double[rows];
        var refRstd = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < cols; c++) sum += x[r * cols + c];
            double m = sum / cols, sq = 0;
            for (int c = 0; c < cols; c++) { double d = x[r * cols + c] - m; sq += d * d; }
            refMean[r] = m;
            refRstd[r] = 1.0 / Math.Sqrt(sq / cols + Epsilon);
            maxRstd = Math.Max(maxRstd, refRstd[r]);
        }
        double maxGamma = 1;
        foreach (float g in gamma) maxGamma = Math.Max(maxGamma, Math.Abs(g));
        double bound = (16 + Math.Log(cols, 2)) / (1 << 24) * Math.Max(1.0, 1000.75 * maxRstd * maxGamma);

        for (int r = 0; r < rows; r++)
        {
            Assert.True(Relative(mean[r], refMean[r]) <= bound, $"mean[{r}] {mean[r]} vs {refMean[r]}");
            Assert.True(Relative(rstd[r], refRstd[r]) <= bound, $"rstd[{r}] {rstd[r]} vs {refRstd[r]}");
            for (int c = 0; c < cols; c++)
            {
                double expected = (x[r * cols + c] - refMean[r]) * refRstd[r] * gamma[c] + beta[c];
                Assert.True(Relative(y[r * cols + c], expected) <= bound,
                    $"y[{r},{c}] {y[r * cols + c]} vs {expected} (bound {bound:E3})");
            }
        }
    }

    private static double Relative(float actual, double expected) =>
        Math.Abs(actual - expected) / Math.Max(1.0, Math.Abs(expected));
}
