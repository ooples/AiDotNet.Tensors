// Copyright (c) AiDotNet. All rights reserved.
// The declared-program route (#890 phase 4): CudaBackend.LayerNorm is served by the program
// lowered from its spec, in the hand-written kernel's contract (y, the saved mean and rstd).
//
// The bar is an fp64 reference of that contract, within the fp32 conditioning bound of the
// data: they sit at an offset of 1000, so rounding the mean alone moves y by about
// u * |x| * rstd * |gamma|, and BOTH routes are entitled to that much. Agreement with the
// hand-written kernel would be the wrong bar -- it carries the same rounding, in another order.

using System;
using System.Collections.Generic;
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

    /// <summary>
    /// Every other routed family, at shapes that take the ragged and wrapped layouts (7x7 and
    /// 14x14 BatchNorm planes, 49-element GroupNorm runs, a 100-wide row): the route fires, and
    /// on well-conditioned data -- no offset, where two fp32 evaluations of the same math agree
    /// to about 1e-6 -- every output agrees with the incumbent's within 1e-4.
    /// </summary>
    [SkippableTheory]
    [InlineData("rmsnorm")]
    [InlineData("groupnorm")]
    [InlineData("batchnorm")]
    [InlineData("layernorm_backward")]
    [InlineData("rmsnorm_backward")]
    [InlineData("batchnorm_backward")]
    [InlineData("softmax")]
    [InlineData("softmax_backward")]
    public void EveryRoutedFamilyFiresAndAgreesWithTheIncumbent(string family)
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ codegen conveyor.");
        foreach (var (inputs, outputSizes, call) in FamilyCases(family))
        {
            var results = new float[2][][];
            bool routed = false;
            for (int run = 0; run < 2; run++)
            {
                bool declared = run == 0;
                backend.DeclaredKernelsSwitch = declared;
                var buffers = new IGpuBuffer[inputs.Length + outputSizes.Length];
                try
                {
                    for (int i = 0; i < inputs.Length; i++) buffers[i] = backend.AllocateBuffer(inputs[i]);
                    for (int o = 0; o < outputSizes.Length; o++)
                        buffers[inputs.Length + o] = backend.AllocateBuffer(new float[outputSizes[o]]);
                    long before = backend.DeclaredProgramLaunches;
                    call(backend, buffers);
                    backend.Synchronize();
                    if (declared) routed = backend.DeclaredProgramLaunches > before;
                    results[run] = new float[buffers.Length][];
                    for (int i = 0; i < buffers.Length; i++) results[run][i] = backend.DownloadBuffer(buffers[i]);
                }
                finally
                {
                    foreach (var buffer in buffers) buffer?.Dispose();
                    backend.DeclaredKernelsSwitch = null;
                }
            }
            Assert.True(routed, family + " fell through to the incumbent.");
            // Every buffer -- outputs, and inputs a call updates in place (BatchNorm's running statistics).
            for (int i = 0; i < results[0].Length; i++)
                for (int j = 0; j < results[0][i].Length; j++)
                {
                    float declared = results[0][i][j], incumbent = results[1][i][j];
                    Assert.True(Math.Abs(declared - incumbent) <= 1e-4 * Math.Max(1.0, Math.Abs(incumbent)),
                        $"{family}: buffer {i}[{j}] declared {declared} incumbent {incumbent}");
                }
        }
    }

    private static IEnumerable<(float[][] Inputs, int[] OutputSizes, Action<CudaBackend, IGpuBuffer[]> Call)> FamilyCases(string family)
    {
        float[] R(int n, int seed, double offset = 0)
        {
            var rng = new Random(seed);
            var v = new float[n];
            for (int i = 0; i < n; i++) v[i] = (float)(offset + rng.NextDouble() * 2 - 1);
            return v;
        }
        float[] Positive(int n, int seed)
        {
            var v = R(n, seed, 1.5);
            return v;
        }
        switch (family)
        {
            case "rmsnorm":
                foreach (var (r, c) in new[] { (64, 1024), (100, 100) })
                    yield return (new[] { R(r * c, 1), R(c, 2) }, new[] { r * c, r },
                        (k, a) => k.RmsNorm(a[0], a[2], a[1], a[3], r, c, Epsilon));
                break;
            case "groupnorm":
                foreach (var (n, ch, g, sp) in new[] { (4, 32, 8, 64), (4, 32, 8, 49), (2, 6, 3, 10) })
                    yield return (new[] { R(n * ch * sp, 3), R(ch, 4), R(ch, 5) }, new[] { n * ch * sp, n * g, n * g },
                        (k, a) => k.GroupNorm(a[0], a[3], a[1], a[2], a[4], a[5], n, g, ch, sp, Epsilon));
                break;
            case "batchnorm":
                foreach (var (n, ch, sp) in new[] { (8, 16, 64), (16, 32, 49), (4, 16, 196) })
                    yield return (new[] { R(n * ch * sp, 6), R(ch, 7), R(ch, 8), R(ch, 9), Positive(ch, 10) },
                        new[] { n * ch * sp, ch, ch },
                        (k, a) => k.BatchNorm(a[0], a[5], a[1], a[2], a[3], a[4], a[6], a[7], n, ch, sp, Epsilon, 0.1f, true));
                break;
            case "layernorm_backward":
                foreach (var (r, c) in new[] { (64, 1024), (100, 100) })
                    yield return (new[] { R(r * c, 11), R(r * c, 12), R(c, 13), R(r, 14), Positive(r, 15) }, new[] { r * c, c, c },
                        (k, a) => k.LayerNormBackward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], r, c, Epsilon));
                break;
            case "rmsnorm_backward":
                foreach (var (r, c) in new[] { (64, 1024), (100, 100) })
                    yield return (new[] { R(r * c, 16), R(r * c, 17), R(c, 18), Positive(r, 19) }, new[] { r * c, c },
                        (k, a) => k.RmsNormBackward(a[0], a[1], a[2], a[3], a[4], a[5], r, c, Epsilon));
                break;
            case "batchnorm_backward":
                foreach (var (n, ch, sp) in new[] { (8, 16, 64), (16, 32, 49), (4, 16, 196) })
                    yield return (new[] { R(n * ch * sp, 20), R(n * ch * sp, 21), R(ch, 22), R(ch, 23), Positive(ch, 24) },
                        new[] { n * ch * sp, ch, ch },
                        (k, a) => k.BatchNormBackward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], n, ch, sp, Epsilon));
                break;
            case "softmax":
                foreach (var (r, c) in new[] { (64, 1024), (100, 100), (16, 5000) })
                    yield return (new[] { R(r * c, 25) }, new[] { r * c }, (k, a) => k.Softmax(a[0], a[1], r, c));
                break;
            case "softmax_backward":
                foreach (var (r, c) in new[] { (64, 1024), (100, 100), (16, 5000) })
                    yield return (new[] { R(r * c, 26), Positive(r * c, 27) }, new[] { r * c },
                        (k, a) => k.SoftmaxBackward(a[0], a[1], a[2], r, c));
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(family));
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
