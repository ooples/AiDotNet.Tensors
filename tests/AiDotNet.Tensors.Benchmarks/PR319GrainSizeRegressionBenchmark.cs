using System;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// PR #319 grain-size migration regression check (wall-clock harness).
///
/// Measures Softmax / BatchNorm / TensorNormalize at three sizes spanning
/// the 32K grain threshold and reports best-of-3-runs median to control
/// for OS scheduler / GC / JIT noise. Each run does 200 warm-up iters
/// (discarded) + 300 measurement iters, GC.Collect+WaitForPendingFinalizers
/// between sizes.
/// </summary>
public static class PR319GrainSizeHarness
{
    public static void Run()
    {
        var engine = new CpuEngine();

        var shapes = new (string Label, int B, int C, int H, int W)[]
        {
            ("small  [4×32×8×8]      = 8K  ", 4, 32, 8, 8),
            ("medium [8×64×16×16]    = 131K", 8, 64, 16, 16),
            ("large  [16×128×32×32]  = 2M  ", 16, 128, 32, 32),
        };

        Console.WriteLine();
        Console.WriteLine("╔══════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║   PR #319 Grain-Size Migration — Regression Harness                  ║");
        Console.WriteLine("║   3 runs of (200 warmup + 300 measured); reports min-of-3 medians    ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════════════╝");

        const int warmup = 200;
        const int measure = 300;
        const int outerRuns = 3;

        foreach (var (label, B, C, H, W) in shapes)
        {
            Console.WriteLine();
            Console.WriteLine($"── {label} ──");

            var rng = new Random(42);
            var input  = MakeTensor(new[] { B, C, H, W }, rng);
            var gamma  = MakeTensor(new[] { C }, rng);
            var beta   = MakeTensor(new[] { C }, rng);

            ReportBest("Softmax       ", outerRuns, warmup, measure,
                () => engine.Softmax(input, axis: -1));

            ReportBest("BatchNorm     ", outerRuns, warmup, measure,
                () => engine.BatchNorm(input, gamma, beta, 1e-5, out _, out _));

            ReportBest("L2Normalize   ", outerRuns, warmup, measure,
                () => engine.TensorNormalize(input, axis: 1, epsilon: 1e-10f));

            // MatMul covers MatrixMultiplyHelper + Avx512Sgemm + SimdGemmDouble
            // Reshape input → 2D matrix and multiply against random kernel
            int M = B * C, K = H * W, N = C;
            var matA = MakeTensor(new[] { M, K }, rng);
            var matB = MakeTensor(new[] { K, N }, rng);
            ReportBest("MatMul        ", outerRuns, warmup, measure,
                () => engine.TensorMatMul(matA, matB));

            // Conv2D covers NchwcConv2D / NchwcConv2D16 / FusedConvHelper
            var convInput = MakeTensor(new[] { B, C, H, W }, rng);
            var convKernel = MakeTensor(new[] { C, C, 3, 3 }, rng);
            ReportBest("Conv2D 3×3    ", outerRuns, warmup, measure,
                () => engine.Conv2D(convInput, convKernel,
                    stride: new[] { 1, 1 }, padding: new[] { 1, 1 }, dilation: new[] { 1, 1 }));
        }

        Console.WriteLine();
    }

    private static void ReportBest(string label, int outerRuns, int warmup, int measure, Action op)
    {
        long[] medians = new long[outerRuns];
        for (int r = 0; r < outerRuns; r++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            medians[r] = TimeMedian(op, warmup, measure);
        }
        long bestMed = medians.Min();
        long worstMed = medians.Max();
        double bestUs = (double)bestMed / Stopwatch.Frequency * 1e6;
        double worstUs = (double)worstMed / Stopwatch.Frequency * 1e6;
        double spread = worstUs / bestUs;
        Console.WriteLine($"  {label} best-median {bestUs,8:F1} µs   (worst {worstUs,8:F1}, spread×{spread:F2})");
    }

    private static long TimeMedian(Action op, int warmup, int measure)
    {
        for (int i = 0; i < warmup; i++) op();
        var times = new long[measure];
        var sw = new Stopwatch();
        for (int i = 0; i < measure; i++)
        {
            sw.Restart();
            op();
            sw.Stop();
            times[i] = sw.ElapsedTicks;
        }
        Array.Sort(times);
        return times[measure / 2];
    }

    private static Tensor<float> MakeTensor(int[] shape, Random rng)
    {
        var t = new Tensor<float>(shape);
        var d = t.GetDataArray();
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }
}
