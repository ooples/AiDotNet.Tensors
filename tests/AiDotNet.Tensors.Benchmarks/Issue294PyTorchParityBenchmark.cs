// Copyright (c) AiDotNet. All rights reserved.

#if NET8_0_OR_GREATER
using System;
using System.Diagnostics;
using AiDotNet.Tensors.Benchmarks.BaselineRunners;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using System.Globalization;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue #294 acceptance criterion #6: integration benchmarks vs
/// PyTorch CPU on AMD Zen + Intel + ARM. Five operation classes
/// covered: matmul (3 sizes), Conv2D (small + big), FlashAttention
/// (3 seq lengths), LayerNorm, BCE derivative.
///
/// <para><b>How this runs:</b> not as a BenchmarkDotNet entrypoint —
/// instead invoked via <c>dotnet run -c Release -- --pytorch-parity</c>.
/// Drives a Python subprocess (via
/// <see cref="PythonBaselineRunner"/>) that times the equivalent
/// PyTorch op, then times AiDotNet on the same shape, then prints
/// a side-by-side table. The Python side requires
/// <c>pip install torch</c>; when missing, the benchmark prints
/// "torch unavailable" rather than failing — the C# side still
/// runs and reports its own timings as a regression baseline.</para>
///
/// <para>Tensors should meet-or-beat PyTorch CPU on ≥80% of cases
/// at batch=1 and batch=32. Cases where PyTorch wins are documented
/// in the PR body — typically large GEMM where MKL's pre-tuned
/// kernels still have an edge. The win cases (small-batch / inference,
/// where .NET's ~10ns op dispatch beats Python+ATen's 1-5μs) are
/// where this PR's perf story lives.</para>
/// </summary>
internal static class Issue294PyTorchParityBenchmark
{
    public static void Run()
    {
        Console.WriteLine("=== Issue #294 PyTorch CPU parity benchmarks ===");
        Console.WriteLine();

        using var runner = new PythonBaselineRunner();
        if (!runner.IsAvailable)
        {
            Console.WriteLine("Python not available on PATH — skipping PyTorch comparison.");
            Console.WriteLine("AiDotNet timings will still be reported as a regression baseline.");
            Console.WriteLine();
        }

        var engine = new CpuEngine();

        // ── Matmul: 3 sizes ─────────────────────────────────────────
        Console.WriteLine("--- Matmul ---");
        Console.WriteLine($"{"Shape",-20} {"AiDotNet (ms)",-15} {"PyTorch (ms)",-15} {"Speedup",-10}");
        foreach (var (m, k, n) in new[] { (32, 32, 32), (256, 256, 256), (1024, 1024, 1024) })
        {
            double aiMs = TimeMatMul(engine, m, k, n, warmup: 5, iters: 30);
            var torch = runner.TryRun(PythonBaselineRunner.Baseline.Torch294,
                "matmul", $"{m}x{k}x{n}", warmup: 5, iters: 30);
            string torchMs = torch?.MedianMs.ToString("F3") ?? "skip";
            string speedup = FormatSpeedup(aiMs, torch?.MedianMs);
            Console.WriteLine($"{m}x{k}x{n,-12} {aiMs,-15:F3} {torchMs,-15} {speedup,-10}");
        }
        Console.WriteLine();

        // ── Conv2D: small + big ──────────────────────────────────────
        // Small: ResNet-style residual block conv (single image,
        // 64 channels, 56×56, 3×3 kernel). Big: ImageNet-style stem
        // conv (single image, 3 → 64 channels, 224×224, 7×7 stride 2).
        // Both common shapes in the issue's "Conv2D (small + big)"
        // line item.
        Console.WriteLine("--- Conv2D (single image) ---");
        Console.WriteLine($"{"Shape",-30} {"AiDotNet (ms)",-15} {"PyTorch (ms)",-15} {"Speedup",-10}");
        var convShapes = new[]
        {
            (n: 1, c: 64, h: 56, w: 56, oc: 64, kh: 3, kw: 3, stride: 1, pad: 1, label: "1x64x56x56 / 64x64x3x3 s1p1"),
            (n: 1, c: 3, h: 224, w: 224, oc: 64, kh: 7, kw: 7, stride: 2, pad: 3, label: "1x3x224x224 / 64x3x7x7 s2p3"),
        };
        foreach (var s in convShapes)
        {
            double aiMs = TimeConv2D(engine, s.n, s.c, s.h, s.w, s.oc, s.kh, s.kw, s.stride, s.pad,
                warmup: 3, iters: 10);
            string args = $"{s.n}x{s.c}x{s.h}x{s.w};{s.oc}x{s.c}x{s.kh}x{s.kw};{s.stride};{s.pad}";
            var torch = runner.TryRun(PythonBaselineRunner.Baseline.Torch294,
                "conv2d", args, warmup: 3, iters: 10);
            string torchMs = torch?.MedianMs.ToString("F3") ?? "skip";
            string speedup = FormatSpeedup(aiMs, torch?.MedianMs);
            Console.WriteLine($"{s.label,-30} {aiMs,-15:F3} {torchMs,-15} {speedup,-10}");
        }
        Console.WriteLine();

        // ── FlashAttention: 3 seq lengths ────────────────────────────
        Console.WriteLine("--- FlashAttention<float> (rank-4 [B=2, H=4, Sq, D=32]) ---");
        Console.WriteLine($"{"Sq",-8} {"AiDotNet (ms)",-15} {"PyTorch (ms)",-15} {"Speedup",-10}");
        foreach (var sq in new[] { 64, 128, 512 })
        {
            double aiMs = TimeFlashAttention(2, 4, sq, 32, warmup: 5, iters: 20);
            var torch = runner.TryRun(PythonBaselineRunner.Baseline.Torch294,
                "attn", $"2x4x{sq}x32;false", warmup: 5, iters: 20);
            string torchMs = torch?.MedianMs.ToString("F3") ?? "skip";
            string speedup = FormatSpeedup(aiMs, torch?.MedianMs);
            Console.WriteLine($"{sq,-8} {aiMs,-15:F3} {torchMs,-15} {speedup,-10}");
        }
        Console.WriteLine();

        // ── LayerNorm ────────────────────────────────────────────────
        Console.WriteLine("--- LayerNorm ---");
        Console.WriteLine($"{"Shape",-20} {"AiDotNet (ms)",-15} {"PyTorch (ms)",-15} {"Speedup",-10}");
        foreach (var (b, f) in new[] { (32, 768), (32, 1024) })
        {
            double aiMs = TimeLayerNorm(engine, b, f, warmup: 10, iters: 50);
            var torch = runner.TryRun(PythonBaselineRunner.Baseline.Torch294,
                "layernorm", $"{b}x{f}", warmup: 10, iters: 50);
            string torchMs = torch?.MedianMs.ToString("F3") ?? "skip";
            string speedup = FormatSpeedup(aiMs, torch?.MedianMs);
            Console.WriteLine($"{b}x{f,-12} {aiMs,-15:F3} {torchMs,-15} {speedup,-10}");
        }
        Console.WriteLine();

        // ── BCE forward + backward (the audit's named target) ──────
        Console.WriteLine("--- Binary Cross Entropy forward + backward ---");
        Console.WriteLine($"{"N",-12} {"AiDotNet (ms)",-15} {"PyTorch (ms)",-15} {"Speedup",-10}");
        foreach (var n in new[] { 1024, 65536 })
        {
            double aiMs = TimeBceForwardBackward(engine, n, warmup: 10, iters: 50);
            var torch = runner.TryRun(PythonBaselineRunner.Baseline.Torch294,
                "bcederiv", n.ToString(), warmup: 10, iters: 50);
            string torchMs = torch?.MedianMs.ToString("F3") ?? "skip";
            string speedup = FormatSpeedup(aiMs, torch?.MedianMs);
            Console.WriteLine($"{n,-12} {aiMs,-15:F3} {torchMs,-15} {speedup,-10}");
        }
        Console.WriteLine();
        Console.WriteLine("Done. Both columns are per-iteration medians (not means);");
        Console.WriteLine("speedup = PyTorch median / AiDotNet median. > 1× means AiDotNet wins.");
    }

    /// <summary>
    /// Median speedup formatter that matches the "median vs median"
    /// timing protocol — both <paramref name="aiMedianMs"/> and
    /// <paramref name="torchMedianMs"/> must come from a per-iter median
    /// reduction. Returns <c>"-"</c> when PyTorch was skipped, and
    /// <c>"n/a"</c> when AiDotNet's median is so close to zero that the
    /// ratio would either overflow or be dominated by stopwatch
    /// resolution noise. Threshold is 1µs (1e-3 ms) — anything below
    /// that is below <see cref="Stopwatch"/>'s realistic on-Windows
    /// resolution and the ratio is uninformative.
    /// </summary>
    private static string FormatSpeedup(double aiMedianMs, double? torchMedianMs)
    {
        if (torchMedianMs is null) return "-";
        if (!(aiMedianMs > 1e-3)) return "n/a";
        double ratio = torchMedianMs.Value / aiMedianMs;
        return ratio.ToString("F2", CultureInfo.InvariantCulture) + "×";
    }

    /// <summary>
    /// Returns the median of an unsorted span of per-iteration sample
    /// times in ms. We use median (not mean) so a single GC pause
    /// doesn't blow up the headline number — same protocol the
    /// Python baseline (<c>run_torch_294.py</c>) reports.
    /// </summary>
    private static double Median(double[] samples)
    {
        var sorted = (double[])samples.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        if (n == 0) return 0.0;
        return (n & 1) == 1
            ? sorted[n / 2]
            : 0.5 * (sorted[(n / 2) - 1] + sorted[n / 2]);
    }

    private static double TimeMatMul(CpuEngine engine, int m, int k, int n, int warmup, int iters)
    {
        var rng = new Random(1);
        var aData = new float[m * k];
        var bData = new float[k * n];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);
        var a = new Tensor<float>(aData, new[] { m, k });
        var b = new Tensor<float>(bData, new[] { k, n });
        for (int i = 0; i < warmup; i++) engine.TensorMatMul(a, b);
        // Per-iter median (not mean): a single stop-the-world GC pause
        // would distort the mean by an order of magnitude on small
        // shapes, but the median is robust and matches the protocol
        // run_torch_294.py uses for the PyTorch side.
        var samples = new double[iters];
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            engine.TensorMatMul(a, b);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        return Median(samples);
    }

    private static double TimeConv2D(CpuEngine engine,
        int n, int c, int h, int w, int oc, int kh, int kw, int stride, int pad,
        int warmup, int iters)
    {
        var rng = new Random(5);
        var xData = new float[n * c * h * w];
        var kData = new float[oc * c * kh * kw];
        for (int i = 0; i < xData.Length; i++) xData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kData.Length; i++) kData[i] = (float)(rng.NextDouble() * 2 - 1);
        var x = new Tensor<float>(xData, new[] { n, c, h, w });
        var kT = new Tensor<float>(kData, new[] { oc, c, kh, kw });
        for (int i = 0; i < warmup; i++) engine.Conv2D(x, kT, stride: stride, padding: pad);
        var samples = new double[iters];
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            engine.Conv2D(x, kT, stride: stride, padding: pad);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        return Median(samples);
    }

    private static double TimeFlashAttention(int b, int h, int sq, int d, int warmup, int iters)
    {
        var rng = new Random(2);
        var qData = new float[b * h * sq * d];
        var kData = new float[b * h * sq * d];
        var vData = new float[b * h * sq * d];
        for (int i = 0; i < qData.Length; i++) qData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kData.Length; i++) kData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < vData.Length; i++) vData[i] = (float)(rng.NextDouble() * 2 - 1);
        var q = new Tensor<float>(qData, new[] { b, h, sq, d });
        var kT = new Tensor<float>(kData, new[] { b, h, sq, d });
        var v = new Tensor<float>(vData, new[] { b, h, sq, d });
        for (int i = 0; i < warmup; i++) FlashAttention<float>.Forward(q, kT, v);
        var samples = new double[iters];
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            FlashAttention<float>.Forward(q, kT, v);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        return Median(samples);
    }

    private static double TimeLayerNorm(CpuEngine engine, int b, int f, int warmup, int iters)
    {
        var rng = new Random(3);
        var xData = new float[b * f];
        var gData = new float[f];
        var bData = new float[f];
        for (int i = 0; i < xData.Length; i++) xData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < f; i++) { gData[i] = 1f; bData[i] = 0f; }
        var x = new Tensor<float>(xData, new[] { b, f });
        var gamma = new Tensor<float>(gData, new[] { f });
        var beta = new Tensor<float>(bData, new[] { f });
        for (int i = 0; i < warmup; i++) engine.TensorLayerNorm(x, gamma, beta);
        var samples = new double[iters];
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            engine.TensorLayerNorm(x, gamma, beta);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        return Median(samples);
    }

    private static double TimeBceForwardBackward(CpuEngine engine, int n, int warmup, int iters)
    {
        var rng = new Random(4);
        var pData = new float[n];
        var tData = new float[n];
        for (int i = 0; i < n; i++)
        {
            pData[i] = (float)(rng.NextDouble() * 0.96 + 0.02); // [0.02, 0.98]
            tData[i] = rng.NextDouble() < 0.5 ? 0f : 1f;
        }
        var pred = new Tensor<float>(pData, new[] { n });
        var target = new Tensor<float>(tData, new[] { n });
        float eps = 1e-7f;
        for (int i = 0; i < warmup; i++)
        {
            engine.TensorBinaryCrossEntropy(pred, target, eps);
            engine.TensorBinaryCrossEntropyBackward(pred, target, eps);
        }
        var samples = new double[iters];
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            engine.TensorBinaryCrossEntropy(pred, target, eps);
            engine.TensorBinaryCrossEntropyBackward(pred, target, eps);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        return Median(samples);
    }
}
#endif
