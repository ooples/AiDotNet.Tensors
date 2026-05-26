using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using AiDotNet.Tensors.Engines.BlasManaged;
using TorchSharp;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Layer E (#375) — head-to-head BlasManaged vs TorchSharp eager. Runs an
/// inline list of representative + worst-loss shapes through both backends,
/// capturing median / p95 / p99 / per-call allocation. Emits a Markdown
/// report with the strict-gate verdict (BlasManaged ≥ 1.5× faster = PASS).
///
/// <para>
/// Inline shapes (not the test-project catalog) keep the benchmark project
/// free of a test-project reference. The shapes mirror the
/// <c>--ab-blas-small-square-fp64</c> probe set — the ones that actually
/// stress the dispatcher + microkernels.
/// </para>
/// </summary>
internal static class HeadToHeadCatalogBench
{
    private readonly record struct Shape(string Name, int M, int N, int K, bool Fp64);

    private static readonly Shape[] Shapes =
    {
        new("64cube_fp32",          64,   64,   64,  false),
        new("64cube_fp64",          64,   64,   64,  true),
        new("BERT_attn_score_fp64", 96,   128,  64,  true),
        new("128cube_fp64",         128,  128,  128, true),
        new("ResNet50_l1_fp32",     3136, 64,   64,  false),
        new("MobileNetV2_pw_fp32",  3136, 32,   32,  false),
        new("thinK_512x512x64_fp64",512,  512,  64,  true),
        new("BERT_FFN_up_fp32",     1024, 3072, 768, false),
        new("FFN_128x768x768_fp64", 128,  768,  768, true),
        new("ViT_QKV_fp32",         197,  768,  768, false),
    };

    public static void Run(string outPath)
    {
        Console.WriteLine("=== Layer E — BlasManaged vs TorchSharp eager (inline catalog) ===");
        torch.set_grad_enabled(false);
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}, torch threads={torch.get_num_threads()}");
        Console.WriteLine();

        var rows = new List<Row>();
        foreach (var s in Shapes)
        {
            var row = s.Fp64 ? MeasureDouble(s) : MeasureFloat(s);
            rows.Add(row);
            Console.WriteLine($"  {s.Name,-26} AiDN {row.AiMedianMs,8:F3}  torch {row.TorchMedianMs,8:F3}  " +
                              $"= {row.MedianRatio,5:F2}×  [{Verdict(row.MedianRatio)}]");
        }

        var dir = Path.GetDirectoryName(outPath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(outPath, BuildMarkdown(rows));
        Console.WriteLine();
        Console.WriteLine($"Report: {outPath}");
    }

    private static int ItersFor(Shape s)
    {
        long work = (long)s.M * s.N * s.K;
        return work > 100_000_000L ? 30 : work > 10_000_000L ? 100 : 200;
    }

    private static Row MeasureFloat(Shape s)
    {
        int iters = ItersFor(s);
        var rng = new Random(42);
        var a = new float[s.M * s.K];
        var b = new float[s.K * s.N];
        var c = new float[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var (aiMed, aiP95, aiP99, aiAlloc) = TimeAi(() =>
            BlasManagedLib.Gemm<float>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K), iters);

        using var aT = torch.tensor(a, new long[] { s.M, s.K }, device: torch.CPU);
        using var bT = torch.tensor(b, new long[] { s.K, s.N }, device: torch.CPU);
        var (tMed, tP95, tP99, tAlloc) = TimeTorch(aT, bT, iters);

        return Build(s, aiMed, aiP95, aiP99, aiAlloc, tMed, tP95, tP99, tAlloc);
    }

    private static Row MeasureDouble(Shape s)
    {
        int iters = ItersFor(s);
        var rng = new Random(42);
        var a = new double[s.M * s.K];
        var b = new double[s.K * s.N];
        var c = new double[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        var (aiMed, aiP95, aiP99, aiAlloc) = TimeAi(() =>
            BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K), iters);

        using var aT = torch.tensor(a, new long[] { s.M, s.K }, device: torch.CPU);
        using var bT = torch.tensor(b, new long[] { s.K, s.N }, device: torch.CPU);
        var (tMed, tP95, tP99, tAlloc) = TimeTorch(aT, bT, iters);

        return Build(s, aiMed, aiP95, aiP99, aiAlloc, tMed, tP95, tP99, tAlloc);
    }

    private static (double med, double p95, double p99, double alloc) TimeAi(Action gemm, int iters)
    {
        for (int i = 0; i < 5; i++) gemm();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);

        long allocStart = GC.GetAllocatedBytesForCurrentThread();
        var times = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            gemm();
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }
        double alloc = (GC.GetAllocatedBytesForCurrentThread() - allocStart) / (double)iters;
        return Percentiles(times, alloc);
    }

    private static (double med, double p95, double p99, double alloc) TimeTorch(
        torch.Tensor aT, torch.Tensor bT, int iters)
    {
        for (int i = 0; i < 5; i++) torch.matmul(aT, bT).Dispose();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);

        long allocStart = GC.GetAllocatedBytesForCurrentThread();
        var times = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            using (var r = torch.matmul(aT, bT)) { }
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }
        double alloc = (GC.GetAllocatedBytesForCurrentThread() - allocStart) / (double)iters;
        return Percentiles(times, alloc);
    }

    private static (double med, double p95, double p99, double alloc) Percentiles(double[] times, double alloc)
    {
        Array.Sort(times);
        int n = times.Length;
        return (times[n / 2],
                times[Math.Min(n - 1, (int)(n * 0.95))],
                times[Math.Min(n - 1, (int)(n * 0.99))],
                alloc);
    }

    private static Row Build(Shape s,
        double aiMed, double aiP95, double aiP99, double aiAlloc,
        double tMed, double tP95, double tP99, double tAlloc) =>
        new()
        {
            Name = s.Name, M = s.M, N = s.N, K = s.K, Dtype = s.Fp64 ? "FP64" : "FP32",
            AiMedianMs = aiMed, AiP95Ms = aiP95, AiP99Ms = aiP99, AiAllocBytes = aiAlloc,
            TorchMedianMs = tMed, TorchP95Ms = tP95, TorchP99Ms = tP99, TorchAllocBytes = tAlloc,
        };

    private static string Verdict(double ratio) =>
        ratio <= 0.67 ? "PASS" : ratio <= 1.05 ? "WARN" : "FAIL";

    private static string Glyph(double ratio) =>
        ratio <= 0.67 ? "PASS" : ratio <= 1.05 ? "WARN" : "FAIL";

    private static string BuildMarkdown(List<Row> rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# BlasManaged vs PyTorch (TorchSharp eager) — strict-gate head-to-head");
        sb.AppendLine();
        sb.AppendLine($"Generated: {DateTime.UtcNow:u}");
        sb.AppendLine($"Host: {Environment.ProcessorCount} cores, torch threads {torch.get_num_threads()}");
        sb.AppendLine();
        sb.AppendLine("Ratio = AiDotNet / Torch (lower is better for us). Gate: ≤0.67 (≥1.5× faster) = PASS, ≤1.05 (parity) = WARN, else FAIL.");
        sb.AppendLine();
        sb.AppendLine("| Shape | M×N×K | Dtype | AiDN med | Torch med | Med | AiDN p95 | Torch p95 | p95 | AiDN p99 | Torch p99 | p99 |");
        sb.AppendLine("|---|---|---|---:|---:|:-:|---:|---:|:-:|---:|---:|:-:|");
        int pass = 0, warn = 0, fail = 0;
        foreach (var r in rows)
        {
            string vMed = Glyph(r.MedianRatio);
            if (vMed == "PASS") pass++; else if (vMed == "WARN") warn++; else fail++;
            sb.AppendLine($"| {r.Name} | {r.M}×{r.N}×{r.K} | {r.Dtype} | " +
                          $"{r.AiMedianMs:F3} | {r.TorchMedianMs:F3} | {vMed} | " +
                          $"{r.AiP95Ms:F3} | {r.TorchP95Ms:F3} | {Glyph(r.P95Ratio)} | " +
                          $"{r.AiP99Ms:F3} | {r.TorchP99Ms:F3} | {Glyph(r.P99Ratio)} |");
        }
        sb.AppendLine();
        sb.AppendLine($"**Summary (median gate):** {pass} PASS, {warn} WARN, {fail} FAIL of {rows.Count} shapes.");
        sb.AppendLine();
        sb.AppendLine("> Note: AiDotNet measured on the default dispatch path (EnableAutotuneV2 off). " +
                      "Raw-GEMM parity with MKL-backed torch on compute-bound shapes is the hardest gate; " +
                      "see the differentiator benchmarks (cold-start, determinism, per-call threads, frozen-weight) " +
                      "for the structural wins PyTorch cannot match regardless of GFLOPS.");
        return sb.ToString();
    }

    private sealed class Row
    {
        public string Name = "";
        public int M, N, K;
        public string Dtype = "";
        public double AiMedianMs, AiP95Ms, AiP99Ms, AiAllocBytes;
        public double TorchMedianMs, TorchP95Ms, TorchP99Ms, TorchAllocBytes;
        public double MedianRatio => TorchMedianMs > 0 ? AiMedianMs / TorchMedianMs : double.PositiveInfinity;
        public double P95Ratio => TorchP95Ms > 0 ? AiP95Ms / TorchP95Ms : double.PositiveInfinity;
        public double P99Ratio => TorchP99Ms > 0 ? AiP99Ms / TorchP99Ms : double.PositiveInfinity;
    }
}
