using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using TorchSharp;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Layer E.2 (#375) — investigate WHY BlasManaged loses to TorchSharp/MKL.
/// Separates the two suspected causes:
///   (1) per-call dispatch/setup overhead (dominates tiny shapes), and
///   (2) raw kernel throughput at steady state (dominates large shapes).
/// Also pins both sides to a single thread to isolate kernel quality from the
/// threading model, and verifies torch actually materialises a correct result
/// (rules out a lazy-eval / measurement artifact).
/// </summary>
internal static class GapInvestigationBench
{
    public static void Run()
    {
        torch.set_grad_enabled(false);
        Console.WriteLine("=== Layer E.2 — gap investigation (BlasManaged vs TorchSharp/MKL) ===");
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}");
        Console.WriteLine();

        // 1) Correctness: is torch.matmul actually computing? Compare to naive ref.
        VerifyTorchComputes();
        Console.WriteLine();

        // 2) Tiny shape: 64³ FP32 — dispatch-overhead-dominated.
        Investigate("64³ FP32", 64, 64, 64);
        // 3) Medium: 512×512×64 FP64-ish (use fp32 for torch parity simplicity).
        Investigate("512×512×512 FP32", 512, 512, 512);
        // 4) Large compute-bound: 1024×3072×768 FP32.
        Investigate("1024×3072×768 FP32", 1024, 3072, 768);

        // 5) TRUE scaling curve via the global MaxDegreeOfParallelism (the only
        //    knob PackBoth's parallel loop actually honours). Tells us whether
        //    the medium-shape gap is kernel-bound (flat single-thread) or
        //    scaling-bound (single-thread fast, parallel inefficient).
        Console.WriteLine();
        ScalingCurve("512×512×512 FP32", 512, 512, 512);
        ScalingCurve("1024×3072×768 FP32", 1024, 3072, 768);
    }

    private static void ScalingCurve(string label, int M, int N, int K)
    {
        Console.WriteLine($"--- scaling curve: {label} (global MaxDOP sweep) ---");
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        double flops = 2.0 * M * N * K;
        int iters = (long)M * N * K > 100_000_000 ? 50 : 300;

        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        double baseGflops = 0;
        try
        {
            Console.WriteLine($"  {"MaxDOP",8}{"min µs",12}{"GFLOPS",12}{"scaling",10}");
            foreach (int dop in new[] { 1, 2, 4, 8, 16 })
            {
                if (dop > Environment.ProcessorCount) break;
                CpuParallelSettings.MaxDegreeOfParallelism = dop;
                var (min, _) = TimeAi(() =>
                    BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K), iters);
                double g = flops / (min * 1e-6) / 1e9;
                if (dop == 1) baseGflops = g;
                Console.WriteLine($"  {dop,8}{min,12:F1}{g,12:F2}{(g / baseGflops),9:F2}×");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
        Console.WriteLine();
    }

    private static void VerifyTorchComputes()
    {
        const int M = 32, K = 16, N = 8;
        var rng = new Random(1);
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble());
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble());

        var refC = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float s = 0;
                for (int kk = 0; kk < K; kk++) s += a[i * K + kk] * b[kk * N + j];
                refC[i * N + j] = s;
            }

        using var aT = torch.tensor(a, new long[] { M, K }, device: torch.CPU);
        using var bT = torch.tensor(b, new long[] { K, N }, device: torch.CPU);
        using var cT = torch.matmul(aT, bT);
        var got = cT.data<float>().ToArray();

        double maxErr = 0;
        for (int i = 0; i < refC.Length; i++) maxErr = Math.Max(maxErr, Math.Abs(refC[i] - got[i]));
        Console.WriteLine($"  torch.matmul correctness: maxErr={maxErr:E3} " +
                          (maxErr < 1e-4 ? "(REAL — materialises correct result)" : "(SUSPECT)"));
    }

    private static void Investigate(string label, int M, int N, int K)
    {
        Console.WriteLine($"--- {label}  (M·N·K = {(long)M * N * K:N0}, {2.0 * M * N * K / 1e9:F3} GFLOP) ---");
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        double flops = 2.0 * M * N * K;
        int iters = (long)M * N * K > 100_000_000 ? 50 : 500;

        // NOTE: the autotune cache is NOT keyed on thread count, and
        // PackBoth's parallel loop uses the GLOBAL MaxDegreeOfParallelism rather
        // than the per-call NumThreads — so the "1-thread" row below does NOT
        // actually run single-threaded for PackBoth; it only differs in the
        // cached blocking factors. Both rows use all cores. Treat the 1-thread
        // row as "kernel + blocking at procs=1 heuristic", not a true serial run.
        // AiDotNet multi-thread (default = Deterministic mode).
        var (aiMtMin, aiMtMed) = TimeAi(() =>
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K), iters);
        var (aiStMin, aiStMed) = TimeAi(() =>
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { NumThreads = 1 }), iters);
        // AiDotNet multi-thread in FAST mode (non-deterministic) — tests the
        // hypothesis that the deterministic guard is what kills PackBoth scaling.
        var (aiFastMin, aiFastMed) = TimeAi(() =>
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { Mode = BlasMode.Fast }), iters);

        // Torch single-thread.
        int prevThreads = torch.get_num_threads();
        torch.set_num_threads(1);
        using (var aT = torch.tensor(a, new long[] { M, K }, device: torch.CPU))
        using (var bT = torch.tensor(b, new long[] { K, N }, device: torch.CPU))
        {
            var (tStMin, tStMed) = TimeTorch(aT, bT, iters);
            torch.set_num_threads(prevThreads);
            var (tMtMin, tMtMed) = TimeTorch(aT, bT, iters);

            Console.WriteLine($"  {"",-18}{"min µs",12}{"med µs",12}{"GFLOPS(min)",14}");
            PrintRow("AiDN 1-thread", aiStMin, aiStMed, flops);
            PrintRow("AiDN N-thr (det)", aiMtMin, aiMtMed, flops);
            PrintRow("AiDN N-thr (fast)", aiFastMin, aiFastMed, flops);
            PrintRow("torch 1-thread", tStMin, tStMed, flops);
            PrintRow("torch N-thread", tMtMin, tMtMed, flops);
            Console.WriteLine($"  → 1-thread kernel gap (AiDN/torch min): {aiStMin / tStMin:F1}×");
            Console.WriteLine($"  → N-thread gap (AiDN/torch min):        {aiMtMin / tMtMin:F1}×");
        }
        Console.WriteLine();
    }

    private static void PrintRow(string label, double minUs, double medUs, double flops)
    {
        double gflops = flops / (minUs * 1e-6) / 1e9;
        Console.WriteLine($"  {label,-18}{minUs,12:F2}{medUs,12:F2}{gflops,14:F2}");
    }

    private static (double minUs, double medUs) TimeAi(Action gemm, int iters)
    {
        for (int i = 0; i < 10; i++) gemm();
        var times = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            gemm();
            sw.Stop();
            times[i] = sw.Elapsed.TotalMicroseconds;
        }
        Array.Sort(times);
        return (times[0], times[iters / 2]);
    }

    private static (double minUs, double medUs) TimeTorch(torch.Tensor aT, torch.Tensor bT, int iters)
    {
        for (int i = 0; i < 10; i++) torch.matmul(aT, bT).Dispose();
        var times = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            using (var r = torch.matmul(aT, bT)) { }
            sw.Stop();
            times[i] = sw.Elapsed.TotalMicroseconds;
        }
        Array.Sort(times);
        return (times[0], times[iters / 2]);
    }
}
