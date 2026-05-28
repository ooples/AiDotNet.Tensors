using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Differentiator benchmark D.4 — pre-packed weight inference loop. In a real
/// inference server the model weights (B in C = A·B) are constant across
/// requests. AiDotNet's <c>PrePackB</c> packs B once and reuses the packed
/// buffer per call. PyTorch's matmul re-packs B inside the kernel on every
/// call — the packed layout isn't exposed to userspace.
/// </summary>
internal static class FrozenWeightInferenceBench
{
    public static void Run()
    {
        Console.WriteLine("=== Differentiator: pre-packed weight inference loop ===");
        Console.WriteLine();

        // Inference-style shape: small batch M, square weight N=K. Pack-B fraction
        // is a meaningful share of total time at this shape.
        const int M = 8, N = 1024, K = 1024;
        const int iters = 200;

        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Baseline: live-pack B every call (the path PyTorch is locked into).
        var cBaseline = new float[M * N];
        for (int w = 0; w < 10; w++)
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cBaseline, N, M, N, K);
        var swBaseline = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cBaseline, N, M, N, K);
        swBaseline.Stop();
        double baselineMsPerCall = swBaseline.Elapsed.TotalMilliseconds / iters;

        // Pre-packed: pack B once, reuse the cached pack every call.
        var cPrePack = new float[M * N];
        var handle = BlasManagedLib.PrePackB<float>(b, N, false, K, N);
        try
        {
            var opts = new BlasOptions<float> { PackedB = handle };
            for (int w = 0; w < 10; w++)
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cPrePack, N, M, N, K, opts);
            var swPrePack = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cPrePack, N, M, N, K, opts);
            swPrePack.Stop();
            double prePackMsPerCall = swPrePack.Elapsed.TotalMilliseconds / iters;
            double speedup = baselineMsPerCall / prePackMsPerCall;

            Console.WriteLine($"  Baseline (re-pack per call): {baselineMsPerCall:F3} ms/call");
            Console.WriteLine($"  Pre-pack (B cached):         {prePackMsPerCall:F3} ms/call");
            Console.WriteLine($"  Speedup:                     {speedup:F2}×");
            Console.WriteLine();
            Console.WriteLine($"  Cumulative over {iters} calls: baseline {swBaseline.Elapsed.TotalMilliseconds:F1} ms, " +
                              $"pre-pack {swPrePack.Elapsed.TotalMilliseconds:F1} ms " +
                              $"(saved {swBaseline.Elapsed.TotalMilliseconds - swPrePack.Elapsed.TotalMilliseconds:F1} ms)");
        }
        finally { handle.Dispose(); }

        Console.WriteLine();
        Console.WriteLine("  Why PyTorch can't match: torch.matmul has no userspace pre-pack API. Every");
        Console.WriteLine("  call re-packs B inside the kernel; inference servers can't hoist it out of");
        Console.WriteLine("  the request hot path.");
    }
}
