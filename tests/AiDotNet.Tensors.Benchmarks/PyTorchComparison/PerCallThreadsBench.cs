using System;
using System.Diagnostics;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.BlasManaged;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Differentiator benchmark D.3 — per-call thread budget for multi-tenant
/// inference. PyTorch's torch.set_num_threads() is process-global; two
/// concurrent requests from different tenants cannot get different thread
/// budgets. AiDotNet's BlasOptions{T}.NumThreads is per-call.
///
/// Scenario: two tasks concurrently submit GEMMs with different NumThreads
/// (one tenant 4, the other 12 on a 16-core box). Both complete without one
/// starving the other.
/// </summary>
internal static class PerCallThreadsBench
{
    public static void Run()
    {
        Console.WriteLine("=== Differentiator: per-call thread budget (multi-tenant inference) ===");
        Console.WriteLine();

        if (Environment.ProcessorCount < 8)
        {
            Console.WriteLine("  Skipped — needs ≥ 8 cores to demonstrate the split.");
            return;
        }

        const int M = 512, N = 512, K = 512;
        const int iters = 50;

        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c1 = new float[M * N];
        var c2 = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // BlasOptions<T> is a ref struct — it can't be captured by a lambda, so
        // each task constructs its own per-call thread budget inline.
        for (int i = 0; i < 5; i++)
        {
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c1, N, M, N, K, new BlasOptions<float> { NumThreads = 4 });
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c2, N, M, N, K, new BlasOptions<float> { NumThreads = 12 });
        }

        var sw = Stopwatch.StartNew();
        var taskA = Task.Run(() =>
        {
            for (int i = 0; i < iters; i++)
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c1, N, M, N, K, new BlasOptions<float> { NumThreads = 4 });
        });
        var taskB = Task.Run(() =>
        {
            for (int i = 0; i < iters; i++)
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c2, N, M, N, K, new BlasOptions<float> { NumThreads = 12 });
        });
        Task.WaitAll(taskA, taskB);
        sw.Stop();

        double totalMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"  Concurrent A(4 thr) + B(12 thr) × {iters} each: {totalMs:F1} ms total " +
                          $"= {totalMs / iters:F2} ms/call avg");
        Console.WriteLine();
        Console.WriteLine("  Why PyTorch can't match: torch.set_num_threads(N) is process-global. A");
        Console.WriteLine("  server hosting two tenants must pick ONE thread count for both — starving");
        Console.WriteLine("  the small request or under-utilising the large one. No per-call primitive exists.");
    }
}
