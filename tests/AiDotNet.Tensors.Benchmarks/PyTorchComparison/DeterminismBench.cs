using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Differentiator benchmark D.2 — bit-exact reproducibility across thread
/// counts. Runs the same GEMM in Deterministic mode at 1/2/4/8/16 threads and
/// asserts the output is byte-identical across runs. PyTorch cannot guarantee
/// this on multi-threaded CPU GEMM — its parallel reduction order varies with
/// torch.set_num_threads(), so results differ in the low bits across thread
/// counts.
/// </summary>
internal static class DeterminismBench
{
    public static void Run()
    {
        Console.WriteLine("=== Differentiator: bit-exact determinism across thread counts ===");
        Console.WriteLine();

        const int M = 256, N = 256, K = 256;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var threadCounts = new[] { 1, 2, 4, 8, 16 };
        byte[]? referenceBytes = null;
        bool allMatch = true;

        foreach (int nt in threadCounts)
        {
            var c = new float[M * N];
            var opts = new BlasOptions<float>
            {
                Mode = BlasMode.Deterministic,
                NumThreads = nt,
            };
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K, opts);
            var bytes = ToBytes(c);
            if (referenceBytes is null)
            {
                referenceBytes = bytes;
                Console.WriteLine($"  NumThreads={nt,2}: reference");
            }
            else
            {
                bool match = bytes.AsSpan().SequenceEqual(referenceBytes);
                Console.WriteLine($"  NumThreads={nt,2}: " + (match ? "BIT-EXACT MATCH" : "MISMATCH"));
                if (!match) allMatch = false;
            }
        }

        Console.WriteLine();
        Console.WriteLine(allMatch
            ? "  Result: AiDotNet Deterministic mode is bit-identical across thread counts."
            : "  Result: MISMATCH — deterministic mode regression, investigate.");
        Console.WriteLine();
        Console.WriteLine("  Why PyTorch can't match: torch.use_deterministic_algorithms(True) documents");
        Console.WriteLine("  reproducibility only within a single set_num_threads() value — the CPU");
        Console.WriteLine("  matmul's parallel reduction order changes with thread count.");
    }

    private static byte[] ToBytes(float[] arr)
    {
        var bytes = new byte[arr.Length * sizeof(float)];
        Buffer.BlockCopy(arr, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
