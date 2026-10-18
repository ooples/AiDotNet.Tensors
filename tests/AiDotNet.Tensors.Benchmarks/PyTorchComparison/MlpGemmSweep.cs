using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// #475: where the AIsEval MLP (Dense 784->512->128->10 @ bs=128) spends its time.
/// Times each layer's GEMM through the managed kernels -- SgemmSequential (what MlpForward uses for the small
/// layers), SgemmWithCachedB and Sgemm (the parallel dispatch) -- across thread budgets, and MlpForward
/// end to end. Median of 300 after 50 warmups, microseconds.
/// Run with: <c>dotnet run -c Release -- --ab-mlp-gemm</c>
/// </summary>
internal static class MlpGemmSweep
{
    public static void Run()
    {
        const int m = 128;
        var shapes = new (int K, int N)[] { (784, 512), (512, 128), (128, 10) };
        int[] dops = { 1, 2, 4, 8, 16, 32, 64, Environment.ProcessorCount };
        var rng = new Random(1);
        float[] Rand(int n) { var v = new float[n]; for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1); return v; }

        Console.WriteLine($"=== #475 MLP GEMM sweep (M={m}), cores={Environment.ProcessorCount} ===");
        Console.WriteLine($"{"layer",-14}{"kernel",-18}" + string.Concat(Array.ConvertAll(dops, d => $"{"dop" + d,9}")));
        foreach (var (k, n) in shapes)
        {
            var a = Rand(m * k); var b = Rand(k * n); var c = new float[m * n];
            var kernels = new List<(string, Action)>
            {
                ("SgemmSequential", () => SimdGemm.SgemmSequential(a, b, c, m, k, n)),
                ("SgemmWithCachedB", () => SimdGemm.SgemmWithCachedB(a, b, c, m, k, n)),
                ("Sgemm", () => SimdGemm.Sgemm(a, b, c, m, k, n)),
            };
            foreach (var (name, run) in kernels)
            {
                var row = new List<string>();
                foreach (int dop in dops)
                {
                    CpuParallelSettings.MaxDegreeOfParallelism = dop;
                    row.Add($"{Median(run),9:F1}");
                }
                Console.WriteLine($"{$"{m}x{k}x{n}",-14}{name,-18}" + string.Concat(row));
            }
        }

        var engine = new CpuEngine();
        var input = new Tensor<float>(Rand(m * 784), new[] { m, 784 });
        var weights = new List<Tensor<float>>
        {
            new(Rand(784 * 512), new[] { 784, 512 }), new(Rand(512 * 128), new[] { 512, 128 }), new(Rand(128 * 10), new[] { 128, 10 }),
        };
        var biases = new List<Tensor<float>?> { new(Rand(512), new[] { 512 }), new(Rand(128), new[] { 128 }), new(Rand(10), new[] { 10 }) };
        var mrow = new List<string>();
        foreach (int dop in dops)
        {
            CpuParallelSettings.MaxDegreeOfParallelism = dop;
            mrow.Add($"{Median(() => engine.MlpForward(input, weights, biases, FusedActivationType.ReLU, FusedActivationType.None)),9:F1}");
        }
        Console.WriteLine($"{"MlpForward",-14}{"engine",-18}" + string.Concat(mrow));
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
    }

    private static double Median(Action run)
    {
        for (int i = 0; i < 50; i++) run();
        var t = new double[300];
        for (int i = 0; i < t.Length; i++)
        {
            long s = Stopwatch.GetTimestamp();
            run();
            t[i] = (Stopwatch.GetTimestamp() - s) * 1e6 / Stopwatch.Frequency;
        }
        Array.Sort(t);
        return t[t.Length / 2];
    }
}
