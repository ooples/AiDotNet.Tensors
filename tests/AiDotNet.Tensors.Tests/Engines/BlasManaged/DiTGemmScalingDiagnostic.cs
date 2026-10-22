using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

// #653: thread scaling of the DiT-XL/2 projection GEMMs (M = 256 tokens) through the default GEMM entry the
// engine's FusedLinear uses. Reports GFLOP/s per thread count and whether the per-tile Goto path ran.
// Set AIDOTNET_PERF_RUNNER=1 on a quiet machine; reports data, asserts nothing about speed.
[Trait("Category", "Performance")]
[Collection("BlasManaged-Perf-Serial")]
public sealed class DiTGemmScalingDiagnostic
{
    private readonly ITestOutputHelper _output;
    public DiTGemmScalingDiagnostic(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public void Sweep()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1", "Set AIDOTNET_PERF_RUNNER=1 on a quiet machine.");
        var shapes = new (string Name, int M, int K, int N)[]
        {
            ("qkv", 256, 1152, 3456), ("proj", 256, 1152, 1152), ("fc1", 256, 1152, 4608), ("fc2", 256, 4608, 1152),
        };
        int[] threads = { 1, 8, 16, 32, 64 };
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            foreach (var s in shapes)
            {
                var a = Random(s.M * s.K, 1); var b = Random(s.K * s.N, 2); var c = new float[s.M * s.N];
                double gflop = 2.0 * s.M * s.N * s.K / 1e9;
                var best = new double[threads.Length];
                for (int i = 0; i < best.Length; i++) best[i] = double.MaxValue;
                long[] tiles = new long[threads.Length];
                // Three interleaved rounds over thread counts; best of each round's median.
                for (int round = 0; round < 3; round++)
                {
                    for (int t = 0; t < threads.Length; t++)
                    {
                        CpuParallelSettings.MaxDegreeOfParallelism = threads[t];
                        for (int w = 0; w < 3; w++) BlasManagedLib.Gemm<float>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K);
                        GotoGemmFp32.ResetTiming(); GotoGemmFp32.s_timing = true;
                        BlasManagedLib.Gemm<float>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K);
                        GotoGemmFp32.s_timing = false; tiles[t] = GotoGemmFp32.s_tileCount;
                        var ms = new double[7];
                        for (int r = 0; r < ms.Length; r++)
                        {
                            var sw = Stopwatch.StartNew();
                            BlasManagedLib.Gemm<float>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K);
                            ms[r] = sw.Elapsed.TotalMilliseconds;
                        }
                        Array.Sort(ms);
                        best[t] = Math.Min(best[t], ms[ms.Length / 2]);
                    }
                }
                var line = $"{s.Name} {s.M}x{s.K}x{s.N}:";
                for (int t = 0; t < threads.Length; t++)
                    line += $"  t{threads[t]} {best[t]:F2} ms {gflop / (best[t] / 1e3):F0} GF/s x{best[0] / best[t]:F1} tiles={tiles[t]}";
                _output.WriteLine(line);
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    private static float[] Random(int n, int seed)
    {
        var r = new Random(seed); var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(r.NextDouble() * 2 - 1);
        return v;
    }
}