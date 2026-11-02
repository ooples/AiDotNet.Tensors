// SCRATCH (probe branch only): 512-bit direct tile efficiency, single thread on L1-resident data, and NN GEMM thread scaling.
#if NET9_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

public class ZzKernelBench
{
    private readonly ITestOutputHelper _o;
    public ZzKernelBench(ITestOutputHelper o) => _o = o;
    private void Line(string s) { _o.WriteLine(s); Console.WriteLine(s); }

    private static (double Min, double Med) Measure(Action run)
    {
        var sw0 = Stopwatch.StartNew(); while (sw0.ElapsedMilliseconds < 250) run();
        var t = new List<double>();
        for (int round = 0; round < 25; round++)
        {
            double best = double.MaxValue;
            for (int i = 0; i < 5; i++) { var sw = Stopwatch.StartNew(); run(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
            t.Add(best);
        }
        t.Sort();
        return (t[0], t[t.Count / 2]);
    }

    [Fact]
    public unsafe void Time()
    {
        if (Environment.GetEnvironmentVariable("GITHUB_ACTIONS") != "true") return;
        var r = new Random(3);
        const int k = 256, reps = 2000;
        var a = new float[6 * k]; var strip = new float[k * 32]; var c = new float[6 * 32];
        for (int i = 0; i < a.Length; i++) a[i] = (float)r.NextDouble();
        for (int i = 0; i < strip.Length; i++) strip[i] = (float)r.NextDouble();
        double flops = 2.0 * 6 * 32 * k * reps;
        var t512 = Measure(() =>
        {
            fixed (float* pa = a, pb = strip, pc = c)
                for (int i = 0; i < reps; i++) SimdGemm.DirectKernel6x32Store(pa, k, pb, 32, pc, 32, k);
        });
        Line($"ZZ kern 6x32 zmm k{k} L1: {flops / (t512.Min * 1e6):F1} GFLOP/s (min {t512.Min:F3} ms, med {t512.Med:F3})  vec512 {SimdGemm.UseVector512Direct}");

        int threads0 = CpuParallelSettings.MaxDegreeOfParallelism;
        foreach (var (m, kk, n) in new[] { (128, 784, 512), (128, 512, 256), (64, 128, 3136) })
        {
            var A = new float[m * kk]; var B = new float[kk * n]; var C = new float[m * n];
            for (int i = 0; i < A.Length; i++) A[i] = (float)r.NextDouble();
            for (int i = 0; i < B.Length; i++) B[i] = (float)r.NextDouble();
            double f = 2.0 * m * kk * n;
            var parts = new List<string>();
            foreach (int th in new[] { 1, 2, 4 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = th;
                var tm = Measure(() => SimdGemm.Sgemm(A, kk, false, B, n, false, C, m, kk, n));
                parts.Add($"t{th} {tm.Min:F4} ms {f / (tm.Min * 1e6):F0} GF/s");
            }
            CpuParallelSettings.MaxDegreeOfParallelism = threads0;
            Line($"ZZ kern sgemm {m}x{kk}x{n}: {string.Join("  ", parts)}");
        }
    }
}
#endif
