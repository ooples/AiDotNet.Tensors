// SCRATCH (probe branch only): Aᵀ·B thin-M direct (what DisableAutotune/Auto run) against forced PackBoth, per thread count.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

[Collection("ZzSerial")]
public class ZzDwBench
{
    private readonly ITestOutputHelper _o;
    public ZzDwBench(ITestOutputHelper o) => _o = o;
    private void Line(string s) { _o.WriteLine(s); Console.WriteLine(s); }

    private static double Min(Action run)
    {
        var sw0 = Stopwatch.StartNew(); while (sw0.ElapsedMilliseconds < 200) run();
        double best = double.MaxValue;
        for (int round = 0; round < 25; round++)
            for (int i = 0; i < 5; i++) { var sw = Stopwatch.StartNew(); run(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
        return best;
    }

    [Fact]
    public void Time()
    {
        if (Environment.GetEnvironmentVariable("GITHUB_ACTIONS") != "true" && Environment.GetEnvironmentVariable("ZZ_CONV") != "1") return;
        var r = new Random(5);
        int threads0 = CpuParallelSettings.MaxDegreeOfParallelism;
        int logical = Environment.ProcessorCount;
        // (M, N, K) of C[M,N] = A[K,M]ᵀ · B[K,N]: weight gradients (M = in, N = out, K = batch) and others in the box.
        foreach (var (M, N, K) in new[] { (784, 512, 128), (512, 256, 128), (1024, 512, 128), (512, 512, 64), (256, 256, 128),
                                          (784, 128, 64), (128, 512, 256), (1024, 256, 512), (512, 128, 1024), (256, 64, 256) })
        {
            var A = new float[K * M]; var B = new float[K * N]; var C = new float[M * N];
            for (int i = 0; i < A.Length; i++) A[i] = (float)r.NextDouble();
            for (int i = 0; i < B.Length; i++) B[i] = (float)r.NextDouble();
            var parts = new List<string>();
            var tl = Environment.GetEnvironmentVariable("ZZ_TLIST");
            int[] tlist = tl is null ? new[] { Math.Max(1, logical / 2), logical } : Array.ConvertAll(tl.Split(','), int.Parse);
            foreach (int th in tlist)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = th;
                double thin = 0, pb = 0;
                // interleaved: thin, packboth, thin, packboth
                for (int rep = 0; rep < 2; rep++)
                {
                    double t1 = Min(() => BlasManaged.Gemm<float>(A, M, true, B, N, false, C.AsSpan(), N, M, N, K,
                        new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune }));
                    double t2 = Min(() => BlasManaged.Gemm<float>(A, M, true, B, N, false, C.AsSpan(), N, M, N, K,
                        new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth }));
                    thin = rep == 0 ? t1 : Math.Min(thin, t1); pb = rep == 0 ? t2 : Math.Min(pb, t2);
                }
                parts.Add($"t{th} thin {thin:F4} packboth {pb:F4} ({thin / pb:F2}x)");
            }
            CpuParallelSettings.MaxDegreeOfParallelism = threads0;
            Line($"ZZ dW {M}x{N}x{K} served {BlasManaged.ThinMDirectServesShape(M, N, K, true, false)}: {string.Join("  ", parts)}");
        }
    }
}
