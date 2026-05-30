// Copyright (c) AiDotNet. All rights reserved.
// #477: isolate the LSTM recurrent GEMM shape [M=128, K=64, N=256] and compare the
// SimdGemm dispatch paths head-to-head, reporting MIN-of-many GF/s (the noise-robust
// best-case estimator) so a routing/kernel win is measurable on a busy machine.
// Category=Performance => excluded from the normal/CI run.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

[Trait("Category", "Performance")]
public class LstmGemmShapeMicroBench
{
    private readonly ITestOutputHelper _out;
    public LstmGemmShapeMicroBench(ITestOutputHelper output) => _out = output;

    [Fact]
    public void RecurrentGemmShape_PathComparison_MinGflops()
    {
        const int m = 128, k = 64, n = 256;
        const int warmup = 200, measured = 2000;
        double flops = 2.0 * m * k * n;

        var rng = new Random(12345);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var c = new float[m * n];

        double MinMs(Action call)
        {
            for (int i = 0; i < warmup; i++) call();
            double min = double.MaxValue;
            for (int i = 0; i < measured; i++)
            {
                var sw = Stopwatch.StartNew();
                call();
                sw.Stop();
                double ms = sw.Elapsed.TotalMilliseconds;
                if (ms < min) min = ms;
            }
            return min;
        }

        double cachedB = MinMs(() => SimdGemm.SgemmWithCachedB(a, b, c, m, k, n));
        double seq = MinMs(() => SimdGemm.SgemmSequential(a, b, c, m, k, n));
        double gen = MinMs(() => SimdGemm.Sgemm(a, b, c, m, k, n));

        _out.WriteLine($"LSTM recurrent GEMM [M={m}, K={k}, N={n}] — min over {measured} runs:");
        _out.WriteLine($"  SgemmWithCachedB (current) : {cachedB * 1000:F2} us   {flops / (cachedB * 1e-3) / 1e9:F1} GF/s");
        _out.WriteLine($"  SgemmSequential  (direct)  : {seq * 1000:F2} us   {flops / (seq * 1e-3) / 1e9:F1} GF/s");
        _out.WriteLine($"  Sgemm            (general) : {gen * 1000:F2} us   {flops / (gen * 1e-3) / 1e9:F1} GF/s");

        Assert.True(cachedB > 0 && seq > 0 && gen > 0);
    }
}
