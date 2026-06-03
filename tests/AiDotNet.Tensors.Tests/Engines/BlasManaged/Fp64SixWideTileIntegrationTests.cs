// Copyright (c) AiDotNet. All rights reserved.
// #409 S.4 integration: verify the 6×8 FP64 tile wired into the live PackBoth path in
// FAST mode produces correct results end-to-end through BlasManaged.Gemm, including the
// M-tail (M % 6 != 0) and N-tail (N % 8 != 0, scalar fallback) paths and the strided
// PackAOnly path. ForcePackBoth/ForcePackAOnly bypass the machine-code path so the C#
// tile is what runs. A perf check confirms the 6×8 isn't slower than the deterministic
// 4×8 at a wide-N compute-bound shape.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class Fp64SixWideTileIntegrationTests
{
    private readonly ITestOutputHelper _out;
    public Fp64SixWideTileIntegrationTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData(192, 512, 256)]   // clean: M mult of 6 & 4, N mult of 8
    [InlineData(190, 508, 200)]   // tails: M%6=4 (M-tail), N%8=4 (N-tail scalar fallback)
    [InlineData(50, 24, 130)]     // small-but-multi-tile: M%6=2, N clean, odd K
    public void FastMode_ForcePackBoth_Fp64_MatchesReference(int m, int n, int k)
    {
        Skip.IfNot(Avx2Fp64_6x8.IsSupported, "AVX2/FMA not supported.");
        RunAndAssert(m, n, k, PackingMode.ForcePackBoth);
    }

    [SkippableTheory]
    [InlineData(192, 512, 256)]   // clean
    [InlineData(190, 508, 200)]   // tails — exercises the strided 6×8 + scalar N-tail
    public void FastMode_ForcePackAOnly_Fp64_MatchesReference(int m, int n, int k)
    {
        Skip.IfNot(Avx2Fp64_6x8.IsSupported, "AVX2/FMA not supported.");
        RunAndAssert(m, n, k, PackingMode.ForcePackAOnly);
    }

    private void RunAndAssert(int m, int n, int k, PackingMode mode)
    {
        var rng = new Random(409 + m + n + k);
        var a = RandD(m * k, rng);
        var b = RandD(k * n, rng);
        var c = new double[m * n];
        var expected = new double[m * n];

        bool before = BlasProvider.IsDeterministicMode;
        bool? beforeTl = BlasProvider.GetThreadLocalDeterministicMode();
        if (beforeTl is not null) BlasProvider.SetThreadLocalDeterministicMode(null);
        try
        {
            BlasProvider.SetDeterministicMode(false); // Fast → 6×8 tile
            BlasManagedLib.Gemm<double>(a, k, false, b, n, false, c, n, m, n, k,
                new BlasOptions<double> { PackingMode = mode });
        }
        finally
        {
            BlasProvider.SetDeterministicMode(before);
            BlasProvider.SetThreadLocalDeterministicMode(beforeTl);
        }

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int kk = 0; kk < k; kk++) acc += a[i * k + kk] * b[kk * n + j];
                expected[i * n + j] = acc;
            }

        double tol = 1e-9 * Math.Max(1, k); // double; accumulation grows with k
        for (int i = 0; i < c.Length; i++)
            Assert.True(Math.Abs(expected[i] - c[i]) <= tol, $"[{mode}] idx {i}: {expected[i]} vs {c[i]} (tol={tol})");
    }

    [SkippableFact]
    [Trait("Category", "Performance")]
    public void FastMode6x8_NotSlowerThan_Deterministic4x8_WideN()
    {
        Skip.IfNot(Avx2Fp64_6x8.IsSupported, "AVX2/FMA not supported.");
        const int M = 192, N = 512, K = 256;
        const int warmup = 100, measured = 1000;
        double flops = 2.0 * M * K * N;

        var rng = new Random(7);
        var a = RandD(M * K, rng);
        var b = RandD(K * N, rng);
        var c = new double[M * N];

        double MeasureMin(bool deterministic)
        {
            BlasProvider.SetDeterministicMode(deterministic);
            for (int i = 0; i < warmup; i++)
                BlasManagedLib.Gemm<double>(a, K, false, b, N, false, c, N, M, N, K,
                    new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth });
            double min = double.MaxValue;
            for (int i = 0; i < measured; i++)
            {
                var sw = Stopwatch.StartNew();
                BlasManagedLib.Gemm<double>(a, K, false, b, N, false, c, N, M, N, K,
                    new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth });
                sw.Stop();
                double us = sw.Elapsed.TotalMilliseconds * 1000.0;
                if (us < min) min = us;
            }
            return min;
        }

        bool before = BlasProvider.IsDeterministicMode;
        bool? beforeTl = BlasProvider.GetThreadLocalDeterministicMode();
        if (beforeTl is not null) BlasProvider.SetThreadLocalDeterministicMode(null);
        double det4, fast6;
        try
        {
            det4 = MeasureMin(true);    // 4×8
            fast6 = MeasureMin(false);  // 6×8
        }
        finally
        {
            BlasProvider.SetDeterministicMode(before);
            BlasProvider.SetThreadLocalDeterministicMode(beforeTl);
        }

        _out.WriteLine($"ForcePackBoth FP64 [M={M},N={N},K={K}] min over {measured}:");
        _out.WriteLine($"  Deterministic 4×8: {det4,7:F2} us   {flops / (det4 * 1e-6) / 1e9,6:F1} GF/s");
        _out.WriteLine($"  Fast          6×8: {fast6,7:F2} us   {flops / (fast6 * 1e-6) / 1e9,6:F1} GF/s   ({det4 / fast6:F2}x vs 4×8)");

        Assert.True(det4 > 0 && fast6 > 0);
        Assert.True(fast6 <= det4 * 1.05, $"6×8 is slower: {fast6} > {det4}");
    }

    private static double[] RandD(int n, Random rng)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.NextDouble() * 2 - 1;
        return a;
    }
}
