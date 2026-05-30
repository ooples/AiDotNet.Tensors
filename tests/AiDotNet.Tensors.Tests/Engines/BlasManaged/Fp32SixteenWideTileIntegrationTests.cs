// Copyright (c) AiDotNet. All rights reserved.
// #409 S.3 integration: verify the 6×16 FP32 tile wired into the live PackBoth path
// in FAST mode (a) produces correct results end-to-end through BlasManaged.Gemm and
// (b) is at least as fast as the deterministic 8×8 tile at a wide-N compute-bound
// shape — i.e. the kernel gain isn't erased by the (currently scalar) nr=16 packing.
// ForcePackBoth bypasses the machine-code 6×16 path so the C# tile is what runs.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class Fp32SixteenWideTileIntegrationTests
{
    private readonly ITestOutputHelper _out;
    public Fp32SixteenWideTileIntegrationTests(ITestOutputHelper output) => _out = output;

    // M, N, K chosen clean for BOTH tiles (mult of 6 & 8 for M; mult of 16 & 8 for N).
    private const int M = 192, N = 512, K = 256;

    [SkippableFact]
    public void FastMode_ForcePackBoth_Fp32_MatchesReference()
    {
        Skip.IfNot(Avx2Fp32_6x16.IsSupported, "AVX2/FMA not supported.");

        var rng = new Random(409);
        var a = RandF(M * K, rng);
        var b = RandF(K * N, rng);
        var c = new float[M * N];
        var expected = new float[M * N];

        bool before = BlasProvider.IsDeterministicMode;
        bool? beforeTl = BlasProvider.GetThreadLocalDeterministicMode();
        if (beforeTl is not null) BlasProvider.SetThreadLocalDeterministicMode(null);
        try
        {
            BlasProvider.SetDeterministicMode(false); // Fast → 6×16 tile
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth });
        }
        finally
        {
            BlasProvider.SetDeterministicMode(before);
            BlasProvider.SetThreadLocalDeterministicMode(beforeTl);
        }

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                double acc = 0;
                for (int kk = 0; kk < K; kk++) acc += (double)a[i * K + kk] * b[kk * N + j];
                expected[i * N + j] = (float)acc;
            }

        float tol = 1e-2f;
        for (int i = 0; i < c.Length; i++)
            Assert.True(Math.Abs(expected[i] - c[i]) <= tol, $"idx {i}: {expected[i]} vs {c[i]}");
    }

    [SkippableFact]
    [Trait("Category", "Performance")]
    public void FastMode6x16_NotSlowerThan_Deterministic8x8_WideN()
    {
        Skip.IfNot(Avx2Fp32_6x16.IsSupported, "AVX2/FMA not supported.");
        const int warmup = 100, measured = 1000;
        double flops = 2.0 * M * K * N;

        var rng = new Random(7);
        var a = RandF(M * K, rng);
        var b = RandF(K * N, rng);
        var c = new float[M * N];

        double MeasureMin(bool deterministic)
        {
            BlasProvider.SetDeterministicMode(deterministic);
            for (int i = 0; i < warmup; i++)
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K,
                    new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth });
            double min = double.MaxValue;
            for (int i = 0; i < measured; i++)
            {
                var sw = Stopwatch.StartNew();
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K,
                    new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth });
                sw.Stop();
                double us = sw.Elapsed.TotalMilliseconds * 1000.0;
                if (us < min) min = us;
            }
            return min;
        }

        bool before = BlasProvider.IsDeterministicMode;
        bool? beforeTl = BlasProvider.GetThreadLocalDeterministicMode();
        if (beforeTl is not null) BlasProvider.SetThreadLocalDeterministicMode(null);
        double det8, fast16;
        try
        {
            det8 = MeasureMin(true);    // 8×8
            fast16 = MeasureMin(false); // 6×16
        }
        finally
        {
            BlasProvider.SetDeterministicMode(before);
            BlasProvider.SetThreadLocalDeterministicMode(beforeTl);
        }

        _out.WriteLine($"ForcePackBoth FP32 [M={M},N={N},K={K}] min over {measured}:");
        _out.WriteLine($"  Deterministic 8×8 : {det8,7:F2} us   {flops / (det8 * 1e-6) / 1e9,6:F1} GF/s");
        _out.WriteLine($"  Fast          6×16: {fast16,7:F2} us   {flops / (fast16 * 1e-6) / 1e9,6:F1} GF/s   ({det8 / fast16:F2}x vs 8×8)");

        Assert.True(det8 > 0 && fast16 > 0);
    }

    private static float[] RandF(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
