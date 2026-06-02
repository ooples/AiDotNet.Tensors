// Copyright (c) AiDotNet. All rights reserved.
// #409 S.4 follow-through: PackAOnlyStrategy must compute correct results for ANY (m, n),
// including m % mr != 0 and n % nr != 0. Previously it threw NotSupportedException for the
// remainder; that surfaced on CI as a hard failure when BlasManaged.Gemm split with one mr
// (Fast 6×8) but the strategy ran with a different mr (Deterministic 4×8) — a process-global
// mode flip / AVX-512-vs-AVX2 hardware difference between the split and the dispatch. These
// tests call the strategy DIRECTLY with deliberately-misaligned shapes (and every supported
// mr×nr tile) so the tail paths are guarded deterministically, independent of dispatch heuristics.

using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class PackAOnlyStrategyTailTests
{
    private readonly ITestOutputHelper _out;
    public PackAOnlyStrategyTailTests(ITestOutputHelper output) => _out = output;

    public static TheoryData<int, int, int, int, int> Cases() => new()
    {
        // m, n, k, mr, nr — each row exercises an M-tail (m%mr!=0) and/or N-tail (n%nr!=0).
        { 190, 508, 200, 4, 8 },   // the exact CI failure: 190%4=2, 508%8=4
        { 190, 508, 200, 6, 8 },   // 190%6=4
        { 186, 504, 200, 4, 8 },   // the recursive sub-shape that threw: 186%4=2
        { 50,  24,  130, 6, 8 },   // small, M-tail 50%6=2
        { 17,  13,  64,  4, 8 },   // tiny, both tails
        { 7,   5,   33,  6, 8 },   // m<mr and n<nr → all-edge, no aligned interior
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void Run_Double_MisalignedShapes_MatchReference(int m, int n, int k, int mr, int nr)
    {
        var rng = new Random(409 + m * 31 + n * 7 + k + mr * 3 + nr);
        var a = RandD(m * k, rng);
        var b = RandD(k * n, rng);
        var c = new double[m * n];

        bool before = BlasProvider.IsDeterministicMode;
        bool? beforeTl = BlasProvider.GetThreadLocalDeterministicMode();
        if (beforeTl is not null) BlasProvider.SetThreadLocalDeterministicMode(null);
        try
        {
            BlasProvider.SetDeterministicMode(false);
            // Call the strategy directly so the tail paths are exercised regardless of
            // BlasManaged.Gemm's dispatch heuristics.
            PackAOnlyStrategy.Run<double>(
                a, k, false, b, n, c, n, m, n, k,
                mc: 256, kc: 256, mr: mr, nr: nr, options: default);
        }
        finally
        {
            BlasProvider.SetDeterministicMode(before);
            BlasProvider.SetThreadLocalDeterministicMode(beforeTl);
        }

        var expected = Reference(a, b, m, n, k);
        double tol = 1e-9 * Math.Max(1, k);
        for (int i = 0; i < c.Length; i++)
            Assert.True(Math.Abs(expected[i] - c[i]) <= tol,
                $"[m={m},n={n},k={k},mr={mr},nr={nr}] idx {i}: {expected[i]} vs {c[i]} (tol={tol})");
    }

    [Theory]
    [InlineData(190, 508, 200, 8, 8)]  // FP32 8×8: 190%8=6, 508%8=4
    [InlineData(50, 24, 130, 6, 16)]   // FP32 6×16: 50%6=2, 24%16=8
    public void Run_Float_MisalignedShapes_MatchReference(int m, int n, int k, int mr, int nr)
    {
        var rng = new Random(900 + m + n + k + mr + nr);
        var a = RandF(m * k, rng);
        var b = RandF(k * n, rng);
        var c = new float[m * n];

        bool before = BlasProvider.IsDeterministicMode;
        bool? beforeTl = BlasProvider.GetThreadLocalDeterministicMode();
        if (beforeTl is not null) BlasProvider.SetThreadLocalDeterministicMode(null);
        try
        {
            BlasProvider.SetDeterministicMode(false);
            PackAOnlyStrategy.Run<float>(
                a, k, false, b, n, c, n, m, n, k,
                mc: 256, kc: 256, mr: mr, nr: nr, options: default);
        }
        finally
        {
            BlasProvider.SetDeterministicMode(before);
            BlasProvider.SetThreadLocalDeterministicMode(beforeTl);
        }

        // FP32 accumulation differs in order between SIMD interior and scalar edge/reference;
        // a relative-ish absolute tolerance scaled by k is appropriate.
        float tol = 1e-3f * Math.Max(1, k);
        for (int i = 0; i < c.Length; i++)
        {
            double accRef = 0;
            int row = i / n, col = i % n;
            for (int p = 0; p < k; p++) accRef += (double)a[row * k + p] * b[p * n + col];
            Assert.True(Math.Abs(accRef - c[i]) <= tol,
                $"[m={m},n={n},k={k},mr={mr},nr={nr}] idx {i}: {accRef} vs {c[i]} (tol={tol})");
        }
    }

    private static double[] Reference(double[] a, double[] b, int m, int n, int k)
    {
        var e = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
                e[i * n + j] = acc;
            }
        return e;
    }

    private static double[] RandD(int n, Random rng)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.NextDouble() * 2 - 1;
        return a;
    }

    private static float[] RandF(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
