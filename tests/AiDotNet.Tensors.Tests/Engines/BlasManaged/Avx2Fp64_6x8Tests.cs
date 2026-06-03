// Copyright (c) AiDotNet. All rights reserved.
// #409 S.4: correctness gate for the new Avx2Fp64_6x8 microkernel (the higher-
// arithmetic-intensity FP64 tile — the double analog of the S.3 FP32 6x16 win).
// Validates Run and RunStridedB against a scalar reference across several Kc,
// independent of the dispatch wiring.

using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class Avx2Fp64_6x8Tests
{
    private const int Mr = 6, Nr = 8;

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(64)]
    [InlineData(256)]
    public void Run_PackedB_MatchesScalarReference(int kc)
    {
        Skip.IfNot(Avx2Fp64_6x8.IsSupported, "AVX2/FMA not supported on this CPU.");

        var rng = new Random(409 + kc);
        // Packed-A: [Kc × Mr] row-major; packed-B: [Kc × Nr] row-major.
        var packedA = RandD(kc * Mr, rng);
        var packedB = RandD(kc * Nr, rng);
        var c = new double[Mr * Nr];          // ldc = Nr, kernel accumulates onto C (start 0)
        var expected = new double[Mr * Nr];

        Avx2Fp64_6x8.Run(packedA, packedB, c, Nr, kc);

        // Reference: C[i,j] = Σ_k packedA[k*Mr+i] * packedB[k*Nr+j].
        for (int i = 0; i < Mr; i++)
            for (int j = 0; j < Nr; j++)
            {
                double acc = 0;
                for (int k = 0; k < kc; k++)
                    acc += packedA[k * Mr + i] * packedB[k * Nr + j];
                expected[i * Nr + j] = acc;
            }

        AssertClose(expected, c, kc);
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(64)]
    [InlineData(256)]
    public void RunStridedB_MatchesScalarReference(int kc)
    {
        Skip.IfNot(Avx2Fp64_6x8.IsSupported, "AVX2/FMA not supported on this CPU.");

        int ldb = Nr + 5; // non-packed stride to exercise the strided read
        var rng = new Random(977 + kc);
        var packedA = RandD(kc * Mr, rng);
        var b = RandD(kc * ldb, rng);
        var c = new double[Mr * Nr];
        var expected = new double[Mr * Nr];

        Avx2Fp64_6x8.RunStridedB(packedA, b, ldb, c, Nr, kc);

        for (int i = 0; i < Mr; i++)
            for (int j = 0; j < Nr; j++)
            {
                double acc = 0;
                for (int k = 0; k < kc; k++)
                    acc += packedA[k * Mr + i] * b[k * ldb + j];
                expected[i * Nr + j] = acc;
            }

        AssertClose(expected, c, kc);
    }

    private static void AssertClose(double[] expected, double[] actual, int kc)
    {
        double tol = 1e-9 * Math.Max(1, kc); // accumulation grows with kc; double has ~15 sig figs
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol,
                $"idx {i}: expected {expected[i]}, got {actual[i]} (kc={kc}, tol={tol})");
    }

    private static double[] RandD(int n, Random rng)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.NextDouble() * 2 - 1;
        return a;
    }
}
