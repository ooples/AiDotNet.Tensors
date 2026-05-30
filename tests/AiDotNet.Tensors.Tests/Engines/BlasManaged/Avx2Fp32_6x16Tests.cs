// Copyright (c) AiDotNet. All rights reserved.
// #409 S.3: correctness gate for the new Avx2Fp32_6x16 microkernel (the higher-
// arithmetic-intensity replacement for the load-bound 8x8). Validates Run and
// RunStridedB against a scalar reference across several Kc, BEFORE the kernel is
// wired into the core FP32 GEMM dispatch.

using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class Avx2Fp32_6x16Tests
{
    private const int Mr = 6, Nr = 16;

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(64)]
    [InlineData(256)]
    public void Run_PackedB_MatchesScalarReference(int kc)
    {
        Skip.IfNot(Avx2Fp32_6x16.IsSupported, "AVX2/FMA not supported on this CPU.");

        var rng = new Random(409 + kc);
        // Packed-A: [Kc × Mr] row-major; packed-B: [Kc × Nr] row-major.
        var packedA = RandF(kc * Mr, rng);
        var packedB = RandF(kc * Nr, rng);
        var c = new float[Mr * Nr];          // ldc = Nr, kernel accumulates onto C (start 0)
        var expected = new float[Mr * Nr];

        Avx2Fp32_6x16.Run(packedA, packedB, c, Nr, kc);

        // Reference: C[i,j] = Σ_k packedA[k*Mr+i] * packedB[k*Nr+j].
        for (int i = 0; i < Mr; i++)
            for (int j = 0; j < Nr; j++)
            {
                double acc = 0;
                for (int k = 0; k < kc; k++)
                    acc += (double)packedA[k * Mr + i] * packedB[k * Nr + j];
                expected[i * Nr + j] = (float)acc;
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
        Skip.IfNot(Avx2Fp32_6x16.IsSupported, "AVX2/FMA not supported on this CPU.");

        int ldb = Nr + 5; // non-packed stride to exercise the strided read
        var rng = new Random(977 + kc);
        var packedA = RandF(kc * Mr, rng);
        var b = RandF(kc * ldb, rng);
        var c = new float[Mr * Nr];
        var expected = new float[Mr * Nr];

        Avx2Fp32_6x16.RunStridedB(packedA, b, ldb, c, Nr, kc);

        for (int i = 0; i < Mr; i++)
            for (int j = 0; j < Nr; j++)
            {
                double acc = 0;
                for (int k = 0; k < kc; k++)
                    acc += (double)packedA[k * Mr + i] * b[k * ldb + j];
                expected[i * Nr + j] = (float)acc;
            }

        AssertClose(expected, c, kc);
    }

    private static void AssertClose(float[] expected, float[] actual, int kc)
    {
        float tol = 1e-3f * Math.Max(1, kc); // accumulation grows with kc
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol,
                $"idx {i}: expected {expected[i]}, got {actual[i]} (kc={kc}, tol={tol})");
    }

    private static float[] RandF(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
