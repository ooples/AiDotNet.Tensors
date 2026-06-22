// Copyright (c) AiDotNet. All rights reserved.
// #653: correctness gate for Avx2Fp32_6x16.RunPrefetch (the opt-in software-prefetch
// variant wired behind AIDOTNET_GEMM_PREFETCH=1). The PR's central claim is that prefetch
// is BIT-IDENTICAL to the non-prefetch Run (same FMA order — it only adds PREFETCHT0 hints).
// These tests assert exactly that, directly on the kernel, rather than relying on the broader
// GEMM suite happening to run with the env flag set.

using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class Avx2Fp32_6x16PrefetchTests
{
    private const int Mr = 6, Nr = 16;

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(256)]
    public void RunPrefetch_IsBitIdenticalTo_Run(int kc)
    {
        Skip.IfNot(Avx2Fp32_6x16.IsSupported, "AVX2/FMA not supported on this CPU.");

        var rng = new Random(653 + kc);
        // Packed-A: [Kc × Mr] row-major; packed-B: [Kc × Nr] row-major.
        var packedA = RandF(kc * Mr, rng);
        var packedB = RandF(kc * Nr, rng);

        // Same C start state for both (the kernel accumulates onto C).
        var cRun = new float[Mr * Nr];
        var cPrefetch = new float[Mr * Nr];

        Avx2Fp32_6x16.Run(packedA, packedB, cRun, Nr, kc);
        Avx2Fp32_6x16.RunPrefetch(packedA, packedB, cPrefetch, Nr, kc);

        // Bit-identical: prefetch only adds PREFETCHT0 hints, the FMA sequence is unchanged.
        for (int i = 0; i < Mr * Nr; i++)
            Assert.Equal(cRun[i], cPrefetch[i]);
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(64)]
    [InlineData(256)]
    public void RunPrefetch_MatchesScalarReference(int kc)
    {
        Skip.IfNot(Avx2Fp32_6x16.IsSupported, "AVX2/FMA not supported on this CPU.");

        var rng = new Random(1306 + kc);
        var packedA = RandF(kc * Mr, rng);
        var packedB = RandF(kc * Nr, rng);
        var c = new float[Mr * Nr];

        Avx2Fp32_6x16.RunPrefetch(packedA, packedB, c, Nr, kc);

        // Reference: C[i,j] = Σ_k packedA[k*Mr+i] * packedB[k*Nr+j], FP64-accumulated.
        for (int i = 0; i < Mr; i++)
            for (int j = 0; j < Nr; j++)
            {
                double acc = 0;
                for (int k = 0; k < kc; k++)
                    acc += (double)packedA[k * Mr + i] * packedB[k * Nr + j];
                // The SIMD kernel accumulates in FP32; tolerate the order-of-summation drift
                // (same tolerance shape the sibling Run test uses).
                float expected = (float)acc;
                float diff = Math.Abs(expected - c[i * Nr + j]);
                float tol = 1e-3f * Math.Max(1, kc);
                Assert.True(diff <= tol, $"[i={i},j={j},kc={kc}] {expected} vs {c[i * Nr + j]} (tol={tol})");
            }
    }

    private static float[] RandF(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
