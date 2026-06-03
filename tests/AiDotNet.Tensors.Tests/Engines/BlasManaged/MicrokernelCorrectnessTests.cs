#if NET8_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #405 (Sub-O): direct correctness tests for the BlasManaged register microkernels.
/// Each kernel accumulates <c>C[i,j] += Σₖ packedA[k·Mr+i]·packedB[k·Nr+j]</c> into a
/// row-major C tile (ldc = Nr, read-modify-write). These lock the numerics against a
/// naive double-precision reference so the software-prefetch additions (which are
/// pure cache hints and must NOT change results) are verified — the AVX2 kernels run
/// everywhere; the AVX-512 kernels run on AVX-512 hardware (the CI avx512-verify job).
/// Kernels not supported on the current CPU are skipped (the assert can't run there).
/// </summary>
public class MicrokernelCorrectnessTests
{
    private const int Kc = 64;

    private static float[] RandomF(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    private static double[] RandomD(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.NextDouble() * 2.0 - 1.0;
        return a;
    }

    private static void AssertMatchesReferenceF(float[] a, float[] b, float[] c, int mr, int nr, int kc, double tol)
    {
        for (int i = 0; i < mr; i++)
            for (int j = 0; j < nr; j++)
            {
                double expected = 0.0;
                for (int k = 0; k < kc; k++) expected += (double)a[k * mr + i] * b[k * nr + j];
                double actual = c[i * nr + j];
                Assert.True(Math.Abs(actual - expected) < tol,
                    $"C[{i},{j}] = {actual} != reference {expected} (tol {tol})");
            }
    }

    private static void AssertMatchesReferenceD(double[] a, double[] b, double[] c, int mr, int nr, int kc, double tol)
    {
        for (int i = 0; i < mr; i++)
            for (int j = 0; j < nr; j++)
            {
                double expected = 0.0;
                for (int k = 0; k < kc; k++) expected += a[k * mr + i] * b[k * nr + j];
                double actual = c[i * nr + j];
                Assert.True(Math.Abs(actual - expected) < tol,
                    $"C[{i},{j}] = {actual} != reference {expected} (tol {tol})");
            }
    }

    [Fact]
    public void Avx2Fp32_8x8_MatchesReference()
    {
        if (!Avx2Fp32_8x8.IsSupported) return;
        int mr = Avx2Fp32_8x8.Mr, nr = Avx2Fp32_8x8.Nr;
        var a = RandomF(Kc * mr, 11);
        var b = RandomF(Kc * nr, 22);
        var c = new float[mr * nr];
        Avx2Fp32_8x8.Run(a, b, c, nr, Kc);
        AssertMatchesReferenceF(a, b, c, mr, nr, Kc, 1e-3);
    }

    [Fact]
    public void Avx2Fp64_4x8_MatchesReference()
    {
        if (!Avx2Fp64_4x8.IsSupported) return;
        int mr = Avx2Fp64_4x8.Mr, nr = Avx2Fp64_4x8.Nr;
        var a = RandomD(Kc * mr, 33);
        var b = RandomD(Kc * nr, 44);
        var c = new double[mr * nr];
        Avx2Fp64_4x8.Run(a, b, c, nr, Kc);
        AssertMatchesReferenceD(a, b, c, mr, nr, Kc, 1e-10);
    }

    [Fact]
    public void Avx512Fp32_16x16_MatchesReference()
    {
        if (!Avx512Fp32_16x16.IsSupported) return;
        int mr = Avx512Fp32_16x16.Mr, nr = Avx512Fp32_16x16.Nr;
        var a = RandomF(Kc * mr, 55);
        var b = RandomF(Kc * nr, 66);
        var c = new float[mr * nr];
        Avx512Fp32_16x16.Run(a, b, c, nr, Kc);
        AssertMatchesReferenceF(a, b, c, mr, nr, Kc, 1e-3);
    }

    [Fact]
    public void Avx512Fp64_8x16_MatchesReference()
    {
        if (!Avx512Fp64_8x16.IsSupported) return;
        int mr = Avx512Fp64_8x16.Mr, nr = Avx512Fp64_8x16.Nr;
        var a = RandomD(Kc * mr, 77);
        var b = RandomD(Kc * nr, 88);
        var c = new double[mr * nr];
        Avx512Fp64_8x16.Run(a, b, c, nr, Kc);
        AssertMatchesReferenceD(a, b, c, mr, nr, Kc, 1e-10);
    }

    // #409 S.4: the higher-intensity AVX-512 FP32 candidate (8×32, 1.6 FMA/load vs the
    // 16×16's load-bound 0.94). Verified on AVX-512 hardware via the avx512-verify CI.
    [Fact]
    public void Avx512Fp32_8x32_MatchesReference()
    {
        if (!Avx512Fp32_8x32.IsSupported) return;
        int mr = Avx512Fp32_8x32.Mr, nr = Avx512Fp32_8x32.Nr;
        var a = RandomF(Kc * mr, 99);
        var b = RandomF(Kc * nr, 110);
        var c = new float[mr * nr];
        Avx512Fp32_8x32.Run(a, b, c, nr, Kc);
        AssertMatchesReferenceF(a, b, c, mr, nr, Kc, 1e-3);
    }
}
#endif
