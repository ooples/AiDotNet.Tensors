#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// #378: FP16 (System.Half) and BF16 GEMM microkernels for BlasManaged, float-accumulated
/// upcast emulation. Two layers:
///   1. The kernels (<see cref="HalfKernels.Matmul"/>, <see cref="BFloat16Kernels.Matmul"/>)
///      vs a double-precision reference computed from the SAME rounded inputs — so the test
///      isolates the kernel's accumulation accuracy (float accumulator + low-precision output
///      rounding) from the input-rounding loss.
///   2. The dispatch wiring: <see cref="MatrixMultiplyHelper.TryGemm{T}"/> now routes Half /
///      BFloat16 to those kernels (returns true) instead of declining to the generic scalar
///      path.
/// </summary>
public class Fp16Bf16GemmTests
{
    [Theory]
    [InlineData(4, 32, 8)]    // K=32: pure 8-lane SIMD path
    [InlineData(16, 24, 20)]  // K=20: 2 SIMD blocks + 4-element scalar tail
    [InlineData(3, 5, 7)]     // K=7 < 8: scalar-only path
    [InlineData(1, 64, 1)]    // degenerate K=1
    public void HalfMatmul_MatchesFloatReference(int m, int n, int k)
    {
        var (af, bf) = RandPair(m, k, n, seed: 1000 + m + n + k);
        // Round inputs to FP16 and use the rounded values for BOTH kernel and reference.
        var ah = new Half[m * k]; var afr = new float[m * k];
        for (int i = 0; i < ah.Length; i++) { ah[i] = (Half)af[i]; afr[i] = (float)ah[i]; }
        var bh = new Half[k * n]; var bfr = new float[k * n];
        for (int i = 0; i < bh.Length; i++) { bh[i] = (Half)bf[i]; bfr[i] = (float)bh[i]; }
        var ch = new Half[m * n];

        HalfKernels.Matmul(ah, k, bh, n, ch, n, m, k, n);

        // FP16 output rounding ~ 2^-10 relative; allow slack + magnitude floor.
        AssertCloseRelative(ch, c => (float)c, afr, bfr, m, k, n, relTol: 6e-3, name: "FP16");
    }

    [Theory]
    [InlineData(4, 32, 8)]
    [InlineData(16, 24, 20)]
    [InlineData(3, 5, 7)]
    [InlineData(1, 64, 1)]
    public void BFloat16Matmul_MatchesFloatReference(int m, int n, int k)
    {
        var (af, bf) = RandPair(m, k, n, seed: 2000 + m + n + k);
        var ab = new BFloat16[m * k]; var afr = new float[m * k];
        for (int i = 0; i < ab.Length; i++) { ab[i] = BFloat16.FromFloat(af[i]); afr[i] = BFloat16.ToFloat(ab[i]); }
        var bb = new BFloat16[k * n]; var bfr = new float[k * n];
        for (int i = 0; i < bb.Length; i++) { bb[i] = BFloat16.FromFloat(bf[i]); bfr[i] = BFloat16.ToFloat(bb[i]); }
        var cb = new BFloat16[m * n];

        BFloat16Kernels.Matmul(ab, k, bb, n, cb, n, m, k, n);

        // BF16 has ~8-bit mantissa → ~2^-8 relative output rounding.
        AssertCloseRelative(cb, BFloat16.ToFloat, afr, bfr, m, k, n, relTol: 2e-2, name: "BF16");
    }

    [Fact]
    public void TryGemm_RoutesFp16_AndIsCorrect()
    {
        const int M = 96, N = 96, K = 96; // above the BLAS work threshold → SIMD route
        var (af, bf) = RandPair(M, K, N, seed: 777);
        var ah = new Half[M * K]; var afr = new float[M * K];
        for (int i = 0; i < ah.Length; i++) { ah[i] = (Half)af[i]; afr[i] = (float)ah[i]; }
        var bh = new Half[K * N]; var bfr = new float[K * N];
        for (int i = 0; i < bh.Length; i++) { bh[i] = (Half)bf[i]; bfr[i] = (float)bh[i]; }
        var ch = new Half[M * N];

        bool routed = MatrixMultiplyHelper.TryGemm(
            (ReadOnlyMemory<Half>)ah, 0, (ReadOnlyMemory<Half>)bh, 0, (Memory<Half>)ch, 0, M, K, N);
        Assert.True(routed, "TryGemm must route FP16 to the SIMD microkernel (return true) above the work threshold");
        AssertCloseRelative(ch, c => (float)c, afr, bfr, M, K, N, relTol: 6e-3, name: "FP16-routed");
    }

    [Fact]
    public void TryGemm_RoutesBf16_AndIsCorrect()
    {
        const int M = 96, N = 96, K = 96;
        var (af, bf) = RandPair(M, K, N, seed: 888);
        var ab = new BFloat16[M * K]; var afr = new float[M * K];
        for (int i = 0; i < ab.Length; i++) { ab[i] = BFloat16.FromFloat(af[i]); afr[i] = BFloat16.ToFloat(ab[i]); }
        var bb = new BFloat16[K * N]; var bfr = new float[K * N];
        for (int i = 0; i < bb.Length; i++) { bb[i] = BFloat16.FromFloat(bf[i]); bfr[i] = BFloat16.ToFloat(bb[i]); }
        var cb = new BFloat16[M * N];

        bool routed = MatrixMultiplyHelper.TryGemm(
            (ReadOnlyMemory<BFloat16>)ab, 0, (ReadOnlyMemory<BFloat16>)bb, 0, (Memory<BFloat16>)cb, 0, M, K, N);
        Assert.True(routed, "TryGemm must route BF16 to the SIMD microkernel (return true) above the work threshold");
        AssertCloseRelative(cb, BFloat16.ToFloat, afr, bfr, M, K, N, relTol: 2e-2, name: "BF16-routed");
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private static (float[] a, float[] b) RandPair(int m, int k, int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[m * k]; for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        var b = new float[k * n]; for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        return (a, b);
    }

    private static void AssertCloseRelative<T>(T[] c, Func<T, float> toFloat,
        float[] aRef, float[] bRef, int m, int k, int n, double relTol, string name)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double truth = 0;
                for (int kk = 0; kk < k; kk++) truth += (double)aRef[i * k + kk] * bRef[kk * n + j];
                double got = toFloat(c[i * n + j]);
                double bound = relTol * Math.Max(1.0, Math.Abs(truth));
                Assert.True(Math.Abs(got - truth) <= bound,
                    $"{name} C[{i},{j}] = {got:G6} vs truth {truth:G6} (|err| {Math.Abs(got - truth):G3} > bound {bound:G3})");
            }
    }
}
#endif
