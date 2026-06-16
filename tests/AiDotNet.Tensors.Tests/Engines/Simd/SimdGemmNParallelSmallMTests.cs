// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Correctness of the fp32 N-parallel small-M GEMM and the fused-projection helper (Phase A /
/// #1622, L2). Validates overwrite + accumulate semantics, the m=1 (batch=1) latency case, that
/// the dispatcher routes small-M large-N through it, and that a fused [Wq|Wk|Wv] projection equals
/// the separate projections.
/// </summary>
public class SimdGemmNParallelSmallMTests
{
    private struct Rng
    {
        private ulong _s;
        public Rng(ulong seed) { _s = seed | 1UL; }
        public double NextUnit() { ulong x = _s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; _s = x; return (x >> 11) * (1.0 / (1UL << 53)); }
        public float NextGaussian(double std) { double u1 = Math.Max(1e-12, NextUnit()), u2 = NextUnit(); return (float)(std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)); }
    }

    private static float[] RandArr(ref Rng rng, int n, double std)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = rng.NextGaussian(std);
        return a;
    }

    // C[m,n] = A[m,k]·B[k,n] (+ existing C if !overwrite), B row-major.
    private static float[] Reference(float[] a, float[] b, float[]? c0, int m, int k, int n)
    {
        var c = new float[m * n];
        if (c0 != null) Array.Copy(c0, c, m * n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
                c[i * n + j] += acc;
            }
        return c;
    }

    private static double RelFroErr(float[] x, float[] y)
    {
        double num = 0, den = 0;
        for (int i = 0; i < x.Length; i++) { double d = (double)x[i] - y[i]; num += d * d; den += (double)y[i] * y[i]; }
        return Math.Sqrt(num / Math.Max(1e-30, den));
    }

    [Theory]
    [InlineData(1, 256, 512)]   // batch=1 (latency case)
    [InlineData(2, 320, 768)]
    [InlineData(5, 256, 500)]   // n not vector-aligned
    [InlineData(8, 384, 1024)]
    public void NParallelSmallM_Overwrite_MatchesReference(int m, int k, int n)
    {
        var rng = new Rng((ulong)(m * 31 + k + n));
        var a = RandArr(ref rng, m * k, 0.1);
        var b = RandArr(ref rng, k * n, 0.05);
        var cRef = Reference(a, b, null, m, k, n);

        var c = new float[m * n];
        SimdGemm.SgemmNParallelSmallM(a, k, b, n, c, m, k, n, clearedOutput: true);

        Assert.True(RelFroErr(c, cRef) < 1e-4, $"overwrite rel err {RelFroErr(c, cRef):E3} (m={m},k={k},n={n})");
    }

    [Fact]
    public void NParallelSmallM_Accumulate_AddsIntoExistingC()
    {
        const int m = 4, k = 256, n = 512;
        var rng = new Rng(123);
        var a = RandArr(ref rng, m * k, 0.1);
        var b = RandArr(ref rng, k * n, 0.05);
        var c0 = RandArr(ref rng, m * n, 1.0);
        var cRef = Reference(a, b, c0, m, k, n);

        var c = (float[])c0.Clone();
        SimdGemm.SgemmNParallelSmallM(a, k, b, n, c, m, k, n, clearedOutput: false);

        Assert.True(RelFroErr(c, cRef) < 1e-4, $"accumulate rel err {RelFroErr(c, cRef):E3}");
    }

    [Fact]
    public void Dispatcher_RoutesBatch1LargeN_AndIsCorrect()
    {
        // Drive the real dispatcher (SgemmAddInternal) at m=1, large N → must take the N-parallel
        // branch and produce the correct product.
        const int m = 1, k = 1024, n = 4096;
        var rng = new Rng(7);
        var a = RandArr(ref rng, m * k, 0.1);
        var b = RandArr(ref rng, k * n, 0.05);
        var cRef = Reference(a, b, null, m, k, n);

        var c = new float[m * n];
        SimdGemm.SgemmAddInternal(a, k, false, b, n, false, c, m, k, n, allowParallel: true, clearedOutput: true);

        Assert.True(RelFroErr(c, cRef) < 1e-4, $"dispatcher batch=1 rel err {RelFroErr(c, cRef):E3}");
    }

    [Fact]
    public void FusedProjections_EqualSeparateProjections()
    {
        // QKV fusion: [Wq|Wk|Wv] concatenated along N == three separate projections.
        const int m = 2, k = 512;
        int nq = 512, nk = 512, nv = 512, nTot = nq + nk + nv;
        var rng = new Rng(55);
        var a = RandArr(ref rng, m * k, 0.1);
        var wq = RandArr(ref rng, k * nq, 0.05);
        var wk = RandArr(ref rng, k * nk, 0.05);
        var wv = RandArr(ref rng, k * nv, 0.05);

        // Concatenate along the output (column) dim: bConcat[p, :] = [wq row p | wk row p | wv row p].
        var bConcat = new float[k * nTot];
        for (int p = 0; p < k; p++)
        {
            Array.Copy(wq, p * nq, bConcat, p * nTot, nq);
            Array.Copy(wk, p * nk, bConcat, p * nTot + nq, nk);
            Array.Copy(wv, p * nv, bConcat, p * nTot + nq + nk, nv);
        }

        var cFused = new float[m * nTot];
        SimdGemm.SgemmConcatenatedProjections(a, bConcat, cFused, m, k, nTot);

        var cq = Reference(a, wq, null, m, k, nq);
        var ck = Reference(a, wk, null, m, k, nk);
        var cv = Reference(a, wv, null, m, k, nv);
        // Reassemble the expected concatenated output and compare.
        var expected = new float[m * nTot];
        for (int i = 0; i < m; i++)
        {
            Array.Copy(cq, i * nq, expected, i * nTot, nq);
            Array.Copy(ck, i * nk, expected, i * nTot + nq, nk);
            Array.Copy(cv, i * nv, expected, i * nTot + nq + nk, nv);
        }
        Assert.True(RelFroErr(cFused, expected) < 1e-4, $"fused-vs-separate rel err {RelFroErr(cFused, expected):E3}");
    }
}
