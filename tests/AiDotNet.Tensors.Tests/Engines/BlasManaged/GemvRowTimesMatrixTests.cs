#if NETCOREAPP
using System;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// The M=1 GEMV (row x matrix) keeps eight accumulators across K and splits column blocks across threads. Each output
/// must still be its column's ascending-K FMA chain (the vector columns) or ascending-K multiply-add (the scalar tail
/// past the last full vector), so the result is bit-identical to the serial strip kernel it replaced and to itself at
/// any thread count.
/// </summary>
[Collection("BlasManaged-Perf-Serial")]
public class GemvRowTimesMatrixTests
{
    // k = 1 is routed to the outer-product kernel, so the smallest case here is k = 2; (64, 2) fills whole blocks with
    // no tail.
    [SkippableTheory]
    [InlineData(8, 2)]
    [InlineData(64, 2)]
    [InlineData(13, 37)]
    [InlineData(63, 64)]
    [InlineData(64, 257)]
    [InlineData(65, 300)]
    [InlineData(200, 31)]
    [InlineData(4100, 600)]   // above the parallel threshold, partial last block
    public void Float_MatchesTheAscendingKChain(int n, int k)
    {
        Skip.IfNot(Fma.IsSupported && Avx2.IsSupported, "The row-times-matrix kernel needs AVX2 and FMA.");
        foreach (bool transA in new[] { false, true })
        {
            int lda = transA ? 3 : k;
            var rng = new Random(n * 31 + k);
            var a = new float[transA ? (k - 1) * lda + 1 : k];
            var b = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
            var c = new float[n];
            BlasManagedLib.Gemm<float>(a, lda, transA, b, n, false, c, n, 1, n, k);

            int vectorEnd = n & ~7;
            for (int j = 0; j < n; j++)
            {
                float expected = 0;
                for (int kk = 0; kk < k; kk++)
                {
                    float av = a[transA ? kk * lda : kk], bv = b[kk * n + j];
                    expected = j < vectorEnd ? MathF.FusedMultiplyAdd(av, bv, expected) : expected + av * bv;
                }
                Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(c[j]),
                    $"n={n} k={k} transA={transA} column {j}: {c[j]:R} vs chain {expected:R}");
            }
        }
    }

    [SkippableTheory]
    [InlineData(4, 2)]
    [InlineData(32, 2)]
    [InlineData(31, 40)]
    [InlineData(33, 129)]
    [InlineData(2050, 600)]
    public void Double_MatchesTheAscendingKChain(int n, int k)
    {
        Skip.IfNot(Fma.IsSupported && Avx2.IsSupported, "The row-times-matrix kernel needs AVX2 and FMA.");
        var rng = new Random(n * 17 + k);
        var a = new double[k];
        var b = new double[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        var c = new double[n];
        BlasManagedLib.Gemm<double>(a, k, false, b, n, false, c, n, 1, n, k);

        int vectorEnd = n & ~3;
        for (int j = 0; j < n; j++)
        {
            double expected = 0;
            for (int kk = 0; kk < k; kk++)
                expected = j < vectorEnd ? Math.FusedMultiplyAdd(a[kk], b[kk * n + j], expected) : expected + a[kk] * b[kk * n + j];
            Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(c[j]),
                $"n={n} k={k} column {j}: {c[j]:R} vs chain {expected:R}");
        }
    }

    [Fact]
    public void Float_IsIdenticalAtOneThreadAndAll()
    {
        const int n = 4096, k = 1024;
        var rng = new Random(7);
        var a = new float[k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var parallel = new float[n];
        var serial = new float[n];
        BlasManagedLib.Gemm<float>(a, k, false, b, n, false, parallel, n, 1, n, k);
        int prev = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = 1;
        try { BlasManagedLib.Gemm<float>(a, k, false, b, n, false, serial, n, 1, n, k); }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = prev; }
        Assert.Equal(serial, parallel);
    }
}
#endif
