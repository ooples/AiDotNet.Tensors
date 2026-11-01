// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// The AVX2 streaming kernel's columns past its last 32-wide block (every column of a classifier head's N = 10): four
/// rows at a time over masked 16-column vector pairs. Checked against a double-precision reference with C accumulated
/// into (the kernel is C += op(A) B), with the padding of C between rows left untouched, and bit-identical when the rows
/// or the columns are split across calls as the M- and N-parallel streaming partitions split them.
/// </summary>
public class Avx2StreamingRemainderTests
{
    private static readonly int[] Ms = { 1, 3, 4, 5, 7, 8, 128 };
    private static readonly int[] Ns = { 1, 2, 7, 8, 9, 10, 15, 16, 17, 24, 31, 33, 42, 63 };
    private static readonly int[] Ks = { 1, 5, 256 };
    private const int Pad = 3;
    private const float Guard = 12345f;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MatchesTheReference_AndLeavesThePaddingAlone(bool transA)
    {
        if (!Avx2Streaming.IsSupported) return;
        foreach (int m in Ms)
            foreach (int n in Ns)
                foreach (int k in Ks)
                {
                    var (a, lda, b, c0, ldc) = Operands(m, n, k, transA, seed: m * 1000 + n * 10 + k);
                    var c = (float[])c0.Clone();
                    Avx2Streaming.RunFp32(a, lda, transA, b, n, false, c, ldc, m, n, k);
                    for (int i = 0; i < m; i++)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            double expected = c0[i * ldc + j];
                            for (int kk = 0; kk < k; kk++)
                                expected += (double)a[transA ? kk * lda + i : i * lda + kk] * b[kk * n + j];
                            float actual = c[i * ldc + j];
                            Assert.True(Math.Abs(actual - expected) <= 1e-5 * (k + Math.Abs(expected)),
                                $"m={m} n={n} k={k} transA={transA} [{i},{j}]: expected {expected}, got {actual}");
                        }
                        for (int j = n; j < ldc; j++)
                            Assert.True(c[i * ldc + j] == Guard, $"m={m} n={n} k={k} transA={transA}: padding [{i},{j}] written");
                    }
                }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void IsBitIdentical_WhenRowsOrColumnsAreSplit(bool transA)
    {
        if (!Avx2Streaming.IsSupported) return;
        foreach (int m in new[] { 5, 9, 128 })
            foreach (int n in new[] { 10, 17, 42 })
            {
                const int k = 37;
                var (a, lda, b, c0, ldc) = Operands(m, n, k, transA, seed: m + n);
                var whole = (float[])c0.Clone();
                Avx2Streaming.RunFp32(a, lda, transA, b, n, false, whole, ldc, m, n, k);

                // Rows [0, r) and [r, m): r = 3 puts rows in different four-row blocks than the whole call does.
                const int r = 3;
                var rows = (float[])c0.Clone();
                Avx2Streaming.RunFp32(a, lda, transA, b, n, false, rows, ldc, r, n, k);
                var aTail = transA ? a.AsSpan(r) : a.AsSpan(r * lda);
                Avx2Streaming.RunFp32(aTail, lda, transA, b, n, false, rows.AsSpan(r * ldc), ldc, m - r, n, k);
                Assert.Equal(whole, rows);

                // Columns [0, 8) and [8, n), as an N partition on an 8-column tile boundary splits them.
                var cols = (float[])c0.Clone();
                Avx2Streaming.RunFp32(a, lda, transA, b, n, false, cols, ldc, m, 8, k);
                Avx2Streaming.RunFp32(a, lda, transA, b.AsSpan(8), n, false, cols.AsSpan(8), ldc, m, n - 8, k);
                Assert.Equal(whole, cols);
            }
    }

    private static (float[] A, int Lda, float[] B, float[] C, int Ldc) Operands(int m, int n, int k, bool transA, int seed)
    {
        var rng = new Random(seed);
        int lda = transA ? m : k, ldc = n + Pad;
        var a = new float[(transA ? k : m) * lda];
        var b = new float[k * n];
        var c = new float[m * ldc];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < ldc; j++)
                c[i * ldc + j] = j < n ? (float)(rng.NextDouble() * 2 - 1) : Guard;
        return (a, lda, b, c, ldc);
    }
}
