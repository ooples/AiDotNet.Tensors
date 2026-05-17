// Issue #358 Task G7 — Gate 3 determinism suite.
// Validates that BlasManaged.Gemm produces bit-exact output across
// threadCount = 1, 2, 4, 8, 16 for 12 representative shapes.
// M/N/2D axes write disjoint C cells so they are deterministic by
// construction across thread counts. K-axis is disabled in determinism mode.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class DeterminismTests
{
    // 12 representative shapes covering the spec's Gate 3 set:
    // - small (Streaming path)
    // - medium (PackBoth path)
    // - L2-shape stand-in (transA=true)
    // - tall-thin (M small, K large)
    // - wide-flat (M large, K small)
    // - rectangular variants
    public static readonly object[][] RepresentativeShapes = new object[][]
    {
        new object[] { 8, 8, 8, false, false },          // tiny
        new object[] { 32, 32, 32, false, false },       // small balanced
        new object[] { 64, 64, 64, false, false },       // medium balanced
        new object[] { 128, 128, 128, false, false },    // larger balanced
        new object[] { 32, 16, 128, true, false },       // L2-stand-in (transA=true)
        new object[] { 16, 32, 128, false, true },       // transB=true
        new object[] { 16, 16, 256, true, true },        // both trans
        new object[] { 256, 32, 64, false, false },      // M-heavy (M-axis)
        new object[] { 32, 256, 64, false, false },      // N-heavy
        new object[] { 16, 16, 512, false, false },      // tall-thin (K-heavy)
        new object[] { 64, 64, 16, false, false },       // wide-flat (K-small)
        new object[] { 128, 32, 64, false, false },      // rectangular
    };

    public static IEnumerable<object[]> ShapeData => RepresentativeShapes;

    [Theory]
    [MemberData(nameof(ShapeData))]
    public void Gemm_FP64_BitExactAcrossThreadCounts(int m, int n, int k, bool transA, bool transB)
    {
        // Build random input once.
        int aRows = transA ? k : m;
        int aCols = transA ? m : k;
        int bRows = transB ? n : k;
        int bCols = transB ? k : n;
        var rng = new Random(42);
        double[] a = new double[aRows * aCols];
        double[] b = new double[bRows * bCols];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        int lda = aCols;
        int ldb = bCols;

        // Run at thread counts 1, 2, 4, 8, 16; collect outputs.
        int[] threadCounts = { 1, 2, 4, 8, 16 };
        double[][] outputs = new double[threadCounts.Length][];

        int originalMaxDegree = CpuParallelSettings.MaxDegreeOfParallelism;
        bool originalDeterministic = BlasProvider.IsDeterministicMode;

        try
        {
            BlasProvider.SetDeterministicMode(true);

            for (int t = 0; t < threadCounts.Length; t++)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threadCounts[t];

                double[] actual = new double[m * n];
                BlasManagedLib.Gemm<double>(
                    a, lda, transA,
                    b, ldb, transB,
                    actual, ldc: n,
                    m, n, k);
                outputs[t] = actual;
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = originalMaxDegree;
            BlasProvider.SetDeterministicMode(originalDeterministic);
        }

        // Every run must produce bit-identical output to run 0 (threadCount=1).
        // Use byte-level comparison via MemoryExtensions.SequenceEqual.
        var reference = outputs[0];
        for (int t = 1; t < threadCounts.Length; t++)
        {
            var bytesRef = MemoryMarshal.AsBytes(reference.AsSpan());
            var bytesActual = MemoryMarshal.AsBytes(outputs[t].AsSpan());
            Assert.True(
                bytesRef.SequenceEqual(bytesActual),
                $"Bit-exact mismatch: threadCount=1 vs threadCount={threadCounts[t]} for shape "
                + $"(m={m}, n={n}, k={k}, transA={transA}, transB={transB})");
        }
    }

    [Fact]
    public void Gemm_FP64_NonDeterministicMode_MatchesNaiveWithinTolerance()
    {
        // Sanity: non-deterministic mode still produces correct output (within ULP tolerance).
        int m = 64, n = 64, k = 64;
        var rng = new Random(42);
        double[] a = new double[m * k];
        double[] b = new double[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        bool originalDeterministic = BlasProvider.IsDeterministicMode;
        try
        {
            BlasProvider.SetDeterministicMode(false);

            double[] actual = new double[m * n];
            BlasManagedLib.Gemm<double>(
                a, lda: k, transA: false,
                b, ldb: n, transB: false,
                actual, ldc: n,
                m, n, k);

            // Naive reference.
            double[] expected = new double[m * n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int kk = 0; kk < k; kk++) sum += a[i * k + kk] * b[kk * n + j];
                    expected[i * n + j] = sum;
                }

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i], precision: 10);
        }
        finally
        {
            BlasProvider.SetDeterministicMode(originalDeterministic);
        }
    }
}
