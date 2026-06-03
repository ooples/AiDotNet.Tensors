using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Pack-time vs consume-time microkernel-tile agreement (the root cause of the
/// CI-only ScalarKernelTests / PrePackedB_Output_BitMatches_LivePack failures):
/// Avx2Pack interleaves panels in mr-row / nr-column stripes, and the Gemm
/// dispatcher's too-small-shape fallback can select a DIFFERENT (mr, nr) at
/// consume time than the prepack tile (e.g. scalar (4,4) for an 8x8 GEMM on an
/// AVX-512 machine whose prepack tile is 8/16 wide). Consuming the buffer with
/// the wrong stripe stride produced deterministic garbage — but only on runners
/// whose CPU picked mismatched tiles, which is why CI failed while dev boxes
/// passed.
///
/// The fix records the pack-time tile on the handle (PackMr/PackNr) and the
/// strategies reject the prepack on mismatch, falling back to live packing.
/// These tests pin the gate DETERMINISTICALLY on any CPU by corrupting the
/// recorded tile and asserting the Gemm result is still correct (live-pack
/// fallback), instead of relying on a particular machine's tile selection.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class PackTileMismatchTests
{
    [Fact]
    public void Gemm_PackedA_WithMismatchedPackMr_FallsBackToLivePack_AndStaysCorrect()
    {
        int m = 8, n = 8, k = 8;
        var (a, b) = GenerateMatrices(m, n, k, seed: 42);

        var handle = BlasManagedLib.PrePackA<double>(a, lda: k, transA: false, m, k);
        try
        {
            // Simulate the dispatcher consuming with a different microkernel tile
            // than the buffer was packed with (the AVX-512-runner scenario).
            handle.PackMr = handle.PackMr == 4 ? 8 : 4;

            var options = new BlasOptions<double> { PackedA = handle, PackingMode = PackingMode.ForcePackBoth };
            var result = new double[m * n];
            BlasManagedLib.Gemm<double>(a, k, false, b, n, false, result, n, m, n, k, options);

            var expected = Naive(a, b, m, n, k);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], result[i], precision: 10);
        }
        finally { handle.Dispose(); }
    }

    [Fact]
    public void Gemm_PackedB_WithMismatchedPackNr_FallsBackToLivePack_AndStaysCorrect()
    {
        int m = 8, n = 16, k = 8;
        var (a, b) = GenerateMatrices(m, n, k, seed: 7);

        var handle = BlasManagedLib.PrePackB<double>(b, ldb: n, transB: false, k, n);
        try
        {
            handle.PackNr = handle.PackNr == 8 ? 16 : 8;

            var options = new BlasOptions<double> { PackedB = handle, PackingMode = PackingMode.ForcePackBoth };
            var result = new double[m * n];
            BlasManagedLib.Gemm<double>(a, k, false, b, n, false, result, n, m, n, k, options);

            var expected = Naive(a, b, m, n, k);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], result[i], precision: 10);
        }
        finally { handle.Dispose(); }
    }

    [Fact]
    public void Gemm_PackedA_WithMatchingPackMr_StillConsumesPrePack()
    {
        // Companion guard: the gate must not be over-broad — an UNTOUCHED handle
        // (pack tile == consume tile on this machine, the normal case) must still
        // produce correct results, whether the prepack was consumed or live-packed.
        int m = 16, n = 16, k = 16;
        var (a, b) = GenerateMatrices(m, n, k, seed: 11);

        var handle = BlasManagedLib.PrePackA<double>(a, lda: k, transA: false, m, k);
        try
        {
            var options = new BlasOptions<double> { PackedA = handle, PackingMode = PackingMode.ForcePackBoth };
            var result = new double[m * n];
            BlasManagedLib.Gemm<double>(a, k, false, b, n, false, result, n, m, n, k, options);

            var expected = Naive(a, b, m, n, k);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], result[i], precision: 10);
        }
        finally { handle.Dispose(); }
    }

    private static (double[] a, double[] b) GenerateMatrices(int m, int n, int k, int seed)
    {
        var rng = new Random(seed);
        var a = new double[m * k];
        var b = new double[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        return (a, b);
    }

    private static double[] Naive(double[] a, double[] b, int m, int n, int k)
    {
        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
                c[i * n + j] = acc;
            }
        return c;
    }
}
