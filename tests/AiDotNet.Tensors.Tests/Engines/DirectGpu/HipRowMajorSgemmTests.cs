#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>#377: the row-major HIP SGEMM against a CPU reference, every tile, on shapes that leave partial tiles in M,
/// N and K, plus the beta contract.</summary>
[Collection("DirectGpuSerial")]
public sealed class HipRowMajorSgemmTests
{
    private static HipBackend? CreateBackend()
    {
        var backend = new HipBackend();
        if (backend.IsAvailable && backend.EnsureRowMajorGemm()) return backend;
        backend.Dispose();
        return null;
    }

    [SkippableTheory]
    [InlineData(1, 1, 1)]
    [InlineData(37, 53, 29)]
    [InlineData(129, 65, 17)]
    [InlineData(4096, 64, 1024)]
    [InlineData(32, 1024, 1024)]
    [InlineData(64, 4096, 1024)]
    [InlineData(200, 500, 300)]
    [InlineData(68, 132, 36)]
    public void EveryTile_MatchesCpu(int m, int n, int k)
    {
        using var backend = CreateBackend();
        Skip.If(backend is null, "A HIP device is required.");
        float[] a = Values(m * k, 3), b = Values(k * n, 5);
        float[] expected = CpuGemm(a, b, m, n, k);
        using var gA = backend.AllocateBuffer(a);
        using var gB = backend.AllocateBuffer(b);
        using var gC = backend.AllocateBuffer(new float[m * n]);
        double tolerance = 2e-6 * k + 1e-5;
        int ran = 0;
        foreach (var tile in HipRowMajorSgemmKernels.Tiles)
        {
            if (!tile.Supports(m, n, k)) { Assert.False(backend.TryExecuteRowMajorGemm(tile, gA, gB, gC, m, n, k, 1f, 0f)); continue; }
            ran++;
            Assert.True(backend.TryExecuteRowMajorGemm(tile, gA, gB, gC, m, n, k, 1f, 0f), tile.KernelName);
            double error = MaxError(expected, backend.DownloadBuffer(gC));
            Assert.True(error <= tolerance, $"{tile.KernelName} {m}x{n}x{k}: max error {error:E2} > {tolerance:E2}");
        }
        Assert.True(ran >= 6, $"only {ran} tiles ran {m}x{n}x{k}");
    }

    [SkippableFact]
    public void BetaZeroIgnoresNaNInC_BetaOneAccumulates_AlphaScales()
    {
        using var backend = CreateBackend();
        Skip.If(backend is null, "A HIP device is required.");
        const int m = 70, n = 90, k = 33;
        float[] a = Values(m * k, 7), b = Values(k * n, 9);
        float[] expected = CpuGemm(a, b, m, n, k);
        using var gA = backend.AllocateBuffer(a);
        using var gB = backend.AllocateBuffer(b);
        var nan = new float[m * n];
        for (int i = 0; i < nan.Length; i++) nan[i] = float.NaN;
        using var gC = backend.AllocateBuffer(nan);
        var tile = HipRowMajorSgemmKernels.Tiles[0];
        Assert.True(backend.TryExecuteRowMajorGemm(tile, gA, gB, gC, m, n, k, 1f, 0f));
        Assert.True(MaxError(expected, backend.DownloadBuffer(gC)) <= 1e-4);
        Assert.True(backend.TryExecuteRowMajorGemm(tile, gA, gB, gC, m, n, k, 2f, 1f));
        var tripled = new float[expected.Length];
        for (int i = 0; i < tripled.Length; i++) tripled[i] = 3 * expected[i];
        Assert.True(MaxError(tripled, backend.DownloadBuffer(gC)) <= 3e-4);
    }

    /// <summary>Split-K sums the K pieces in a fixed order and applies alpha/beta once, on partial tiles and on a K
    /// that does not divide evenly; a split larger than K's 16-blocks falls back to fewer pieces.</summary>
    [SkippableTheory]
    [InlineData(32, 1024, 1024, 4)]
    [InlineData(37, 53, 300, 7)]
    [InlineData(4096, 64, 1024, 2)]
    [InlineData(20, 20, 40, 8)]
    [InlineData(256, 256, 2048, 8)]
    [InlineData(256, 256, 2048, 1)]
    public void SplitK_MatchesCpu_WithAlphaBeta(int m, int n, int k, int splits)
    {
        using var backend = CreateBackend();
        Skip.If(backend is null, "A HIP device is required.");
        float[] a = Values(m * k, 13), b = Values(k * n, 17), c0 = Values(m * n, 19);
        float[] ab = CpuGemm(a, b, m, n, k);
        var expected = new float[m * n];
        for (int i = 0; i < expected.Length; i++) expected[i] = 0.5f * ab[i] + 2f * c0[i];
        using var gA = backend.AllocateBuffer(a);
        using var gB = backend.AllocateBuffer(b);
        double tolerance = 2e-6 * k + 1e-5;
        foreach (var tile in HipRowMajorSgemmKernels.Tiles)
        {
            if (!tile.Supports(m, n, k)) continue;
            using var gC = backend.AllocateBuffer((float[])c0.Clone());
            Assert.True(backend.TryExecuteRowMajorGemm(tile, gA, gB, gC, m, n, k, 0.5f, 2f, splits), tile.KernelName);
            double error = MaxError(expected, backend.DownloadBuffer(gC));
            Assert.True(error <= tolerance, $"{tile.KernelName} {m}x{n}x{k} split {splits}: max error {error:E2}");
        }
    }

    /// <summary>On RDNA1, Gemm sends a shape to the row-major kernel exactly when the measured table says so, and the
    /// result is right on either route. The table's anchor shapes are pinned so a change to it is a visible decision.</summary>
    [SkippableTheory]
    [InlineData(256, 256, 2304, true)]
    [InlineData(256, 256, 256, true)]
    [InlineData(200, 500, 300, true)]
    [InlineData(32, 1024, 1024, true)]
    [InlineData(1024, 1024, 1024, false)]
    [InlineData(1024, 4096, 1024, false)]
    public void Gemm_RoutesByTheMeasuredTable_AndStaysCorrect(int m, int n, int k, bool rowMajorExpected)
    {
        using var backend = CreateBackend();
        Skip.If(backend is null, "A HIP device is required.");
        Skip.IfNot(backend.RowMajorGemmTableApplies, "The routing table was measured on RDNA1 (gfx1010/gfx1012) only.");
        var choice = backend.ChooseRowMajorGemm(m, n, k, backend.IsHipBlasAvailable);
        Assert.Equal(rowMajorExpected, choice is not null);
        float[] a = Values(m * k, 21), b = Values(k * n, 23);
        float[] expected = CpuGemm(a, b, m, n, k);
        using var gA = backend.AllocateBuffer(a);
        using var gB = backend.AllocateBuffer(b);
        using var gC = backend.AllocateBuffer(new float[m * n]);
        long before = backend.RowMajorGemmRuns;
        backend.Gemm(gA, gB, gC, m, n, k);
        Assert.Equal(rowMajorExpected ? 1 : 0, backend.RowMajorGemmRuns - before);
        double error = MaxError(expected, backend.DownloadBuffer(gC));
        Assert.True(error <= 2e-6 * k + 1e-5, $"{m}x{n}x{k}: max error {error:E2}");
    }

    /// <summary>At a size the CPU reference would make slow, every tile's split-K result matches its own unsplit result
    /// (same tile, so only the K split and the reduce differ).</summary>
    [SkippableTheory]
    [InlineData(1024, 1024, 1024)]
    [InlineData(1024, 4096, 1024)]
    public void SplitK_MatchesUnsplit_AtLargeSizes(int m, int n, int k)
    {
        using var backend = CreateBackend();
        Skip.If(backend is null, "A HIP device is required.");
        float[] a = Values(m * k, 31), b = Values(k * n, 37);
        using var gA = backend.AllocateBuffer(a);
        using var gB = backend.AllocateBuffer(b);
        using var gRef = backend.AllocateBuffer(new float[m * n]);
        using var gC = backend.AllocateBuffer(new float[m * n]);
        var report = new System.Text.StringBuilder();
        foreach (var tile in HipRowMajorSgemmKernels.Tiles)
        {
            if (!tile.Supports(m, n, k)) continue;
            Assert.True(backend.TryExecuteRowMajorGemm(tile, gA, gB, gRef, m, n, k, 1f, 0f, 1));
            float[] reference = backend.DownloadBuffer(gRef);
            foreach (int split in new[] { 2, 4, 8 })
            {
                Assert.True(backend.TryExecuteRowMajorGemm(tile, gA, gB, gC, m, n, k, 1f, 0f, split));
                double error = MaxError(reference, backend.DownloadBuffer(gC));
                if (error > 1e-2) report.AppendLine($"{tile.KernelName} split {split}: max error {error:E2}");
            }
        }
        Assert.True(report.Length == 0, report.ToString());
    }

    private static float[] Values(int count, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        return values;
    }

    private static float[] CpuGemm(float[] a, float[] b, int m, int n, int k)
    {
        var c = new float[m * n];
        var row = new double[n];
        for (int i = 0; i < m; i++)
        {
            Array.Clear(row, 0, n);
            for (int p = 0; p < k; p++)
            {
                double av = a[i * k + p];
                int bo = p * n;
                for (int j = 0; j < n; j++) row[j] += av * b[bo + j];
            }
            for (int j = 0; j < n; j++) c[i * n + j] = (float)row[j];
        }
        return c;
    }

    private static double MaxError(float[] expected, float[] actual)
    {
        double max = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs(expected[i] - actual[i]);
            if (double.IsNaN(d)) return double.PositiveInfinity;
            max = Math.Max(max, d);
        }
        return max;
    }
}
#endif
