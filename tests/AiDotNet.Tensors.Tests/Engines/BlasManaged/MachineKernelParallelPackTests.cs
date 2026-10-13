#if NET5_0_OR_GREATER
using System;
using MachineKernelGemm = AiDotNet.Tensors.Engines.BlasManaged.MachineKernelGemm;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #1218: <c>MachineKernelGemm.RunPacked</c> packs A and B inside parallel dispatches and runs partial M-stripes /
/// N-tiles on zero-padded panels through a scratch tile. Called directly (not through Auto routing, whose
/// thresholds decide whether the machine kernel engages) on aligned, ragged and sub-tile shapes, multi-K-block sizes
/// and padded leading dimensions, against a naive reference. The padding of C is a NaN sentinel that must stay
/// untouched, so an edge tile that wrote outside the m×n window fails.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class MachineKernelParallelPackTests
{
    public static TheoryData<int, int, int, int> Shapes => new()
    {
        { 6, 8, 4, 0 },        // one whole tile
        { 1, 1, 1, 0 },        // smaller than a tile in both dims: all edge
        { 5, 7, 300, 0 },      // sub-tile, two K-blocks
        { 7, 9, 64, 3 },       // one-row M edge, one-col N edge, corner; padded strides
        { 50, 100, 256, 0 },   // m%6=2, n%8=4 (n%16=4, n%32=4)
        { 515, 333, 700, 5 },  // ragged, three K-blocks, padded strides
        { 1000, 1000, 600, 0 },// the traced shape class (m%6=4); many N-panels
        { 96, 1031, 257, 1 },  // wide ragged N, K just over one block
    };

    private static double[] Rand(int count, int seed)
    {
        var rng = new Random(seed);
        var r = new double[count];
        for (int i = 0; i < count; i++) r[i] = rng.NextDouble() * 2 - 1;
        return r;
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Fp64_AnyShape_MatchesReference_AndLeavesPaddingAlone(int m, int n, int k, int pad)
    {
        Skip.IfNot(MachineKernelGemm.IsFp64Available, "no FP64 machine kernel on this host");
        int lda = k + pad, ldb = n + pad, ldc = n + pad;
        double[] a = Rand(m * lda, 11 + m), b = Rand(k * ldb, 13 + n);
        var c = new double[m * ldc];
        for (int i = 0; i < c.Length; i++) c[i] = (i % ldc) < n ? 0.0 : double.NaN;

        Assert.True(MachineKernelGemm.TryGemmFp64(a, lda, false, b, ldb, false, c, ldc, m, n, k, false, false));

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double want = 0;
                for (int p = 0; p < k; p++) want += a[i * lda + p] * b[p * ldb + j];
                Assert.True(Math.Abs(c[i * ldc + j] - want) <= 1e-12 * k + 1e-12,
                    $"[{i},{j}] of {m}x{n}x{k}: got {c[i * ldc + j]:R}, want {want:R}");
            }
            for (int j = n; j < ldc; j++)
                Assert.True(double.IsNaN(c[i * ldc + j]), $"padding [{i},{j}] was written");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Fp32_AnyShape_MatchesReference_AndLeavesPaddingAlone(int m, int n, int k, int pad)
    {
        Skip.IfNot(MachineKernelGemm.IsFp32Available, "no FP32 machine kernel on this host");
        int lda = k + pad, ldb = n + pad, ldc = n + pad;
        double[] ad = Rand(m * lda, 17 + m), bd = Rand(k * ldb, 19 + n);
        var a = Array.ConvertAll(ad, x => (float)x);
        var b = Array.ConvertAll(bd, x => (float)x);
        var c = new float[m * ldc];
        for (int i = 0; i < c.Length; i++) c[i] = (i % ldc) < n ? 0f : float.NaN;

        Assert.True(MachineKernelGemm.TryGemmFp32(a, lda, false, b, ldb, false, c, ldc, m, n, k, false, false));

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double want = 0;
                for (int p = 0; p < k; p++) want += (double)a[i * lda + p] * b[p * ldb + j];
                Assert.True(Math.Abs(c[i * ldc + j] - want) <= 2e-6 * k + 1e-5,
                    $"[{i},{j}] of {m}x{n}x{k}: got {c[i * ldc + j]:R}, want {want:R}");
            }
            for (int j = n; j < ldc; j++)
                Assert.True(float.IsNaN(c[i * ldc + j]), $"padding [{i},{j}] was written");
        }
    }

    [SkippableFact]
    public void Fp64_IsBitIdenticalAcrossThreadCounts()
    {
        Skip.IfNot(MachineKernelGemm.IsFp64Available, "no FP64 machine kernel on this host");
        const int m = 203, n = 517, k = 600;
        double[] a = Rand(m * k, 23), b = Rand(k * n, 29);
        var serial = new double[m * n];
        var parallel = new double[m * n];
        int saved = AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
            Assert.True(MachineKernelGemm.TryGemmFp64(a, k, false, b, n, false, serial, n, m, n, k, false, false));
        }
        finally
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = saved;
        }
        Assert.True(MachineKernelGemm.TryGemmFp64(a, k, false, b, n, false, parallel, n, m, n, k, false, false));
        Assert.Equal(serial, parallel);
    }
}
#endif
