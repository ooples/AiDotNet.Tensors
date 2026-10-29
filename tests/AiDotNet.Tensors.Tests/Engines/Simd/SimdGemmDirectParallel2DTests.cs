// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// The small-M direct GEMM partitioned over both output axes (SimdGemm.DirectParallel2D.cs): values against a
/// double-precision reference on full and ragged (masked-tail) shapes, bit-identity across every partition the
/// thread count can produce, and that SimdGemm.Sgemm routes training-batch shapes to it.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]   // varies the process-wide MaxDegreeOfParallelism
public class SimdGemmDirectParallel2DTests
{
    private static float[] Rand(int n, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static double[] Reference(float[] a, float[] b, int m, int k, int n)
    {
        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++)
            {
                double av = a[i * k + p];
                for (int j = 0; j < n; j++) c[i * n + j] += av * b[p * n + j];
            }
        return c;
    }

    private static void AssertMatchesReference(double[] expected, float[] actual, int k, string what)
    {
        // Each output is a k-term float FMA chain of O(1) terms: allow a few ulps per term of accumulated error.
        double tol = 4e-7 * k + 1e-6;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= tol,
                $"{what}[{i}] = {actual[i]:G9}, expected {expected[i]:G9} (tol {tol:G3})");
    }

    private static int[] ThreadCounts => new[] { 1, 2, 3, 5, 16, Math.Max(2, Environment.ProcessorCount) };

    [SkippableTheory]
    [InlineData(64, 784, 512)]    // parity-MLP layer 1 forward (k > 512: previously the packed path)
    [InlineData(64, 512, 128)]    // parity-MLP layer 2 forward
    [InlineData(37, 300, 77)]     // ragged: last row block has 1 row, last column strip 13 columns (n % 8 != 0)
    [InlineData(192, 1024, 16)]   // the gate's corners: max m, max k, one column strip
    [InlineData(6, 256, 1000)]    // a single row block: all parallelism from columns
    public void MatchesReference_AndIsBitIdenticalAcrossPartitions(int m, int k, int n)
    {
        // The direct kernels are register FMA chains with no portable form; production reaches them only behind an FMA gate.
        Skip.IfNot(System.Runtime.Intrinsics.X86.Fma.IsSupported, "the direct GEMM kernels need FMA");
        var a = Rand(m * k, m * 31 + k);
        var b = Rand(k * n, n * 17 + k);
        var expected = Reference(a, b, m, k, n);

        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            float[]? first = null;
            foreach (int threads in ThreadCounts)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                foreach (bool allowParallel in new[] { true, false })
                {
                    var c = new float[m * n];
                    for (int i = 0; i < c.Length; i++) c[i] = float.NaN;   // every element must be written
                    Assert.True(SimdGemm.TrySgemmDirectParallel2D(a, k, b, n, c, m, k, n, allowParallel));
                    AssertMatchesReference(expected, c, k, $"threads={threads} parallel={allowParallel} C");
                    if (first is null) first = c;
                    else
                        for (int i = 0; i < c.Length; i++)
                            Assert.True(BitConverter.SingleToInt32Bits(first[i]) == BitConverter.SingleToInt32Bits(c[i]),
                                $"threads={threads} parallel={allowParallel}: C[{i}] = {c[i]:G9} differs from {first[i]:G9}");
                }
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }

#if NET9_0_OR_GREATER
    // The 512-bit 6x32 tile computes each element as the same FMA chain the 256-bit tiles do, so the output may not
    // depend on which one the host runs. Forced on, it runs in software where the CPU lacks AVX-512 (same bits).
    [SkippableTheory]
    [InlineData(64, 784, 512)]    // every strip a full 32 columns
    [InlineData(37, 300, 77)]     // 2 full strips, then a 13-column strip on the masked 16-wide kernel
    [InlineData(66, 96, 48)]      // a full strip and a 16-column remainder on the 6x16 kernel
    [InlineData(7, 40, 32)]       // one full row block (6x32 tile) and a 1-row masked block
    public void Vector512Tile_IsBitIdenticalTo256BitTiles(int m, int k, int n)
    {
        // The direct kernels are register FMA chains with no portable form; production reaches them only behind an FMA gate.
        Skip.IfNot(System.Runtime.Intrinsics.X86.Fma.IsSupported, "the direct GEMM kernels need FMA");
        var a = Rand(m * k, m * 13 + k);
        var b = Rand(k * n, n * 7 + k);
        var narrow = new float[m * n];
        var wide = new float[m * n];
        bool? before = SimdGemm.s_forceVector512Direct;
        int threads = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 3;
            SimdGemm.s_forceVector512Direct = false;
            Assert.True(SimdGemm.TrySgemmDirectParallel2D(a, k, b, n, narrow, m, k, n, allowParallel: true));
            SimdGemm.s_forceVector512Direct = true;
            Assert.True(SimdGemm.TrySgemmDirectParallel2D(a, k, b, n, wide, m, k, n, allowParallel: true));
        }
        finally
        {
            SimdGemm.s_forceVector512Direct = before;
            CpuParallelSettings.MaxDegreeOfParallelism = threads;
        }
        for (int i = 0; i < narrow.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(narrow[i]) == BitConverter.SingleToInt32Bits(wide[i]),
                $"C[{i / n},{i % n}]: 256-bit {narrow[i]:G9}, 512-bit {wide[i]:G9}");
        AssertMatchesReference(Reference(a, b, m, k, n), wide, k, "512-bit");
    }

    // The transposed-A GEMM (a dense layer's dW = Xt*dY): full 6-row blocks take 32-column steps on the 512-bit tile.
    [SkippableTheory]
    [InlineData(784, 64, 512)]    // the parity MLP's first-layer dW
    [InlineData(30, 17, 80)]      // 2 full 32-column steps, then a 16-column step
    [InlineData(29, 9, 70)]       // a 4-row remainder block, a scalar row, and a 6-column tail (mFull = 28)
    public void Vector512TransATile_IsBitIdenticalTo256BitTiles(int m, int k, int n)
    {
        // The direct kernels are register FMA chains with no portable form; production reaches them only behind an FMA gate.
        Skip.IfNot(System.Runtime.Intrinsics.X86.Fma.IsSupported, "the direct GEMM kernels need FMA");
        var a = Rand(k * m, m * 5 + k);   // stored [k, m]
        var b = Rand(k * n, n * 3 + k);
        var narrow = new float[m * n];
        var wide = new float[m * n];
        bool? before = SimdGemm.s_forceVector512Direct;
        try
        {
            SimdGemm.s_forceVector512Direct = false;
            SimdGemm.SgemmDirectParallelMIntoTransA(a, b, narrow, m, k, n);
            SimdGemm.s_forceVector512Direct = true;
            SimdGemm.SgemmDirectParallelMIntoTransA(a, b, wide, m, k, n);
        }
        finally
        {
            SimdGemm.s_forceVector512Direct = before;
        }
        for (int i = 0; i < narrow.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(narrow[i]) == BitConverter.SingleToInt32Bits(wide[i]),
                $"C[{i / n},{i % n}]: 256-bit {narrow[i]:G9}, 512-bit {wide[i]:G9}");
        var expected = new double[m * n];
        for (int p = 0; p < k; p++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    expected[i * n + j] += (double)a[p * m + i] * b[p * n + j];
        AssertMatchesReference(expected, wide, k, "512-bit");
    }
#endif

    [SkippableTheory]
    [InlineData(128, 784, 512)]   // parity-MLP layer 1 forward at batch 128: 21 row blocks + a 2-row tail
    [InlineData(128, 512, 256)]   // layer 2
    [InlineData(131, 300, 250)]   // 5-row tail and a 10-column tail in the same call
    [InlineData(66, 96, 40)]      // n < 3 tiles: some column groups are just the masked tail
    public void ParallelMOverwrite_MatchesReference_AndIsBitIdenticalAcrossThreadCounts(int m, int k, int n)
    {
        // The direct kernels are register FMA chains with no portable form; production reaches them only behind an FMA gate.
        Skip.IfNot(System.Runtime.Intrinsics.X86.Fma.IsSupported, "the direct GEMM kernels need FMA");
        // The row x column-group split (with the m % 6 tail rows inside the last row chunk's tasks) must write every
        // element exactly once, whatever the pool size, and produce the same bits for every partition.
        var a = Rand(m * k, m * 13 + k);
        var b = Rand(k * n, n * 7 + k);
        var expected = Reference(a, b, m, k, n);
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            float[]? first = null;
            foreach (int threads in ThreadCounts)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                var c = new float[m * n];
                for (int i = 0; i < c.Length; i++) c[i] = float.NaN;   // every element must be written
                SimdGemm.SgemmDirectParallelMOverwrite(a, b, c, m, k, n);
                AssertMatchesReference(expected, c, k, $"threads={threads} C");
                if (first is null) first = c;
                else
                    for (int i = 0; i < c.Length; i++)
                        Assert.True(BitConverter.SingleToInt32Bits(first[i]) == BitConverter.SingleToInt32Bits(c[i]),
                            $"threads={threads}: C[{i}] = {c[i]:G9} differs from {first[i]:G9}");
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }
    [Fact]
    public void Sgemm_RoutesTrainingBatchShapeToTheDirect2DPath()
    {
        const int m = 64, k = 784, n = 512;
        var a = Rand(m * k, 1);
        var b = Rand(k * n, 2);
        if (!(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported))
        {
            // Without AVX2/FMA the gate must decline (the kernels would throw PlatformNotSupportedException) and the
            // GEMM still be computed, on whatever route this machine can run.
            var portable = new float[m * n];
            int runsBeforePortable = SimdGemm.t_directParallel2DRuns;
            SimdGemm.SgemmAddInternal(a, k, false, b, n, false, portable, m, k, n, allowParallel: true, clearedOutput: true);
            Assert.Equal(runsBeforePortable, SimdGemm.t_directParallel2DRuns);
            AssertMatchesReference(Reference(a, b, m, k, n), portable, k, "portable route C");
            return;
        }
        var direct = new float[m * n];
        Assert.True(SimdGemm.TrySgemmDirectParallel2D(a, k, b, n, direct, m, k, n, allowParallel: true));

        bool before = SimdGemm.UseDirectParallel2D;
        try
        {
            // Matching bits alone do not prove the route: the per-thread run counter must move for the gated shape,
            // and must not move when the path is switched off (the control that the counter itself discriminates).
            SimdGemm.UseDirectParallel2D = false;
            int runsBeforeControl = SimdGemm.t_directParallel2DRuns;
            SimdGemm.SgemmAddInternal(a, k, false, b, n, false, new float[m * n], m, k, n, allowParallel: true, clearedOutput: true);
            Assert.Equal(runsBeforeControl, SimdGemm.t_directParallel2DRuns);

            SimdGemm.UseDirectParallel2D = true;
            var routed = new float[m * n];
            int runsBefore = SimdGemm.t_directParallel2DRuns;
            SimdGemm.SgemmAddInternal(a, k, false, b, n, false, routed, m, k, n, allowParallel: true, clearedOutput: true);
            Assert.Equal(runsBefore + 1, SimdGemm.t_directParallel2DRuns);
            for (int i = 0; i < routed.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(direct[i]) == BitConverter.SingleToInt32Bits(routed[i]),
                    $"routed C[{i}] = {routed[i]:G9}, direct path {direct[i]:G9}");
        }
        finally
        {
            SimdGemm.UseDirectParallel2D = before;
        }
    }

    [SkippableTheory]
    [InlineData(64, 784, 512)]
    [InlineData(67, 100, 37)]    // ragged rows (67 = 11*6 + 1) and columns (37 = 2*16 + 5)
    [InlineData(6, 9, 16)]
    public void OverwritesEveryElementOfC_WithoutAClearedOutput(int m, int k, int n)
    {
        // The direct kernels are register FMA chains with no portable form; production reaches them only behind an FMA gate.
        Skip.IfNot(System.Runtime.Intrinsics.X86.Fma.IsSupported, "the direct GEMM kernels need FMA");
        // Sgemm and BlasManaged hand the 2D kernel an uncleared C: every element must be stored, edges included.
        var a = Rand(m * k, 5);
        var b = Rand(k * n, 6);
        var c = new float[m * n];
        Array.Fill(c, float.NaN);
        Assert.True(SimdGemm.TrySgemmDirectParallel2D(a, k, b, n, c, m, k, n, allowParallel: true));
        AssertMatchesReference(Reference(a, b, m, k, n), c, k, "C over a NaN-filled destination");
    }

    [Fact]
    public void BlasManagedThinM_RoutesABatchSizedNoTransposeGemmToTheDirect2DPath()
    {
        // BlasManaged's thin-M kernel split rows, so every worker streamed all of B; in a training step B (the weights)
        // arrives cold from the optimizer and that ran 2-3x slower than the column-first 2D split. The training
        // forward routes through BlasManaged, so its batch-sized no-transpose GEMMs must land on the 2D kernel.
        const int m = 64, k = 784, n = 512;
        var a = Rand(m * k, 3);
        var b = Rand(k * n, 4);
        if (!System.Runtime.Intrinsics.X86.Fma.IsSupported)
        {
            // BlasManaged's thin-M route is gated on FMA: without it the 2D kernel must not be entered, and the GEMM
            // must still come out right through the strategy paths.
            var portable = new float[m * n];
            int runsBeforePortable = SimdGemm.t_directParallel2DRuns;
            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, n, false, portable, n, m, n, k,
                new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float> { PackingMode = AiDotNet.Tensors.Engines.BlasManaged.PackingMode.DisableAutotune });
            Assert.Equal(runsBeforePortable, SimdGemm.t_directParallel2DRuns);
            AssertMatchesReference(Reference(a, b, m, k, n), portable, k, "portable route C");
            return;
        }
        var direct = new float[m * n];
        Assert.True(SimdGemm.TrySgemmDirectParallel2D(a, k, b, n, direct, m, k, n, allowParallel: true));

        var routed = new float[m * n];
        int runsBefore = SimdGemm.t_directParallel2DRuns;
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, n, false, routed, n, m, n, k,
            new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float> { PackingMode = AiDotNet.Tensors.Engines.BlasManaged.PackingMode.DisableAutotune });
        Assert.Equal(runsBefore + 1, SimdGemm.t_directParallel2DRuns);
        for (int i = 0; i < routed.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(direct[i]) == BitConverter.SingleToInt32Bits(routed[i]),
                $"routed C[{i}] = {routed[i]:G9}, direct path {direct[i]:G9}");
    }

    [Fact]
    public void TooNarrowForTheKernel_ReturnsFalseAndLeavesOutputUntouched()
    {
        var a = Rand(5 * 64, 3);
        var b = Rand(64 * 32, 4);
        var c = new float[5 * 32];
        Array.Fill(c, 7f);
        Assert.False(SimdGemm.TrySgemmDirectParallel2D(a, 64, b, 32, c, 5, 64, 32, allowParallel: true));   // m < Mr
        Assert.All(c, v => Assert.Equal(7f, v));
    }
}
#endif
