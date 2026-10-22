using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public sealed class GotoGemmRoutingTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(32)]
    [InlineData(47)]
    public void IsPreferredForThreadBudget_RejectsSmallAndMidSizedMachines(int threadBudget)
    {
        Assert.False(GotoGemmFp32.IsPreferredForThreadBudget(threadBudget));
    }

    [Theory]
    [InlineData(48)]
    [InlineData(64)]
    [InlineData(128)]
    public void IsPreferredForThreadBudget_AllowsManyCoreMachines(int threadBudget)
    {
        Assert.True(GotoGemmFp32.IsPreferredForThreadBudget(threadBudget));
    }

    // #653: below the 48-thread gate, PackBoth splits along M only, at no fewer than 64 rows per
    // block. A transformer-sized M therefore cannot fill the budget, and those shapes route to the
    // 2D-tiled kernel instead. Large-M GEMMs keep PackBoth.
    [Theory]
    [InlineData(256, 16)]
    [InlineData(256, 32)]
    [InlineData(512, 16)]
    [InlineData(64, 2)]
    public void PackBothUnderOccupies_WhenItsMBlocksCannotFillTheBudget(int m, int threadBudget)
    {
        Assert.True(GotoGemmFp32.PackBothUnderOccupies(m, threadBudget));
    }

    [Theory]
    [InlineData(1024, 16)]
    [InlineData(4096, 32)]
    [InlineData(256, 4)]
    [InlineData(256, 1)]
    public void PackBothUnderOccupies_IsFalseWhenPackBothCanFillTheBudget(int m, int threadBudget)
    {
        Assert.False(GotoGemmFp32.PackBothUnderOccupies(m, threadBudget));
    }

    // A thin-M no-transpose GEMM with deep K (and at least 128 rows) runs packed (GotoGemm). Its tile width depends on the
    // thread count, so the result may not: with n a multiple of 16 every element is the same K-blocked FMA chain at every
    // count. (N stays above 128, out of the JIT generator's box, which takes small shapes only on one thread.)
    [SkippableTheory]
    [InlineData(128, 784, 512)]   // the parity MLP's first layer
    [InlineData(130, 600, 160)]   // a 4-row M-tail, K not a multiple of the 256 K-block
    public void ThinMDeepK_MatchesADoubleReference_AtEveryThreadCount(int m, int k, int n)
    {
        // Without the machine-code microkernel the packed route is never taken and this would pass through another
        // kernel, covering nothing it claims to.
        Skip.IfNot(GotoGemmFp32.IsAvailable, "The packed GotoGemm route is unavailable on this CPU/OS/runtime.");
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        var rng = RandomHelper.CreateSeededRandom(1055);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);
        float[]? first = null;
        try
        {
            foreach (int threads in new[] { 1, 3, 7, 16 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                var c = new float[m * n];
                for (int i = 0; i < c.Length; i++) c[i] = float.NaN;
                AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k);
                if (first is null)
                {
                    first = c;
                    double worst = 0;
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < n; j++)
                        {
                            double e = 0;
                            for (int p = 0; p < k; p++) e += (double)a[i * k + p] * b[p * n + j];
                            worst = Math.Max(worst, Math.Abs(e - c[i * n + j]));
                        }
                    Assert.True(worst < 1e-4, $"max |C - C_ref| = {worst:E3}");
                }
                else
                {
                    for (int i = 0; i < c.Length; i++)
                        Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(first[i]) == TestHelpers.MathCompat.SingleToInt32Bits(c[i]),
                            $"C[{i / n},{i % n}] at {threads} threads: {c[i]:G9}, at 1 thread {first[i]:G9}");
                }
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }
    // M from 64: a 77-token cross-attention context's K/V projection (M77 K1024 N2304) left PackBoth on 2 threads.
    // Thin-M very-wide-N with M a multiple of the 6-row microkernel stays on PackBoth's N-axis path.
    [Theory]
    [InlineData(77, 2304, 1024)]
    [InlineData(64, 1152, 1152)]
    [InlineData(256, 3456, 1152)]
    public void BeatsPackBoth_ClaimsShortMShapes(int m, int n, int k)
    {
        Assert.True(GotoGemmFp32.BeatsPackBoth(m, n, k));
    }

    [Theory]
    [InlineData(63, 2304, 1024)]
    [InlineData(96, 4096, 1024)]
    public void BeatsPackBoth_CedesTinyMAndSixAlignedVeryWideN(int m, int n, int k)
    {
        Assert.False(GotoGemmFp32.BeatsPackBoth(m, n, k));
    }

    [Theory]
    [InlineData(256, 768, 768)]
    [InlineData(256, 768, 3072)]
    [InlineData(250, 768, 520)]
    [InlineData(77, 1024, 2304)]
    public void TransformerShapes_OnTheNewRoute_MatchADoubleReference(int m, int k, int n)
    {
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 16;
            var rng = RandomHelper.CreateSeededRandom(653);
            var a = new Tensor<float>(new[] { m, k });
            var b = new Tensor<float>(new[] { k, n });
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);

            var c = new CpuEngine().BatchMatMul(a, b);

            Assert.Equal(new[] { m, n }, c.Shape.ToArray());
            // The reference reads flat copies: through the rank-2 indexer the 256x768x3072 case made 604M indexer
            // calls (41 s on a 16-core desktop) and outlasted the CI blame-hang timeout on a loaded 4-vCPU runner.
            float[] av = a.ToArray(), bv = b.ToArray(), cv = c.ToArray();
            double worst = 0;
            var row = new double[n];
            for (int i = 0; i < m; i++)
            {
                Array.Clear(row, 0, n);
                for (int p = 0; p < k; p++)
                {
                    double aip = av[i * k + p];
                    int bRow = p * n;
                    for (int j = 0; j < n; j++) row[j] += aip * bv[bRow + j];
                }
                for (int j = 0; j < n; j++) worst = Math.Max(worst, Math.Abs(row[j] - cv[i * n + j]));
            }

            // K=768 products of values in [-0.5, 0.5] sum to O(5); float accumulation error is ~1e-5,
            // and a mis-tiled or skipped block is off by O(0.1) or more.
            Assert.True(worst < 1e-4, $"max |C - C_ref| = {worst:E3}");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }

    // The routed per-tile kernel writes C first (overwrite kernel on the first K-panel, tail strips
    // zeroed per tile), so BlasManaged no longer clears C before it. A NaN-filled destination proves
    // every element is written: any cell left to a "+=" over stale memory stays NaN.
    [Theory]
    [InlineData(256, 768, 768)]
    [InlineData(250, 768, 520)]
    [InlineData(256, 3072, 768)]
    [InlineData(77, 1024, 2304)]
    public void RoutedPath_OverwritesAGarbageDestination(int m, int k, int n)
    {
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 16;
            var rng = RandomHelper.CreateSeededRandom(9);
            var a = new float[m * k];
            var b = new float[k * n];
            var c = new float[m * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < c.Length; i++) c[i] = float.NaN;

            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
                a, k, false, b, n, false, c, n, m, n, k,
                new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune, BetaZero = true });

            double worst = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double expected = 0;
                    for (int p = 0; p < k; p++) expected += (double)a[i * k + p] * b[p * n + j];
                    float actual = c[i * n + j];
                    Assert.False(float.IsNaN(actual), $"C[{i},{j}] was never written");
                    worst = Math.Max(worst, Math.Abs(expected - actual));
                }
            Assert.True(worst < 1e-3, $"max |C - C_ref| = {worst:E3}");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }

    // #653: PackBoth's fp32 6x16 kernel needs m % 6 == 0. An unaligned M (DiT's 256-token projections) was tail-split, or
    // on a many-core host sent whole to Streaming, which at 2-4 threads ran 256x4608x1152 in 316 ms against 34 ms on one.
    [Theory]
    [InlineData(256, true)]
    [InlineData(130, true)]
    [InlineData(252, false)]
    [InlineData(768, false)]
    public void PackBothNeedsTailSplit_ExactlyWhenMIsNotAMultipleOfItsTile(int m, bool expected)
    {
        Assert.Equal(expected, GotoGemmFp32.PackBothNeedsTailSplit(m));
    }

    // The DiT projection that ran 10x slower at 2 threads than at 1 (#653) now takes the per-tile route at every
    // budget, which gives the same bits whatever the thread count.
    [SkippableFact]
    public void AnUnalignedMFusedLinear_IsBitIdenticalAtEveryThreadCount()
    {
        Skip.IfNot(GotoGemmFp32.IsAvailable, "The packed GotoGemm route is unavailable on this CPU/OS/runtime.");
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        var engine = new CpuEngine();
        var rng = RandomHelper.CreateSeededRandom(653);
        Tensor<float> Rand(params int[] shape)
        {
            var t = new Tensor<float>(shape);
            for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
            return t;
        }
        var x = Rand(256, 4608);
        var w = Rand(4608, 1152);
        var bias = Rand(1152);
        float[]? first = null;
        try
        {
            foreach (int threads in new[] { 1, 2, 3, 4, 8, 16 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                float[] y = engine.FusedLinear(x, w, bias, FusedActivationType.None).ToArray();
                if (first is null) { first = y; continue; }
                for (int i = 0; i < y.Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(first[i]) == BitConverter.SingleToInt32Bits(y[i]),
                        $"threads={threads}: element {i} is {y[i]}, at 1 thread {first[i]}");
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }
}
