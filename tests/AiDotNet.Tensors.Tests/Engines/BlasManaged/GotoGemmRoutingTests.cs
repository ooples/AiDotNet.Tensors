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
        // Where the 512-bit direct tile runs, deep-K thin-M products leave the packed route; keep them on it here.
        bool beforeGoto512 = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_thinMGotoOn512;
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_thinMGotoOn512 = true;
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
            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_thinMGotoOn512 = beforeGoto512;
        }
    }

#if NET9_0_OR_GREATER
    [SkippableFact]
    public void ThinMDeepK_ProductionGemmHonorsPackedRouteOverride()
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Fma.IsSupported && GotoGemmFp32.IsAvailable,
            "Both direct FMA and packed GotoGemm routes must be available.");
        const int m = 128, k = 784, n = 512;
        var a = new float[m * k];
        var b = new float[k * n];
        var c = new float[m * n];
        var rng = RandomHelper.CreateSeededRandom(1644);
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);
        bool? beforeForce = AiDotNet.Tensors.Engines.Simd.SimdGemm.s_forceVector512Direct;
        bool beforeGoto = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_thinMGotoOn512;
        bool beforeDirect = AiDotNet.Tensors.Engines.Simd.SimdGemm.UseDirectParallel2D;
        try
        {
            AiDotNet.Tensors.Engines.Simd.SimdGemm.s_forceVector512Direct = true;
            AiDotNet.Tensors.Engines.Simd.SimdGemm.UseDirectParallel2D = true;
            foreach (bool packed in new[] { false, true })
            {
                AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_thinMGotoOn512 = packed;
                int before = AiDotNet.Tensors.Engines.Simd.SimdGemm.t_directParallel2DRuns;
                AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k,
                    new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune });
                Assert.Equal(before + (packed ? 0 : 1), AiDotNet.Tensors.Engines.Simd.SimdGemm.t_directParallel2DRuns);
            }
        }
        finally
        {
            AiDotNet.Tensors.Engines.Simd.SimdGemm.s_forceVector512Direct = beforeForce;
            AiDotNet.Tensors.Engines.Simd.SimdGemm.UseDirectParallel2D = beforeDirect;
            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_thinMGotoOn512 = beforeGoto;
        }
    }

    // The packed route's microkernel is 256-bit: where the 512-bit direct tile runs, a deep-K thin-M product takes
    // that tile instead, unless AIDOTNET_THINM_GOTO_512 keeps the packed route.
    [Fact]
    public void ThinMDeepK_LeavesThePackedRoute_WhereThe512BitTileRuns()
    {
        bool? beforeForce = AiDotNet.Tensors.Engines.Simd.SimdGemm.s_forceVector512Direct;
        bool beforeGoto512 = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_thinMGotoOn512;
        try
        {
            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_thinMGotoOn512 = false;
            AiDotNet.Tensors.Engines.Simd.SimdGemm.s_forceVector512Direct = true;
            Assert.False(AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.ThinMDeepKTakesGoto(128, 784));
            AiDotNet.Tensors.Engines.Simd.SimdGemm.s_forceVector512Direct = false;
            Assert.True(AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.ThinMDeepKTakesGoto(128, 784));
            AiDotNet.Tensors.Engines.Simd.SimdGemm.s_forceVector512Direct = true;
            // More rows than the direct tile serves keep the packed route.
            Assert.True(AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.ThinMDeepKTakesGoto(
                AiDotNet.Tensors.Engines.Simd.SimdGemm.DirectParallel2DMaxM + 1, 784));
            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_thinMGotoOn512 = true;
            Assert.True(AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.ThinMDeepKTakesGoto(128, 784));
        }
        finally
        {
            AiDotNet.Tensors.Engines.Simd.SimdGemm.s_forceVector512Direct = beforeForce;
            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_thinMGotoOn512 = beforeGoto512;
        }
    }
#endif
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
                    Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(first[i]) == TestHelpers.MathCompat.SingleToInt32Bits(y[i]),
                        $"threads={threads}: element {i} is {y[i]}, at 1 thread {first[i]}");
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }


    // #653: under the default deterministic mode the engine's float matmul (DisableAutotune) now takes GotoGemm when n
    // is a multiple of its 16-column panel. That route must give the same bits at every thread count and agree with an
    // fp64 reference; with n off the panel it must not be taken.
    [SkippableTheory]
    [InlineData(256, 640, 768, true)]   // M=256: a 4-row tail on the 6-row tile, as in the DiT linears
    [InlineData(130, 600, 528, true)]   // K off the 256 K-block
    [InlineData(192, 512, 768, true)]
    [InlineData(256, 1152, 1152, true)] // a DiT-XL projection
    [InlineData(256, 640, 776, false)]  // n % 16 != 0: a scalar N-tail, so not thread-count invariant
    public void DeterministicDisableAutotune_TakesGotoWhereThreadCountInvariant(int m, int k, int n, bool expectGoto)
    {
        Skip.IfNot(GotoGemmFp32.IsAvailable, "The packed GotoGemm route is unavailable on this CPU/OS/runtime.");
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        bool? beforeDeterministic = BlasProvider.GetThreadLocalDeterministicMode();
        var rng = RandomHelper.CreateSeededRandom(6530);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);
        float[]? first = null;
        try
        {
            BlasProvider.SetThreadLocalDeterministicMode(true);
            foreach (int threads in new[] { 1, 2, 3, 7, 16, 32 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                var c = new float[m * n];
                for (int i = 0; i < c.Length; i++) c[i] = float.NaN;
                int runs = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.t_deterministicGotoRuns;
                AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k,
                    new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune });
                Assert.True(runs + (expectGoto ? 1 : 0) == AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.t_deterministicGotoRuns, $"route at {threads} threads: expected GotoGemm={expectGoto}");
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
                    continue;
                }
                for (int i = 0; i < c.Length; i++)
                    Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(first[i]) == TestHelpers.MathCompat.SingleToInt32Bits(c[i]),
                        $"C[{i / n},{i % n}] at {threads} threads: {c[i]:G9}, at 1 thread {first[i]:G9}");
            }
            // An explicit thread count (BlasOptions.NumThreads) takes the same route and gives the same bits.
            CpuParallelSettings.MaxDegreeOfParallelism = 16;
            foreach (int explicitThreads in new[] { -1, 1, 3, 64 })
            {
                var c = new float[m * n];
                int runs = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.t_deterministicGotoRuns;
                AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k,
                    new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune, NumThreads = explicitThreads });
                Assert.True(runs + (expectGoto ? 1 : 0) == AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.t_deterministicGotoRuns,
                    $"route at NumThreads={explicitThreads}: expected GotoGemm={expectGoto}");
                for (int i = 0; i < c.Length; i++)
                    Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(first![i]) == TestHelpers.MathCompat.SingleToInt32Bits(c[i]),
                        $"C[{i / n},{i % n}] at NumThreads={explicitThreads}: {c[i]:G9}, at 1 thread {first[i]:G9}");
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
            BlasProvider.SetThreadLocalDeterministicMode(beforeDeterministic);
        }
    }

    // The engine's float matmul is the caller this serves; the kill switch restores the previous routing.
    [SkippableFact]
    public void EngineMatMul_TakesDeterministicGoto_AndTheKillSwitchRestoresTheOldRoute()
    {
        Skip.IfNot(GotoGemmFp32.IsAvailable, "The packed GotoGemm route is unavailable on this CPU/OS/runtime.");
        bool? beforeDeterministic = BlasProvider.GetThreadLocalDeterministicMode();
        bool beforeSwitch = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_disableDeterministicGoto;
        var engine = new CpuEngine();
        var rng = RandomHelper.CreateSeededRandom(6531);
        Tensor<float> Rand(params int[] shape)
        {
            var t = new Tensor<float>(shape);
            for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
            return t;
        }
        var x = Rand(256, 640);
        var w = Rand(640, 768);
        try
        {
            BlasProvider.SetThreadLocalDeterministicMode(true);
            int runs = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.t_deterministicGotoRuns;
            float[] viaGoto = engine.BatchMatMul(x, w).ToArray();
            Assert.Equal(runs + 1, AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.t_deterministicGotoRuns);

            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_disableDeterministicGoto = true;
            runs = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.t_deterministicGotoRuns;
            float[] viaOld = engine.BatchMatMul(x, w).ToArray();
            Assert.Equal(runs, AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.t_deterministicGotoRuns);
            double worst = 0;
            for (int i = 0; i < viaGoto.Length; i++) worst = Math.Max(worst, Math.Abs((double)viaGoto[i] - viaOld[i]));
            Assert.True(worst < 1e-4, $"max |goto - old| = {worst:E3}");
        }
        finally
        {
            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_disableDeterministicGoto = beforeSwitch;
            BlasProvider.SetThreadLocalDeterministicMode(beforeDeterministic);
        }
    }

    // Where the thin-M direct route serves one thread count (n <= 512, k <= 1024) and this route the others, the bits
    // must still agree across counts.
    [SkippableTheory]
    [InlineData(256, 640, 512)]
    [InlineData(132, 1024, 496)]
    public void DeterministicDisableAutotune_AtTheThinMBoundary_IsBitIdenticalAtEveryThreadCount(int m, int k, int n)
    {
        Skip.IfNot(GotoGemmFp32.IsAvailable, "The packed GotoGemm route is unavailable on this CPU/OS/runtime.");
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        bool? beforeDeterministic = BlasProvider.GetThreadLocalDeterministicMode();
        var rng = RandomHelper.CreateSeededRandom(6532);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);
        float[]? first = null;
        try
        {
            BlasProvider.SetThreadLocalDeterministicMode(true);
            foreach (int threads in new[] { 1, 2, 3, 7, 16, 32 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                var c = new float[m * n];
                AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k,
                    new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune });
                if (first is null) { first = c; continue; }
                for (int i = 0; i < c.Length; i++)
                    Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(first[i]) == TestHelpers.MathCompat.SingleToInt32Bits(c[i]),
                        $"C[{i / n},{i % n}] at {threads} threads: {c[i]:G9}, at 1 thread {first[i]:G9}");
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
            BlasProvider.SetThreadLocalDeterministicMode(beforeDeterministic);
        }
    }
}
