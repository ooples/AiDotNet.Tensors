#if NET9_0_OR_GREATER
using System;
using Bm = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// An FP32 Aᵀ·B (a weight gradient Xᵀ·dY) in the thin-M box goes to PackBoth where the thin-M kernel runs its 256-bit
/// tile and the shape is one PackBoth measured faster on (<see cref="BlasManaged.ThinMTransAYieldsToPackBoth"/>), and the
/// result stays right either way.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class ThinMTransAPackBothRouteTests
{
    [Theory]
    // (m, n, k, 512-bit tile, threads, yields)
    [InlineData(784, 512, 128, false, 2, true)]   // MLP layer 1 dW
    [InlineData(512, 256, 128, false, 2, true)]   // MLP layer 2 dW
    [InlineData(784, 512, 128, false, 4, false)]  // k <= 128 past two threads: thin-M scales across cores
    [InlineData(1024, 256, 512, false, 4, true)]  // deep k: PackBoth at any thread count
    [InlineData(512, 128, 1024, false, 8, true)]
    [InlineData(784, 128, 64, false, 2, false)]   // shallow k
    [InlineData(512, 512, 64, false, 2, false)]
    [InlineData(128, 512, 256, false, 2, false)]  // few rows
    [InlineData(784, 128, 128, false, 2, false)]  // narrow
    [InlineData(784, 512, 128, true, 2, false)]   // 512-bit tile: thin-M everywhere
    [InlineData(1024, 256, 512, true, 2, false)]
    public void Predicate_FollowsTileShapeAndThreads(int m, int n, int k, bool v512, int threads, bool expected)
    {
        var (force0, threads0) = (SimdGemm.s_forceVector512Direct, CpuParallelSettings.MaxDegreeOfParallelism);
        try
        {
            SimdGemm.s_forceVector512Direct = v512;
            CpuParallelSettings.MaxDegreeOfParallelism = threads;
            Assert.Equal(expected, Bm.ThinMTransAYieldsToPackBoth(m, n, k));
        }
        finally
        {
            SimdGemm.s_forceVector512Direct = force0;
            CpuParallelSettings.MaxDegreeOfParallelism = threads0;
        }
    }

    [Theory]
    [InlineData(784, 512, 128, 2, true)]
    [InlineData(1024, 256, 512, 4, true)]
    [InlineData(790, 264, 520, 2, true)]   // m off the 6-row tile, n off 16, k off the panel
    [InlineData(784, 128, 64, 2, false)]
    [InlineData(784, 512, 128, 4, false)]
    public void Gemm_TransA_RoutesAndMatchesReference(int m, int n, int k, int threads, bool expectYield)
    {
        if (!System.Runtime.Intrinsics.X86.Fma.IsSupported) return; // the thin-M route needs FMA
        var (force0, threads0) = (SimdGemm.s_forceVector512Direct, CpuParallelSettings.MaxDegreeOfParallelism);
        try
        {
            SimdGemm.s_forceVector512Direct = false;
            CpuParallelSettings.MaxDegreeOfParallelism = threads;
            var rng = new Random(m * 31 + n * 7 + k);
            var a = new float[k * m]; var b = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);

            foreach (var mode in new[] { PackingMode.DisableAutotune, PackingMode.Auto })
            {
                var c = new float[m * n];
                for (int i = 0; i < c.Length; i++) c[i] = float.NaN; // every element must be written
                int before = Bm.t_thinMTransAYieldRuns;
                Bm.Gemm<float>(a, m, true, b, n, false, c.AsSpan(), n, m, n, k,
                    new BlasOptions<float> { PackingMode = mode });
                Assert.Equal(expectYield ? 1 : 0, Bm.t_thinMTransAYieldRuns - before);

                double maxErr = 0, maxRef = 0;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double want = 0;
                        for (int p = 0; p < k; p++) want += (double)a[p * m + i] * b[p * n + j];
                        maxErr = Math.Max(maxErr, Math.Abs(c[i * n + j] - want));
                        maxRef = Math.Max(maxRef, Math.Abs(want));
                    }
                Assert.True(maxErr <= 1e-5 * k * Math.Max(1, maxRef), $"{mode} {m}x{n}x{k}: max error {maxErr}");
            }
        }
        finally
        {
            SimdGemm.s_forceVector512Direct = force0;
            CpuParallelSettings.MaxDegreeOfParallelism = threads0;
        }
    }
}
#endif
