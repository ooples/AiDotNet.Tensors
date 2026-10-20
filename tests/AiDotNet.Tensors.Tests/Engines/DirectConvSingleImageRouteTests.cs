#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #653: single-image stride-1 3x3 convolutions on 40x40-80x80 planes with 64+ input channels take the direct kernels
/// (measured 1.10-2.29x over the im2col GEMM routes at 1 and 16 threads). Pins the rule's edges and checks the direct
/// result against the full im2col GEMM.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public sealed class DirectConvSingleImageRouteTests
{
    private static DirectConvShape Forward(int batch, int channels, int hw, int stride = 1) =>
        new(DirectConvPass.Forward, batch, channels, channels, hw, hw, 3, 3, stride, stride, 1, 1, 1, 1);

    [Theory]
    [InlineData(1, 64, 64, 1, true)]     // inside the rule
    [InlineData(1, 64, 40, 1, true)]     // 1600 positions: just past the batch rule's 1024
    [InlineData(1, 640, 80, 1, true)]
    [InlineData(1, 64, 128, 1, true)]    // 128x128 planes win once the output store is staged
    [InlineData(1, 64, 160, 1, true)]    // the 160x160 upper bound
    [InlineData(1, 64, 161, 1, false)]   // past it (unmeasured)
    [InlineData(1, 32, 64, 1, false)]    // 32 channels measured slower at 16 threads
    [InlineData(1, 768, 32, 1, true)]    // a 32x32 plane, exactly the batch rule's 1024 (since the cache-sized input passes)
    [InlineData(1, 128, 32, 1, true)]    // 32x32 at 128 channels: 16 input blocks, two 3x3 passes -> direct
    [InlineData(1, 112, 32, 1, false)]   // 32x32 at 112 channels: 14 blocks fit ONE pass -> stays on GEMM
    [InlineData(1, 64, 32, 1, false)]    // 32x32 at 64 channels: one pass, measured 0.94-0.97x -> GEMM
    [InlineData(1, 64, 31, 1, false)]    // 961 positions: below it
    [InlineData(1, 256, 16, 1, false)]   // a 256-position plane stays on the GEMM routes for a single image
    [InlineData(1, 64, 64, 2, false)]    // stride 2 is outside the single-image rule
    [InlineData(2, 64, 16, 1, true)]     // the batch rule is unchanged
    [InlineData(2, 64, 64, 1, false)]    // ...including its 1024-position limit for batches
    public void DefaultChoosesDirect_SingleImageRule(int batch, int channels, int hw, int stride, bool expected)
    {
        Assert.Equal(expected, DirectConvAvx2.DefaultChoosesDirect(Forward(batch, channels, hw, stride)));
    }

    [SkippableTheory]
    [InlineData(64, 64, 1)]
    [InlineData(96, 64, 16)]
    [InlineData(64, 48, 4)]
    [InlineData(64, 128, 16)]
    [InlineData(64, 41, 1)]      // 1681 positions: the last staged run is partial
    [InlineData(256, 32, 4)]     // 32x32, 256 channels: the input channels in two cache-sized passes
    public void SingleImageDirectConv_MatchesFullIm2Col(int channels, int hw, int threads)
    {
        Skip.IfNot(DirectConvAvx2.IsSupported, "The direct kernels need AVX2 and FMA.");
        var shape = new DirectConvShape(DirectConvPass.Forward, 1, channels, channels, hw, hw, 3, 3, 1, 1, 1, 1, 1, 1);
        Skip.IfNot(DirectConvAvx2.TryChoose(shape, out _), "A tuned entry routes this shape elsewhere on this host.");
        var e = new CpuEngine();
        var x = Rand(new[] { 1, channels, hw, hw }, 707);
        var kernel = Rand(new[] { channels, channels, 3, 3 }, 808);
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        float[] direct, full;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = threads;
            direct = e.Conv2D(x, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }).ToArray();
            using (CpuEngine.ForceFullIm2ColScope())
                full = e.Conv2D(x, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }).ToArray();
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = previous;
        }
        // Different summation order (direct vs GEMM): compare against the size of a K = 9 * channels dot product.
        double tolerance = 1e-5 * Math.Sqrt(9.0 * channels) + 1e-6;
        Assert.Equal(full.Length, direct.Length);
        bool anyDifferent = false;
        for (int i = 0; i < full.Length; i++)
        {
            Assert.True(Math.Abs(full[i] - direct[i]) <= tolerance, $"index {i}: full {full[i]} direct {direct[i]}");
            anyDifferent |= full[i] != direct[i];
        }
        // Positive control: identical bits everywhere would mean both calls ran the same route.
        Assert.True(anyDifferent, "the default call did not take a different route from the pinned full im2col");
    }

    /// <summary>The staged output store only reorders the writes: it must match the per-position store bit for bit,
    /// including a partial last run and both thread counts.</summary>
    [SkippableTheory]
    [InlineData(64, 41, 1)]
    [InlineData(64, 128, 16)]
    [InlineData(96, 64, 4)]
    public void StagedOutputStore_IsBitIdentical_ToPerPositionStore(int channels, int hw, int threads)
    {
        Skip.IfNot(DirectConvAvx2.IsSupported, "The direct kernels need AVX2 and FMA.");
        var e = new CpuEngine();
        var x = Rand(new[] { 1, channels, hw, hw }, 909);
        var kernel = Rand(new[] { channels, channels, 3, 3 }, 1010);
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        bool previousStaged = DirectConvAvx2.s_stagedStore;
        float[] staged, perPosition;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = threads;
            DirectConvAvx2.s_stagedStore = true;
            staged = e.Conv2D(x, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }).ToArray();
            DirectConvAvx2.s_stagedStore = false;
            perPosition = e.Conv2D(x, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }).ToArray();
        }
        finally
        {
            DirectConvAvx2.s_stagedStore = previousStaged;
            CpuParallelSettings.MaxDegreeOfParallelism = previous;
        }
        Assert.Equal(perPosition.Length, staged.Length);
        for (int i = 0; i < staged.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(staged[i]) == BitConverter.SingleToInt32Bits(perPosition[i]),
                $"index {i}: staged {staged[i]} per-position {perPosition[i]}");
    }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }
}
#endif
