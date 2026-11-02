// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Winograd F(4x4, 3x3) against a double-precision direct sum: the forward and the input gradient, written and
/// accumulated, on planes that are and are not multiples of the 4x4 tile, with the error bounded relative to the sum of
/// the products' magnitudes (the transforms' rounding grows with it). Then the engine's routing: the forward and the
/// input gradient of a 3x3 stride-1 convolution take the kernel and agree with the direct route.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class WinogradF43Tests
{
    /// <summary>Whether the kernel itself can run on this host.</summary>
    private static bool KernelRuns =>
        System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported;

    /// <summary>Whether the engine may route a convolution to the kernel: it runs, and the operator opt-out is not set.</summary>
    private static bool WinogradRoutes =>
        KernelRuns && Environment.GetEnvironmentVariable("AIDOTNET_CONV_WINOGRAD") != "0";

    public static TheoryData<int, int, int, int, int> Shapes => new()
    {
        { 2, 8, 16, 8, 8 },
        { 1, 16, 8, 9, 11 },
        { 3, 24, 32, 14, 14 },
        { 2, 64, 64, 16, 16 },
        { 1, 8, 8, 13, 8 },
        { 8, 32, 16, 8, 8 },
        { 1, 16, 24, 30, 33 },   // blocked: 9 tiles a row -> bands of 3 tile rows, the last one partial
        { 2, 32, 64, 32, 32 },   // blocked: two bands of 4 tile rows
        { 3, 32, 32, 30, 33 },   // blocked weight gradient: nine bands over eight splits, the last split adds two
        { 9, 40, 64, 16, 12 },   // blocked weight gradient: 40 columns (32 + an 8-wide tail), 64 rows (21 three-row tiles + 1)
        { 8, 128, 128, 4, 4 },   // few patches: one tile per image, eight patches
        { 3, 128, 144, 4, 4 },   // few patches: three patches padded to a group of eight; nine output-channel blocks
        { 2, 128, 128, 5, 6 },   // few patches: 2x2 partial tiles per image
        { 3, 128, 136, 4, 4 },   // input gradient few-patch (136 input channels: a 16 + 8 slab tail); forward batch-wide
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Forward_MatchesTheDirectSum(int n, int c, int k, int h, int w)
    {
        Skip.IfNot(KernelRuns, "The Winograd kernel needs AVX2 + FMA.");
        var rng = new Random(n * 100 + c + k + h * 7 + w);
        float[] x = Fill(rng, n * c * h * w), wt = Fill(rng, k * c * 9);
        foreach (bool accumulate in new[] { false, true })
        {
            float[] y = Fill(new Random(5), n * k * h * w);
            float[] y0 = (float[])y.Clone();
            WinogradF43.Forward(x, 0, wt, 0, y, 0, accumulate, n, c, h, w, k);
            for (int b = 0; b < n; b++)
                for (int o = 0; o < k; o++)
                    for (int i = 0; i < h; i++)
                        for (int j = 0; j < w; j++)
                        {
                            double sum = 0, mag = 0;
                            for (int ic = 0; ic < c; ic++)
                                for (int di = 0; di < 3; di++)
                                    for (int dj = 0; dj < 3; dj++)
                                    {
                                        int yi = i - 1 + di, xj = j - 1 + dj;
                                        if (yi < 0 || yi >= h || xj < 0 || xj >= w) continue;
                                        double p = (double)x[((b * c + ic) * h + yi) * w + xj] * wt[(o * c + ic) * 9 + di * 3 + dj];
                                        sum += p; mag += Math.Abs(p);
                                    }
                            int at = ((b * k + o) * h + i) * w + j;
                            double expected = (accumulate ? y0[at] : 0) + sum;
                            Assert.True(Math.Abs(y[at] - expected) <= 2e-6 * (mag + 1) * 8,
                                $"n={n} c={c} k={k} {h}x{w} acc={accumulate} [{b},{o},{i},{j}]: expected {expected}, got {y[at]}");
                        }
        }
    }

    // A wide layer forced through the blocked form agrees with the batch-wide passes (both are F(4x4); only the GEMMs
    // sum in different orders).
    [SkippableFact]
    public void Blocked_AgreesWithTheBatchWidePasses_OnAWideLayer()
    {
        Skip.IfNot(KernelRuns, "The Winograd kernel needs AVX2 + FMA.");
        const int n = 3, c = 64, k = 72, h = 20, w = 18;
        var rng = new Random(91);
        float[] x = Fill(rng, n * c * h * w), wt = Fill(rng, k * c * 9);
        float[] dy = Fill(rng, n * k * h * w);
        float[] wide = new float[n * k * h * w], blocked = new float[wide.Length];
        float[] wideDw = new float[k * c * 9], blockedDw = new float[wideDw.Length];
        WinogradF43.Forward(x, 0, wt, 0, wide, 0, false, n, c, h, w, k);
        WinogradF43.BackwardKernel(x, 0, dy, 0, wideDw, 0, false, n, c, h, w, k);
        bool previous = WinogradF43.s_forceBlockedForTest;
        try
        {
            WinogradF43.s_forceBlockedForTest = true;
            WinogradF43.Forward(x, 0, wt, 0, blocked, 0, false, n, c, h, w, k);
            WinogradF43.BackwardKernel(x, 0, dy, 0, blockedDw, 0, false, n, c, h, w, k);
        }
        finally
        {
            WinogradF43.s_forceBlockedForTest = previous;
        }
        for (int i = 0; i < wide.Length; i++)
            Assert.True(Math.Abs(wide[i] - blocked[i]) <= 1e-4 * (1 + Math.Abs(wide[i])), $"[{i}] {wide[i]} vs {blocked[i]}");
        // About a thousand products per weight gradient (~0.25 each).
        for (int i = 0; i < wideDw.Length; i++)
            Assert.True(Math.Abs(wideDw[i] - blockedDw[i]) <= 1e-3 * (1 + Math.Abs(wideDw[i])), $"dw[{i}] {wideDw[i]} vs {blockedDw[i]}");
    }

#if NET9_0_OR_GREATER
    // The blocked passes' 512-bit GEMM tile (eight rows x 32 columns, then 16- and 8-wide column tails) sums every output's
    // depth terms in the same order as the 256-bit tile: bit-identical forward, input and weight gradients. Forced on
    // hosts without AVX-512 (Vector512 then runs in software).
    [SkippableTheory]
    [InlineData(2, 64, 64, 16, 16)]    // 32 patches a band: four 8-row tiles; 64 columns: two 32-wide tiles
    [InlineData(3, 40, 56, 30, 33)]    // 40 and 56 columns: a 32-wide tile and an 8- or a 16 + 8-wide tail; partial bands
    [InlineData(9, 48, 64, 16, 12)]    // weight gradient rows 64, columns 48 (32 + 16); bands of 12 patches (8 + 4 rows)
    public void BlockedGemm512_IsBitIdenticalTo256BitTile(int n, int c, int k, int h, int w)
    {
        Skip.IfNot(KernelRuns, "The Winograd kernel needs AVX2 + FMA.");
        var rng = new Random(n * 7 + c + k + h + w);
        float[] x = Fill(rng, n * c * h * w), wt = Fill(rng, k * c * 9), dy = Fill(rng, n * k * h * w);
        bool previousForce = WinogradF43.s_forceBlockedForTest, previous512 = WinogradF43.s_blockGemm512;
        bool? previousVector = SimdGemm.s_forceVector512Direct;
        (float[] Y, float[] Dx, float[] Dw) Run(bool wide)
        {
            WinogradF43.s_blockGemm512 = wide;
            SimdGemm.s_forceVector512Direct = wide ? true : previousVector;
            float[] y = Fill(new Random(1), n * k * h * w), dx = Fill(new Random(2), n * c * h * w), dw = Fill(new Random(3), k * c * 9);
            WinogradF43.Forward(x, 0, wt, 0, y, 0, true, n, c, h, w, k);
            WinogradF43.BackwardInput(dy, 0, wt, 0, dx, 0, true, n, c, h, w, k);
            WinogradF43.BackwardKernel(x, 0, dy, 0, dw, 0, true, n, c, h, w, k);
            return (y, dx, dw);
        }
        try
        {
            WinogradF43.s_forceBlockedForTest = true;
            var narrow = Run(false);
            var wide = Run(true);
            Assert.Equal(narrow.Y, wide.Y);
            Assert.Equal(narrow.Dx, wide.Dx);
            Assert.Equal(narrow.Dw, wide.Dw);
        }
        finally
        {
            WinogradF43.s_forceBlockedForTest = previousForce;
            WinogradF43.s_blockGemm512 = previous512;
            SimdGemm.s_forceVector512Direct = previousVector;
        }
    }
#endif

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void BackwardInput_MatchesTheDirectSum(int n, int c, int k, int h, int w)
    {
        Skip.IfNot(KernelRuns, "The Winograd kernel needs AVX2 + FMA.");
        var rng = new Random(n * 31 + c + k + h + w * 3);
        float[] dy = Fill(rng, n * k * h * w), wt = Fill(rng, k * c * 9);
        foreach (bool accumulate in new[] { false, true })
        {
            float[] dx = Fill(new Random(9), n * c * h * w);
            float[] dx0 = (float[])dx.Clone();
            WinogradF43.BackwardInput(dy, 0, wt, 0, dx, 0, accumulate, n, c, h, w, k);
            for (int b = 0; b < n; b++)
                for (int ic = 0; ic < c; ic++)
                    for (int i = 0; i < h; i++)
                        for (int j = 0; j < w; j++)
                        {
                            // dX[ic][i][j] = sum over o, di, dj of dY[o][i + 1 - di][j + 1 - dj] w[o][ic][di][dj].
                            double sum = 0, mag = 0;
                            for (int o = 0; o < k; o++)
                                for (int di = 0; di < 3; di++)
                                    for (int dj = 0; dj < 3; dj++)
                                    {
                                        int yi = i + 1 - di, xj = j + 1 - dj;
                                        if (yi < 0 || yi >= h || xj < 0 || xj >= w) continue;
                                        double p = (double)dy[((b * k + o) * h + yi) * w + xj] * wt[(o * c + ic) * 9 + di * 3 + dj];
                                        sum += p; mag += Math.Abs(p);
                                    }
                            int at = ((b * c + ic) * h + i) * w + j;
                            double expected = (accumulate ? dx0[at] : 0) + sum;
                            Assert.True(Math.Abs(dx[at] - expected) <= 2e-6 * (mag + 1) * 8,
                                $"n={n} c={c} k={k} {h}x{w} acc={accumulate} [{b},{ic},{i},{j}]: expected {expected}, got {dx[at]}");
                        }
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void BackwardKernel_MatchesTheDirectSum(int n, int c, int k, int h, int w)
    {
        Skip.IfNot(KernelRuns, "The Winograd kernel needs AVX2 + FMA.");
        var rng = new Random(n * 17 + c * 3 + k + h + w * 5);
        float[] x = Fill(rng, n * c * h * w), dy = Fill(rng, n * k * h * w);
        foreach (bool accumulate in new[] { false, true })
        {
            float[] dw = Fill(new Random(13), k * c * 9);
            float[] dw0 = (float[])dw.Clone();
            WinogradF43.BackwardKernel(x, 0, dy, 0, dw, 0, accumulate, n, c, h, w, k);
            for (int o = 0; o < k; o++)
                for (int ic = 0; ic < c; ic++)
                    for (int di = 0; di < 3; di++)
                        for (int dj = 0; dj < 3; dj++)
                        {
                            // dW[o][ic][di][dj] = sum over b, i, j of dY[b][o][i][j] x[b][ic][i - 1 + di][j - 1 + dj].
                            double sum = 0, mag = 0;
                            for (int b = 0; b < n; b++)
                                for (int i = 0; i < h; i++)
                                    for (int j = 0; j < w; j++)
                                    {
                                        int yi = i - 1 + di, xj = j - 1 + dj;
                                        if (yi < 0 || yi >= h || xj < 0 || xj >= w) continue;
                                        double p = (double)dy[((b * k + o) * h + i) * w + j] * x[((b * c + ic) * h + yi) * w + xj];
                                        sum += p; mag += Math.Abs(p);
                                    }
                            int at = (o * c + ic) * 9 + di * 3 + dj;
                            double expected = (accumulate ? dw0[at] : 0) + sum;
                            Assert.True(Math.Abs(dw[at] - expected) <= 2e-6 * (mag + 1) * 8,
                                $"n={n} c={c} k={k} {h}x{w} acc={accumulate} [{o},{ic},{di},{dj}]: expected {expected}, got {dw[at]}");
                        }
        }
    }

    [SkippableFact]
    public void Engine_WeightGradient_TakesTheKernel_AndAgreesWithTheDirectRoute()
    {
        Skip.IfNot(WinogradRoutes, "Winograd routing needs AVX2 + FMA and AIDOTNET_CONV_WINOGRAD not set to 0.");
        const int n = 8, c = 128, k = 136, h = 16, w = 16;
        var rng = new Random(78);
        var x = new Tensor<float>(Fill(rng, n * c * h * w), new[] { n, c, h, w });
        var dy = new Tensor<float>(Fill(rng, n * k * h * w), new[] { n, k, h, w });
        var engine = new CpuEngine();
        int[] stride = { 1, 1 }, pad = { 1, 1 }, dil = { 1, 1 }, kernelShape = { k, c, 3, 3 };
        float[] allocWino = engine.Conv2DBackwardKernel(dy, x, kernelShape, stride, pad, dil).ToArray();
        var intoWino = new Tensor<float>(kernelShape);
        engine.Conv2DBackwardKernelInto(intoWino, dy, x, kernelShape, stride, pad, dil, false);
        float[] allocDirect;
        var intoDirect = new Tensor<float>(kernelShape);
        WinogradF43.s_disable = true;
        try
        {
            allocDirect = engine.Conv2DBackwardKernel(dy, x, kernelShape, stride, pad, dil).ToArray();
            engine.Conv2DBackwardKernelInto(intoDirect, dy, x, kernelShape, stride, pad, dil, false);
        }
        finally
        {
            WinogradF43.s_disable = false;
        }
        // 2048 products per weight (~0.25 each): allow 1e-3 of their typical sum.
        AssertClose(allocDirect, allocWino, "weight gradient", 2e-2);
        AssertClose(intoDirect.ToArray(), intoWino.ToArray(), "weight gradient (into)", 2e-2);
        Assert.NotEqual(allocDirect, allocWino);
        Assert.NotEqual(intoDirect.ToArray(), intoWino.ToArray());
    }

    // The first ResNet stage (64 -> 64 on 32x32 planes) takes the blocked weight gradient through the engine and agrees
    // with the direct route.
    [SkippableFact]
    public void Engine_NarrowWeightGradient_TakesTheBlockedKernel_AndAgreesWithTheDirectRoute()
    {
        Skip.IfNot(WinogradRoutes, "Winograd routing needs AVX2 + FMA and AIDOTNET_CONV_WINOGRAD not set to 0.");
        Skip.IfNot(WinogradF43.s_blockedNarrow && WinogradF43.s_minBackwardPatches <= 512, "Default blocked routing only.");
        const int n = 8, c = 64, k = 64, h = 32, w = 32;
        var rng = new Random(79);
        var x = new Tensor<float>(Fill(rng, n * c * h * w), new[] { n, c, h, w });
        var dy = new Tensor<float>(Fill(rng, n * k * h * w), new[] { n, k, h, w });
        var engine = new CpuEngine();
        int[] stride = { 1, 1 }, pad = { 1, 1 }, dil = { 1, 1 }, kernelShape = { k, c, 3, 3 };
        Assert.True(WinogradF43.UsesBlockedWeightGradient(c, k));
        var wino = new Tensor<float>(kernelShape);
        engine.Conv2DBackwardKernelInto(wino, dy, x, kernelShape, stride, pad, dil, false);
        var direct = new Tensor<float>(kernelShape);
        WinogradF43.s_disable = true;
        try
        {
            engine.Conv2DBackwardKernelInto(direct, dy, x, kernelShape, stride, pad, dil, false);
        }
        finally
        {
            WinogradF43.s_disable = false;
        }
        // 8192 products per weight (~0.25 each).
        AssertClose(direct.ToArray(), wino.ToArray(), "narrow weight gradient", 4e-2);
        Assert.NotEqual(direct.ToArray(), wino.ToArray());
    }

    [SkippableFact]
    public void Applies_OnlyTo3x3Stride1Padding1_WithChannelMultiplesOfEight()
    {
        Skip.IfNot(WinogradRoutes, "Winograd routing needs AVX2 + FMA and AIDOTNET_CONV_WINOGRAD not set to 0.");
        Skip.IfNot(WinogradF43.s_blockedNarrow && !WinogradF43.s_forceBlockedForTest
                   && WinogradF43.PatchFloorsAreDefault,
            "Default routing only (AIDOTNET_WINOGRAD_BLOCKED and the patch floors unset).");
        bool fullRate = WinogradF43.DirectKernelIsFullRate512;
        Assert.Equal(fullRate ? WinogradF43.FullRateMinPatches : 32, WinogradF43.s_minPatches);
        Assert.True(WinogradF43.Applies(8, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.False(WinogradF43.Applies(8, 64, 64, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1));   // stride 2
        Assert.False(WinogradF43.Applies(8, 64, 64, 32, 32, 3, 3, 1, 1, 0, 0, 1, 1));   // padding 0
        Assert.False(WinogradF43.Applies(8, 64, 64, 32, 32, 3, 3, 1, 1, 2, 2, 2, 2));   // dilated
        Assert.False(WinogradF43.Applies(8, 3, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1));    // 3 input channels
        Assert.True(WinogradF43.Applies(64, 32, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1));   // 32 input channels: blocked
        Assert.False(WinogradF43.Applies(64, 8, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1));   // 8 input channels
        Assert.Equal(WinogradF43.s_fewMaxPatches >= 32, WinogradF43.Applies(8, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));  // 32 patches: few-patch form
        Assert.Equal(!fullRate, WinogradF43.Applies(8, 96, 96, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));     // 32 patches, 96 channels: the floor
        Assert.Equal(!fullRate, WinogradF43.Applies(16, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));  // 64 patches
        Assert.False(WinogradF43.Applies(8, 64, 60, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1));   // 60 output channels
        Assert.False(WinogradF43.Applies(8, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1));     // 4x4 plane
        // Wide layers with few patches take the few-patch form, on planes down to 4x4; not the weight gradient.
        Assert.Equal(WinogradF43.s_fewMaxPatches >= 8, WinogradF43.Applies(8, 512, 512, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.Equal(WinogradF43.s_fewMaxPatches >= 8, WinogradF43.AppliesToInputGradient(8, 512, 512, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.False(WinogradF43.AppliesToWeightGradient(8, 512, 512, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.False(WinogradF43.Applies(8, 512, 512, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1));    // 3x3 plane
        Assert.False(WinogradF43.Applies(8, 512, 520, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1));    // 520 output channels
        Assert.False(WinogradF43.Applies(1, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));     // 4 patches
        Assert.False(WinogradF43.Applies(4, 96, 96, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));     // 16 patches, 96 channels
        Assert.False(WinogradF43.Applies(8, 64, 64, 32, 32, 5, 5, 1, 1, 2, 2, 1, 1));   // 5x5
        // The batch-wide weight gradient wants wider layers (C K >= 64 (C + K)); layers of 32 to 64 channels run blocked.
        Assert.True(WinogradF43.AppliesToWeightGradient(8, 128, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.True(WinogradF43.AppliesToWeightGradient(8, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.True(WinogradF43.AppliesToWeightGradient(8, 32, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.False(WinogradF43.AppliesToWeightGradient(8, 16, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.False(WinogradF43.AppliesToWeightGradient(8, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1));   // 128 patches
        Assert.True(WinogradF43.AppliesToWeightGradient(16, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1));   // 256 patches
        Assert.False(WinogradF43.AppliesToWeightGradient(8, 64, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1));
        // The forward and input gradient run blocked up to 64 channels each side, batch-wide past it.
        Assert.True(WinogradF43.UseBlocked(64, 64));
        Assert.True(WinogradF43.UseBlocked(32, 128));
        Assert.False(WinogradF43.UseBlocked(64, 128));
        Assert.False(WinogradF43.UseBlocked(96, 96));
    }

    [SkippableFact]
    public void Gradients_TakeFewerPatchesThanTheForward()
    {
        Skip.IfNot(WinogradRoutes, "Winograd routing needs AVX2 + FMA and AIDOTNET_CONV_WINOGRAD not set to 0.");
        Skip.IfNot(WinogradF43.PatchFloorsAreDefault, "Default patch floors only (AIDOTNET_WINOGRAD_MIN_*PATCHES unset).");
        // ResNet-18 256 -> 256 on 8x8 planes at batch 8: 32 patches. The weight gradient takes it; the forward and input
        // gradient do unless a full-rate 512-bit unit runs the direct kernel (floors 128 and 64).
        bool fullRate = WinogradF43.DirectKernelIsFullRate512;
        Assert.Equal(fullRate ? WinogradF43.FullRateInputGradientMinPatches : 32, WinogradF43.s_minInputGradientPatches);
        Assert.Equal(!fullRate, WinogradF43.AppliesToInputGradient(8, 96, 96, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));
        // 256 channels: the few-patch form takes 32 patches on every host.
        Assert.Equal(WinogradF43.s_fewMaxPatches >= 32, WinogradF43.AppliesToInputGradient(8, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.True(WinogradF43.AppliesToInputGradient(16, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));   // 64 patches
        Assert.True(WinogradF43.AppliesToWeightGradient(8, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));
        // 16 patches: the weight gradient stays direct; the input gradient takes only the few-patch form (wide layers).
        Assert.False(WinogradF43.AppliesToWeightGradient(4, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.Equal(WinogradF43.s_fewMaxPatches >= 16, WinogradF43.AppliesToInputGradient(4, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.False(WinogradF43.AppliesToInputGradient(4, 96, 96, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));
        // The input gradient keeps the forward's other rules.
        Assert.False(WinogradF43.AppliesToInputGradient(8, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1));
    }

    [Fact]
    public void HalfRateVector512_IsZen4Only()
    {
        Assert.True(CpuMicroarchitecture.IsHalfRateVector512("AuthenticAMD", 0x19, avx512: true));    // Zen 4
        Assert.False(CpuMicroarchitecture.IsHalfRateVector512("AuthenticAMD", 0x19, avx512: false));  // Zen 3
        Assert.False(CpuMicroarchitecture.IsHalfRateVector512("AuthenticAMD", 0x1A, avx512: true));   // Zen 5
        Assert.False(CpuMicroarchitecture.IsHalfRateVector512("GenuineIntel", 0x6, avx512: true));
    }

    [SkippableFact]
    public void WeightGradient_FallsBackPastItsWorkspaceCap()
    {
        Skip.IfNot(WinogradRoutes, "The Winograd kernel needs AVX2 + FMA and must not be opted out.");
        // A ResNet 64 -> 64 layer at 56x56, batch 128, fits the cap; 128 -> 128 at batch 256 does not, though its
        // forward still applies.
        Assert.True(WinogradF43.WeightGradientWorkspaceFloats(128, 64, 64, 56, 56) <= WinogradF43.MaxWeightGradientWorkspaceFloats);
        Assert.True(WinogradF43.WeightGradientWorkspaceFloats(256, 128, 128, 56, 56) > WinogradF43.MaxWeightGradientWorkspaceFloats);
        Assert.True(WinogradF43.Applies(256, 128, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.False(WinogradF43.AppliesToWeightGradient(256, 128, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1));
    }

    [SkippableFact]
    public void Engine_ForwardAndInputGradient_TakeTheKernel_AndAgreeWithTheDirectRoute()
    {
        Skip.IfNot(WinogradRoutes, "Winograd routing needs AVX2 + FMA and AIDOTNET_CONV_WINOGRAD not set to 0.");
        const int n = 8, c = 64, k = 72, h = 16, w = 16;
        var rng = new Random(77);
        var x = new Tensor<float>(Fill(rng, n * c * h * w), new[] { n, c, h, w });
        var wt = new Tensor<float>(Fill(rng, k * c * 9), new[] { k, c, 3, 3 });
        var dy = new Tensor<float>(Fill(rng, n * k * h * w), new[] { n, k, h, w });
        var engine = new CpuEngine();
        int[] stride = { 1, 1 }, pad = { 1, 1 }, dil = { 1, 1 };
        float[] yWino = engine.Conv2D(x, wt, stride, pad, dil).ToArray();
        float[] dxWino = engine.Conv2DBackwardInput(dy, wt, x._shape, stride, pad, dil).ToArray();
        float[] yDirect, dxDirect;
        WinogradF43.s_disable = true;
        try
        {
            yDirect = engine.Conv2D(x, wt, stride, pad, dil).ToArray();
            dxDirect = engine.Conv2DBackwardInput(dy, wt, x._shape, stride, pad, dil).ToArray();
        }
        finally
        {
            WinogradF43.s_disable = false;
        }
        AssertClose(yDirect, yWino, "forward");
        AssertClose(dxDirect, dxWino, "input gradient");
        // The two routes round differently: equal arrays would mean the engine never reached the kernel.
        Assert.NotEqual(yDirect, yWino);
        Assert.NotEqual(dxDirect, dxWino);
    }

#if NET9_0_OR_GREATER
    // The 512-bit few-patch tiles (32 output channels per job, slabs of 8 input channels) run the 256-bit tiles'
    // multiply-adds in the same order per lane, so they agree bit for bit.
    [SkippableTheory]
    [InlineData(8, 128, 128, 4, 4, false)]
    [InlineData(2, 128, 160, 5, 6, false)]
    [InlineData(3, 160, 128, 4, 4, true)]
    [InlineData(4, 128, 128, 8, 8, true)]
    public void FewPatches512_IsBitIdenticalTo256BitTiles(int n, int c, int k, int h, int w, bool inputGradient)
    {
        Skip.IfNot(KernelRuns, "The Winograd kernel needs AVX2 + FMA.");
        var rng = new Random(n + c + k + h * w);
        float[] x = Fill(rng, n * (inputGradient ? k : c) * h * w), wt = Fill(rng, k * c * 9);
        int outLength = n * (inputGradient ? c : k) * h * w;
        float[] Run()
        {
            var y = new float[outLength];
            if (inputGradient) WinogradF43.BackwardInput(x, 0, wt, 0, y, 0, false, n, c, h, w, k);
            else WinogradF43.Forward(x, 0, wt, 0, y, 0, false, n, c, h, w, k);
            return y;
        }
        bool previous512 = WinogradF43.s_fewPatches512, previousForce = WinogradF43.s_forceFewPatches512ForTest;
        int previousMax = WinogradF43.s_fewMaxPatches;
        try
        {
            WinogradF43.s_fewMaxPatches = 32;
            WinogradF43.s_fewPatches512 = false;
            float[] narrow = Run();
            WinogradF43.s_fewPatches512 = true;
            WinogradF43.s_forceFewPatches512ForTest = true;
            Assert.Equal(narrow, Run());
        }
        finally
        {
            WinogradF43.s_fewPatches512 = previous512;
            WinogradF43.s_forceFewPatches512ForTest = previousForce;
            WinogradF43.s_fewMaxPatches = previousMax;
        }
    }
#endif

    [SkippableFact]
    public void Engine_FewPatchLayer_TakesTheKernel_AndAgreesWithTheDirectRoute()
    {
        Skip.IfNot(WinogradRoutes, "Winograd routing needs AVX2 + FMA and AIDOTNET_CONV_WINOGRAD not set to 0.");
        Skip.IfNot(WinogradF43.s_fewMaxPatches >= 8, "AIDOTNET_WINOGRAD_FEW_MAX_PATCHES below 8.");
        // ResNet-18's last stage at batch 8: 512 -> 512 on 4x4 planes, one tile per image.
        const int n = 8, c = 512, k = 512, h = 4, w = 4;
        var rng = new Random(78);
        var x = new Tensor<float>(Fill(rng, n * c * h * w), new[] { n, c, h, w });
        var wt = new Tensor<float>(Fill(rng, k * c * 9), new[] { k, c, 3, 3 });
        var dy = new Tensor<float>(Fill(rng, n * k * h * w), new[] { n, k, h, w });
        var engine = new CpuEngine();
        int[] stride = { 1, 1 }, pad = { 1, 1 }, dil = { 1, 1 };
        float[] yWino = engine.Conv2D(x, wt, stride, pad, dil).ToArray();
        float[] dxWino = engine.Conv2DBackwardInput(dy, wt, x._shape, stride, pad, dil).ToArray();
        float[] yDirect, dxDirect;
        WinogradF43.s_disable = true;
        try
        {
            yDirect = engine.Conv2D(x, wt, stride, pad, dil).ToArray();
            dxDirect = engine.Conv2DBackwardInput(dy, wt, x._shape, stride, pad, dil).ToArray();
        }
        finally
        {
            WinogradF43.s_disable = false;
        }
        // 4608 products per output (~0.25 each).
        AssertClose(yDirect, yWino, "forward", 5e-3);
        AssertClose(dxDirect, dxWino, "input gradient", 5e-3);
        Assert.NotEqual(yDirect, yWino);
        Assert.NotEqual(dxDirect, dxWino);
    }

    private static void AssertClose(float[] expected, float[] actual, string what, double absolute = 1e-3)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            // 576 products per output: F(4x4)'s rounding is about 2e-6 of their summed magnitude (~0.25 each here).
            Assert.True(Math.Abs(expected[i] - actual[i]) <= absolute * (1 + Math.Abs(expected[i])),
                $"{what} [{i}]: direct {expected[i]}, Winograd {actual[i]}");
    }

    private static float[] Fill(Random rng, int length)
    {
        var a = new float[length];
        for (int i = 0; i < length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
#endif
