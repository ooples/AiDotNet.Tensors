// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// 3x3, stride-1, padding-1 convolution (NCHW, float) as Winograd F(4x4, 3x3) (Lavin and Gray, 2016): each 4x4 output
/// tile is A^T [ (G g G^T) . (B^T d B) ] A over its 6x6 input patch d, so the channel reduction becomes 36 independent
/// GEMMs over the tiles, M_xi [P x K] = V_xi [P x C] U_xi [C x K] (P = batch x tiles), with 36 multiplies per tile and
/// channel pair against the direct form's 144. The direct kernel already runs near the core's FMA peak (ResNet-18's
/// 128-channel 16x16 layer on an EPYC 7763: ours 3.58 ms, PyTorch's oneDNN 3.04 ms, both close to two cores' peak);
/// only fewer multiplies goes past that.
/// <para>
/// Layout: eight channels share a vector. The input transform transposes each block of eight input planes into a
/// zero-bordered [(4 tilesH + 2), (4 tilesW + 2), 8] tile (8x8 AVX transposes), and writes each patch's 36 transformed
/// vectors to V [36][P][C]; the weight transform writes U [36][C][K]; the GEMMs (single-threaded each, run in parallel
/// over the 36 points) write M [36][P][K]; the output transform reads eight output channels of a tile's 36 points, writes
/// a blocked [4 tilesH, 4 tilesW, 8] tile and transposes it into the eight output planes (adding to them when
/// accumulating). The input gradient of the same convolution is this forward of dY with the kernel turned 180 degrees and
/// its channel axes swapped, so it shares everything but the weight transform's indexing.
/// </para>
/// <para>
/// Every output element is the same sequence of operations whatever the thread count (the GEMMs are single-threaded per
/// point and each transform writes disjoint data), so the result is deterministic. Against the direct sum it differs by
/// rounding (the transforms scale by up to 8 and 1/24): about 1e-6 relative to the output's magnitude at these shapes.
/// </para>
/// </summary>
internal static unsafe partial class WinogradF43
{
    private const int Alpha = 6;              // patch side
    private const int Points = Alpha * Alpha; // 36 transformed points
    private const int Tile = 4;               // output tile side

    /// <summary>Test switch: false routes every convolution away from this kernel.</summary>
    internal static bool s_disable;

    // Operator opt-out (A/B switch).
    private static readonly bool s_env = Environment.GetEnvironmentVariable("AIDOTNET_CONV_WINOGRAD") != "0";

    /// <summary>Fewest patches (batch x tiles) the transform-domain GEMMs take: below it the weight transform (36 x C x K
    /// values) outweighs the multiplies saved, and the direct kernel stays.</summary>
    internal static readonly int s_minPatches =
        int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_WINOGRAD_MIN_PATCHES"), out int minPatches) && minPatches > 0 ? minPatches : 128;

    /// <summary>Fewest patches the input and weight gradients take. Lower than the forward's: the direct backward kernels
    /// cost more than the direct forward, so the transforms pay off sooner (ResNet-18 256 -> 256 on 8x8 planes at batch 8,
    /// 32 patches, ran the input gradient in 0.55-0.85x and the weight gradient in 0.5-0.7x the direct kernels' time on
    /// four EPYC hosts, where the forward ran up to 1.8x slower).</summary>
    internal static readonly int s_minBackwardPatches =
        int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_WINOGRAD_MIN_BACKWARD_PATCHES"), out int minBackward) && minBackward > 0 ? minBackward : 32;

    /// <summary>Fewest input and output channels of the batch-wide passes: with fewer the transformed operands (36 values
    /// per patch and channel) cost more memory traffic than the multiplies they save (a 32 -> 64 channel 14x14 layer at
    /// batch 64 ran 2.6x slower than the direct kernel). Narrower layers run blocked (WinogradF43.Blocked.cs).</summary>
    internal static readonly int s_minChannels = 64;

    /// <summary>Whether this kernel takes the convolution: 3x3, stride 1, padding 1, undilated, channel counts multiples
    /// of eight, planes at least 8x8, and enough patches (<see cref="s_minPatches"/>).</summary>
    internal static bool Applies(int batch, int inChannels, int outChannels, int height, int width, int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
        => Applies(batch, inChannels, outChannels, height, width, kernelH, kernelW, strideH, strideW, padH, padW,
            dilationH, dilationW, s_minPatches);

    /// <summary>Whether this kernel takes the input gradient (dY convolved with the turned kernel, dY's channels as the
    /// input's): <see cref="Applies"/> with <see cref="s_minBackwardPatches"/> as the patch floor.</summary>
    internal static bool AppliesToInputGradient(int batch, int inChannels, int outChannels, int height, int width,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
        => Applies(batch, inChannels, outChannels, height, width, kernelH, kernelW, strideH, strideW, padH, padW,
            dilationH, dilationW, s_minBackwardPatches);

    private static bool Applies(int batch, int inChannels, int outChannels, int height, int width, int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int minPatches)
    {
        if (s_disable || !s_env || DirectConvAvx2.SuppressedOnThisThread || !Avx2.IsSupported || !Fma.IsSupported) return false;
        if (kernelH != 3 || kernelW != 3 || strideH != 1 || strideW != 1 || padH != 1 || padW != 1
            || dilationH != 1 || dilationW != 1) return false;
        int minChannels = s_blockedNarrow || s_forceBlockedForTest ? MinBlockedChannels : s_minChannels;
        if (inChannels % 8 != 0 || outChannels % 8 != 0 || inChannels < minChannels || outChannels < minChannels
            || batch <= 0) return false;
        if (height < 8 || width < 8) return false;
        long patches = (long)batch * ((height + 3) / 4) * ((width + 3) / 4);
        if (patches < minPatches) return false;
        // V and M stay addressable with int offsets.
        return (long)Points * patches * Math.Max(inChannels, outChannels) <= int.MaxValue
            && (long)Points * inChannels * outChannels <= int.MaxValue;
    }

    /// <summary>
    /// Whether <see cref="BackwardKernel"/> takes the weight gradient: <see cref="Applies"/> with
    /// <see cref="s_minBackwardPatches"/> as the patch floor, and either the patch-blocked form
    /// (<see cref="UsesBlockedWeightGradient"/>) or channel counts with C K >= 64 (C + K). The batch-wide form streams V
    /// and D (36 values per patch and channel, C + K channels) through memory beside GEMMs that save work in proportion
    /// to C K: at 64 -> 64 channels on 32x32 planes, batch 8, it ran no faster than the direct kernel on Xeon 6973P and
    /// 8573C, and slower at batch 32 on EPYC 9V45; at 128 -> 128 on 16x16 it ran in 0.4-0.7x the direct kernel's time.
    /// </summary>
    /// <remarks>
    /// Also refuses a shape whose workspace passes <see cref="MaxWeightGradientWorkspaceFloats"/>: unlike the forward,
    /// which works tile by tile in a small scratch, <see cref="BackwardKernel"/> holds V, D and M for the whole batch at
    /// once, so its memory grows with batch x plane x channels. Past the cap the direct route takes it.
    /// </remarks>
    internal static bool AppliesToWeightGradient(int batch, int inChannels, int outChannels, int height, int width,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
        => Applies(batch, inChannels, outChannels, height, width, kernelH, kernelW, strideH, strideW, padH, padW,
               dilationH, dilationW, s_minBackwardPatches)
           && ((UsesBlockedWeightGradient(inChannels, outChannels)
                && (long)batch * ((height + Tile - 1) / Tile) * ((width + Tile - 1) / Tile) >= BlockedWeightGradientMinPatches)
               || ((long)inChannels * outChannels >= 64L * (inChannels + outChannels)
                   && WeightGradientWorkspaceFloats(batch, inChannels, outChannels, height, width) <= MaxWeightGradientWorkspaceFloats));

    /// <summary>Whether the weight gradient runs patch-blocked (<see cref="BackwardKernelBlocked"/>): layers the blocked
    /// forward takes with at least 32 channels each side. Its workspace is a fixed number of 36 x C x K partials, not
    /// the batch-wide V and D, so neither the C K rule nor the workspace cap applies. In-process A/B against the direct
    /// kernel (four threads, 64 -> 64, ms): batch 8 on 32x32 planes 1.57 vs 2.14 on Xeon 8573C, 1.67 vs 2.35 on 6975P,
    /// 1.62 vs 2.94 on EPYC 9V74, 1.25 vs 1.25 on 9V45; batch 8 on 56x56 3.9 vs 11.0 (8573C) and 2.65 vs 9.25 (9V45);
    /// batch 32 on 56x56 15.3 vs 142 (8573C).</summary>
    internal static bool UsesBlockedWeightGradient(int inChannels, int outChannels)
        => UseBlocked(inChannels, outChannels) && Math.Min(inChannels, outChannels) >= 32;

    /// <summary>Fewest patches the blocked weight gradient takes: at 128 (64 -> 64 on 16x16 planes, batch 8) each split
    /// holds one band and its partial costs as much as the band's multiplies; it ran 0.39 ms against the direct kernel's
    /// 0.32 on EPYC 9V45 and 0.55 against 0.52 on Xeon 8573C.</summary>
    internal const int BlockedWeightGradientMinPatches = 256;

    /// <summary>The most floats <see cref="BackwardKernel"/> may hold at once: 2^27, 512 MiB. A ResNet 64 -> 64 layer on
    /// 56x56 planes at batch 128 needs 116M and stays under it.</summary>
    internal const long MaxWeightGradientWorkspaceFloats = 1L << 27;

    /// <summary>The floats <see cref="BackwardKernel"/> rents together: V (36 per patch and input channel), D (36 per
    /// patch and output channel) and M (36 per channel pair).</summary>
    internal static long WeightGradientWorkspaceFloats(int batch, int inChannels, int outChannels, int height, int width)
    {
        long patches = (long)batch * ((height + Tile - 1) / Tile) * ((width + Tile - 1) / Tile);
        return Points * (patches * ((long)inChannels + outChannels) + (long)inChannels * outChannels);
    }

    /// <summary>y [N, K, H, W] (=, or += when accumulate) = the 3x3 padding-1 convolution of x [N, C, H, W] with w [K, C, 3, 3].</summary>
    internal static void Forward(float[] x, int xOff, float[] w, int wOff, float[] y, int yOff, bool accumulate,
        int batch, int inChannels, int height, int width, int outChannels)
        => Run(x, xOff, w, wOff, y, yOff, accumulate, batch, inChannels, height, width, outChannels, inputGradient: false);

    /// <summary>dx [N, C, H, W] (=, or += when accumulate) = the input gradient of that convolution from dy [N, K, H, W]
    /// and the forward weights w [K, C, 3, 3].</summary>
    internal static void BackwardInput(float[] dy, int dyOff, float[] w, int wOff, float[] dx, int dxOff, bool accumulate,
        int batch, int inChannels, int height, int width, int outChannels)
        => Run(dy, dyOff, w, wOff, dx, dxOff, accumulate, batch, outChannels, height, width, inChannels, inputGradient: true);

    /// <summary>
    /// dw [K, C, 3, 3] (=, or += when accumulate) = the weight gradient of that convolution from x [N, C, H, W] and
    /// dy [N, K, H, W]. The forward is linear in w through U = G w G^T, so its adjoint is dw = G^T [ sum over patches of
    /// (B^T d B) . (A dy_tile A^T) ] G: the input transform of the forward, the output gradient's 4x4 tiles taken to 6x6
    /// by A, 36 GEMMs over the patches (K^T x P times P x C), and G^T . G per channel pair.
    /// </summary>
    internal static void BackwardKernel(float[] x, int xOff, float[] dy, int dyOff, float[] dw, int dwOff, bool accumulate,
        int batch, int inChannels, int height, int width, int outChannels)
    {
        if (UsesBlockedWeightGradient(inChannels, outChannels))
        {
            BackwardKernelBlocked(x, xOff, dy, dyOff, dw, dwOff, accumulate, batch, inChannels, height, width, outChannels);
            return;
        }
        int tilesH = (height + Tile - 1) / Tile, tilesW = (width + Tile - 1) / Tile;
        int patches = batch * tilesH * tilesW;
        float[] v = ArrayPool<float>.Shared.Rent(Points * patches * inChannels);
        float[] d = ArrayPool<float>.Shared.Rent(Points * patches * outChannels);
        float[] m = ArrayPool<float>.Shared.Rent(Points * inChannels * outChannels);
        try
        {
            TransformInput(x, xOff, v, batch, inChannels, height, width, tilesH, tilesW);
            TransformOutputGradient(dy, dyOff, d, batch, outChannels, height, width, tilesH, tilesW);
            MultiplyOverPatches(d, v, m, patches, inChannels, outChannels);
            TransformWeightGradient(m, 1, 0, dw, dwOff, accumulate, inChannels, outChannels);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(v);
            ArrayPool<float>.Shared.Return(d);
            ArrayPool<float>.Shared.Return(m);
        }
    }

    // D [36][P][cOut] = A t A^T per patch and output channel, t the 4x4 output-gradient tile (zero past the plane);
    // one job per (image, block of eight output channels).
    private static void TransformOutputGradient(float[] dy, int dyOff, float[] d, int batch, int cOut, int height, int width,
        int tilesH, int tilesW)
    {
        int blocks = cOut / 8, tiles = tilesH * tilesW, patches = batch * tiles;
        int outW = tilesW * Tile, outH = tilesH * Tile;
        int tileFloats = outH * outW * 8;
        int hw = height * width;
        CpuParallelSettings.ParallelForOrSerial(0, batch * blocks, (long)batch * cOut * hw * 12, [MethodImpl(MethodImplOptions.AggressiveOptimization)] (job) =>
        {
            int n = job / blocks, kb = job - n * blocks;
            float[] scratch = RentScratch(tileFloats);
            try
            {
                fixed (float* gp = dy)
                fixed (float* dp = d)
                fixed (float* tp = scratch)
                {
                    new Span<float>(tp, tileFloats).Clear();
                    float* planes = gp + dyOff + ((long)n * cOut + kb * 8) * hw;
                    for (int h = 0; h < height; h++)
                    {
                        float* src = planes + h * width;
                        float* dst = tp + h * outW * 8;
                        int q = 0;
                        for (; q + 8 <= width; q += 8) Transpose8x8(src + q, hw, dst + q * 8, 8);
                        for (; q < width; q++)
                            for (int l = 0; l < 8; l++) dst[q * 8 + l] = src[(long)l * hw + q];
                    }
                    var g = stackalloc Vector256<float>[16];
                    var t = stackalloc Vector256<float>[24];
                    var r = stackalloc Vector256<float>[36];
                    long pointStride = (long)patches * cOut;
                    for (int th = 0; th < tilesH; th++)
                        for (int tw = 0; tw < tilesW; tw++)
                        {
                            LoadTile(tp + (th * Tile * outW + tw * Tile) * 8, outW * 8, 4, 4, g);
                            // Columns: t[., b] = A g[., b] (6 x 4); rows: r[i, .] = t[i, .] A^T (6 x 6).
                            for (int b = 0; b < 4; b++) GradientTransform(g, b, 4, t, b, 4);
                            for (int i = 0; i < 6; i++) GradientTransform(t, i * 4, 1, r, i * 6, 1);
                            int p = n * tiles + th * tilesW + tw;
                            StorePoints(r, dp + (long)p * cOut + kb * 8, pointStride);
                        }
                }
            }
            finally
            {
                ReturnScratch(scratch, tileFloats);
            }
        }, deterministicSafe: true);
    }

    // A (6 x 4) applied to the four vectors src[s + r * ss], written as six to dst[o + r * os]: A is the transpose of
    // the output transform's A^T.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GradientTransform(Vector256<float>* src, int s, int ss, Vector256<float>* dst, int o, int os)
    {
        var r0 = src[s]; var r1 = src[s + ss]; var r2 = src[s + 2 * ss]; var r3 = src[s + 3 * ss];
        var e = Avx.Add(r0, r2); var od = Avx.Add(r1, r3);
        var e4 = Fma.MultiplyAdd(r2, Vector256.Create(4f), r0);
        var o8 = Fma.MultiplyAdd(r3, Vector256.Create(4f), r1);   // r1 + 4 r3; doubled below
        dst[o] = r0;
        dst[o + os] = Avx.Add(e, od);
        dst[o + 2 * os] = Avx.Subtract(e, od);
        dst[o + 3 * os] = Fma.MultiplyAdd(o8, Vector256.Create(2f), e4);
        dst[o + 4 * os] = Fma.MultiplyAddNegated(o8, Vector256.Create(2f), e4);
        dst[o + 5 * os] = r3;
    }

    // M_xi [cOut x cIn] = D_xi^T [cOut x P] V_xi [P x cIn]: 36 single-threaded GEMMs in parallel, each summing every
    // patch in one order (deterministic whatever the thread count).
    private static void MultiplyOverPatches(float[] d, float[] v, float[] m, int patches, int cIn, int cOut)
    {
        CpuParallelSettings.ParallelForOrSerial(0, Points, (long)Points * patches * cIn * cOut, (xi) =>
        {
            var a = new ReadOnlySpan<float>(d, (int)((long)xi * patches * cOut), patches * cOut);
            var b = new ReadOnlySpan<float>(v, (int)((long)xi * patches * cIn), patches * cIn);
            var c = new Span<float>(m, xi * cOut * cIn, cOut * cIn);
            var options = new BlasManaged.BlasOptions<float> { NumThreads = -1, BetaZero = true };
            BlasManaged.BlasManaged.Gemm(a, cOut, true, b, cIn, false, c, cIn, cOut, cIn, patches, in options);
        }, deterministicSafe: true);
    }

    // dw[o][i] (3 x 3) = G^T M[., o, i] G, eight input channels per vector; one job per output channel. M is the sum of
    // `partials` blocks partialStride floats apart, added in order as they load.
    private static void TransformWeightGradient(float[] m, int partials, long partialStride, float[] dw, int dwOff,
        bool accumulate, int cIn, int cOut)
    {
        CpuParallelSettings.ParallelForOrSerial(0, cOut, (long)cOut * cIn * 60, [MethodImpl(MethodImplOptions.AggressiveOptimization)] (o) =>
        {
            var mm = stackalloc Vector256<float>[36];
            var t = stackalloc Vector256<float>[18];
            var r = stackalloc Vector256<float>[9];
            float* lanes = stackalloc float[72];
            long pointStride = (long)cOut * cIn;
            fixed (float* mp = m)
            fixed (float* wp = dw)
            {
                for (int i0 = 0; i0 < cIn; i0 += 8)
                {
                    float* src = mp + (long)o * cIn + i0;
                    for (int xi = 0; xi < Points; xi++) mm[xi] = Avx.LoadVector256(src + xi * pointStride);
                    for (int s = 1; s < partials; s++)
                    {
                        float* part = src + s * partialStride;
                        for (int xi = 0; xi < Points; xi++) mm[xi] = Avx.Add(mm[xi], Avx.LoadVector256(part + xi * pointStride));
                    }
                    // Columns: t[., j] = G^T mm[., j] (3 x 6); rows: r[a, .] = t[a, .] G (3 x 3).
                    for (int j = 0; j < 6; j++) KernelGradientTransform(mm, j, 6, t, j, 6);
                    for (int a = 0; a < 3; a++) KernelGradientTransform(t, a * 6, 1, r, a * 3, 1);
                    for (int e = 0; e < 9; e++) Avx.Store(lanes + e * 8, r[e]);
                    float* dst = wp + dwOff + ((long)o * cIn + i0) * 9;
                    for (int l = 0; l < 8; l++)
                        for (int e = 0; e < 9; e++)
                        {
                            float value = lanes[e * 8 + l];
                            if (accumulate) dst[l * 9 + e] += value;
                            else dst[l * 9 + e] = value;
                        }
                }
            }
        }, deterministicSafe: true);
    }

    // G^T (3 x 6) applied to the six vectors src[s + r * ss], written as three to dst[o + r * os].
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void KernelGradientTransform(Vector256<float>* src, int s, int ss, Vector256<float>* dst, int o, int os)
    {
        var m0 = src[s]; var m1 = src[s + ss]; var m2 = src[s + 2 * ss];
        var m3 = src[s + 3 * ss]; var m4 = src[s + 4 * ss]; var m5 = src[s + 5 * ss];
        var p12 = Avx.Add(m1, m2); var n12 = Avx.Subtract(m2, m1);
        var p34 = Avx.Add(m3, m4); var n34 = Avx.Subtract(m3, m4);
        var sixth = Vector256.Create(1f / 6f);
        // m0/4 - (m1 + m2)/6 + (m3 + m4)/24
        dst[o] = Fma.MultiplyAdd(m0, Vector256.Create(0.25f),
            Fma.MultiplyAdd(p34, Vector256.Create(1f / 24f), Avx.Multiply(p12, Vector256.Create(-1f / 6f))));
        // (m2 - m1)/6 + (m3 - m4)/12
        dst[o + os] = Fma.MultiplyAdd(n12, sixth, Avx.Multiply(n34, Vector256.Create(1f / 12f)));
        // (m3 + m4 - m1 - m2)/6 + m5
        dst[o + 2 * os] = Fma.MultiplyAdd(Avx.Subtract(p34, p12), sixth, m5);
    }

    // A convolution with cIn input and cOut output channels. inputGradient: w is [cIn][cOut][3][3] (the forward weights,
    // whose output channels are this pass's inputs) and is read turned 180 degrees; otherwise [cOut][cIn][3][3].
    private static void Run(float[] x, int xOff, float[] w, int wOff, float[] y, int yOff, bool accumulate,
        int batch, int cIn, int height, int width, int cOut, bool inputGradient)
    {
        int tilesH = (height + Tile - 1) / Tile, tilesW = (width + Tile - 1) / Tile;
        int tiles = tilesH * tilesW, patches = batch * tiles;
        int vLength = Points * patches * cIn, mLength = Points * patches * cOut, uLength = Points * cIn * cOut;
        float[] u = ArrayPool<float>.Shared.Rent(uLength);
        if (UseBlocked(cIn, cOut))
        {
            try
            {
                TransformWeights(w, wOff, u, cIn, cOut, inputGradient);
                RunBlocked(x, xOff, u, y, yOff, accumulate, batch, cIn, height, width, cOut);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(u);
            }
            return;
        }
        float[] v = ArrayPool<float>.Shared.Rent(vLength);
        float[] m = ArrayPool<float>.Shared.Rent(mLength);
        try
        {
            TransformWeights(w, wOff, u, cIn, cOut, inputGradient);
            TransformInput(x, xOff, v, batch, cIn, height, width, tilesH, tilesW);
            Multiply(v, u, m, patches, cIn, cOut);
            TransformOutput(m, y, yOff, accumulate, batch, cOut, height, width, tilesH, tilesW);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(v);
            ArrayPool<float>.Shared.Return(m);
            ArrayPool<float>.Shared.Return(u);
        }
    }

    [ThreadStatic] private static float[]? t_scratch;

    // Largest tile scratch a thread keeps between jobs (1 MB): a 112x112 plane's eight channels (~400 KB) stay reused,
    // while a larger plane's buffer goes back to the pool after its job instead of living on every persistent worker.
    private const int MaxRetainedScratchFloats = 1 << 18;

    // A job's tile scratch (the input and output transforms never overlap on a thread). Pair with ReturnScratch.
    private static float[] RentScratch(int length)
    {
        if (length > MaxRetainedScratchFloats) return ArrayPool<float>.Shared.Rent(length);
        var s = t_scratch;
        if (s is null || s.Length < length) t_scratch = s = new float[length];
        return s;
    }

    private static void ReturnScratch(float[] scratch, int length)
    {
        if (length > MaxRetainedScratchFloats) ArrayPool<float>.Shared.Return(scratch);
    }

    // The 36 transform-domain points of one patch, stride floats apart, to and from 36 vectors.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StorePoints(Vector256<float>* src, float* dst, long stride)
    {
        for (int xi = 0; xi < Points; xi++) Avx.Store(dst + xi * stride, src[xi]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LoadPoints(float* src, long stride, Vector256<float>* dst)
    {
        for (int xi = 0; xi < Points; xi++) dst[xi] = Avx.LoadVector256(src + xi * stride);
    }

    // A rows x cols tile of eight-channel vectors, rowFloats floats between rows, to and from dst[r * cols + c].
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LoadTile(float* src, int rowFloats, int rows, int cols, Vector256<float>* dst)
    {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                dst[r * cols + c] = Avx.LoadVector256(src + r * rowFloats + c * 8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreTile(Vector256<float>* src, int rows, int cols, float* dst, int rowFloats)
    {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Avx.Store(dst + r * rowFloats + c * 8, src[r * cols + c]);
    }

    // U [36][cIn][cOut] = G g G^T per (input channel, output channel), eight output channels per vector.
    private static void TransformWeights(float[] w, int wOff, float[] u, int cIn, int cOut, bool inputGradient)
    {
        int outBlocks = cOut / 8;
        // Lane l of block ob reads output channel 8 ob + l: forward w[o][i] at o * cIn * 9 + i * 9; input gradient
        // w[i][o] (the forward's [cIn = its outputs][cOut = its inputs]) at i * cOut * 9 + o * 9, taps reversed.
        int laneStride = inputGradient ? 9 : cIn * 9;
        var lanes = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7) * laneStride;
        CpuParallelSettings.ParallelForOrSerial(0, outBlocks * cIn, (long)cIn * cOut * 162, [MethodImpl(MethodImplOptions.AggressiveOptimization)] (job) =>
        {
            int ob = job / cIn, ic = job - ob * cIn;
            int o0 = ob * 8;
            fixed (float* wp = w)
            fixed (float* up = u)
            {
                float* basePtr = wp + wOff + (inputGradient ? (long)ic * cOut * 9 + (long)o0 * 9 : (long)o0 * cIn * 9 + (long)ic * 9);
                var g = stackalloc Vector256<float>[9];
                for (int e = 0; e < 9; e++)
                    g[e] = Avx2.GatherVector256(basePtr + (inputGradient ? 8 - e : e), lanes, 4);
                // t = G g (6 x 3), column by column.
                var t = stackalloc Vector256<float>[18];
                for (int j = 0; j < 3; j++) TransformKernelColumn(g[j], g[3 + j], g[6 + j], t, j, 3);
                // U = t G^T (6 x 6), row by row.
                var uu = stackalloc Vector256<float>[36];
                for (int i = 0; i < 6; i++) TransformKernelColumn(t[i * 3], t[i * 3 + 1], t[i * 3 + 2], uu, i * 6, 1);
                StorePoints(uu, up + (long)ic * cOut + o0, (long)cIn * cOut);
            }
        }, deterministicSafe: true);
    }

    // The six values of G [a b c]^T, written at dst[offset + r * step].
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void TransformKernelColumn(Vector256<float> a, Vector256<float> b, Vector256<float> c,
        Vector256<float>* dst, int offset, int step)
    {
        var quarter = Vector256.Create(0.25f);
        var sixth = Vector256.Create(1f / 6f);
        var twelfth = Vector256.Create(1f / 12f);
        var twentyFourth = Vector256.Create(1f / 24f);
        var ac = Avx.Add(a, c);
        dst[offset] = Avx.Multiply(a, quarter);
        dst[offset + step] = Avx.Multiply(Avx.Add(ac, b), Vector256.Create(-1f / 6f));
        dst[offset + 2 * step] = Avx.Multiply(Avx.Subtract(ac, b), Vector256.Create(-1f / 6f));
        var even = Fma.MultiplyAdd(a, twentyFourth, Avx.Multiply(c, sixth));
        var odd = Avx.Multiply(b, twelfth);
        dst[offset + 3 * step] = Avx.Add(even, odd);
        dst[offset + 4 * step] = Avx.Subtract(even, odd);
        dst[offset + 5 * step] = c;
    }

    // V [36][P][cIn] = B^T d B per patch and input channel; one job per (image, block of eight input channels).
    private static void TransformInput(float[] x, int xOff, float[] v, int batch, int cIn, int height, int width,
        int tilesH, int tilesW)
    {
        int blocks = cIn / 8, tiles = tilesH * tilesW, patches = batch * tiles;
        int padW = tilesW * Tile + 2, padH = tilesH * Tile + 2;
        int tileFloats = padH * padW * 8;
        int hw = height * width;
        CpuParallelSettings.ParallelForOrSerial(0, batch * blocks, (long)batch * cIn * hw * 12, [MethodImpl(MethodImplOptions.AggressiveOptimization)] (job) =>
        {
            int n = job / blocks, cb = job - n * blocks;
            float[] scratch = RentScratch(tileFloats);
            try
            {
                fixed (float* xp = x)
                fixed (float* vp = v)
                fixed (float* tp = scratch)
                {
                    new Span<float>(tp, tileFloats).Clear();
                    float* planes = xp + xOff + ((long)n * cIn + cb * 8) * hw;
                    for (int h = 0; h < height; h++)
                    {
                        float* src = planes + h * width;
                        float* dst = tp + ((h + 1) * padW + 1) * 8;
                        int q = 0;
                        for (; q + 8 <= width; q += 8) Transpose8x8(src + q, hw, dst + q * 8, 8);
                        for (; q < width; q++)
                            for (int l = 0; l < 8; l++) dst[q * 8 + l] = src[(long)l * hw + q];
                    }
                    var d = stackalloc Vector256<float>[36];
                    var t = stackalloc Vector256<float>[36];
                    long pointStride = (long)patches * cIn;
                    for (int th = 0; th < tilesH; th++)
                        for (int tw = 0; tw < tilesW; tw++)
                        {
                            LoadTile(tp + (th * Tile * padW + tw * Tile) * 8, padW * 8, 6, 6, d);
                            // Rows: t[., j] = B^T d[., j]; then columns: V[i, .] = t[i, .] B.
                            for (int j = 0; j < 6; j++) InputTransform(d, j, 6, t, j, 6);
                            for (int i = 0; i < 6; i++) InputTransform(t, i * 6, 1, d, i * 6, 1);
                            int p = n * tiles + th * tilesW + tw;
                            StorePoints(d, vp + (long)p * cIn + cb * 8, pointStride);
                        }
                }
            }
            finally
            {
                ReturnScratch(scratch, tileFloats);
            }
        }, deterministicSafe: true);
    }

    // B^T applied to the six vectors src[s + r * ss], written to dst[o + r * os].
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void InputTransform(Vector256<float>* src, int s, int ss, Vector256<float>* dst, int o, int os)
    {
        var d0 = src[s]; var d1 = src[s + ss]; var d2 = src[s + 2 * ss];
        var d3 = src[s + 3 * ss]; var d4 = src[s + 4 * ss]; var d5 = src[s + 5 * ss];
        var four = Vector256.Create(4f); var five = Vector256.Create(5f); var two = Vector256.Create(2f);
        // 4 d0 - 5 d2 + d4
        dst[o] = Fma.MultiplyAdd(d0, four, Fma.MultiplyAddNegated(d2, five, d4));
        // -4 (d1 + d2) + (d3 + d4); 4 (d1 - d2) + (d4 - d3)
        var s12 = Avx.Add(d1, d2); var s34 = Avx.Add(d3, d4);
        dst[o + os] = Fma.MultiplyAddNegated(s12, four, s34);
        dst[o + 2 * os] = Fma.MultiplyAdd(Avx.Subtract(d1, d2), four, Avx.Subtract(d4, d3));
        // 2 (d3 - d1) + (d4 - d2); 2 (d1 - d3) + (d4 - d2)
        var d31 = Avx.Subtract(d3, d1); var d42 = Avx.Subtract(d4, d2);
        dst[o + 3 * os] = Fma.MultiplyAdd(d31, two, d42);
        dst[o + 4 * os] = Fma.MultiplyAddNegated(d31, two, d42);
        // 4 d1 - 5 d3 + d5
        dst[o + 5 * os] = Fma.MultiplyAdd(d1, four, Fma.MultiplyAddNegated(d3, five, d5));
    }

    // M_xi [P x cOut] = V_xi [P x cIn] U_xi [cIn x cOut]: 36 single-threaded GEMMs in parallel (each split along P when
    // the points alone would leave threads idle).
    private static void Multiply(float[] v, float[] u, float[] m, int patches, int cIn, int cOut)
    {
        int procs = Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism);
        int splits = 1;
        while (Points * splits < 2 * procs && patches / (splits * 2) >= 64) splits *= 2;
        int rowsPerSplit = (patches + splits - 1) / splits;
        CpuParallelSettings.ParallelForOrSerial(0, Points * splits, (long)Points * patches * cIn * cOut, (job) =>
        {
            int xi = job / splits, part = job - xi * splits;
            int row0 = part * rowsPerSplit, rows = Math.Min(rowsPerSplit, patches - row0);
            if (rows <= 0) return;
            var a = new ReadOnlySpan<float>(v, (int)(((long)xi * patches + row0) * cIn), rows * cIn);
            var b = new ReadOnlySpan<float>(u, xi * cIn * cOut, cIn * cOut);
            var c = new Span<float>(m, (int)(((long)xi * patches + row0) * cOut), rows * cOut);
            var options = new BlasManaged.BlasOptions<float> { NumThreads = -1, BetaZero = true };
            BlasManaged.BlasManaged.Gemm(a, cIn, false, b, cOut, false, c, cOut, rows, cOut, cIn, in options);
        }, deterministicSafe: true);
    }

    // y = A^T m A per patch and output channel; one job per (image, block of eight output channels).
    private static void TransformOutput(float[] m, float[] y, int yOff, bool accumulate, int batch, int cOut,
        int height, int width, int tilesH, int tilesW)
    {
        int blocks = cOut / 8, tiles = tilesH * tilesW, patches = batch * tiles;
        int outW = tilesW * Tile, outH = tilesH * Tile;
        int tileFloats = outH * outW * 8;
        int hw = height * width;
        CpuParallelSettings.ParallelForOrSerial(0, batch * blocks, (long)batch * cOut * hw * 12, [MethodImpl(MethodImplOptions.AggressiveOptimization)] (job) =>
        {
            int n = job / blocks, kb = job - n * blocks;
            float[] scratch = RentScratch(tileFloats + 64);
            try
            {
                fixed (float* mp = m)
                fixed (float* yp = y)
                fixed (float* op = scratch)
                {
                    var mm = stackalloc Vector256<float>[36];
                    var t = stackalloc Vector256<float>[24];
                    var r = stackalloc Vector256<float>[16];
                    long pointStride = (long)patches * cOut;
                    for (int th = 0; th < tilesH; th++)
                        for (int tw = 0; tw < tilesW; tw++)
                        {
                            int p = n * tiles + th * tilesW + tw;
                            LoadPoints(mp + (long)p * cOut + kb * 8, pointStride, mm);
                            // Rows: t[a, j] = A^T m[., j] (4 x 6); columns: r[a, b] = t[a, .] A (4 x 4).
                            for (int j = 0; j < 6; j++) OutputTransform(mm, j, 6, t, j, 6);
                            for (int a = 0; a < 4; a++) OutputTransform(t, a * 6, 1, r, a * 4, 1);
                            StoreTile(r, 4, 4, op + (th * Tile * outW + tw * Tile) * 8, outW * 8);
                        }
                    float* planes = yp + yOff + ((long)n * cOut + kb * 8) * hw;
                    float* tmp = op + tileFloats;   // 64 floats
                    for (int h = 0; h < height; h++)
                    {
                        float* src = op + h * outW * 8;
                        float* dstRow = planes + h * width;
                        int q = 0;
                        if (accumulate)
                        {
                            for (; q + 8 <= width; q += 8)
                            {
                                Transpose8x8(src + q * 8, 8, tmp, 8);
                                for (int l = 0; l < 8; l++)
                                {
                                    float* row = dstRow + (long)l * hw + q;
                                    Avx.Store(row, Avx.Add(Avx.LoadVector256(row), Avx.LoadVector256(tmp + l * 8)));
                                }
                            }
                            for (; q < width; q++)
                                for (int l = 0; l < 8; l++) dstRow[(long)l * hw + q] += src[q * 8 + l];
                        }
                        else
                        {
                            for (; q + 8 <= width; q += 8) Transpose8x8(src + q * 8, 8, dstRow + q, hw);
                            for (; q < width; q++)
                                for (int l = 0; l < 8; l++) dstRow[(long)l * hw + q] = src[q * 8 + l];
                        }
                    }
                }
            }
            finally
            {
                ReturnScratch(scratch, tileFloats + 64);
            }
        }, deterministicSafe: true);
    }

    // A^T applied to the six vectors src[s + r * ss], written as four to dst[o + r * os].
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void OutputTransform(Vector256<float>* src, int s, int ss, Vector256<float>* dst, int o, int os)
    {
        var m0 = src[s]; var m1 = src[s + ss]; var m2 = src[s + 2 * ss];
        var m3 = src[s + 3 * ss]; var m4 = src[s + 4 * ss]; var m5 = src[s + 5 * ss];
        var p12 = Avx.Add(m1, m2); var n12 = Avx.Subtract(m1, m2);
        var p34 = Avx.Add(m3, m4); var n34 = Avx.Subtract(m3, m4);
        dst[o] = Avx.Add(Avx.Add(m0, p12), p34);
        dst[o + os] = Fma.MultiplyAdd(n34, Vector256.Create(2f), n12);
        dst[o + 2 * os] = Fma.MultiplyAdd(p34, Vector256.Create(4f), p12);
        dst[o + 3 * os] = Avx.Add(Fma.MultiplyAdd(n34, Vector256.Create(8f), n12), m5);
    }

    // d[j * dStride + c] = s[c * sStride + j] for an 8x8 block.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose8x8(float* s, long sStride, float* d, long dStride)
    {
        var r0 = Avx.LoadVector256(s); var r1 = Avx.LoadVector256(s + sStride);
        var r2 = Avx.LoadVector256(s + 2 * sStride); var r3 = Avx.LoadVector256(s + 3 * sStride);
        var r4 = Avx.LoadVector256(s + 4 * sStride); var r5 = Avx.LoadVector256(s + 5 * sStride);
        var r6 = Avx.LoadVector256(s + 6 * sStride); var r7 = Avx.LoadVector256(s + 7 * sStride);
        var t0 = Avx.UnpackLow(r0, r1); var t1 = Avx.UnpackHigh(r0, r1);
        var t2 = Avx.UnpackLow(r2, r3); var t3 = Avx.UnpackHigh(r2, r3);
        var t4 = Avx.UnpackLow(r4, r5); var t5 = Avx.UnpackHigh(r4, r5);
        var t6 = Avx.UnpackLow(r6, r7); var t7 = Avx.UnpackHigh(r6, r7);
        var u0 = Avx.Shuffle(t0, t2, 0x44); var u1 = Avx.Shuffle(t0, t2, 0xEE);
        var u2 = Avx.Shuffle(t1, t3, 0x44); var u3 = Avx.Shuffle(t1, t3, 0xEE);
        var u4 = Avx.Shuffle(t4, t6, 0x44); var u5 = Avx.Shuffle(t4, t6, 0xEE);
        var u6 = Avx.Shuffle(t5, t7, 0x44); var u7 = Avx.Shuffle(t5, t7, 0xEE);
        Avx.Store(d, Avx.Permute2x128(u0, u4, 0x20));
        Avx.Store(d + dStride, Avx.Permute2x128(u1, u5, 0x20));
        Avx.Store(d + 2 * dStride, Avx.Permute2x128(u2, u6, 0x20));
        Avx.Store(d + 3 * dStride, Avx.Permute2x128(u3, u7, 0x20));
        Avx.Store(d + 4 * dStride, Avx.Permute2x128(u0, u4, 0x31));
        Avx.Store(d + 5 * dStride, Avx.Permute2x128(u1, u5, 0x31));
        Avx.Store(d + 6 * dStride, Avx.Permute2x128(u2, u6, 0x31));
        Avx.Store(d + 7 * dStride, Avx.Permute2x128(u3, u7, 0x31));
    }
}
#endif
