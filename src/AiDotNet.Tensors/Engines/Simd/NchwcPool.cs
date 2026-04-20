using System;
using System.Threading.Tasks;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Max / Average / Global pooling kernels for NCHWc8 float tensors.
///
/// <para>Each output spatial cell reduces a <c>kH × kW</c> window across
/// exactly 8 channel-lanes that sit contiguous in memory — a single
/// <c>Vector256&lt;float&gt;</c> load/store per source cell, and one SIMD
/// Max/Add per window step. No broadcast, no lane-crossing.</para>
///
/// <para>The 8 lanes correspond to 8 different global channels, so the
/// pool reduction is trivially per-channel. No shuffles needed.</para>
/// </summary>
internal static class NchwcPool
{
    public const int CBlock = 8;

    /// <summary>
    /// MaxPool on NCHWc8 <c>[N, cg, H, W, 8]</c> → <c>[N, cg, oH, oW, 8]</c>.
    /// </summary>
    public static void MaxPoolNchwc8(
        float[] input, float[] output,
        int N, int C, int H, int W, int oH, int oW,
        int kH, int kW, int sH, int sW, int padH, int padW)
    {
        int cg = C / CBlock;
        int inStrideN = cg * H * W * CBlock;
        int inStrideCg = H * W * CBlock;
        int inStrideH = W * CBlock;
        int outStrideN = cg * oH * oW * CBlock;
        int outStrideCg = oH * oW * CBlock;
        int outStrideH = oW * CBlock;

        // Closure captures the caller's arrays directly — no ToArray()
        // copies, no allocation of a temporary output buffer.
        var inArr = input;
        var outArr = output;
#if NET5_0_OR_GREATER
        bool useSimd = Avx.IsSupported;
#endif

        Parallel.For(0, N * cg, task =>
        {
            int n = task / cg;
            int ocg = task % cg;
            int inBase = n * inStrideN + ocg * inStrideCg;
            int outBase = n * outStrideN + ocg * outStrideCg;
            var acc = new float[CBlock];

            for (int oh = 0; oh < oH; oh++)
            {
                for (int ow = 0; ow < oW; ow++)
                {
                    // Initialize reduction with -Inf. SIMD path uses vector, scalar array.
#if NET5_0_OR_GREATER
                    var vMax = Vector256.Create(float.NegativeInfinity);
#endif
                    for (int i = 0; i < CBlock; i++) acc[i] = float.NegativeInfinity;

                    for (int kh = 0; kh < kH; kh++)
                    {
                        int ih = oh * sH + kh - padH;
                        if ((uint)ih >= (uint)H) continue;
                        int inHBase = inBase + ih * inStrideH;
                        for (int kw = 0; kw < kW; kw++)
                        {
                            int iw = ow * sW + kw - padW;
                            if ((uint)iw >= (uint)W) continue;
                            int idx = inHBase + iw * CBlock;
#if NET5_0_OR_GREATER
                            if (useSimd)
                            {
                                var vIn = Vector256.Create(
                                    inArr[idx + 0], inArr[idx + 1], inArr[idx + 2], inArr[idx + 3],
                                    inArr[idx + 4], inArr[idx + 5], inArr[idx + 6], inArr[idx + 7]);
                                vMax = Avx.Max(vMax, vIn);
                                continue;
                            }
#endif
                            for (int cb = 0; cb < CBlock; cb++)
                                if (inArr[idx + cb] > acc[cb]) acc[cb] = inArr[idx + cb];
                        }
                    }

                    int outIdx = outBase + oh * outStrideH + ow * CBlock;
#if NET5_0_OR_GREATER
                    if (useSimd)
                    {
                        vMax.CopyTo(acc);
                    }
#endif
                    for (int cb = 0; cb < CBlock; cb++) outArr[outIdx + cb] = acc[cb];
                }
            }
        });
        // `outArr` IS the caller's `output` — writes land directly.
    }

    /// <summary>
    /// AvgPool on NCHWc8 <c>[N, cg, H, W, 8]</c> → <c>[N, cg, oH, oW, 8]</c>.
    /// Divisor is the full kernel area (matches ONNX default
    /// <c>count_include_pad=0</c> only for interior cells; a proper
    /// count-excluding-pad mode would require per-output area tracking).
    /// </summary>
    public static void AvgPoolNchwc8(
        float[] input, float[] output,
        int N, int C, int H, int W, int oH, int oW,
        int kH, int kW, int sH, int sW, int padH, int padW,
        bool countIncludePad)
    {
        int cg = C / CBlock;
        int inStrideN = cg * H * W * CBlock;
        int inStrideCg = H * W * CBlock;
        int inStrideH = W * CBlock;
        int outStrideN = cg * oH * oW * CBlock;
        int outStrideCg = oH * oW * CBlock;
        int outStrideH = oW * CBlock;
        int kernelArea = kH * kW;

        var inArr = input;
        var outArr = output;
#if NET5_0_OR_GREATER
        bool useSimd = Avx.IsSupported;
#endif

        Parallel.For(0, N * cg, task =>
        {
            int n = task / cg;
            int ocg = task % cg;
            int inBase = n * inStrideN + ocg * inStrideCg;
            int outBase = n * outStrideN + ocg * outStrideCg;
            var acc = new float[CBlock];

            for (int oh = 0; oh < oH; oh++)
            {
                for (int ow = 0; ow < oW; ow++)
                {
#if NET5_0_OR_GREATER
                    var vSum = Vector256<float>.Zero;
#endif
                    for (int i = 0; i < CBlock; i++) acc[i] = 0f;
                    int count = 0;

                    for (int kh = 0; kh < kH; kh++)
                    {
                        int ih = oh * sH + kh - padH;
                        if ((uint)ih >= (uint)H) continue;
                        int inHBase = inBase + ih * inStrideH;
                        for (int kw = 0; kw < kW; kw++)
                        {
                            int iw = ow * sW + kw - padW;
                            if ((uint)iw >= (uint)W) continue;
                            int idx = inHBase + iw * CBlock;
                            count++;
#if NET5_0_OR_GREATER
                            if (useSimd)
                            {
                                var vIn = Vector256.Create(
                                    inArr[idx + 0], inArr[idx + 1], inArr[idx + 2], inArr[idx + 3],
                                    inArr[idx + 4], inArr[idx + 5], inArr[idx + 6], inArr[idx + 7]);
                                vSum = Avx.Add(vSum, vIn);
                                continue;
                            }
#endif
                            for (int cb = 0; cb < CBlock; cb++) acc[cb] += inArr[idx + cb];
                        }
                    }

                    float divisor = countIncludePad ? kernelArea : Math.Max(count, 1);
                    int outIdx = outBase + oh * outStrideH + ow * CBlock;
#if NET5_0_OR_GREATER
                    if (useSimd)
                    {
                        vSum.CopyTo(acc);
                    }
#endif
                    float inv = 1f / divisor;
                    for (int cb = 0; cb < CBlock; cb++) outArr[outIdx + cb] = acc[cb] * inv;
                }
            }
        });
        // `outArr` IS the caller's `output` — writes land directly.
    }

    /// <summary>
    /// GlobalAvgPool on NCHWc8: <c>[N, cg, H, W, 8]</c> → <c>[N, C]</c> flat,
    /// then reshaped by caller to <c>[N, C, 1, 1]</c>. Divisor is <c>H*W</c>.
    /// </summary>
    public static void GlobalAvgPoolNchwc8(
        float[] input, float[] output,
        int N, int C, int H, int W)
    {
        int cg = C / CBlock;
        int spatial = H * W;
        int inStrideN = cg * spatial * CBlock;
        int inStrideCg = spatial * CBlock;
        float inv = 1f / spatial;

        var inArr = input;
        var outArr = output;
#if NET5_0_OR_GREATER
        bool useSimd = Avx.IsSupported;
#endif

        Parallel.For(0, N * cg, task =>
        {
            int n = task / cg;
            int ocg = task % cg;
            int inBase = n * inStrideN + ocg * inStrideCg;
            var acc = new float[CBlock];
#if NET5_0_OR_GREATER
            var vSum = Vector256<float>.Zero;
#endif
            for (int sp = 0; sp < spatial; sp++)
            {
                int idx = inBase + sp * CBlock;
#if NET5_0_OR_GREATER
                if (useSimd)
                {
                    var vIn = Vector256.Create(
                        inArr[idx + 0], inArr[idx + 1], inArr[idx + 2], inArr[idx + 3],
                        inArr[idx + 4], inArr[idx + 5], inArr[idx + 6], inArr[idx + 7]);
                    vSum = Avx.Add(vSum, vIn);
                    continue;
                }
#endif
                for (int cb = 0; cb < CBlock; cb++) acc[cb] += inArr[idx + cb];
            }
#if NET5_0_OR_GREATER
            if (useSimd) vSum.CopyTo(acc);
#endif
            // Store into [n, ocg*CBlock + cb] of flat [N, C].
            int outBase = n * C + ocg * CBlock;
            for (int cb = 0; cb < CBlock; cb++) outArr[outBase + cb] = acc[cb] * inv;
        });
        // `outArr` IS the caller's `output`.
    }
}
