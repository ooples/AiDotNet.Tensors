using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Layout reorder between NCHW and NCHWc (channel-packed). NCHW is the
/// ONNX-canonical <c>[N, C, H, W]</c>; NCHWc partitions the channel axis
/// into <c>C/cBlock</c> outer and <c>cBlock</c> inner, yielding
/// <c>[N, C/cBlock, H, W, cBlock]</c>. The inner <c>cBlock</c> runs
/// contiguous in memory — one SIMD load per spatial position per group —
/// which is what makes per-channel ops (Conv/BN/pool) SIMD-friendly.
///
/// <para><b>Padding.</b> When <c>C</c> is not divisible by <c>cBlock</c>,
/// callers must decide up front whether to round up (pad with zeros) or
/// reject. The reorder APIs here enforce divisibility and throw on
/// mismatch; the layout planner upstream is responsible for only issuing
/// a reorder when divisibility holds.</para>
///
/// <para>Both directions are implemented as a single contiguous scatter
/// over the flat element indices, bounded to a sequential <c>Parallel.For</c>
/// across N for batched inputs. Bit-exact round-trip is guaranteed when
/// <c>C % cBlock == 0</c>.</para>
/// </summary>
internal static class NchwcReorder
{
    /// <summary>
    /// Reorder a row-major NCHW float tensor (shape <c>[N, C, H, W]</c>)
    /// into the NCHWc layout with <paramref name="cBlock"/>-sized channel
    /// groups. Output shape: <c>[N, C/cBlock, H, W, cBlock]</c>, still
    /// flat-length <c>N·C·H·W</c>.
    /// </summary>
    public static void ToNchwc(
        ReadOnlySpan<float> src, Span<float> dst,
        int n, int c, int h, int w, int cBlock)
    {
        if (c % cBlock != 0)
            throw new ArgumentException(
                $"NCHW → NCHWc reorder requires C ({c}) divisible by cBlock ({cBlock}).");
        int hw = h * w;
        int outerC = c / cBlock;
        int totalPerN = c * hw;
        // dst flat index: (((n · outerC + oc) · h + hi) · w + wi) · cBlock + cb
        //                = n · (outerC · hw · cBlock) + oc · (hw · cBlock)
        //                  + (hi · w + wi) · cBlock + cb
        // src flat index: (((n · c + (oc·cBlock + cb)) · h + hi) · w + wi)
        //                = n · c · hw + (oc·cBlock + cb) · hw + hi · w + wi
        for (int ni = 0; ni < n; ni++)
        {
            int srcN = ni * totalPerN;
            int dstN = ni * totalPerN;
            for (int oc = 0; oc < outerC; oc++)
            {
                int srcOc = srcN + oc * cBlock * hw;
                int dstOc = dstN + oc * hw * cBlock;
                for (int p = 0; p < hw; p++)
                {
                    int dstBase = dstOc + p * cBlock;
                    // Read cBlock channels at this spatial position, write
                    // them contiguously.
                    for (int cb = 0; cb < cBlock; cb++)
                        dst[dstBase + cb] = src[srcOc + cb * hw + p];
                }
            }
        }
    }

    /// <summary>
    /// Inverse of <see cref="ToNchwc"/> — takes an NCHWc tensor back to
    /// NCHW. Same divisibility requirement.
    /// </summary>
    public static void ToNchw(
        ReadOnlySpan<float> src, Span<float> dst,
        int n, int c, int h, int w, int cBlock)
    {
        if (c % cBlock != 0)
            throw new ArgumentException(
                $"NCHWc → NCHW reorder requires C ({c}) divisible by cBlock ({cBlock}).");
        int hw = h * w;
        int outerC = c / cBlock;
        int totalPerN = c * hw;
        for (int ni = 0; ni < n; ni++)
        {
            int srcN = ni * totalPerN;
            int dstN = ni * totalPerN;
            for (int oc = 0; oc < outerC; oc++)
            {
                int srcOc = srcN + oc * hw * cBlock;
                int dstOc = dstN + oc * cBlock * hw;
                for (int p = 0; p < hw; p++)
                {
                    int srcBase = srcOc + p * cBlock;
                    for (int cb = 0; cb < cBlock; cb++)
                        dst[dstOc + cb * hw + p] = src[srcBase + cb];
                }
            }
        }
    }

    /// <summary>
    /// Reorder a Conv2D kernel tensor from NCHW-style
    /// <c>[outC, inC, kH, kW]</c> into the NCHWc-compatible
    /// <c>[outC/cBlock, inC/cBlock, kH, kW, cBlock_in, cBlock_out]</c>
    /// (OIHWio layout). This is the kernel layout the Conv NCHWc
    /// microkernel expects: the inner two dims are cBlock_in × cBlock_out
    /// so the FMA inner loop is a contiguous outer-product across SIMD
    /// register tiles.
    /// </summary>
    public static void KernelToOihwIo(
        ReadOnlySpan<float> src, Span<float> dst,
        int outC, int inC, int kH, int kW, int cBlock)
    {
        if (outC % cBlock != 0)
            throw new ArgumentException(
                $"Kernel reorder requires outC ({outC}) divisible by cBlock ({cBlock}).");
        if (inC % cBlock != 0)
            throw new ArgumentException(
                $"Kernel reorder requires inC ({inC}) divisible by cBlock ({cBlock}).");
        int outerOut = outC / cBlock;
        int outerIn = inC / cBlock;
        // src index: ((oco·cBlock + ocb) · inC + (ici·cBlock + icb)) · kH·kW
        //            + kh · kW + kw
        // dst index: ((((oco · outerIn + ici) · kH + kh) · kW + kw) · cBlock + icb) · cBlock + ocb
        for (int oco = 0; oco < outerOut; oco++)
        {
            for (int ici = 0; ici < outerIn; ici++)
            {
                for (int kh = 0; kh < kH; kh++)
                {
                    for (int kw = 0; kw < kW; kw++)
                    {
                        int dstBase = ((((oco * outerIn + ici) * kH + kh) * kW + kw) * cBlock) * cBlock;
                        for (int icb = 0; icb < cBlock; icb++)
                        {
                            int dstRow = dstBase + icb * cBlock;
                            int srcICh = (ici * cBlock + icb) * kH * kW + kh * kW + kw;
                            for (int ocb = 0; ocb < cBlock; ocb++)
                            {
                                int srcOCh = (oco * cBlock + ocb) * inC * kH * kW + srcICh;
                                dst[dstRow + ocb] = src[srcOCh];
                            }
                        }
                    }
                }
            }
        }
    }
}
