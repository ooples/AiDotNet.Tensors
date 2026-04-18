using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Conv2D kernel operating on NCHWc-layout tensors with <c>cBlock = 8</c>.
/// Consumes:
/// <list type="bullet">
/// <item>Input in NCHWc8 layout <c>[N, C/8, H, W, 8]</c>.</item>
/// <item>Kernel in OIHWio layout <c>[outC/8, inC/8, kH, kW, 8_in, 8_out]</c>
/// (see <see cref="NchwcReorder.KernelToOihwIo"/>).</item>
/// </list>
/// Emits NCHWc8 output <c>[N, outC/8, oH, oW, 8]</c>.
///
/// <para>The inner loop accumulates one <c>Vector256&lt;float&gt;</c> of
/// partial output per spatial position — each <c>cb_in</c> input lane is
/// broadcast to all 8 <c>cb_out</c> slots and FMA'd against the matching
/// kernel row. Eight input channels × 8 output channels contributes 64
/// FMAs per kernel-cell per output position; the SIMD FMA collapses the
/// output-lane dimension so the effective inner cost is <c>8 × Fma.Add</c>
/// per kernel cell, which matches oneDNN's NCHWc8 direct kernel on AVX2.</para>
///
/// <para>When <c>Avx2</c>/<c>Fma</c> aren't available, the scalar fallback
/// produces bit-exact output — parity tests verify this.</para>
/// </summary>
internal static class NchwcConv2D
{
    public const int CBlock = 8;

    /// <summary>
    /// NCHWc8 direct conv. All shapes derive from the channel-block tiling:
    /// <c>cgIn = inC / 8</c>, <c>cgOut = outC / 8</c>. Pad is symmetric.
    /// </summary>
    public static void Run(
        ReadOnlySpan<float> input,           // [N, cgIn, H, W, 8]
        ReadOnlySpan<float> kernel,          // [cgOut, cgIn, kH, kW, 8_in, 8_out]
        Span<float> output,                  // [N, cgOut, oH, oW, 8]
        int N, int inC, int H, int W,
        int outC, int kH, int kW,
        int oH, int oW,
        int sH, int sW, int padH, int padW, int dH, int dW)
    {
        int cgIn = inC / CBlock;
        int cgOut = outC / CBlock;
        // Strides (flat offsets).
        int inStrideN = cgIn * H * W * CBlock;
        int inStrideCg = H * W * CBlock;
        int inStrideH = W * CBlock;
        int inStrideW = CBlock;

        int outStrideN = cgOut * oH * oW * CBlock;
        int outStrideCg = oH * oW * CBlock;
        int outStrideH = oW * CBlock;
        int outStrideW = CBlock;

        int kStrideOcg = cgIn * kH * kW * CBlock * CBlock;
        int kStrideIcg = kH * kW * CBlock * CBlock;
        int kStrideKh = kW * CBlock * CBlock;
        int kStrideKw = CBlock * CBlock;

        output.Clear();

        // Local copies for lambda capture.
        int _N = N, _cgIn = cgIn, _cgOut = cgOut, _H = H, _W = W, _oH = oH, _oW = oW;
        int _kH = kH, _kW = kW, _sH = sH, _sW = sW, _padH = padH, _padW = padW, _dH = dH, _dW = dW;
        int _inStrideN = inStrideN, _inStrideCg = inStrideCg, _inStrideH = inStrideH, _inStrideW = inStrideW;
        int _outStrideN = outStrideN, _outStrideCg = outStrideCg, _outStrideH = outStrideH, _outStrideW = outStrideW;
        int _kStrideOcg = kStrideOcg, _kStrideIcg = kStrideIcg, _kStrideKh = kStrideKh, _kStrideKw = kStrideKw;

        // Parallelize across (N × cgOut). Each task owns one output channel
        // group of one batch image — no write conflicts, good cache reuse
        // (all kernels for this ocg fit in L2 for typical ResNet shapes).
        int totalTasks = N * cgOut;
        var inArr = input.ToArray();        // .NET lacks a way to capture Span in closure
        var kArr = kernel.ToArray();
        var outArr = new float[output.Length];

        bool useSimd = Avx2.IsSupported && Fma.IsSupported;

        Parallel.For(0, totalTasks, taskIdx =>
        {
            int n = taskIdx / _cgOut;
            int ocg = taskIdx % _cgOut;
            int nBase = n * _inStrideN;
            int oBase = n * _outStrideN + ocg * _outStrideCg;
            int kOBase = ocg * _kStrideOcg;
            // Per-task scratch reused across every output cell. Avoids
            // stackalloc-in-loop (CA2014) and keeps allocations linear
            // in task count, not oH*oW.
            var scalarAcc = new float[CBlock];
            var simdStore = new float[CBlock];

            for (int oh = 0; oh < _oH; oh++)
            {
                for (int ow = 0; ow < _oW; ow++)
                {
                    int outIdx = oBase + oh * _outStrideH + ow * _outStrideW;
                    var acc = Vector256<float>.Zero;
                    if (!useSimd) Array.Clear(scalarAcc, 0, CBlock);

                    for (int icg = 0; icg < _cgIn; icg++)
                    {
                        int inCgBase = nBase + icg * _inStrideCg;
                        int kIBase = kOBase + icg * _kStrideIcg;
                        for (int kh = 0; kh < _kH; kh++)
                        {
                            int ih = oh * _sH + kh * _dH - _padH;
                            if ((uint)ih >= (uint)_H) continue;
                            int kHBase = kIBase + kh * _kStrideKh;
                            int inHBase = inCgBase + ih * _inStrideH;
                            for (int kw = 0; kw < _kW; kw++)
                            {
                                int iw = ow * _sW + kw * _dW - _padW;
                                if ((uint)iw >= (uint)_W) continue;
                                int inIdx = inHBase + iw * _inStrideW;
                                int kIdx = kHBase + kw * _kStrideKw;
                                if (useSimd)
                                {
                                    for (int icb = 0; icb < CBlock; icb++)
                                    {
                                        var vIn = Vector256.Create(inArr[inIdx + icb]);
                                        var vK = Vector256.Create(
                                            kArr[kIdx + icb * CBlock + 0], kArr[kIdx + icb * CBlock + 1],
                                            kArr[kIdx + icb * CBlock + 2], kArr[kIdx + icb * CBlock + 3],
                                            kArr[kIdx + icb * CBlock + 4], kArr[kIdx + icb * CBlock + 5],
                                            kArr[kIdx + icb * CBlock + 6], kArr[kIdx + icb * CBlock + 7]);
                                        acc = Fma.MultiplyAdd(vIn, vK, acc);
                                    }
                                }
                                else
                                {
                                    for (int icb = 0; icb < CBlock; icb++)
                                    {
                                        float xv = inArr[inIdx + icb];
                                        int kRow = kIdx + icb * CBlock;
                                        for (int ocb = 0; ocb < CBlock; ocb++)
                                            scalarAcc[ocb] += xv * kArr[kRow + ocb];
                                    }
                                }
                            }
                        }
                    }

                    if (useSimd)
                    {
                        acc.CopyTo(simdStore);
                        for (int ocb = 0; ocb < CBlock; ocb++) outArr[outIdx + ocb] = simdStore[ocb];
                    }
                    else
                    {
                        for (int ocb = 0; ocb < CBlock; ocb++) outArr[outIdx + ocb] = scalarAcc[ocb];
                    }
                }
            }
        });

        // Copy back.
        outArr.AsSpan().CopyTo(output);
    }
}
