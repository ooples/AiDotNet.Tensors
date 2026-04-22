using System;
using System.Threading.Tasks;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// AVX-512 Conv2D kernel for NCHWc16 layout (cBlock = 16). Mirror of
/// <see cref="NchwcConv2D"/> but emits one <see cref="Vector512"/>
/// accumulator per output spatial cell — 16 output lanes collapsed into a
/// single FMA.
///
/// <para>Inner loop: broadcast one of 16 cb_in lanes of the input, FMA
/// against the matching [16_out] kernel row. 16 input channels × 16 output
/// channels = 256 FMAs per kernel-cell per output position, collapsing to
/// 16 SIMD FMAs with AVX-512. This doubles the per-cycle FMA throughput of
/// the NchwcConv2D cb=8 kernel on Skylake-X and later.</para>
/// </summary>
internal static class NchwcConv2D16
{
    public const int CBlock = 16;

    public static void Run(
        float[] input,                       // [N, cgIn, H, W, 16]
        float[] kernel,                      // [cgOut, cgIn, kH, kW, 16_in, 16_out]
        float[] output,                      // [N, cgOut, oH, oW, 16]
        int N, int inC, int H, int W,
        int outC, int kH, int kW,
        int oH, int oW,
        int sH, int sW, int padH, int padW, int dH, int dW)
    {
        int cgIn  = inC  / CBlock;
        int cgOut = outC / CBlock;

        int inStrideN  = cgIn  * H  * W  * CBlock;
        int inStrideCg = H * W * CBlock;
        int inStrideH  = W * CBlock;

        int outStrideN  = cgOut * oH * oW * CBlock;
        int outStrideCg = oH * oW * CBlock;
        int outStrideH  = oW * CBlock;

        int kStrideOcg = cgIn * kH * kW * CBlock * CBlock;
        int kStrideIcg = kH * kW * CBlock * CBlock;
        int kStrideKh  = kW * CBlock * CBlock;
        int kStrideKw  = CBlock * CBlock;

        Array.Clear(output, 0, output.Length);

        int _N = N, _cgIn = cgIn, _cgOut = cgOut, _H = H, _W = W, _oH = oH, _oW = oW;
        int _kH = kH, _kW = kW, _sH = sH, _sW = sW, _padH = padH, _padW = padW, _dH = dH, _dW = dW;
        int _inStrideN = inStrideN, _inStrideCg = inStrideCg, _inStrideH = inStrideH;
        int _outStrideN = outStrideN, _outStrideCg = outStrideCg, _outStrideH = outStrideH;
        int _kStrideOcg = kStrideOcg, _kStrideIcg = kStrideIcg, _kStrideKh = kStrideKh, _kStrideKw = kStrideKw;

        // Same zero-copy refactor as NchwcConv2D: closure captures the
        // caller-owned float[] arrays directly, no temporary allocations,
        // no copy-back.
        var inArr = input;
        var kArr = kernel;
        var outArr = output;

#if NET8_0_OR_GREATER
        bool useSimd = Avx512F.IsSupported;
#endif

        Parallel.For(0, N * cgOut, task =>
        {
            int n = task / _cgOut;
            int ocg = task % _cgOut;
            int nBase = n * _inStrideN;
            int oBase = n * _outStrideN + ocg * _outStrideCg;
            int kOBase = ocg * _kStrideOcg;
            var scalarAcc = new float[CBlock];
            var simdStore = new float[CBlock];

            for (int oh = 0; oh < _oH; oh++)
            {
                for (int ow = 0; ow < _oW; ow++)
                {
                    int outIdx = oBase + oh * _outStrideH + ow * CBlock;
#if NET8_0_OR_GREATER
                    var acc = Vector512<float>.Zero;
#endif
                    Array.Clear(scalarAcc, 0, CBlock);

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
                                int inIdx = inHBase + iw * CBlock;
                                int kIdx = kHBase + kw * _kStrideKw;
#if NET8_0_OR_GREATER
                                if (useSimd)
                                {
                                    for (int icb = 0; icb < CBlock; icb++)
                                    {
                                        var vIn = Vector512.Create(inArr[inIdx + icb]);
                                        int kRow = kIdx + icb * CBlock;
                                        var vK = Vector512.Create(
                                            kArr[kRow + 0],  kArr[kRow + 1],  kArr[kRow + 2],  kArr[kRow + 3],
                                            kArr[kRow + 4],  kArr[kRow + 5],  kArr[kRow + 6],  kArr[kRow + 7],
                                            kArr[kRow + 8],  kArr[kRow + 9],  kArr[kRow + 10], kArr[kRow + 11],
                                            kArr[kRow + 12], kArr[kRow + 13], kArr[kRow + 14], kArr[kRow + 15]);
                                        acc = Avx512F.FusedMultiplyAdd(vIn, vK, acc);
                                    }
                                    continue;
                                }
#endif
                                // Scalar fallback (also used for net<8 TFMs).
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

#if NET8_0_OR_GREATER
                    if (useSimd)
                    {
                        acc.CopyTo(simdStore);
                        for (int ocb = 0; ocb < CBlock; ocb++) outArr[outIdx + ocb] = simdStore[ocb];
                    }
                    else
#endif
                    {
                        for (int ocb = 0; ocb < CBlock; ocb++) outArr[outIdx + ocb] = scalarAcc[ocb];
                    }
                }
            }
        });
        // `outArr` IS the caller's `output` array.
    }
}
