namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    public const string DcConv2dDirectFp16Hw = "dc_p_conv2d_direct_fp16hw";

    // #682: the direct FP16 convolution (CUDA's conv2d_direct_fp16hw, #671) for the backends that build the shared
    // source. FP32 input rounded to FP16 per access, FP16 weights [outC, inC, kH, kW], FP32 multiply-accumulate in
    // (ic, kh, kw) order, FP32 output. One thread per output element; no column matrix.
    //
    // Both operands are read through int words, the one pointer type every translation carries: the weights as
    // packed halves (DL_HALF, from the decode-linear source), the input as the bit patterns of its floats.
    // DC_ROUND_HALF gives the value a float has after rounding to FP16 (round to nearest, ties to even), in integer
    // arithmetic on the bit pattern, so no compiler can fold the rounding away (a pack/unpack round trip was folded
    // to the identity on one driver). Exponent field e of the float:
    //   e == 255        infinity and NaN pass through;
    //   e <  113        below 2^-14, the FP16 subnormal range, spaced 2^-24 apart: the 24-bit significand shifted
    //                   right by 126 - e gives the count of 2^-24 steps, rounded on the bits shifted out
    //                   (a shift above 24 is less than half a step: zero);
    //   otherwise       a normal half keeps 10 of the 23 mantissa bits: drop 13, ties to even. A carry out of the
    //                   mantissa bumps the exponent, which is the right result; 65536 or more overflows to infinity.
    internal const string DirectConvSource = @"
#define DC_ROUND_HALF(V, BITS) { int dc_u = BITS; int dc_mag = dc_u & 2147483647; int dc_e = dc_mag >> 23; if (dc_e == 255) { V = __int_as_float(dc_mag); } else { if (dc_e < 113) { int dc_q = 0; int dc_s = 126 - dc_e; if (dc_e != 0 && dc_s <= 24) { int dc_full = (dc_mag & 8388607) | 8388608; dc_q = dc_full >> dc_s; int dc_rem = dc_full - (dc_q << dc_s); int dc_tie = 1 << (dc_s - 1); if (dc_rem > dc_tie || (dc_rem == dc_tie && (dc_q & 1) != 0)) { dc_q = dc_q + 1; } } float dc_f = dc_q; V = dc_f * 5.9604644775390625e-8f; } else { dc_mag = dc_mag + 4095 + ((dc_mag >> 13) & 1); dc_mag = (dc_mag >> 13) << 13; if (dc_mag >= 1199570944) { dc_mag = 2139095040; } V = __int_as_float(dc_mag); } } if (dc_u < 0) { V = -V; } }
extern ""C"" __global__ __launch_bounds__(256) void dc_p_conv2d_direct_fp16hw(const int* inBits, const int* wh, float* y, int batch, int inC, int inH, int inW, int outC, int outH, int outW, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW) {
    int t = bx_gid(); if (t >= batch * outC * outH * outW) return;
    int r1 = t / outW; int ow = t - r1 * outW;
    int r2 = r1 / outH; int oh = r1 - r2 * outH;
    int b = r2 / outC; int oc = r2 - b * outC;
    float acc = 0.0f;
    for (int ic = 0; ic < inC; ic++) {
        for (int kh = 0; kh < kH; kh++) {
            int ih = oh * sH - pH + kh * dH;
            if (ih >= 0 && ih < inH) {
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow * sW - pW + kw * dW;
                    if (iw >= 0 && iw < inW) {
                        float xv; float wv;
                        int wi = ((oc * inC + ic) * kH + kh) * kW + kw;
                        DC_ROUND_HALF(xv, inBits[((b * inC + ic) * inH + ih) * inW + iw])
                        DL_HALF(wv, wh, wi)
                        acc += xv * wv;
                    }
                }
            }
        }
    }
    y[t] = acc;
}
";
}
