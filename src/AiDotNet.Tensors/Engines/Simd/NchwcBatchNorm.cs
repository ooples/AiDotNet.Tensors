using System;
using System.Threading.Tasks;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Inference-mode batch normalization for NCHWc8 and NCHW float tensors.
///
/// <para>Decomposes ONNX BN into a single pre-computed FMA:
/// <c>out = scale' * x + bias'</c> where
/// <c>scale' = gamma / sqrt(var + eps)</c> and
/// <c>bias' = beta - scale' * mean</c>. That collapses six broadcast ops
/// (sub, add, sqrt, div, mul, add) into one fused FMA per element.</para>
///
/// <para>The NCHWc8 path processes contiguous 8-wide lanes with a single
/// <c>Vector256&lt;float&gt;</c> FMA per spatial cell — bandwidth-bound
/// on modern CPUs (one load, one store, effectively one FMA).</para>
/// </summary>
internal static class NchwcBatchNorm
{
    public const int CBlock = 8;

    /// <summary>
    /// Inference BN on NCHWc8 layout <c>[N, C/8, H, W, 8]</c>.
    /// <paramref name="scaleArr"/>/<paramref name="biasArr"/> have shape [C] in
    /// channel-major order; the method packs them on the fly into [C/8, 8] to
    /// match the lane layout.
    /// </summary>
    public static void RunNchwc8(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> variance,
        float epsilon,
        Span<float> output,
        int N, int C, int H, int W)
    {
        int cg = C / CBlock;
        int spatial = H * W;
        int hwC = spatial * CBlock;

        // Pre-combine: scale[c] = gamma[c] / sqrt(var[c] + eps); bias[c] = beta[c] - scale[c] * mean[c].
        // Pack scale/bias into [cg, 8] so lane loads are contiguous.
        var packedScale = new float[C];
        var packedBias = new float[C];
        for (int c = 0; c < C; c++)
        {
            double s = gamma[c] / Math.Sqrt((double)variance[c] + epsilon);
            packedScale[c] = (float)s;
            packedBias[c] = beta[c] - (float)(s * mean[c]);
        }

#if NET5_0_OR_GREATER
        bool useSimd = Avx.IsSupported && Fma.IsSupported;
#endif
        var inArr = input.ToArray();
        var outArr = new float[output.Length];

        Parallel.For(0, N * cg, task =>
        {
            int n = task / cg;
            int ocg = task % cg;
            int groupBase = (n * cg + ocg) * hwC;
            int scaleBase = ocg * CBlock;

#if NET5_0_OR_GREATER
            if (useSimd)
            {
                var vScale = Vector256.Create(
                    packedScale[scaleBase + 0], packedScale[scaleBase + 1],
                    packedScale[scaleBase + 2], packedScale[scaleBase + 3],
                    packedScale[scaleBase + 4], packedScale[scaleBase + 5],
                    packedScale[scaleBase + 6], packedScale[scaleBase + 7]);
                var vBias = Vector256.Create(
                    packedBias[scaleBase + 0], packedBias[scaleBase + 1],
                    packedBias[scaleBase + 2], packedBias[scaleBase + 3],
                    packedBias[scaleBase + 4], packedBias[scaleBase + 5],
                    packedBias[scaleBase + 6], packedBias[scaleBase + 7]);

                for (int sp = 0; sp < spatial; sp++)
                {
                    int idx = groupBase + sp * CBlock;
                    var vIn = Vector256.Create(
                        inArr[idx + 0], inArr[idx + 1], inArr[idx + 2], inArr[idx + 3],
                        inArr[idx + 4], inArr[idx + 5], inArr[idx + 6], inArr[idx + 7]);
                    var vOut = Fma.MultiplyAdd(vIn, vScale, vBias);
                    var store = new float[CBlock];
                    vOut.CopyTo(store);
                    for (int i = 0; i < CBlock; i++) outArr[idx + i] = store[i];
                }
            }
            else
#endif
            {
                for (int sp = 0; sp < spatial; sp++)
                {
                    int idx = groupBase + sp * CBlock;
                    for (int cb = 0; cb < CBlock; cb++)
                    {
                        outArr[idx + cb] =
                            inArr[idx + cb] * packedScale[scaleBase + cb] + packedBias[scaleBase + cb];
                    }
                }
            }
        });

        outArr.AsSpan().CopyTo(output);
    }

    /// <summary>
    /// Inference BN on NCHW layout <c>[N, C, H, W]</c>. Shared with the NCHWc
    /// path — we always collapse to pre-combined scale/bias and emit one FMA.
    /// </summary>
    public static void RunNchw(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> variance,
        float epsilon,
        Span<float> output,
        int N, int C, int H, int W)
    {
        int spatial = H * W;
        var packedScale = new float[C];
        var packedBias = new float[C];
        for (int c = 0; c < C; c++)
        {
            double s = gamma[c] / Math.Sqrt((double)variance[c] + epsilon);
            packedScale[c] = (float)s;
            packedBias[c] = beta[c] - (float)(s * mean[c]);
        }

#if NET5_0_OR_GREATER
        bool useSimd = Avx.IsSupported && Fma.IsSupported;
#endif
        var inArr = input.ToArray();
        var outArr = new float[output.Length];

        Parallel.For(0, N * C, task =>
        {
            int n = task / C;
            int c = task % C;
            int chanBase = (n * C + c) * spatial;
            float s = packedScale[c];
            float b = packedBias[c];

#if NET5_0_OR_GREATER
            if (useSimd && spatial >= 8)
            {
                var vS = Vector256.Create(s);
                var vB = Vector256.Create(b);
                int i = 0;
                for (; i + 8 <= spatial; i += 8)
                {
                    int idx = chanBase + i;
                    var vIn = Vector256.Create(
                        inArr[idx + 0], inArr[idx + 1], inArr[idx + 2], inArr[idx + 3],
                        inArr[idx + 4], inArr[idx + 5], inArr[idx + 6], inArr[idx + 7]);
                    var vOut = Fma.MultiplyAdd(vIn, vS, vB);
                    var store = new float[8];
                    vOut.CopyTo(store);
                    for (int k = 0; k < 8; k++) outArr[idx + k] = store[k];
                }
                for (; i < spatial; i++)
                    outArr[chanBase + i] = inArr[chanBase + i] * s + b;
            }
            else
#endif
            {
                for (int i = 0; i < spatial; i++)
                    outArr[chanBase + i] = inArr[chanBase + i] * s + b;
            }
        });

        outArr.AsSpan().CopyTo(output);
    }
}
