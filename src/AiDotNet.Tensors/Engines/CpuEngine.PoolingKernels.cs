using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// One plane of the tiling 2x2 stride-2 max-pool backward (<see cref="MaxPool2DBackwardRecomputeInto{T}"/>), eight
    /// windows per AVX step: the two input rows are de-interleaved into the four taps, each window's winner is chosen
    /// with the scalar rule (start at float.MinValue, strict ordered greater-than in tap order, so the first maximum
    /// wins and NaN never does), and the four output cells are written as <c>0 + g</c> on the winner and +0 elsewhere,
    /// re-interleaved into the two rows. A window with no winner sends its gradient to plane cell 0, added in window
    /// order after its block is stored: the same additions in the same order as the scalar loop, so the plane is
    /// bit-identical. The columns past the last full block of eight windows run the scalar loop.
    /// </summary>
    private static unsafe void MaxPool2x2Stride2TilesBackwardPlane(
        float[] x, int xBase, float[] g, int gBase, float[] d, int dBase, int width, int outH, int outW)
    {
        fixed (float* px = &x[xBase])
        fixed (float* pg = &g[gBase])
        fixed (float* pd = &d[dBase])
        {
            for (int oh = 0; oh < outH; oh++)
            {
                float* x0 = px + 2 * oh * width, x1 = x0 + width;
                float* d0 = pd + 2 * oh * width, d1 = d0 + width;
                float* gr = pg + oh * outW;
                int ow = 0;
#if NET5_0_OR_GREATER
                if (Avx2.IsSupported)
                {
                    var minV = Vector256.Create(float.MinValue);
                    var zero = Vector256<float>.Zero;
                    var none = Vector256.Create(-1f);
                    var t0 = Vector256.Create(0f); var t1 = Vector256.Create(1f);
                    var t2 = Vector256.Create(2f); var t3 = Vector256.Create(3f);
                    for (; ow + 8 <= outW; ow += 8)
                    {
                        int c = 2 * ow;
                        var r0lo = Avx.LoadVector256(x0 + c);
                        var r0hi = Avx.LoadVector256(x0 + c + 8);
                        var r1lo = Avx.LoadVector256(x1 + c);
                        var r1hi = Avx.LoadVector256(x1 + c + 8);
                        var a = DeinterleaveEven(r0lo, r0hi);
                        var b = DeinterleaveOdd(r0lo, r0hi);
                        var cc = DeinterleaveEven(r1lo, r1hi);
                        var dd = DeinterleaveOdd(r1lo, r1hi);
                        var m = minV;
                        var sel = none;
                        var gt = Avx.Compare(a, m, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                        m = Avx.BlendVariable(m, a, gt); sel = Avx.BlendVariable(sel, t0, gt);
                        gt = Avx.Compare(b, m, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                        m = Avx.BlendVariable(m, b, gt); sel = Avx.BlendVariable(sel, t1, gt);
                        gt = Avx.Compare(cc, m, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                        m = Avx.BlendVariable(m, cc, gt); sel = Avx.BlendVariable(sel, t2, gt);
                        gt = Avx.Compare(dd, m, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                        sel = Avx.BlendVariable(sel, t3, gt);
                        var gv = Avx.Add(zero, Avx.LoadVector256(gr + ow));
                        var dA = Avx.And(gv, Avx.Compare(sel, t0, FloatComparisonMode.OrderedEqualNonSignaling));
                        var dB = Avx.And(gv, Avx.Compare(sel, t1, FloatComparisonMode.OrderedEqualNonSignaling));
                        var dC = Avx.And(gv, Avx.Compare(sel, t2, FloatComparisonMode.OrderedEqualNonSignaling));
                        var dD = Avx.And(gv, Avx.Compare(sel, t3, FloatComparisonMode.OrderedEqualNonSignaling));
                        StoreInterleaved(d0 + c, dA, dB);
                        StoreInterleaved(d1 + c, dC, dD);
                        int orphan = Avx.MoveMask(Avx.Compare(sel, none, FloatComparisonMode.OrderedEqualNonSignaling));
                        if (orphan != 0)
                            for (int lane = 0; lane < 8; lane++)
                                if ((orphan & (1 << lane)) != 0) pd[0] += gr[ow + lane];
                    }
                }
#endif
                for (; ow < outW; ow++)
                {
                    int iw0 = 2 * ow;
                    float maxVal = float.MinValue;
                    int winner = -1;
                    float v = x0[iw0]; if (v > maxVal) { maxVal = v; winner = 0; }
                    v = x0[iw0 + 1]; if (v > maxVal) { maxVal = v; winner = 1; }
                    v = x1[iw0]; if (v > maxVal) { maxVal = v; winner = 2; }
                    v = x1[iw0 + 1]; if (v > maxVal) { winner = 3; }
                    d0[iw0] = 0f; d0[iw0 + 1] = 0f; d1[iw0] = 0f; d1[iw0 + 1] = 0f;
                    float* target = winner switch { 0 => d0 + iw0, 1 => d0 + iw0 + 1, 2 => d1 + iw0, 3 => d1 + iw0 + 1, _ => pd };
                    *target += gr[ow];
                }
            }
        }
    }

#if NET5_0_OR_GREATER
    /// <summary>Elements 0, 2, 4, ... of the 16 floats <paramref name="lo"/>:<paramref name="hi"/>, in order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> DeinterleaveEven(Vector256<float> lo, Vector256<float> hi)
        => Avx2.Permute4x64(Avx.Shuffle(lo, hi, 0b10_00_10_00).AsDouble(), 0b11_01_10_00).AsSingle();

    /// <summary>Elements 1, 3, 5, ... of the 16 floats <paramref name="lo"/>:<paramref name="hi"/>, in order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> DeinterleaveOdd(Vector256<float> lo, Vector256<float> hi)
        => Avx2.Permute4x64(Avx.Shuffle(lo, hi, 0b11_01_11_01).AsDouble(), 0b11_01_10_00).AsSingle();

    /// <summary>Stores a0 b0 a1 b1 ... a7 b7 (16 floats) at <paramref name="dst"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreInterleaved(float* dst, Vector256<float> a, Vector256<float> b)
    {
        var lo = Avx.UnpackLow(a, b);
        var hi = Avx.UnpackHigh(a, b);
        Avx.Store(dst, Avx.Permute2x128(lo, hi, 0x20));
        Avx.Store(dst + 8, Avx.Permute2x128(lo, hi, 0x31));
    }
#endif

    /// <summary>
    /// One output row of the unpadded 2x2 stride-2 max pool: <c>dst[ow] = max</c> over the window at columns
    /// 2ow, 2ow+1 of input rows <paramref name="r0"/> and <paramref name="r1"/>, eight windows per AVX step. Same rule
    /// as the scalar kernel (start from the window's first tap, take a later tap only when strictly greater, so a NaN
    /// first tap stays and a later NaN never replaces), so the output is bit-identical.
    /// </summary>
    internal static unsafe void MaxPool2x2Stride2Row(float* r0, float* r1, float* dst, int outW)
    {
        int ow = 0;
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported)
        {
            for (; ow + 8 <= outW; ow += 8)
            {
                int c = 2 * ow;
                var r0lo = Avx.LoadVector256(r0 + c);
                var r0hi = Avx.LoadVector256(r0 + c + 8);
                var r1lo = Avx.LoadVector256(r1 + c);
                var r1hi = Avx.LoadVector256(r1 + c + 8);
                var m = DeinterleaveEven(r0lo, r0hi);
                var v = DeinterleaveOdd(r0lo, r0hi);
                m = Avx.BlendVariable(m, v, Avx.Compare(v, m, FloatComparisonMode.OrderedGreaterThanNonSignaling));
                v = DeinterleaveEven(r1lo, r1hi);
                m = Avx.BlendVariable(m, v, Avx.Compare(v, m, FloatComparisonMode.OrderedGreaterThanNonSignaling));
                v = DeinterleaveOdd(r1lo, r1hi);
                m = Avx.BlendVariable(m, v, Avx.Compare(v, m, FloatComparisonMode.OrderedGreaterThanNonSignaling));
                Avx.Store(dst + ow, m);
            }
        }
#endif
        for (; ow < outW; ow++)
        {
            int iw = 2 * ow;
            float m = r0[iw];
            float v = r0[iw + 1]; if (v > m) m = v;
            v = r1[iw]; if (v > m) m = v;
            v = r1[iw + 1]; if (v > m) m = v;
            dst[ow] = m;
        }
    }
}