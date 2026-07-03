using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Shared hardware-SIMD Adam / AMSGrad step kernel (#1757). Performs the in-place, zero-allocation
/// Adam update over raw parameter / gradient / moment spans:
/// <code>
/// m = β1·m + (1-β1)·g;   v = β2·v + (1-β2)·g²
/// m̂ = m / bc1;           v̂ = AMSGrad ? max(v̂, v̂_prev) : v / bc2
/// p -= lr · m̂ / (√v̂ + ε)
/// </code>
/// This is the single source of truth for the CPU eager-optimizer Adam inner loop: the consumer's
/// <c>AdamOptimizer</c> tape-step calls it (InternalsVisibleTo "AiDotNet") instead of hand-rolling SIMD.
///
/// <para><b>Numerics — SIMD is bit-identical to the scalar tail.</b> Both use a fused multiply-add for
/// the two moment updates (<see cref="System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(Vector256{float},Vector256{float},Vector256{float})"/>
/// in the vector path, <see cref="MathF.FusedMultiplyAdd"/> / <see cref="Math.FusedMultiplyAdd"/> in the
/// scalar path), the SAME correctly-rounded hardware sqrt (<c>Avx.Sqrt</c> == <see cref="MathF.Sqrt"/> /
/// <see cref="Math.Sqrt"/>), plain IEEE divide/multiply/subtract, and an ordered greater-than compare +
/// blend for the AMSGrad running-max (matching the scalar <c>&gt;</c> ternary, incl. NaN → keep previous).
/// Because the vector lanes and the sub-width tail run identical operations, the result does not depend
/// on the vector width or the remainder split. (This uses FMA for speed, so it is intentionally NOT
/// bit-identical to a non-FMA separate-multiply-add loop.)</para>
///
/// <para>On net471 (no <c>System.Runtime.Intrinsics</c>) or any host without AVX+FMA, the scalar path
/// runs over the whole span. There, <see cref="MathF"/>/FMA intrinsics are unavailable, so it falls back
/// to separate multiply-add and <c>(float)Math.Sqrt</c> — correct, just not bit-matched to the SIMD path
/// that never runs on those hosts.</para>
/// </summary>
internal static class AdamMomentKernels
{
#if NET5_0_OR_GREATER
    private static float FmaF(float a, float b, float c) => MathF.FusedMultiplyAdd(a, b, c);
    private static double FmaD(double a, double b, double c) => Math.FusedMultiplyAdd(a, b, c);
    private static float SqrtF(float x) => MathF.Sqrt(x);
#else
    private static float FmaF(float a, float b, float c) => a * b + c;
    private static double FmaD(double a, double b, double c) => a * b + c;
    private static float SqrtF(float x) => (float)Math.Sqrt(x);
#endif

    /// <summary>fp32 Adam/AMSGrad step, in place over <paramref name="param"/>/<paramref name="m"/>/<paramref name="v"/> (and <paramref name="vMax"/> when <paramref name="useAmsgrad"/>).</summary>
    internal static unsafe void AdamStep(
        Span<float> param, ReadOnlySpan<float> grad, Span<float> m, Span<float> v, Span<float> vMax,
        float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2,
        float bc1, float bc2, float lr, float eps, bool useAmsgrad)
    {
        int n = param.Length;
        int i = 0;
        fixed (float* pParam = param, pGrad = grad, pM = m, pV = v, pVMax = vMax)
        {
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && Fma.IsSupported && n >= Vector256<float>.Count)
            {
                var vB1 = Vector256.Create(beta1);
                var vB2 = Vector256.Create(beta2);
                var v1mB1 = Vector256.Create(oneMinusBeta1);
                var v1mB2 = Vector256.Create(oneMinusBeta2);
                var vBc1 = Vector256.Create(bc1);
                var vBc2 = Vector256.Create(bc2);
                var vLr = Vector256.Create(lr);
                var vEps = Vector256.Create(eps);
                int simdLen = n & ~(Vector256<float>.Count - 1);
                for (; i < simdLen; i += Vector256<float>.Count)
                {
                    var g = Avx.LoadVector256(pGrad + i);
                    var mNew = Fma.MultiplyAdd(vB1, Avx.LoadVector256(pM + i), Avx.Multiply(v1mB1, g));
                    Avx.Store(pM + i, mNew);
                    var vNew = Fma.MultiplyAdd(vB2, Avx.LoadVector256(pV + i), Avx.Multiply(v1mB2, Avx.Multiply(g, g)));
                    Avx.Store(pV + i, vNew);
                    var mHat = Avx.Divide(mNew, vBc1);
                    Vector256<float> vHatEff;
                    if (useAmsgrad)
                    {
                        var vHatNow = Avx.Divide(vNew, vBc2);
                        var vMaxPrev = Avx.LoadVector256(pVMax + i);
                        var mask = Avx.Compare(vHatNow, vMaxPrev, FloatComparisonMode.OrderedGreaterThanSignaling);
                        var vMaxNew = Avx.BlendVariable(vMaxPrev, vHatNow, mask);
                        Avx.Store(pVMax + i, vMaxNew);
                        vHatEff = vMaxNew;
                    }
                    else
                    {
                        vHatEff = Avx.Divide(vNew, vBc2);
                    }
                    var denom = Avx.Add(Avx.Sqrt(vHatEff), vEps);
                    var update = Avx.Divide(Avx.Multiply(vLr, mHat), denom);
                    Avx.Store(pParam + i, Avx.Subtract(Avx.LoadVector256(pParam + i), update));
                }
            }
#endif
            for (; i < n; i++)
            {
                float g = pGrad[i];
                float mNew = FmaF(beta1, pM[i], oneMinusBeta1 * g);
                float vNew = FmaF(beta2, pV[i], oneMinusBeta2 * (g * g));
                pM[i] = mNew;
                pV[i] = vNew;
                float mHat = mNew / bc1;
                float vHatEff;
                if (useAmsgrad)
                {
                    float vHatNow = vNew / bc2;
                    float prev = pVMax[i];
                    float mx = vHatNow > prev ? vHatNow : prev;
                    pVMax[i] = mx;
                    vHatEff = mx;
                }
                else
                {
                    vHatEff = vNew / bc2;
                }
                pParam[i] -= lr * mHat / (SqrtF(vHatEff) + eps);
            }
        }
    }

    /// <summary>fp64 Adam/AMSGrad step, in place over <paramref name="param"/>/<paramref name="m"/>/<paramref name="v"/> (and <paramref name="vMax"/> when <paramref name="useAmsgrad"/>).</summary>
    internal static unsafe void AdamStep(
        Span<double> param, ReadOnlySpan<double> grad, Span<double> m, Span<double> v, Span<double> vMax,
        double beta1, double beta2, double oneMinusBeta1, double oneMinusBeta2,
        double bc1, double bc2, double lr, double eps, bool useAmsgrad)
    {
        int n = param.Length;
        int i = 0;
        fixed (double* pParam = param, pGrad = grad, pM = m, pV = v, pVMax = vMax)
        {
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && Fma.IsSupported && n >= Vector256<double>.Count)
            {
                var vB1 = Vector256.Create(beta1);
                var vB2 = Vector256.Create(beta2);
                var v1mB1 = Vector256.Create(oneMinusBeta1);
                var v1mB2 = Vector256.Create(oneMinusBeta2);
                var vBc1 = Vector256.Create(bc1);
                var vBc2 = Vector256.Create(bc2);
                var vLr = Vector256.Create(lr);
                var vEps = Vector256.Create(eps);
                int simdLen = n & ~(Vector256<double>.Count - 1);
                for (; i < simdLen; i += Vector256<double>.Count)
                {
                    var g = Avx.LoadVector256(pGrad + i);
                    var mNew = Fma.MultiplyAdd(vB1, Avx.LoadVector256(pM + i), Avx.Multiply(v1mB1, g));
                    Avx.Store(pM + i, mNew);
                    var vNew = Fma.MultiplyAdd(vB2, Avx.LoadVector256(pV + i), Avx.Multiply(v1mB2, Avx.Multiply(g, g)));
                    Avx.Store(pV + i, vNew);
                    var mHat = Avx.Divide(mNew, vBc1);
                    Vector256<double> vHatEff;
                    if (useAmsgrad)
                    {
                        var vHatNow = Avx.Divide(vNew, vBc2);
                        var vMaxPrev = Avx.LoadVector256(pVMax + i);
                        var mask = Avx.Compare(vHatNow, vMaxPrev, FloatComparisonMode.OrderedGreaterThanSignaling);
                        var vMaxNew = Avx.BlendVariable(vMaxPrev, vHatNow, mask);
                        Avx.Store(pVMax + i, vMaxNew);
                        vHatEff = vMaxNew;
                    }
                    else
                    {
                        vHatEff = Avx.Divide(vNew, vBc2);
                    }
                    var denom = Avx.Add(Avx.Sqrt(vHatEff), vEps);
                    var update = Avx.Divide(Avx.Multiply(vLr, mHat), denom);
                    Avx.Store(pParam + i, Avx.Subtract(Avx.LoadVector256(pParam + i), update));
                }
            }
#endif
            for (; i < n; i++)
            {
                double g = pGrad[i];
                double mNew = FmaD(beta1, pM[i], oneMinusBeta1 * g);
                double vNew = FmaD(beta2, pV[i], oneMinusBeta2 * (g * g));
                pM[i] = mNew;
                pV[i] = vNew;
                double mHat = mNew / bc1;
                double vHatEff;
                if (useAmsgrad)
                {
                    double vHatNow = vNew / bc2;
                    double prev = pVMax[i];
                    double mx = vHatNow > prev ? vHatNow : prev;
                    pVMax[i] = mx;
                    vHatEff = mx;
                }
                else
                {
                    vHatEff = vNew / bc2;
                }
                pParam[i] -= lr * mHat / (Math.Sqrt(vHatEff) + eps);
            }
        }
    }
}
