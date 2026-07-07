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
/// <para><b>Numerics — SIMD is bit-identical to the scalar tail AND to the consumer's scalar eager
/// Adam.</b> The two moment updates use a SEPARATE multiply then add (NOT a fused multiply-add), the
/// SAME correctly-rounded hardware sqrt (<c>Avx.Sqrt</c> == <see cref="MathF.Sqrt"/> / <see cref="Math.Sqrt"/>),
/// plain IEEE divide/multiply/subtract, and an ordered greater-than compare + blend for the AMSGrad
/// running-max (matching the scalar <c>&gt;</c> ternary, incl. NaN → keep previous). Because Adam is
/// fully elementwise (no cross-lane reduction), the vector lanes and the sub-width tail run identical
/// per-element operations, so the result does not depend on the vector width or the remainder split.
/// Separate-multiply-add (rather than FMA) is deliberate: it keeps this kernel bit-exact with
/// <c>AdamOptimizer&lt;T&gt;</c>'s eager fp32/fp64 loop (<c>beta*m + (1-beta)*g</c>), so wiring the eager
/// optimizer onto this SIMD path does not perturb saved golden trajectories.</para>
/// </summary>
internal static class AdamMomentKernels
{
    // ── DUAL-PATH mode select (AiDotNet #1804) ────────────────────────────────
    // Default (env UNSET): the bit-exact, separate-multiply-add, double-rounded-sqrt
    // path below — reproduces saved golden N-BEATS trajectories to the last ULP
    // (forecast[0]=47.395111, MAE=0.023372). This is production reproducibility.
    //
    // AIDOTNET_FAST_MATH=1: the FMA / single-precision-sqrt fast path — one fused
    // multiply-add per moment (vs a separate multiply+add) and a hardware single-
    // precision Avx.Sqrt / MathF.Sqrt (vs widen→double-sqrt→narrow). Faster, but the
    // fused rounding drifts ~1 ULP/step which the chaotic N-BEATS training amplifies
    // past 1e-4 — near-golden, not bit-exact. For research / PyTorch-parity benchmarks.
    //
    // Read ONCE into a static readonly bool so the JIT hoists the branch out of the
    // hot loop (effectively a compile-time constant per process).
    private static readonly bool FastMath =
        Environment.GetEnvironmentVariable("AIDOTNET_FAST_MATH") == "1";

    // Separate multiply-then-add (NOT fused). The consumer AdamOptimizer's eager
    // fp32/fp64 fast path computes the two moment updates as `beta*m + (1-beta)*g`
    // with a distinct multiply and add, so to stay bit-exact with that reference
    // (and hence with saved golden trajectories) this kernel must NOT contract the
    // pair into an FMA — an FMA keeps the intermediate at full width and rounds once,
    // which drifts from the scalar path and, over a full training run, amplifies well
    // past 1e-4. Because Adam is fully elementwise (no cross-lane reduction), the SIMD
    // path below and this scalar tail produce identical results regardless of vector
    // width or the remainder split.
    // Default: double-rounded (widen to double, IEEE-sqrt, narrow) — matches the scalar
    // eager path's `(float)Math.Sqrt((double)x)` exactly. Fast mode: single-precision
    // sqrt (up to 1 ULP off, but ~2× cheaper — no widen/narrow round trip).
    private static float SqrtF(float x) =>
#if NET5_0_OR_GREATER
        FastMath ? MathF.Sqrt(x) : (float)Math.Sqrt(x);
#else
        (float)Math.Sqrt(x);
#endif

    // Fused multiply-add helpers for the fast-mode scalar tail (mirrors the SIMD FMA
    // lanes for width-independence). Guarded: MathF.FusedMultiplyAdd / Math.FusedMultiplyAdd
    // are .NET Core 3.0+ only; net471 (never a benchmark target) falls back to separate
    // multiply-add — fast mode is a NET5+ research/benchmark path.
    private static float FmaTail(float a, float b, float c) =>
#if NET5_0_OR_GREATER
        MathF.FusedMultiplyAdd(a, b, c);
#else
        a * b + c;
#endif

    private static double FmaTail(double a, double b, double c) =>
#if NET5_0_OR_GREATER
        Math.FusedMultiplyAdd(a, b, c);
#else
        a * b + c;
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
                    // Default (bit-exact) vs fast (FMA). Default matches AdamOptimizer's scalar
                    // eager fp32 loop with a SEPARATE multiply-add:
                    //   mNew = (β1·m) + ((1-β1)·g)      — separate multiply-add, not FMA
                    //   vNew = (β2·v) + (((1-β2)·g)·g)  — LEFT-assoc `(1-β2)*g*g`, matching C# `f1mb2 * g * g`
                    // Fast mode fuses the accumulate into one FMA (rounds once, ~1 ULP drift).
                    Vector256<float> mNew, vNew;
                    if (FastMath)
                    {
                        mNew = Fma.MultiplyAdd(v1mB1, g, Avx.Multiply(vB1, Avx.LoadVector256(pM + i)));
                        vNew = Fma.MultiplyAdd(Avx.Multiply(v1mB2, g), g, Avx.Multiply(vB2, Avx.LoadVector256(pV + i)));
                    }
                    else
                    {
                        mNew = Avx.Add(Avx.Multiply(vB1, Avx.LoadVector256(pM + i)), Avx.Multiply(v1mB1, g));
                        vNew = Avx.Add(Avx.Multiply(vB2, Avx.LoadVector256(pV + i)), Avx.Multiply(Avx.Multiply(v1mB2, g), g));
                    }
                    Avx.Store(pM + i, mNew);
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
                    // Default: DOUBLE-ROUNDED sqrt — (float)Math.Sqrt((double)vHat) — to bit-match
                    // the scalar path (single-precision Avx.Sqrt differs by up to 1 ULP, which the
                    // chaotic N-BEATS training amplifies past 1e-4). Widen each 4-float half to double,
                    // IEEE-sqrt, narrow round-to-nearest. Fast mode: one hardware single-precision
                    // Avx.Sqrt over all 8 lanes (no widen/narrow round trip).
                    Vector256<float> denom;
                    if (FastMath)
                    {
                        denom = Avx.Add(Avx.Sqrt(vHatEff), vEps);
                    }
                    else
                    {
                        var sqLo = Avx.ConvertToVector128Single(Avx.Sqrt(Avx.ConvertToVector256Double(vHatEff.GetLower())));
                        var sqHi = Avx.ConvertToVector128Single(Avx.Sqrt(Avx.ConvertToVector256Double(vHatEff.GetUpper())));
                        denom = Avx.Add(Vector256.Create(sqLo, sqHi), vEps);
                    }
                    var update = Avx.Divide(Avx.Multiply(vLr, mHat), denom);
                    Avx.Store(pParam + i, Avx.Subtract(Avx.LoadVector256(pParam + i), update));
                }
            }
#endif
            for (; i < n; i++)
            {
                float g = pGrad[i];
                // Default: separate multiply-add (not FMA), left-assoc `(1-β2)*g*g`, DOUBLE-ROUNDED
                // sqrt (SqrtF) — bit-exact with the scalar eager fp32 loop. Fast mode: FMA-fused
                // moments + single-precision sqrt, mirroring the SIMD lanes (width-independence).
                float mNew, vNew;
                if (FastMath)
                {
                    mNew = FmaTail(oneMinusBeta1, g, beta1 * pM[i]);
                    vNew = FmaTail(oneMinusBeta2 * g, g, beta2 * pV[i]);
                }
                else
                {
                    mNew = beta1 * pM[i] + oneMinusBeta1 * g;
                    vNew = beta2 * pV[i] + oneMinusBeta2 * g * g;
                }
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
                    // Default: separate multiply-add (not FMA), left-assoc `(1-β2)*g*g` — bit-matches
                    // the scalar eager fp64 loop. Fast mode: FMA-fused accumulate (rounds once).
                    Vector256<double> mNew, vNew;
                    if (FastMath)
                    {
                        mNew = Fma.MultiplyAdd(v1mB1, g, Avx.Multiply(vB1, Avx.LoadVector256(pM + i)));
                        vNew = Fma.MultiplyAdd(Avx.Multiply(v1mB2, g), g, Avx.Multiply(vB2, Avx.LoadVector256(pV + i)));
                    }
                    else
                    {
                        mNew = Avx.Add(Avx.Multiply(vB1, Avx.LoadVector256(pM + i)), Avx.Multiply(v1mB1, g));
                        vNew = Avx.Add(Avx.Multiply(vB2, Avx.LoadVector256(pV + i)), Avx.Multiply(Avx.Multiply(v1mB2, g), g));
                    }
                    Avx.Store(pM + i, mNew);
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
                double mNew, vNew;
                if (FastMath)
                {
                    mNew = FmaTail(oneMinusBeta1, g, beta1 * pM[i]);
                    vNew = FmaTail(oneMinusBeta2 * g, g, beta2 * pV[i]);
                }
                else
                {
                    mNew = beta1 * pM[i] + oneMinusBeta1 * g;
                    vNew = beta2 * pV[i] + oneMinusBeta2 * g * g;
                }
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
