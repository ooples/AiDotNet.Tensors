// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Bit-exact parity for <see cref="AdamMomentKernels"/> (#1757, dual-path #1804): the SIMD (AVX) lanes must
/// produce results identical to the scalar tail, so a training step's output does not depend on the vector
/// width or the remainder split. The kernel is DUAL-PATH: with <c>AIDOTNET_FAST_MATH</c> UNSET (production
/// default) it uses separate-multiply-add moments and a double-rounded sqrt (bit-exact with the eager scalar
/// optimizer and saved golden trajectories); with <c>AIDOTNET_FAST_MATH=1</c> it uses FMA moments and a
/// single-precision sqrt. This reference mirrors the kernel's scalar tail for WHICHEVER mode the process is
/// running in (same env read), so the SIMD == scalar invariant is proven in the active mode. On net471
/// (scalar-only kernel) the reference mirrors the same fallback, so the comparison still holds.
/// </summary>
public class AdamMomentKernelsParityTests
{
    // Same env read as the kernel: default = bit-exact, "1" = FMA fast path.
    private static readonly bool FastMath =
        Environment.GetEnvironmentVariable("AIDOTNET_FAST_MATH") == "1";

    // Mirror the kernel's scalar tail EXACTLY per mode / TFM.
    private static float FmaF(float a, float b, float c) =>
#if NET5_0_OR_GREATER
        MathF.FusedMultiplyAdd(a, b, c);
#else
        a * b + c;
#endif
    private static double FmaD(double a, double b, double c) =>
#if NET5_0_OR_GREATER
        Math.FusedMultiplyAdd(a, b, c);
#else
        a * b + c;
#endif
    private static float SqrtF(float x) =>
#if NET5_0_OR_GREATER
        FastMath ? MathF.Sqrt(x) : (float)Math.Sqrt(x);
#else
        (float)Math.Sqrt(x);
#endif

    private static int FloatBits(float f) => BitConverter.ToInt32(BitConverter.GetBytes(f), 0);

    private static void RefStepF(
        float[] param, float[] grad, float[] m, float[] v, float[] vMax,
        float b1, float b2, float om1, float om2, float bc1, float bc2, float lr, float eps, bool ams)
    {
        for (int i = 0; i < param.Length; i++)
        {
            float g = grad[i];
            // Fast: FMA moments (matches kernel FmaTail(om1,g,b1*m) / FmaTail(om2*g,g,b2*v)).
            // Default: separate multiply-add, left-assoc (om2*g)*g.
            float mNew = FastMath ? FmaF(om1, g, b1 * m[i]) : b1 * m[i] + om1 * g;
            float vNew = FastMath ? FmaF(om2 * g, g, b2 * v[i]) : b2 * v[i] + om2 * g * g;
            m[i] = mNew;
            v[i] = vNew;
            float mHat = mNew / bc1;
            float vHatEff;
            if (ams) { float now = vNew / bc2; float prev = vMax[i]; float mx = now > prev ? now : prev; vMax[i] = mx; vHatEff = mx; }
            else vHatEff = vNew / bc2;
            param[i] -= lr * mHat / (SqrtF(vHatEff) + eps);
        }
    }

    private static void RefStepD(
        double[] param, double[] grad, double[] m, double[] v, double[] vMax,
        double b1, double b2, double om1, double om2, double bc1, double bc2, double lr, double eps, bool ams)
    {
        for (int i = 0; i < param.Length; i++)
        {
            double g = grad[i];
            double mNew = FastMath ? FmaD(om1, g, b1 * m[i]) : b1 * m[i] + om1 * g;
            double vNew = FastMath ? FmaD(om2 * g, g, b2 * v[i]) : b2 * v[i] + om2 * g * g;
            m[i] = mNew;
            v[i] = vNew;
            double mHat = mNew / bc1;
            double vHatEff;
            if (ams) { double now = vNew / bc2; double prev = vMax[i]; double mx = now > prev ? now : prev; vMax[i] = mx; vHatEff = mx; }
            else vHatEff = vNew / bc2;
            param[i] -= lr * mHat / (Math.Sqrt(vHatEff) + eps);
        }
    }

    private static float[] RandF(Random r, int n, bool positive = false)
    { var a = new float[n]; for (int i = 0; i < n; i++) { double x = r.NextDouble(); a[i] = (float)(positive ? x + 1e-3 : (x - 0.5) * 2.0); } return a; }
    private static double[] RandD(Random r, int n, bool positive = false)
    { var a = new double[n]; for (int i = 0; i < n; i++) { double x = r.NextDouble(); a[i] = positive ? x + 1e-3 : (x - 0.5) * 2.0; } return a; }

    // Sizes include sub-vector-width (1..7), exact multiples, and multiple+tail.
    [Theory]
    [InlineData(1, false)] [InlineData(3, false)] [InlineData(7, false)] [InlineData(8, false)]
    [InlineData(8, true)] [InlineData(15, false)] [InlineData(16, true)] [InlineData(100, false)] [InlineData(100, true)]
    public void Float_SimdMatchesScalar_BitForBit(int n, bool ams)
    {
        var r = new Random(1757 + n + (ams ? 1 : 0));
        float b1 = 0.9f, b2 = 0.999f, om1 = 1f - b1, om2 = 1f - b2, bc1 = 0.271f, bc2 = 0.02f, lr = 1e-3f, eps = 1e-8f;
        float[] p = RandF(r, n), g = RandF(r, n), m = RandF(r, n), v = RandF(r, n, true), vm = RandF(r, n, true);

        float[] pR = (float[])p.Clone(), mR = (float[])m.Clone(), vR = (float[])v.Clone(), vmR = (float[])vm.Clone();
        RefStepF(pR, g, mR, vR, vmR, b1, b2, om1, om2, bc1, bc2, lr, eps, ams);

        float[] pK = (float[])p.Clone(), mK = (float[])m.Clone(), vK = (float[])v.Clone(), vmK = (float[])vm.Clone();
        AdamMomentKernels.AdamStep(pK, g, mK, vK, vmK, b1, b2, om1, om2, bc1, bc2, lr, eps, ams);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(FloatBits(pR[i]), FloatBits(pK[i]));
            Assert.Equal(FloatBits(mR[i]), FloatBits(mK[i]));
            Assert.Equal(FloatBits(vR[i]), FloatBits(vK[i]));
            if (ams) Assert.Equal(FloatBits(vmR[i]), FloatBits(vmK[i]));
        }
    }

    [Theory]
    [InlineData(1, false)] [InlineData(3, false)] [InlineData(4, false)] [InlineData(4, true)]
    [InlineData(7, false)] [InlineData(8, true)] [InlineData(100, false)] [InlineData(100, true)]
    public void Double_SimdMatchesScalar_BitForBit(int n, bool ams)
    {
        var r = new Random(4200 + n + (ams ? 1 : 0));
        double b1 = 0.9, b2 = 0.999, om1 = 1 - b1, om2 = 1 - b2, bc1 = 0.271, bc2 = 0.02, lr = 1e-3, eps = 1e-8;
        double[] p = RandD(r, n), g = RandD(r, n), m = RandD(r, n), v = RandD(r, n, true), vm = RandD(r, n, true);

        double[] pR = (double[])p.Clone(), mR = (double[])m.Clone(), vR = (double[])v.Clone(), vmR = (double[])vm.Clone();
        RefStepD(pR, g, mR, vR, vmR, b1, b2, om1, om2, bc1, bc2, lr, eps, ams);

        double[] pK = (double[])p.Clone(), mK = (double[])m.Clone(), vK = (double[])v.Clone(), vmK = (double[])vm.Clone();
        AdamMomentKernels.AdamStep(pK, g, mK, vK, vmK, b1, b2, om1, om2, bc1, bc2, lr, eps, ams);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(BitConverter.DoubleToInt64Bits(pR[i]), BitConverter.DoubleToInt64Bits(pK[i]));
            Assert.Equal(BitConverter.DoubleToInt64Bits(mR[i]), BitConverter.DoubleToInt64Bits(mK[i]));
            Assert.Equal(BitConverter.DoubleToInt64Bits(vR[i]), BitConverter.DoubleToInt64Bits(vK[i]));
            if (ams) Assert.Equal(BitConverter.DoubleToInt64Bits(vmR[i]), BitConverter.DoubleToInt64Bits(vmK[i]));
        }
    }
}
