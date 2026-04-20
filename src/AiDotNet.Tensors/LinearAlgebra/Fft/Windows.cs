// Copyright (c) AiDotNet. All rights reserved.
// Window functions for STFT, spectral analysis, filter design.
// Parameters match torch.signal.windows / scipy.signal.windows.

using System;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Fft;

/// <summary>
/// Canonical window functions (Hann, Hamming, Blackman, …). Each returns a
/// 1D <see cref="Tensor{T}"/> of length <c>n</c> with values in <c>[0, 1]</c>
/// (Gaussian / exponential can lie slightly outside the endpoint convention).
///
/// <para>Every window has a <paramref name="periodic"/> flag:
/// <list type="bullet">
///   <item><term><c>periodic = true</c></term>
///         <description>(default for STFT) divides by <c>N</c> in the phase — produces a
///         window whose endpoints are NOT equal, which is what the overlap-add
///         reconstruction identity expects.</description></item>
///   <item><term><c>periodic = false</c></term>
///         <description>divides by <c>N − 1</c> — produces a symmetric window where
///         <c>w[0] == w[N−1]</c>, appropriate for single-shot filter design.</description></item>
/// </list>
/// This matches torch's <c>periodic</c> argument and scipy's
/// <c>sym=True</c>/<c>sym=False</c> inversion convention.</para>
/// </summary>
public static class Windows
{
    /// <summary>Hann window: <c>w[n] = 0.5 · (1 − cos(2πn / M))</c>.</summary>
    public static Tensor<T> Hann<T>(int length, bool periodic = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => Cosine2Term<T>(length, periodic, a0: 0.5, a1: -0.5);

    /// <summary>Hamming window: <c>w[n] = 0.54 − 0.46 · cos(2πn / M)</c>.</summary>
    public static Tensor<T> Hamming<T>(int length, bool periodic = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => Cosine2Term<T>(length, periodic, a0: 0.54, a1: -0.46);

    /// <summary>
    /// Generalized Hamming with parameter <paramref name="alpha"/>:
    /// <c>w[n] = α − (1 − α) · cos(2πn / M)</c>. <c>α = 0.5</c> is Hann,
    /// <c>α = 0.54</c> is Hamming.
    /// </summary>
    public static Tensor<T> GeneralHamming<T>(int length, double alpha, bool periodic = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => Cosine2Term<T>(length, periodic, a0: alpha, a1: -(1.0 - alpha));

    /// <summary>Blackman window (3-term): <c>w[n] = 0.42 − 0.5·cos(2πn/M) + 0.08·cos(4πn/M)</c>.</summary>
    public static Tensor<T> Blackman<T>(int length, bool periodic = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => GeneralCosine<T>(length, new[] { 0.42, -0.5, 0.08 }, periodic);

    /// <summary>
    /// Nuttall window (4-term, strong sidelobe rejection):
    /// coefficients <c>[0.3635819, −0.4891775, 0.1365995, −0.0106411]</c>.
    /// </summary>
    public static Tensor<T> Nuttall<T>(int length, bool periodic = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => GeneralCosine<T>(length, new[] { 0.3635819, -0.4891775, 0.1365995, -0.0106411 }, periodic);

    /// <summary>
    /// Generalized cosine window: <c>w[n] = Σ_k a_k · cos(2πnk / M)</c>
    /// (<c>cos(0) = 1</c> handles the DC term). Blackman and Nuttall are
    /// specializations.
    /// </summary>
    public static Tensor<T> GeneralCosine<T>(int length, double[] coefficients, bool periodic = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (coefficients is null) throw new ArgumentNullException(nameof(coefficients));
        if (length <= 0) throw new ArgumentException("length must be positive.", nameof(length));
        var result = new Tensor<T>(new[] { length });
        if (length == 1)
        {
            // Sum of coefficients at n = 0 (cos 0 = 1 for every term).
            double sum = 0;
            for (int k = 0; k < coefficients.Length; k++) sum += coefficients[k];
            result[0] = FromDouble<T>(sum);
            return result;
        }
        int M = periodic ? length : length - 1;
        var data = result.GetDataArray();
        for (int i = 0; i < length; i++)
        {
            double v = 0;
            for (int k = 0; k < coefficients.Length; k++)
                v += coefficients[k] * Math.Cos(2.0 * Math.PI * k * i / M);
            data[i] = FromDouble<T>(v);
        }
        return result;
    }

    /// <summary>
    /// Bartlett (triangular) window: linear ramp up to the midpoint and back
    /// down. <c>w[n] = 1 − |2n − M| / M</c>.
    /// </summary>
    public static Tensor<T> Bartlett<T>(int length, bool periodic = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (length <= 0) throw new ArgumentException("length must be positive.", nameof(length));
        var result = new Tensor<T>(new[] { length });
        if (length == 1) { result[0] = FromDouble<T>(1.0); return result; }
        int M = periodic ? length : length - 1;
        var data = result.GetDataArray();
        for (int i = 0; i < length; i++)
        {
            double v = 1.0 - Math.Abs((2.0 * i - M) / M);
            data[i] = FromDouble<T>(v);
        }
        return result;
    }

    /// <summary>
    /// Cosine window (also called sine window):
    /// <c>w[n] = sin(π(n + 0.5) / M)</c>. Matches torch's <c>cosine_window</c>.
    /// </summary>
    public static Tensor<T> Cosine<T>(int length, bool periodic = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (length <= 0) throw new ArgumentException("length must be positive.", nameof(length));
        var result = new Tensor<T>(new[] { length });
        if (length == 1) { result[0] = FromDouble<T>(1.0); return result; }
        int M = periodic ? length : length - 1;
        var data = result.GetDataArray();
        for (int i = 0; i < length; i++)
        {
            double v = Math.Sin(Math.PI * (i + 0.5) / M);
            data[i] = FromDouble<T>(v);
        }
        return result;
    }

    /// <summary>
    /// Gaussian window: <c>w[n] = exp(−½ · ((n − (M/2)) / std)²)</c>. The
    /// <paramref name="std"/> parameter controls the width; smaller values
    /// yield a more concentrated window.
    /// </summary>
    public static Tensor<T> Gaussian<T>(int length, double std, bool periodic = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (length <= 0) throw new ArgumentException("length must be positive.", nameof(length));
        if (std <= 0) throw new ArgumentException("std must be positive.", nameof(std));
        var result = new Tensor<T>(new[] { length });
        if (length == 1) { result[0] = FromDouble<T>(1.0); return result; }
        double mid = (periodic ? length : length - 1) / 2.0;
        var data = result.GetDataArray();
        for (int i = 0; i < length; i++)
        {
            double d = (i - mid) / std;
            double v = Math.Exp(-0.5 * d * d);
            data[i] = FromDouble<T>(v);
        }
        return result;
    }

    /// <summary>
    /// Exponential window: <c>w[n] = exp(−|n − center| / tau)</c>.
    /// <paramref name="center"/> defaults to <c>(M−1)/2</c> (symmetric).
    /// <paramref name="tau"/> controls the decay rate; required to be positive.
    /// </summary>
    public static Tensor<T> Exponential<T>(int length, double? center = null, double tau = 1.0, bool periodic = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (length <= 0) throw new ArgumentException("length must be positive.", nameof(length));
        if (tau <= 0) throw new ArgumentException("tau must be positive.", nameof(tau));
        if (periodic && center.HasValue)
            throw new ArgumentException("Exponential window requires periodic=false when center is specified.");
        var result = new Tensor<T>(new[] { length });
        double c = center ?? (periodic ? length - 1 : length - 1) / 2.0;
        var data = result.GetDataArray();
        for (int i = 0; i < length; i++)
        {
            double v = Math.Exp(-Math.Abs(i - c) / tau);
            data[i] = FromDouble<T>(v);
        }
        return result;
    }

    /// <summary>
    /// Kaiser window: <c>w[n] = I₀(β · √(1 − ((2n − M) / M)²)) / I₀(β)</c>,
    /// where <c>I₀</c> is the modified Bessel function of the first kind, zero
    /// order. <paramref name="beta"/> trades main-lobe width for sidelobe
    /// attenuation — typical values are 5 (−40 dB), 8.6 (−60 dB), 14 (−100 dB).
    /// </summary>
    public static Tensor<T> Kaiser<T>(int length, double beta, bool periodic = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (length <= 0) throw new ArgumentException("length must be positive.", nameof(length));
        var result = new Tensor<T>(new[] { length });
        if (length == 1) { result[0] = FromDouble<T>(1.0); return result; }
        int M = periodic ? length : length - 1;
        double i0Beta = BesselI0(beta);
        var data = result.GetDataArray();
        for (int i = 0; i < length; i++)
        {
            double r = (2.0 * i - M) / M; // −1 .. 1
            double arg = beta * Math.Sqrt(Math.Max(0.0, 1.0 - r * r));
            data[i] = FromDouble<T>(BesselI0(arg) / i0Beta);
        }
        return result;
    }

    // ── Helpers ────────────────────────────────────────────────────────────
    private static Tensor<T> Cosine2Term<T>(int length, bool periodic, double a0, double a1)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => GeneralCosine<T>(length, new[] { a0, a1 }, periodic);

    /// <summary>
    /// Modified Bessel function of the first kind, zero order, for real
    /// arguments. Uses a truncated power series around 0 (converges quickly for
    /// the Kaiser window's input range |x| ≤ β which is typically ≤ 20).
    /// </summary>
    internal static double BesselI0(double x)
    {
        double ax = Math.Abs(x);
        // Power series: I₀(x) = Σ_{k=0}^∞ (x/2)^(2k) / (k!)²
        // Each term ratio is (x²/4) / k², converging quickly for |x| < ~20.
        double term = 1.0;
        double sum = 1.0;
        double xHalfSq = 0.25 * ax * ax;
        for (int k = 1; k < 64; k++)
        {
            term *= xHalfSq / (k * k);
            sum += term;
            if (term < 1e-18 * sum) break;
        }
        return sum;
    }

    private static T FromDouble<T>(double v)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => MathHelper.GetNumericOperations<T>().FromDouble(v);
}
