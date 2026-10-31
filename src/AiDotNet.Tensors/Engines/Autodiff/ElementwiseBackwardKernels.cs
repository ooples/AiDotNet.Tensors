using System;
using System.Numerics;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Typed host kernels for elementwise backward functions whose generic form went through a per-element
/// <c>GetFlat</c>/<c>SetFlat</c> plus a virtual <see cref="Interfaces.INumericOperations{T}"/> call.
/// </summary>
/// <remarks>
/// <para>Every kernel here performs exactly the arithmetic of the generic form it replaces, in the same operation
/// order and in the same precision (float in float, double in double), so its results are bit-identical. The vector
/// loops use only lane-wise IEEE operations (compare, select, add, multiply, divide), which round exactly like their
/// scalar counterparts; no fused multiply-add is introduced.</para>
/// </remarks>
internal static class ElementwiseBackwardKernels
{
    /// <summary>
    /// ELU backward over <c>[start, end)</c>: <c>v = g * (y &gt; 0 ? 1 : y + alpha)</c>, stored into <paramref name="d"/>
    /// when <paramref name="overwrite"/> and added to it (<c>d + v</c>) otherwise. <c>y</c> is the ELU output.
    /// </summary>
    internal static void EluBackward(float[] g, int gOff, float[] y, int yOff, float[] d, int dOff,
        int start, int end, float alpha, bool overwrite)
    {
        int i = start;
        if (Vector.IsHardwareAccelerated)
        {
            int w = Vector<float>.Count;
            var vAlpha = new Vector<float>(alpha);
            var one = Vector<float>.One;
            var zero = Vector<float>.Zero;
            for (; i + w <= end; i += w)
            {
                var yv = new Vector<float>(y, yOff + i);
                var derivative = Vector.ConditionalSelect(Vector.GreaterThan(yv, zero), one, yv + vAlpha);
                var v = new Vector<float>(g, gOff + i) * derivative;
                if (!overwrite) v = new Vector<float>(d, dOff + i) + v;
                v.CopyTo(d, dOff + i);
            }
        }
        for (; i < end; i++)
        {
            float yi = y[yOff + i];
            float v = g[gOff + i] * (yi > 0f ? 1f : yi + alpha);
            d[dOff + i] = overwrite ? v : d[dOff + i] + v;
        }
    }

    /// <inheritdoc cref="EluBackward(float[], int, float[], int, float[], int, int, int, float, bool)"/>
    internal static void EluBackward(double[] g, int gOff, double[] y, int yOff, double[] d, int dOff,
        int start, int end, double alpha, bool overwrite)
    {
        int i = start;
        if (Vector.IsHardwareAccelerated)
        {
            int w = Vector<double>.Count;
            var vAlpha = new Vector<double>(alpha);
            var one = Vector<double>.One;
            var zero = Vector<double>.Zero;
            for (; i + w <= end; i += w)
            {
                var yv = new Vector<double>(y, yOff + i);
                var derivative = Vector.ConditionalSelect(Vector.GreaterThan(yv, zero), one, yv + vAlpha);
                var v = new Vector<double>(g, gOff + i) * derivative;
                if (!overwrite) v = new Vector<double>(d, dOff + i) + v;
                v.CopyTo(d, dOff + i);
            }
        }
        for (; i < end; i++)
        {
            double yi = y[yOff + i];
            double v = g[gOff + i] * (yi > 0d ? 1d : yi + alpha);
            d[dOff + i] = overwrite ? v : d[dOff + i] + v;
        }
    }

    /// <summary>
    /// The engine-op ELU backward (CpuEngine.EluBackward) for float: the derivative is chosen on the INPUT
    /// (<c>x &gt; 0</c>) and evaluated in double, <c>(float)((double)g * (x &gt; 0 ? 1.0 : (double)y + alpha))</c>.
    /// </summary>
    internal static void EluBackwardFromInput(ReadOnlySpan<float> g, ReadOnlySpan<float> x, ReadOnlySpan<float> y,
        Span<float> result, double alpha)
    {
        for (int i = 0; i < result.Length; i++)
        {
            double xi = x[i];
            double derivative = xi > 0 ? 1.0 : (double)y[i] + alpha;
            result[i] = (float)((double)g[i] * derivative);
        }
    }

    /// <inheritdoc cref="EluBackwardFromInput(ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float}, double)"/>
    internal static void EluBackwardFromInput(ReadOnlySpan<double> g, ReadOnlySpan<double> x, ReadOnlySpan<double> y,
        Span<double> result, double alpha)
    {
        for (int i = 0; i < result.Length; i++)
        {
            double derivative = x[i] > 0 ? 1.0 : y[i] + alpha;
            result[i] = g[i] * derivative;
        }
    }

    /// <summary>
    /// Complex multiply backward over interleaved (re, im) pairs: <c>gradA = g * conj(b)</c> and
    /// <c>gradB = conj(a) * g</c>, each product rounded before the add/subtract as in the generic form.
    /// </summary>
    internal static void ComplexMultiplyBackward(ReadOnlySpan<float> g, ReadOnlySpan<float> a, ReadOnlySpan<float> b,
        Span<float> gradA, Span<float> gradB, int pairs)
    {
        for (int p = 0; p < pairs; p++)
        {
            int idx = p * 2;
            float gRe = g[idx], gIm = g[idx + 1];
            float bRe = b[idx], bIm = b[idx + 1];
            gradA[idx] = gRe * bRe + gIm * bIm;
            gradA[idx + 1] = gIm * bRe - gRe * bIm;
        }
        for (int p = 0; p < pairs; p++)
        {
            int idx = p * 2;
            float gRe = g[idx], gIm = g[idx + 1];
            float aRe = a[idx], aIm = a[idx + 1];
            gradB[idx] = aRe * gRe + aIm * gIm;
            gradB[idx + 1] = aRe * gIm - aIm * gRe;
        }
    }

    /// <inheritdoc cref="ComplexMultiplyBackward(ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float}, Span{float}, int)"/>
    internal static void ComplexMultiplyBackward(ReadOnlySpan<double> g, ReadOnlySpan<double> a, ReadOnlySpan<double> b,
        Span<double> gradA, Span<double> gradB, int pairs)
    {
        for (int p = 0; p < pairs; p++)
        {
            int idx = p * 2;
            double gRe = g[idx], gIm = g[idx + 1];
            double bRe = b[idx], bIm = b[idx + 1];
            gradA[idx] = gRe * bRe + gIm * bIm;
            gradA[idx + 1] = gIm * bRe - gRe * bIm;
        }
        for (int p = 0; p < pairs; p++)
        {
            int idx = p * 2;
            double gRe = g[idx], gIm = g[idx + 1];
            double aRe = a[idx], aIm = a[idx + 1];
            gradB[idx] = aRe * gRe + aIm * gIm;
            gradB[idx + 1] = aRe * gIm - aIm * gRe;
        }
    }

    /// <summary>
    /// Complex magnitude backward: <c>gradA[re] = g * (re / (|z| + eps))</c>, <c>gradA[im] = g * (im / (|z| + eps))</c>,
    /// with <paramref name="eps"/> already converted to the element type.
    /// </summary>
    internal static void ComplexMagnitudeBackward(ReadOnlySpan<float> g, ReadOnlySpan<float> a, ReadOnlySpan<float> magnitude,
        Span<float> gradA, int pairs, float eps)
    {
        for (int p = 0; p < pairs; p++)
        {
            int idx = p * 2;
            float safeMag = magnitude[p] + eps;
            float gOut = g[p];
            gradA[idx] = gOut * (a[idx] / safeMag);
            gradA[idx + 1] = gOut * (a[idx + 1] / safeMag);
        }
    }

    /// <inheritdoc cref="ComplexMagnitudeBackward(ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float}, int, float)"/>
    internal static void ComplexMagnitudeBackward(ReadOnlySpan<double> g, ReadOnlySpan<double> a, ReadOnlySpan<double> magnitude,
        Span<double> gradA, int pairs, double eps)
    {
        for (int p = 0; p < pairs; p++)
        {
            int idx = p * 2;
            double safeMag = magnitude[p] + eps;
            double gOut = g[p];
            gradA[idx] = gOut * (a[idx] / safeMag);
            gradA[idx + 1] = gOut * (a[idx + 1] / safeMag);
        }
    }
}
