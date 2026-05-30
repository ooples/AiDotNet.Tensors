using System;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Activations that need per-row (across the feature dimension) or per-column
/// (channel) context rather than a pointwise scalar map. Shared by the
/// MlpForward/FusedLinear bias-activation epilogue (CpuFusedOperations) and the
/// BlasManaged GEMM <c>ActivationEpilogue</c> so both fused paths apply identical
/// math. Operates in place on a row-major matrix view: element (i,j) at
/// <c>buffer[i*ldc + j]</c>, i in [0,m), j in [0,n).
/// </summary>
internal static class RowwiseFusedActivations
{
    /// <summary>True for activations this helper owns (channel/row-wise).</summary>
    internal static bool Handles(FusedActivationType a) => a is
        FusedActivationType.PReLU
        or FusedActivationType.Softmax or FusedActivationType.Softmin
        or FusedActivationType.LogSoftmax or FusedActivationType.LogSoftmin
        or FusedActivationType.SphericalSoftmax or FusedActivationType.TaylorSoftmax
        or FusedActivationType.GumbelSoftmax or FusedActivationType.Sparsemax
        or FusedActivationType.Squash;

    internal static void ApplyFloat(Span<float> c, int ldc, int m, int n, FusedActivationType act, FusedActivationParams? p)
    {
        float[]? scratch = act == FusedActivationType.Sparsemax ? new float[n] : null;
        for (int i = 0; i < m; i++)
        {
            int row = i * ldc;
            switch (act)
            {
                case FusedActivationType.PReLU:
                {
                    var slope = p?.PReluSlope;
                    for (int j = 0; j < n; j++)
                    {
                        float s = slope == null ? 0.25f : (slope.Length == 1 ? slope[0] : slope[j]);
                        float v = c[row + j];
                        c[row + j] = v > 0f ? v : s * v;
                    }
                    break;
                }
                case FusedActivationType.Softmax: SoftmaxF(c, row, n, negate: false); break;
                case FusedActivationType.Softmin: SoftmaxF(c, row, n, negate: true); break;
                case FusedActivationType.GumbelSoftmax:
                {
                    float tau = p?.Alpha ?? 1f;
                    if (tau != 1f) for (int j = 0; j < n; j++) c[row + j] /= tau;
                    SoftmaxF(c, row, n, negate: false);
                    break;
                }
                case FusedActivationType.SphericalSoftmax:
                {
                    // softmax(x / ||x||2); values bounded to [-1,1] so the softmax is stable.
                    double ss = 0; for (int j = 0; j < n; j++) ss += (double)c[row + j] * c[row + j];
                    float norm = (float)Math.Sqrt(ss);
                    if (norm > 0f) for (int j = 0; j < n; j++) c[row + j] /= norm;
                    SoftmaxF(c, row, n, negate: false);
                    break;
                }
                case FusedActivationType.LogSoftmax: LogSoftmaxF(c, row, n, negate: false); break;
                case FusedActivationType.LogSoftmin: LogSoftmaxF(c, row, n, negate: true); break;
                case FusedActivationType.TaylorSoftmax:
                {
                    // T(x)=1+x+x²/2 (2nd-order Taylor of exp); strictly positive (=0.5(x+1)²+0.5).
                    float sum = 0f;
                    for (int j = 0; j < n; j++) { float x = c[row + j]; float t = 1f + x + 0.5f * x * x; c[row + j] = t; sum += t; }
                    float inv = 1f / sum;
                    for (int j = 0; j < n; j++) c[row + j] *= inv;
                    break;
                }
                case FusedActivationType.Squash:
                {
                    double ss = 0; for (int j = 0; j < n; j++) ss += (double)c[row + j] * c[row + j];
                    float normSq = (float)ss; float norm = (float)Math.Sqrt(ss);
                    float k = norm > 0f ? (normSq / (1f + normSq)) / norm : 0f;
                    for (int j = 0; j < n; j++) c[row + j] *= k;
                    break;
                }
                case FusedActivationType.Sparsemax: SparsemaxF(c, row, n, scratch!); break;
                default: throw new NotSupportedException($"RowwiseFusedActivations: {act} not handled.");
            }
        }
    }

    private static void SoftmaxF(Span<float> c, int row, int n, bool negate)
    {
        float mx = float.NegativeInfinity;
        for (int j = 0; j < n; j++) { float v = negate ? -c[row + j] : c[row + j]; if (v > mx) mx = v; }
        float sum = 0f;
        for (int j = 0; j < n; j++) { float v = negate ? -c[row + j] : c[row + j]; float e = MathF.Exp(v - mx); c[row + j] = e; sum += e; }
        float inv = 1f / sum;
        for (int j = 0; j < n; j++) c[row + j] *= inv;
    }

    private static void LogSoftmaxF(Span<float> c, int row, int n, bool negate)
    {
        float mx = float.NegativeInfinity;
        for (int j = 0; j < n; j++) { float v = negate ? -c[row + j] : c[row + j]; if (v > mx) mx = v; }
        float sum = 0f;
        for (int j = 0; j < n; j++) { float v = negate ? -c[row + j] : c[row + j]; sum += MathF.Exp(v - mx); }
        float lse = mx + MathF.Log(sum); // log Σ exp(v)
        for (int j = 0; j < n; j++) { float v = negate ? -c[row + j] : c[row + j]; c[row + j] = v - lse; }
    }

    private static void SparsemaxF(Span<float> c, int row, int n, float[] sorted)
    {
        for (int j = 0; j < n; j++) sorted[j] = c[row + j];
        Array.Sort(sorted, 0, n);                 // ascending
        // walk descending to find support size k and cumulative sum
        float cum = 0f; int k = 1; float cumK = 0f;
        for (int i = 0; i < n; i++)
        {
            float z = sorted[n - 1 - i];          // i-th largest
            cum += z;
            if (1f + (i + 1) * z > cum) { k = i + 1; cumK = cum; }
        }
        float tau = (cumK - 1f) / k;
        for (int j = 0; j < n; j++) { float v = c[row + j] - tau; c[row + j] = v > 0f ? v : 0f; }
    }

    internal static void ApplyDouble(Span<double> c, int ldc, int m, int n, FusedActivationType act, FusedActivationParams? p)
    {
        double[]? scratch = act == FusedActivationType.Sparsemax ? new double[n] : null;
        for (int i = 0; i < m; i++)
        {
            int row = i * ldc;
            switch (act)
            {
                case FusedActivationType.PReLU:
                {
                    var slope = p?.PReluSlope;
                    for (int j = 0; j < n; j++)
                    {
                        double s = slope == null ? 0.25 : (slope.Length == 1 ? slope[0] : slope[j]);
                        double v = c[row + j];
                        c[row + j] = v > 0.0 ? v : s * v;
                    }
                    break;
                }
                case FusedActivationType.Softmax: SoftmaxD(c, row, n, negate: false); break;
                case FusedActivationType.Softmin: SoftmaxD(c, row, n, negate: true); break;
                case FusedActivationType.GumbelSoftmax:
                {
                    double tau = p?.Alpha ?? 1.0;
                    if (tau != 1.0) for (int j = 0; j < n; j++) c[row + j] /= tau;
                    SoftmaxD(c, row, n, negate: false);
                    break;
                }
                case FusedActivationType.SphericalSoftmax:
                {
                    double ss = 0; for (int j = 0; j < n; j++) ss += c[row + j] * c[row + j];
                    double norm = Math.Sqrt(ss);
                    if (norm > 0.0) for (int j = 0; j < n; j++) c[row + j] /= norm;
                    SoftmaxD(c, row, n, negate: false);
                    break;
                }
                case FusedActivationType.LogSoftmax: LogSoftmaxD(c, row, n, negate: false); break;
                case FusedActivationType.LogSoftmin: LogSoftmaxD(c, row, n, negate: true); break;
                case FusedActivationType.TaylorSoftmax:
                {
                    double sum = 0; for (int j = 0; j < n; j++) { double x = c[row + j]; double t = 1.0 + x + 0.5 * x * x; c[row + j] = t; sum += t; }
                    double inv = 1.0 / sum;
                    for (int j = 0; j < n; j++) c[row + j] *= inv;
                    break;
                }
                case FusedActivationType.Squash:
                {
                    double normSq = 0; for (int j = 0; j < n; j++) normSq += c[row + j] * c[row + j];
                    double norm = Math.Sqrt(normSq);
                    double k = norm > 0.0 ? (normSq / (1.0 + normSq)) / norm : 0.0;
                    for (int j = 0; j < n; j++) c[row + j] *= k;
                    break;
                }
                case FusedActivationType.Sparsemax: SparsemaxD(c, row, n, scratch!); break;
                default: throw new NotSupportedException($"RowwiseFusedActivations: {act} not handled.");
            }
        }
    }

    private static void SoftmaxD(Span<double> c, int row, int n, bool negate)
    {
        double mx = double.NegativeInfinity;
        for (int j = 0; j < n; j++) { double v = negate ? -c[row + j] : c[row + j]; if (v > mx) mx = v; }
        double sum = 0; for (int j = 0; j < n; j++) { double v = negate ? -c[row + j] : c[row + j]; double e = Math.Exp(v - mx); c[row + j] = e; sum += e; }
        double inv = 1.0 / sum; for (int j = 0; j < n; j++) c[row + j] *= inv;
    }

    private static void LogSoftmaxD(Span<double> c, int row, int n, bool negate)
    {
        double mx = double.NegativeInfinity;
        for (int j = 0; j < n; j++) { double v = negate ? -c[row + j] : c[row + j]; if (v > mx) mx = v; }
        double sum = 0; for (int j = 0; j < n; j++) { double v = negate ? -c[row + j] : c[row + j]; sum += Math.Exp(v - mx); }
        double lse = mx + Math.Log(sum);
        for (int j = 0; j < n; j++) { double v = negate ? -c[row + j] : c[row + j]; c[row + j] = v - lse; }
    }

    private static void SparsemaxD(Span<double> c, int row, int n, double[] sorted)
    {
        for (int j = 0; j < n; j++) sorted[j] = c[row + j];
        Array.Sort(sorted, 0, n);
        double cum = 0; int k = 1; double cumK = 0;
        for (int i = 0; i < n; i++)
        {
            double z = sorted[n - 1 - i];
            cum += z;
            if (1.0 + (i + 1) * z > cum) { k = i + 1; cumK = cum; }
        }
        double tau = (cumK - 1.0) / k;
        for (int j = 0; j < n; j++) { double v = c[row + j] - tau; c[row + j] = v > 0.0 ? v : 0.0; }
    }
}
