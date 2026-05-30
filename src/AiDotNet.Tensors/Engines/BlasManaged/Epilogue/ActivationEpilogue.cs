using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Applies an element-wise activation to each cell of the C output matrix.
/// Supports the activations enumerated in <see cref="FusedActivationType"/>.
/// </summary>
internal static class ActivationEpilogue
{
    public static void Apply<T>(Span<T> c, int ldc, int m, int n, FusedActivationType activation, FusedActivationParams? p = null)
        where T : unmanaged
    {
        if (activation == FusedActivationType.None) return;

        if (typeof(T) == typeof(double))
        {
            var cd = MemoryMarshal.Cast<T, double>(c);
            ApplyFp64(cd, ldc, m, n, activation, p);
            return;
        }
        if (typeof(T) == typeof(float))
        {
            var cf = MemoryMarshal.Cast<T, float>(c);
            ApplyFp32(cf, ldc, m, n, activation, p);
            return;
        }
        throw new NotSupportedException($"ActivationEpilogue does not support T={typeof(T).Name}.");
    }

    private static void ApplyFp64(Span<double> c, int ldc, int m, int n, FusedActivationType activation, FusedActivationParams? p = null)
    {
        switch (activation)
        {
            case FusedActivationType.ReLU:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double v = c[i * ldc + j];
                        if (v < 0) c[i * ldc + j] = 0;
                    }
                break;
            case FusedActivationType.LeakyReLU:
            {
                double slope = p?.Alpha ?? 0.01; // default leaky slope 0.01
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double v = c[i * ldc + j];
                        c[i * ldc + j] = v >= 0 ? v : slope * v;
                    }
                break;
            }
            case FusedActivationType.Sigmoid:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double v = c[i * ldc + j];
                        c[i * ldc + j] = 1.0 / (1.0 + Math.Exp(-v));
                    }
                break;
            case FusedActivationType.Tanh:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        c[i * ldc + j] = Math.Tanh(c[i * ldc + j]);
                break;
            case FusedActivationType.GELU:
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                const double sqrt2OverPi = 0.7978845608028654;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        double inner = sqrt2OverPi * (x + 0.044715 * x * x * x);
                        c[i * ldc + j] = 0.5 * x * (1.0 + Math.Tanh(inner));
                    }
                break;
            case FusedActivationType.Swish:
                // f(x) = x * sigmoid(x)
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        c[i * ldc + j] = x / (1.0 + Math.Exp(-x));
                    }
                break;
            case FusedActivationType.Mish:
                // f(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        double sp = Math.Log(1.0 + Math.Exp(x));
                        c[i * ldc + j] = x * Math.Tanh(sp);
                    }
                break;
            case FusedActivationType.ELU:
            {
                // f(x) = x if x > 0 else alpha * (exp(x) - 1); alpha default 1.
                double a = p?.Alpha ?? 1.0;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        c[i * ldc + j] = x > 0 ? x : a * (Math.Exp(x) - 1.0);
                    }
                break;
            }
            case FusedActivationType.CELU:
            {
                // max(0,x)+min(0,a*(exp(x/a)-1)) → piecewise: x>=0 ? x : a*(exp(x/a)-1).
                double a = p?.Alpha ?? 1.0;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        c[i * ldc + j] = x >= 0 ? x : a * (Math.Exp(x / a) - 1.0);
                    }
                break;
            }
            case FusedActivationType.ThresholdedReLU:
            {
                double t = p?.Theta ?? 1.0;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        c[i * ldc + j] = x > t ? x : 0.0;
                    }
                break;
            }
            case FusedActivationType.ScaledTanh:
            {
                double a = p?.Alpha ?? 1.0, b = p?.Beta ?? 1.0;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        c[i * ldc + j] = a * Math.Tanh(b * c[i * ldc + j]);
                break;
            }
            case FusedActivationType.SELU:
                // scale * (x>0 ? x : alpha*(exp(x)-1)); Klambauer et al. 2017 constants.
                const double seluAlpha = 1.6732632423543772, seluScale = 1.0507009873554805;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        c[i * ldc + j] = seluScale * (x > 0 ? x : seluAlpha * (Math.Exp(x) - 1.0));
                    }
                break;
            case FusedActivationType.Softplus:
                // log(1+exp(x)); x>20 linear cutoff avoids exp overflow (PyTorch threshold).
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        c[i * ldc + j] = x > 20.0 ? x : Math.Log(1.0 + Math.Exp(x));
                    }
                break;
            case FusedActivationType.HardSwish:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        double t = (x + 3.0) / 6.0;
                        t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
                        c[i * ldc + j] = x * t;
                    }
                break;
            case FusedActivationType.HardSigmoid:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double t = (c[i * ldc + j] + 3.0) / 6.0;
                        c[i * ldc + j] = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
                    }
                break;
            case FusedActivationType.HardTanh:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        c[i * ldc + j] = x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x);
                    }
                break;
            case FusedActivationType.ReLU6:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        c[i * ldc + j] = x < 0.0 ? 0.0 : (x > 6.0 ? 6.0 : x);
                    }
                break;
            case FusedActivationType.SoftSign:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        c[i * ldc + j] = x / (1.0 + Math.Abs(x));
                    }
                break;
            case FusedActivationType.None:
                break;
            default:
                throw new NotSupportedException($"ActivationEpilogue: {activation} not yet implemented.");
        }
    }

    private static void ApplyFp32(Span<float> c, int ldc, int m, int n, FusedActivationType activation, FusedActivationParams? p = null)
    {
        switch (activation)
        {
            case FusedActivationType.ReLU:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float v = c[i * ldc + j];
                        if (v < 0) c[i * ldc + j] = 0;
                    }
                break;
            case FusedActivationType.LeakyReLU:
            {
                float slope = p?.Alpha ?? 0.01f;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float v = c[i * ldc + j];
                        c[i * ldc + j] = v >= 0 ? v : slope * v;
                    }
                break;
            }
            case FusedActivationType.Sigmoid:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float v = c[i * ldc + j];
                        c[i * ldc + j] = (float)(1.0 / (1.0 + Math.Exp(-v)));
                    }
                break;
            case FusedActivationType.Tanh:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        c[i * ldc + j] = (float)Math.Tanh(c[i * ldc + j]);
                break;
            case FusedActivationType.GELU:
                const float sqrt2OverPiF = 0.79788458f;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        float inner = sqrt2OverPiF * (x + 0.044715f * x * x * x);
                        c[i * ldc + j] = 0.5f * x * (1.0f + (float)Math.Tanh(inner));
                    }
                break;
            case FusedActivationType.Swish:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        c[i * ldc + j] = (float)(x / (1.0 + Math.Exp(-x)));
                    }
                break;
            case FusedActivationType.Mish:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        float sp = (float)Math.Log(1.0 + Math.Exp(x));
                        c[i * ldc + j] = x * (float)Math.Tanh(sp);
                    }
                break;
            case FusedActivationType.ELU:
            {
                float a = p?.Alpha ?? 1f;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        c[i * ldc + j] = x > 0 ? x : a * (float)(Math.Exp(x) - 1.0);
                    }
                break;
            }
            case FusedActivationType.CELU:
            {
                float a = p?.Alpha ?? 1f;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        c[i * ldc + j] = x >= 0 ? x : a * (float)(Math.Exp(x / a) - 1.0);
                    }
                break;
            }
            case FusedActivationType.ThresholdedReLU:
            {
                float t = p?.Theta ?? 1f;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        c[i * ldc + j] = x > t ? x : 0f;
                    }
                break;
            }
            case FusedActivationType.ScaledTanh:
            {
                float a = p?.Alpha ?? 1f, b = p?.Beta ?? 1f;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        c[i * ldc + j] = a * (float)Math.Tanh(b * c[i * ldc + j]);
                break;
            }
            case FusedActivationType.SELU:
                const float seluAlphaF = 1.6732632f, seluScaleF = 1.0507010f;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        c[i * ldc + j] = seluScaleF * (x > 0 ? x : seluAlphaF * (float)(Math.Exp(x) - 1.0));
                    }
                break;
            case FusedActivationType.Softplus:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        c[i * ldc + j] = x > 20f ? x : (float)Math.Log(1.0 + Math.Exp(x));
                    }
                break;
            case FusedActivationType.HardSwish:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        float t = (x + 3f) / 6f;
                        t = t < 0f ? 0f : (t > 1f ? 1f : t);
                        c[i * ldc + j] = x * t;
                    }
                break;
            case FusedActivationType.HardSigmoid:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float t = (c[i * ldc + j] + 3f) / 6f;
                        c[i * ldc + j] = t < 0f ? 0f : (t > 1f ? 1f : t);
                    }
                break;
            case FusedActivationType.HardTanh:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        c[i * ldc + j] = x < -1f ? -1f : (x > 1f ? 1f : x);
                    }
                break;
            case FusedActivationType.ReLU6:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        c[i * ldc + j] = x < 0f ? 0f : (x > 6f ? 6f : x);
                    }
                break;
            case FusedActivationType.SoftSign:
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        c[i * ldc + j] = x / (1f + Math.Abs(x));
                    }
                break;
            case FusedActivationType.None:
                break;
            default:
                throw new NotSupportedException($"ActivationEpilogue: {activation} not yet implemented.");
        }
    }
}
