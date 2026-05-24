using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Applies an element-wise activation to each cell of the C output matrix.
/// Supports the activations enumerated in <see cref="FusedActivationType"/>.
/// </summary>
internal static class ActivationEpilogue
{
    public static void Apply<T>(Span<T> c, int ldc, int m, int n, FusedActivationType activation)
        where T : unmanaged
    {
        if (activation == FusedActivationType.None) return;

        if (typeof(T) == typeof(double))
        {
            var cd = MemoryMarshal.Cast<T, double>(c);
            ApplyFp64(cd, ldc, m, n, activation);
            return;
        }
        if (typeof(T) == typeof(float))
        {
            var cf = MemoryMarshal.Cast<T, float>(c);
            ApplyFp32(cf, ldc, m, n, activation);
            return;
        }
        throw new NotSupportedException($"ActivationEpilogue does not support T={typeof(T).Name}.");
    }

    private static void ApplyFp64(Span<double> c, int ldc, int m, int n, FusedActivationType activation)
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
                // Default leaky slope = 0.01.
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double v = c[i * ldc + j];
                        c[i * ldc + j] = v >= 0 ? v : 0.01 * v;
                    }
                break;
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
                // f(x) = x if x > 0 else alpha * (exp(x) - 1); alpha = 1.
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double x = c[i * ldc + j];
                        c[i * ldc + j] = x > 0 ? x : (Math.Exp(x) - 1.0);
                    }
                break;
            case FusedActivationType.None:
                break;
            default:
                throw new NotSupportedException($"ActivationEpilogue: {activation} not yet implemented.");
        }
    }

    private static void ApplyFp32(Span<float> c, int ldc, int m, int n, FusedActivationType activation)
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
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float v = c[i * ldc + j];
                        c[i * ldc + j] = v >= 0 ? v : 0.01f * v;
                    }
                break;
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
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float x = c[i * ldc + j];
                        c[i * ldc + j] = x > 0 ? x : (float)(Math.Exp(x) - 1.0);
                    }
                break;
            case FusedActivationType.None:
                break;
            default:
                throw new NotSupportedException($"ActivationEpilogue: {activation} not yet implemented.");
        }
    }
}
