// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.NN.Activations;

/// <summary>
/// Activations missing from the engine surface, plus the newer GLU
/// family that powers modern LLMs (SwiGLU, GEGLU, ReGLU). Engine
/// already covers ReLU / Sigmoid / Tanh / GELU / Mish / Softplus /
/// HardSwish / HardSigmoid / ELU / SELU; this file fills in the
/// long-tail PyTorch ships under <c>torch.nn.functional</c>.
/// </summary>
public static class Activations
{
    /// <summary>Hard shrinkage: <c>x</c> when <c>|x| &gt; lambda</c>, else 0.</summary>
    public static Tensor<T> Hardshrink<T>(Tensor<T> input, double lambda = 0.5)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input._shape.Clone());
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
        {
            double v = ops.ToDouble(src[i]);
            dst[i] = Math.Abs(v) > lambda ? src[i] : ops.Zero;
        }
        return output;
    }

    /// <summary>Soft shrinkage: <c>x − sign(x) · lambda</c> when
    /// <c>|x| &gt; lambda</c>, else 0.</summary>
    public static Tensor<T> Softshrink<T>(Tensor<T> input, double lambda = 0.5)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input._shape.Clone());
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
        {
            double v = ops.ToDouble(src[i]);
            double r = v > lambda ? v - lambda
                     : v < -lambda ? v + lambda
                     : 0.0;
            dst[i] = ops.FromDouble(r);
        }
        return output;
    }

    /// <summary>Tanh shrinkage: <c>x − tanh(x)</c>.</summary>
    public static Tensor<T> Tanhshrink<T>(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input._shape.Clone());
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
        {
            double v = ops.ToDouble(src[i]);
            dst[i] = ops.FromDouble(v - Math.Tanh(v));
        }
        return output;
    }

    /// <summary>Threshold: <c>x</c> when <c>x &gt; threshold</c>, else
    /// <paramref name="value"/>.</summary>
    public static Tensor<T> Threshold<T>(Tensor<T> input, double threshold, double value)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input._shape.Clone());
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();
        T fill = ops.FromDouble(value);
        for (int i = 0; i < src.Length; i++)
        {
            double v = ops.ToDouble(src[i]);
            dst[i] = v > threshold ? src[i] : fill;
        }
        return output;
    }

    /// <summary>Quick GELU — <c>x · sigmoid(1.702 · x)</c>. Cheaper
    /// than the standard GELU and used in CLIP / GPT-2.</summary>
    public static Tensor<T> QuickGelu<T>(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input._shape.Clone());
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
        {
            double v = ops.ToDouble(src[i]);
            double sig = 1.0 / (1.0 + Math.Exp(-1.702 * v));
            dst[i] = ops.FromDouble(v * sig);
        }
        return output;
    }

    /// <summary>GELU with the tanh approximation —
    /// <c>0.5 · x · (1 + tanh(sqrt(2/π) · (x + 0.044715 · x³)))</c>.
    /// Mirrors PyTorch's <c>F.gelu(x, approximate="tanh")</c>.</summary>
    public static Tensor<T> GeluTanh<T>(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input._shape.Clone());
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();
        const double C = 0.7978845608028654; // sqrt(2/π)
        for (int i = 0; i < src.Length; i++)
        {
            double v = ops.ToDouble(src[i]);
            double inner = C * (v + 0.044715 * v * v * v);
            dst[i] = ops.FromDouble(0.5 * v * (1.0 + Math.Tanh(inner)));
        }
        return output;
    }

    /// <summary>SwiGLU: <c>silu(x_a) · x_b</c> where the input's last
    /// axis is split in half (a, b). Used in LLaMA, PaLM, Mistral.</summary>
    public static Tensor<T> SwiGLU<T>(Tensor<T> input)
        => GLUFamily(input, GLUKind.SwiGLU);

    /// <summary>GEGLU: <c>gelu(x_a) · x_b</c>. Used in T5-v1.1,
    /// Gemma, GLaM.</summary>
    public static Tensor<T> GEGLU<T>(Tensor<T> input)
        => GLUFamily(input, GLUKind.GEGLU);

    /// <summary>ReGLU: <c>relu(x_a) · x_b</c>. Used in Noam
    /// Shazeer's "GLU Variants Improve Transformer" study.</summary>
    public static Tensor<T> ReGLU<T>(Tensor<T> input)
        => GLUFamily(input, GLUKind.ReGLU);

    private enum GLUKind { SwiGLU, GEGLU, ReGLU }

    private static Tensor<T> GLUFamily<T>(Tensor<T> input, GLUKind kind)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank == 0)
            throw new ArgumentException("GLU family requires rank ≥ 1.", nameof(input));
        int last = input._shape[input.Rank - 1];
        if (last == 0)
            throw new ArgumentException("GLU family requires last dim > 0.", nameof(input));
        if (last % 2 != 0)
            throw new ArgumentException(
                $"GLU family requires last dim be even; got {last}.", nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        int half = last / 2;
        var newShape = (int[])input._shape.Clone();
        newShape[input.Rank - 1] = half;
        var output = new Tensor<T>(newShape);

        int outer = 1;
        for (int d = 0; d < input.Rank - 1; d++) outer *= input._shape[d];
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();

        for (int o = 0; o < outer; o++)
        {
            for (int j = 0; j < half; j++)
            {
                double a = ops.ToDouble(src[o * last + j]);
                double b = ops.ToDouble(src[o * last + half + j]);
                double act = kind switch
                {
                    GLUKind.SwiGLU => a / (1.0 + Math.Exp(-a)),
                    GLUKind.GEGLU => 0.5 * a * (1.0 + Math.Tanh(0.7978845608028654 * (a + 0.044715 * a * a * a))),
                    GLUKind.ReGLU => Math.Max(0, a),
                    _ => a,
                };
                dst[o * half + j] = ops.FromDouble(act * b);
            }
        }
        return output;
    }
}
