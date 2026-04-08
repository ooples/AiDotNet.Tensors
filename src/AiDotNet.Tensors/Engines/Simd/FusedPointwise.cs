using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Fused pointwise kernel: applies a chain of element-wise operations in a single
/// pass over memory, halving memory bandwidth compared to separate kernels.
///
/// Standard: sigmoid(x) then x*sigmoid(x) = 2 passes over memory (read+write each).
/// Fused: single pass computing sigmoid(x[i]) * x[i] per element.
///
/// This uses delegate composition at compile time to build a single fused function
/// from a chain of pointwise operations, then applies it element-wise.
/// </summary>
internal static class FusedPointwise
{
    /// <summary>
    /// Builds a fused delegate from a chain of pointwise operation names.
    /// The delegate applies all operations in sequence to a single element.
    /// </summary>
    internal static Func<float, float>? BuildFusedDelegate(string[] opChain)
    {
        if (opChain.Length == 0) return null;

        // Build composed function: f(x) = opN(...(op2(op1(x))))
        Func<float, float>? composed = null;
        foreach (var opName in opChain)
        {
            var op = GetPointwiseOp(opName);
            if (op is null) return null; // Can't fuse if any op is unknown

            if (composed is null)
                composed = op;
            else
            {
                var prev = composed;
                var curr = op;
                composed = x => curr(prev(x));
            }
        }

        return composed;
    }

    /// <summary>
    /// Applies a fused pointwise delegate to an entire array in a single pass.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void ApplyFused(
        float* input, float* output, int length, Func<float, float> fusedOp)
    {
        // Process 4 elements at a time for ILP (instruction-level parallelism)
        int i = 0;
        int unrolled = length - 3;
        for (; i < unrolled; i += 4)
        {
            output[i] = fusedOp(input[i]);
            output[i + 1] = fusedOp(input[i + 1]);
            output[i + 2] = fusedOp(input[i + 2]);
            output[i + 3] = fusedOp(input[i + 3]);
        }
        for (; i < length; i++)
            output[i] = fusedOp(input[i]);
    }

    /// <summary>
    /// Applies a fused pointwise delegate in-place.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void ApplyFusedInPlace(float* data, int length, Func<float, float> fusedOp)
    {
        int i = 0;
        int unrolled = length - 3;
        for (; i < unrolled; i += 4)
        {
            data[i] = fusedOp(data[i]);
            data[i + 1] = fusedOp(data[i + 1]);
            data[i + 2] = fusedOp(data[i + 2]);
            data[i + 3] = fusedOp(data[i + 3]);
        }
        for (; i < length; i++)
            data[i] = fusedOp(data[i]);
    }

    /// <summary>Maps operation name to element-wise float function.</summary>
    private static Func<float, float>? GetPointwiseOp(string opName)
    {
        return opName switch
        {
            "ReLU" => x => x > 0f ? x : 0f,
            "Sigmoid" => x => 1f / (1f + MathF.Exp(-x)),
            "Tanh" => MathF.Tanh,
            "GELU" => x =>
            {
                const float sqrt2OverPi = 0.7978845608028654f;
                const float coeff = 0.044715f;
                float xCubed = x * x * x;
                float inner = sqrt2OverPi * (x + coeff * xCubed);
                return 0.5f * x * (1f + MathF.Tanh(inner));
            },
            "Swish" => x => x * (1f / (1f + MathF.Exp(-x))),
            "Mish" => x => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x))),
            "Exp" => MathF.Exp,
            "Log" => MathF.Log,
            "Abs" => MathF.Abs,
            "Sqrt" => MathF.Sqrt,
            "Negate" => x => -x,
            "Sign" => x => MathF.Sign(x),
            "Softplus" => x => x > 20f ? x : MathF.Log(1f + MathF.Exp(x)),
            "HardSwish" => x =>
            {
                float clip = MathHelperClamp((x + 3f) / 6f, 0f, 1f);
                return x * clip;
            },
            "HardSigmoid" => x => MathHelperClamp((x + 3f) / 6f, 0f, 1f),
            "ELU" => x => x >= 0f ? x : MathF.Exp(x) - 1f,
            "SELU" => x =>
            {
                const float alpha = 1.6732632423543772f;
                const float scale = 1.0507009873554805f;
                return scale * (x >= 0f ? x : alpha * (MathF.Exp(x) - 1f));
            },
            "Reciprocal" => x => 1f / x,
            "Floor" => MathF.Floor,
            "Ceiling" => MathF.Ceiling,
            "Round" => x => MathF.Round(x),
            _ => null
        };
    }

    /// <summary>Clamp helper for net471 compatibility (Math.Clamp not available).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float MathHelperClamp(float value, float min, float max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }
}
