// Copyright (c) AiDotNet. All rights reserved.
// Metal Shading Language (MSL) compute kernels for GPU-accelerated tensor operations.
// These kernels are compiled at runtime and executed on Apple Silicon GPUs.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Contains Metal Shading Language (MSL) compute kernel source code for GPU operations.
/// </summary>
/// <remarks>
/// <para><b>Architecture:</b></para>
/// <para>
/// Metal compute kernels are written in MSL (Metal Shading Language), which is based on C++14.
/// Each kernel is a function marked with the 'kernel' attribute that runs in parallel across
/// GPU threads organized into threadgroups.
/// </para>
/// <para><b>Memory Model:</b></para>
/// <list type="bullet">
/// <item><b>device</b> - Global GPU memory, accessible by all threads</item>
/// <item><b>threadgroup</b> - Shared memory within a threadgroup (32KB typical)</item>
/// <item><b>thread</b> - Per-thread private memory (registers)</item>
/// </list>
/// <para><b>Threading Model:</b></para>
/// <list type="bullet">
/// <item><b>thread_position_in_grid</b> - Global thread ID</item>
/// <item><b>thread_position_in_threadgroup</b> - Local thread ID within threadgroup</item>
/// <item><b>threadgroup_position_in_grid</b> - Threadgroup ID</item>
/// <item><b>threads_per_threadgroup</b> - Threadgroup dimensions</item>
/// </list>
/// </remarks>
public static class MetalKernels
{
    #region Common Header

    /// <summary>
    /// Common includes and type definitions used across all kernels.
    /// </summary>
    private const string CommonHeader = @"
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// Constants
constant float EPSILON = 1e-7f;
constant float PI = 3.14159265358979323846f;
constant float SQRT_2_OVER_PI = 0.7978845608028654f;
constant float GELU_COEF = 0.044715f;

// Helper functions
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float safe_divide(float a, float b) {
    return a / (b + EPSILON);
}

inline float clamp_value(float x, float min_val, float max_val) {
    return max(min_val, min(max_val, x));
}
";

    #endregion

    #region Element-wise Operations

    /// <summary>
    /// Element-wise operation kernels: add, subtract, multiply, divide, etc.
    /// </summary>
    public const string ElementWiseKernels = CommonHeader + @"

// Vector addition: C = A + B
kernel void add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = A[gid] + B[gid];
    }
}

// Vector subtraction: C = A - B
kernel void subtract(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = A[gid] - B[gid];
    }
}

// Element-wise multiplication: C = A * B
kernel void multiply(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = A[gid] * B[gid];
    }
}

// Element-wise division: C = A / B
kernel void divide(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = safe_divide(A[gid], B[gid]);
    }
}

// Element-wise minimum: C = min(A, B)
kernel void minimum(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = min(A[gid], B[gid]);
    }
}

// Element-wise maximum: C = max(A, B)
kernel void maximum(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = max(A[gid], B[gid]);
    }
}

// Scalar multiplication: B = A * scalar
kernel void multiply_scalar(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = A[gid] * scalar;
    }
}

// Scalar addition: B = A + scalar
kernel void add_scalar(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = A[gid] + scalar;
    }
}

// Power: B = A^power
kernel void pow_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& power [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = pow(A[gid], power);
    }
}

// Absolute value: B = |A|
kernel void abs_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = abs(A[gid]);
    }
}

// Exponential: B = exp(A)
kernel void exp_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = exp(A[gid]);
    }
}

// Natural logarithm: B = log(A)
kernel void log_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = log(A[gid] + EPSILON);
    }
}

// Square root: B = sqrt(A)
kernel void sqrt_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = sqrt(A[gid]);
    }
}

// Reciprocal: B = 1/A
kernel void reciprocal_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = safe_divide(1.0f, A[gid]);
    }
}

// Negate: B = -A
kernel void negate_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = -A[gid];
    }
}

// Square: B = A^2
kernel void square_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float val = A[gid];
        B[gid] = val * val;
    }
}

// Copy buffer
kernel void copy_buffer(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        dst[gid] = src[gid];
    }
}

// Fill buffer with constant value
kernel void fill_buffer(
    device float* buffer [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        buffer[gid] = value;
    }
}

// Fused multiply-add: D = A * B + C
kernel void fma_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* C [[buffer(2)]],
    device float* D [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        D[gid] = fma(A[gid], B[gid], C[gid]);
    }
}

// Clamp values: B = clamp(A, min, max)
kernel void clamp_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& min_val [[buffer(2)]],
    constant float& max_val [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = clamp_value(A[gid], min_val, max_val);
    }
}

// Broadcast add along last axis: C[i,j] = A[i,j] + B[j]
kernel void broadcast_add_last(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.y;
    uint j = gid.x;
    if (i < outer_size && j < inner_size) {
        uint idx = i * inner_size + j;
        C[idx] = A[idx] + B[j];
    }
}

// Broadcast multiply along last axis: C[i,j] = A[i,j] * B[j]
kernel void broadcast_mul_last(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.y;
    uint j = gid.x;
    if (i < outer_size && j < inner_size) {
        uint idx = i * inner_size + j;
        C[idx] = A[idx] * B[j];
    }
}

// Broadcast multiply along first axis: C[i,j] = A[i,j] * B[i]
kernel void broadcast_mul_first(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.y;
    uint j = gid.x;
    if (i < outer_size && j < inner_size) {
        uint idx = i * inner_size + j;
        C[idx] = A[idx] * B[i];
    }
}

// Broadcast add along first axis: C[i,j] = A[i,j] + B[i]
kernel void broadcast_add_first(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.y;
    uint j = gid.x;
    if (i < outer_size && j < inner_size) {
        uint idx = i * inner_size + j;
        C[idx] = A[idx] + B[i];
    }
}

// Fused linear interpolation: output = a + t * (b - a)
kernel void lerp_fused(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant float& t [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = fma(t, B[gid] - A[gid], A[gid]);
    }
}

// Fused scaled addition: output = scaleA * a + scaleB * b
kernel void add_scaled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant float& scaleA [[buffer(3)]],
    constant float& scaleB [[buffer(4)]],
    constant uint& size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = fma(scaleA, A[gid], scaleB * B[gid]);
    }
}
";

    #endregion

    #region Activation Functions

    /// <summary>
    /// Neural network activation function kernels.
    /// </summary>
    public const string ActivationKernels = CommonHeader + @"

// ReLU activation: B = max(0, A)
kernel void relu(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = max(0.0f, A[gid]);
    }
}

// ReLU backward: gradInput = gradOutput * (input > 0)
kernel void relu_backward(
    device const float* gradOutput [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        gradInput[gid] = input[gid] > 0.0f ? gradOutput[gid] : 0.0f;
    }
}

// Leaky ReLU: B = x > 0 ? x : alpha * x
kernel void leaky_relu(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = A[gid];
        B[gid] = x > 0.0f ? x : alpha * x;
    }
}

// Leaky ReLU backward
kernel void leaky_relu_backward(
    device const float* gradOutput [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        gradInput[gid] = input[gid] > 0.0f ? gradOutput[gid] : alpha * gradOutput[gid];
    }
}

// Sigmoid activation: B = 1 / (1 + exp(-A))
kernel void sigmoid(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = A[gid];
        // Numerically stable sigmoid
        if (x >= 0.0f) {
            float exp_neg_x = exp(-x);
            B[gid] = 1.0f / (1.0f + exp_neg_x);
        } else {
            float exp_x = exp(x);
            B[gid] = exp_x / (1.0f + exp_x);
        }
    }
}

// Sigmoid backward: gradInput = gradOutput * sigmoid * (1 - sigmoid)
kernel void sigmoid_backward(
    device const float* gradOutput [[buffer(0)]],
    device const float* output [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float s = output[gid];
        gradInput[gid] = gradOutput[gid] * s * (1.0f - s);
    }
}

// Tanh activation
kernel void tanh_activation(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = tanh(A[gid]);
    }
}

// Tanh backward: gradInput = gradOutput * (1 - tanh^2)
kernel void tanh_backward(
    device const float* gradOutput [[buffer(0)]],
    device const float* output [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float t = output[gid];
        gradInput[gid] = gradOutput[gid] * (1.0f - t * t);
    }
}

// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = A[gid];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        B[gid] = 0.5f * x * (1.0f + tanh(inner));
    }
}

// GELU backward (approximate)
kernel void gelu_backward(
    device const float* gradOutput [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = input[gid];
        float x2 = x * x;
        float x3 = x2 * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        float tanh_inner = tanh(inner);
        float sech2 = 1.0f - tanh_inner * tanh_inner;
        float inner_deriv = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x2);
        float grad = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * inner_deriv;
        gradInput[gid] = gradOutput[gid] * grad;
    }
}

// SiLU/Swish activation: x * sigmoid(x)
kernel void silu(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = A[gid];
        float sig = 1.0f / (1.0f + exp(-x));
        B[gid] = x * sig;
    }
}

// SiLU backward
kernel void silu_backward(
    device const float* gradOutput [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = input[gid];
        float sig = 1.0f / (1.0f + exp(-x));
        float grad = sig * (1.0f + x * (1.0f - sig));
        gradInput[gid] = gradOutput[gid] * grad;
    }
}

// ELU activation: x > 0 ? x : alpha * (exp(x) - 1)
kernel void elu(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = A[gid];
        B[gid] = x > 0.0f ? x : alpha * (exp(x) - 1.0f);
    }
}

// SELU activation: scale * (x > 0 ? x : alpha * (exp(x) - 1))
kernel void selu(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        const float alpha = 1.6732632423543772f;
        const float scale = 1.0507009873554805f;
        float x = A[gid];
        B[gid] = scale * (x > 0.0f ? x : alpha * (exp(x) - 1.0f));
    }
}

// Softplus: log(1 + exp(x))
kernel void softplus(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& beta [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = A[gid] * beta;
        // Numerically stable softplus
        if (x > 20.0f) {
            B[gid] = A[gid];
        } else if (x < -20.0f) {
            B[gid] = 0.0f;
        } else {
            B[gid] = log(1.0f + exp(x)) / beta;
        }
    }
}

// Mish: x * tanh(softplus(x))
kernel void mish(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = A[gid];
        float sp = log(1.0f + exp(x));
        B[gid] = x * tanh(sp);
    }
}

// Hardswish: x * min(max(x + 3, 0), 6) / 6
kernel void hardswish(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = A[gid];
        float relu6 = min(max(x + 3.0f, 0.0f), 6.0f);
        B[gid] = x * relu6 / 6.0f;
    }
}

// Hardsigmoid: min(max(x + 3, 0), 6) / 6
kernel void hardsigmoid(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = A[gid];
        B[gid] = min(max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    }
}

// PReLU: x > 0 ? x : weight * x (per-channel weights)
kernel void prelu(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& spatial_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        uint channel = (gid / spatial_size) % channels;
        float x = input[gid];
        float w = weight[channel];
        output[gid] = x > 0.0f ? x : w * x;
    }
}
";

    #endregion

    #region Trigonometric Functions

    /// <summary>
    /// Trigonometric and hyperbolic function kernels.
    /// </summary>
    public const string TrigKernels = CommonHeader + @"

// Sine
kernel void sin_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = sin(A[gid]);
    }
}

// Cosine
kernel void cos_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = cos(A[gid]);
    }
}

// Tangent
kernel void tan_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = tan(A[gid]);
    }
}

// Arc sine
kernel void asin_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = asin(clamp_value(A[gid], -1.0f, 1.0f));
    }
}

// Arc cosine
kernel void acos_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = acos(clamp_value(A[gid], -1.0f, 1.0f));
    }
}

// Arc tangent
kernel void atan_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = atan(A[gid]);
    }
}

// Hyperbolic sine
kernel void sinh_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = sinh(A[gid]);
    }
}

// Hyperbolic cosine
kernel void cosh_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = cosh(A[gid]);
    }
}

// Hyperbolic tangent (already defined in activations as tanh_activation)

// Inverse hyperbolic sine
kernel void asinh_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = A[gid];
        B[gid] = log(x + sqrt(x * x + 1.0f));
    }
}

// Inverse hyperbolic cosine
kernel void acosh_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = max(A[gid], 1.0f);
        B[gid] = log(x + sqrt(x * x - 1.0f));
    }
}

// Inverse hyperbolic tangent
kernel void atanh_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = clamp_value(A[gid], -0.9999999f, 0.9999999f);
        B[gid] = 0.5f * log((1.0f + x) / (1.0f - x));
    }
}

// Floor
kernel void floor_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = floor(A[gid]);
    }
}

// Ceiling
kernel void ceil_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = ceil(A[gid]);
    }
}

// Round
kernel void round_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = round(A[gid]);
    }
}

// Sign
kernel void sign_kernel(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = A[gid];
        B[gid] = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }
}
";

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Reduction operation kernels using parallel reduction algorithm.
    /// </summary>
    public const string ReductionKernels = CommonHeader + @"

// Threadgroup size for reductions (should match dispatch)
constant uint REDUCTION_THREADGROUP_SIZE = 256;

// Parallel sum reduction within threadgroup
kernel void sum_reduce(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]])
{
    // Load data to shared memory
    float sum = 0.0f;
    for (uint i = gid; i < size; i += group_size * 256) {
        sum += input[i];
    }
    shared[lid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in shared memory
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread writes result
    if (lid == 0) {
        output[group_id] = shared[0];
    }
}

// Parallel max reduction
kernel void max_reduce(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]])
{
    float max_val = -INFINITY;
    for (uint i = gid; i < size; i += group_size * 256) {
        max_val = max(max_val, input[i]);
    }
    shared[lid] = max_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[group_id] = shared[0];
    }
}

// Parallel min reduction
kernel void min_reduce(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]])
{
    float min_val = INFINITY;
    for (uint i = gid; i < size; i += group_size * 256) {
        min_val = min(min_val, input[i]);
    }
    shared[lid] = min_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared[lid] = min(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[group_id] = shared[0];
    }
}

// Sum along axis (row-wise)
kernel void sum_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < outer_size) {
        float sum = 0.0f;
        uint offset = gid * reduce_size;
        for (uint i = 0; i < reduce_size; i++) {
            sum += input[offset + i];
        }
        output[gid] = sum;
    }
}

// Mean along axis
kernel void mean_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < outer_size) {
        float sum = 0.0f;
        uint offset = gid * reduce_size;
        for (uint i = 0; i < reduce_size; i++) {
            sum += input[offset + i];
        }
        output[gid] = sum / float(reduce_size);
    }
}

// Variance along axis
kernel void var_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < outer_size) {
        uint offset = gid * reduce_size;

        // Compute mean
        float sum = 0.0f;
        for (uint i = 0; i < reduce_size; i++) {
            sum += input[offset + i];
        }
        float mean = sum / float(reduce_size);

        // Compute variance
        float var_sum = 0.0f;
        for (uint i = 0; i < reduce_size; i++) {
            float diff = input[offset + i] - mean;
            var_sum += diff * diff;
        }
        output[gid] = var_sum / float(reduce_size);
    }
}

// Max along axis
kernel void max_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < outer_size) {
        float max_val = -INFINITY;
        uint offset = gid * reduce_size;
        for (uint i = 0; i < reduce_size; i++) {
            max_val = max(max_val, input[offset + i]);
        }
        output[gid] = max_val;
    }
}

// ArgMax along axis
kernel void argmax_axis(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < outer_size) {
        float max_val = -INFINITY;
        int max_idx = 0;
        uint offset = gid * reduce_size;
        for (uint i = 0; i < reduce_size; i++) {
            float val = input[offset + i];
            if (val > max_val) {
                max_val = val;
                max_idx = int(i);
            }
        }
        output[gid] = max_idx;
    }
}

// ArgMin along axis
kernel void argmin_axis(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < outer_size) {
        float min_val = INFINITY;
        int min_idx = 0;
        uint offset = gid * reduce_size;
        for (uint i = 0; i < reduce_size; i++) {
            float val = input[offset + i];
            if (val < min_val) {
                min_val = val;
                min_idx = int(i);
            }
        }
        output[gid] = min_idx;
    }
}
";

    #endregion

    #region Matrix Operations

    /// <summary>
    /// Matrix operation kernels including optimized GEMM.
    /// </summary>
    public const string MatrixKernels = CommonHeader + @"

// Tile size for matrix multiplication (adjust based on GPU)
constant uint TILE_SIZE = 16;

// Naive matrix multiplication (for small matrices)
kernel void matmul_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication for better cache utilization
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]])
{
    uint row = group_id.y * TILE_SIZE + lid.y;
    uint col = group_id.x * TILE_SIZE + lid.x;

    float sum = 0.0f;

    // Loop over tiles
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        uint aRow = row;
        uint aCol = t * TILE_SIZE + lid.x;
        if (aRow < M && aCol < K) {
            tileA[lid.y * TILE_SIZE + lid.x] = A[aRow * K + aCol];
        } else {
            tileA[lid.y * TILE_SIZE + lid.x] = 0.0f;
        }

        // Load tile of B into shared memory
        uint bRow = t * TILE_SIZE + lid.y;
        uint bCol = col;
        if (bRow < K && bCol < N) {
            tileB[lid.y * TILE_SIZE + lid.x] = B[bRow * N + bCol];
        } else {
            tileB[lid.y * TILE_SIZE + lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[lid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Batched matrix multiplication
kernel void batch_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& batchCount [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (batch < batchCount && row < M && col < N) {
        uint aOffset = batch * M * K;
        uint bOffset = batch * K * N;
        uint cOffset = batch * M * N;

        float sum = 0.0f;
        for (uint k = 0; k < K; k++) {
            sum += A[aOffset + row * K + k] * B[bOffset + k * N + col];
        }
        C[cOffset + row * N + col] = sum;
    }
}

// Matrix transpose
kernel void transpose(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// Batched transpose
kernel void batched_transpose(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (b < batch && row < rows && col < cols) {
        uint inOffset = b * rows * cols;
        uint outOffset = b * cols * rows;
        output[outOffset + col * rows + row] = input[inOffset + row * cols + col];
    }
}

// Matrix-vector multiplication: y = A * x
kernel void matvec(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < M) {
        float sum = 0.0f;
        for (uint j = 0; j < N; j++) {
            sum += A[gid * N + j] * x[j];
        }
        y[gid] = sum;
    }
}

// Outer product: C = a * b^T
kernel void outer_product(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row < M && col < N) {
        C[row * N + col] = a[row] * b[col];
    }
}

// Dot product (partial - needs reduction to complete)
kernel void dot_product_partial(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]])
{
    float sum = 0.0f;
    for (uint i = gid; i < size; i += group_size * 256) {
        sum += A[i] * B[i];
    }
    shared[lid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[group_id] = shared[0];
    }
}
";

    #endregion

    #region Normalization Operations

    /// <summary>
    /// Normalization layer kernels (BatchNorm, LayerNorm, etc.).
    /// </summary>
    public const string NormalizationKernels = CommonHeader + @"

// Batch normalization forward (inference mode)
kernel void batchnorm_forward_inference(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device const float* running_mean [[buffer(4)]],
    device const float* running_var [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant uint& batch [[buffer(7)]],
    constant uint& channels [[buffer(8)]],
    constant uint& spatial_size [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = batch * channels * spatial_size;
    if (gid < total) {
        uint c = (gid / spatial_size) % channels;

        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrt(var + epsilon);

        float x = input[gid];
        float normalized = (x - mean) * inv_std;
        output[gid] = gamma[c] * normalized + beta[c];
    }
}

// Layer normalization forward
kernel void layernorm_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device float* mean [[buffer(4)]],
    device float* rstd [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant uint& batch [[buffer(7)]],
    constant uint& norm_size [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < batch) {
        uint offset = gid * norm_size;

        // Compute mean
        float sum = 0.0f;
        for (uint i = 0; i < norm_size; i++) {
            sum += input[offset + i];
        }
        float m = sum / float(norm_size);
        mean[gid] = m;

        // Compute variance
        float var_sum = 0.0f;
        for (uint i = 0; i < norm_size; i++) {
            float diff = input[offset + i] - m;
            var_sum += diff * diff;
        }
        float variance = var_sum / float(norm_size);
        float inv_std = rsqrt(variance + epsilon);
        rstd[gid] = inv_std;

        // Normalize and scale
        for (uint i = 0; i < norm_size; i++) {
            float normalized = (input[offset + i] - m) * inv_std;
            output[offset + i] = gamma[i] * normalized + beta[i];
        }
    }
}

// RMS normalization forward
kernel void rmsnorm_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device float* rstd [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& norm_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < batch) {
        uint offset = gid * norm_size;

        // Compute RMS
        float sum_sq = 0.0f;
        for (uint i = 0; i < norm_size; i++) {
            float val = input[offset + i];
            sum_sq += val * val;
        }
        float rms = sqrt(sum_sq / float(norm_size) + epsilon);
        float inv_rms = 1.0f / rms;
        rstd[gid] = inv_rms;

        // Normalize and scale
        for (uint i = 0; i < norm_size; i++) {
            output[offset + i] = input[offset + i] * inv_rms * gamma[i];
        }
    }
}

// Softmax forward (row-wise)
kernel void softmax_row(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& features [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < batch) {
        uint offset = gid * features;

        // Find max for numerical stability
        float max_val = -INFINITY;
        for (uint i = 0; i < features; i++) {
            max_val = max(max_val, input[offset + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (uint i = 0; i < features; i++) {
            float exp_val = exp(input[offset + i] - max_val);
            output[offset + i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (uint i = 0; i < features; i++) {
            output[offset + i] *= inv_sum;
        }
    }
}

// Log softmax forward
kernel void log_softmax_row(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& features [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < batch) {
        uint offset = gid * features;

        // Find max
        float max_val = -INFINITY;
        for (uint i = 0; i < features; i++) {
            max_val = max(max_val, input[offset + i]);
        }

        // Compute log-sum-exp
        float sum = 0.0f;
        for (uint i = 0; i < features; i++) {
            sum += exp(input[offset + i] - max_val);
        }
        float log_sum = log(sum) + max_val;

        // Compute log softmax
        for (uint i = 0; i < features; i++) {
            output[offset + i] = input[offset + i] - log_sum;
        }
    }
}

// L2 normalization
kernel void l2_normalize(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < batch) {
        uint offset = gid * size;

        // Compute L2 norm
        float sum_sq = 0.0f;
        for (uint i = 0; i < size; i++) {
            float val = input[offset + i];
            sum_sq += val * val;
        }
        float inv_norm = rsqrt(sum_sq + epsilon);

        // Normalize
        for (uint i = 0; i < size; i++) {
            output[offset + i] = input[offset + i] * inv_norm;
        }
    }
}

// Dropout forward (with mask generation)
kernel void dropout_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device uchar* mask [[buffer(2)]],
    device const uint* random_state [[buffer(3)]],
    constant float& keep_prob [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    constant uint& size [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        // Simple LCG random number generator
        uint state = random_state[gid % 1024] + gid;
        state = state * 1103515245u + 12345u;
        float rand_val = float(state & 0x7FFFFFFF) / float(0x7FFFFFFF);

        if (rand_val < keep_prob) {
            mask[gid] = 1;
            output[gid] = input[gid] * scale;
        } else {
            mask[gid] = 0;
            output[gid] = 0.0f;
        }
    }
}

// Dropout backward
kernel void dropout_backward(
    device const float* grad_output [[buffer(0)]],
    device const uchar* mask [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        grad_input[gid] = mask[gid] ? grad_output[gid] * scale : 0.0f;
    }
}

// Embedding lookup
kernel void embedding_forward(
    device const int* indices [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& embedding_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint idx = gid.y;
    uint dim = gid.x;

    if (idx < num_indices && dim < embedding_dim) {
        int embed_idx = indices[idx];
        output[idx * embedding_dim + dim] = weight[embed_idx * embedding_dim + dim];
    }
}
";

    #endregion

    #region Convolution Operations

    /// <summary>
    /// Convolution and pooling operation kernels.
    /// </summary>
    public const string ConvolutionKernels = CommonHeader + @"

// Im2col transformation for efficient convolution
kernel void im2col(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& kernel_h [[buffer(6)]],
    constant uint& kernel_w [[buffer(7)]],
    constant uint& stride_h [[buffer(8)]],
    constant uint& stride_w [[buffer(9)]],
    constant uint& pad_h [[buffer(10)]],
    constant uint& pad_w [[buffer(11)]],
    constant uint& out_h [[buffer(12)]],
    constant uint& out_w [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z;
    uint out_idx = gid.y;
    uint col_idx = gid.x;

    if (b >= batch || out_idx >= out_h * out_w) return;

    uint kernel_size = kernel_h * kernel_w;
    if (col_idx >= channels * kernel_size) return;

    uint c = col_idx / kernel_size;
    uint k = col_idx % kernel_size;
    uint kh = k / kernel_w;
    uint kw = k % kernel_w;

    uint oh = out_idx / out_w;
    uint ow = out_idx % out_w;

    int ih = int(oh * stride_h + kh) - int(pad_h);
    int iw = int(ow * stride_w + kw) - int(pad_w);

    float val = 0.0f;
    if (ih >= 0 && ih < int(height) && iw >= 0 && iw < int(width)) {
        val = input[b * channels * height * width + c * height * width + ih * width + iw];
    }

    uint out_offset = b * (channels * kernel_size * out_h * out_w);
    output[out_offset + col_idx * out_h * out_w + out_idx] = val;
}

// Col2im for convolution backward
kernel void col2im(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& kernel_h [[buffer(6)]],
    constant uint& kernel_w [[buffer(7)]],
    constant uint& stride_h [[buffer(8)]],
    constant uint& stride_w [[buffer(9)]],
    constant uint& pad_h [[buffer(10)]],
    constant uint& pad_w [[buffer(11)]],
    constant uint& out_h [[buffer(12)]],
    constant uint& out_w [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z;
    uint c = gid.y;
    uint pixel = gid.x;

    if (b >= batch || c >= channels || pixel >= height * width) return;

    uint ih = pixel / width;
    uint iw = pixel % width;

    float sum = 0.0f;
    uint kernel_size = kernel_h * kernel_w;

    for (uint kh = 0; kh < kernel_h; kh++) {
        for (uint kw = 0; kw < kernel_w; kw++) {
            int oh = int(ih + pad_h - kh);
            int ow = int(iw + pad_w - kw);

            if (oh % stride_h == 0 && ow % stride_w == 0) {
                oh /= stride_h;
                ow /= stride_w;

                if (oh >= 0 && oh < int(out_h) && ow >= 0 && ow < int(out_w)) {
                    uint k = kh * kernel_w + kw;
                    uint col_idx = c * kernel_size + k;
                    uint out_idx = oh * out_w + ow;

                    uint in_offset = b * (channels * kernel_size * out_h * out_w);
                    sum += input[in_offset + col_idx * out_h * out_w + out_idx];
                }
            }
        }
    }

    output[b * channels * height * width + c * height * width + ih * width + iw] = sum;
}

// Max pooling 2D forward
kernel void maxpool2d_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device int* indices [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& in_h [[buffer(5)]],
    constant uint& in_w [[buffer(6)]],
    constant uint& out_h [[buffer(7)]],
    constant uint& out_w [[buffer(8)]],
    constant uint& kernel_h [[buffer(9)]],
    constant uint& kernel_w [[buffer(10)]],
    constant uint& stride_h [[buffer(11)]],
    constant uint& stride_w [[buffer(12)]],
    constant uint& pad_h [[buffer(13)]],
    constant uint& pad_w [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z / channels;
    uint c = gid.z % channels;
    uint oh = gid.y;
    uint ow = gid.x;

    if (b >= batch || oh >= out_h || ow >= out_w) return;

    int h_start = int(oh * stride_h) - int(pad_h);
    int w_start = int(ow * stride_w) - int(pad_w);
    int h_end = min(h_start + int(kernel_h), int(in_h));
    int w_end = min(w_start + int(kernel_w), int(in_w));
    h_start = max(h_start, 0);
    w_start = max(w_start, 0);

    float max_val = -INFINITY;
    int max_idx = 0;
    uint input_offset = b * channels * in_h * in_w + c * in_h * in_w;

    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            float val = input[input_offset + h * in_w + w];
            if (val > max_val) {
                max_val = val;
                max_idx = h * in_w + w;
            }
        }
    }

    uint out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
    output[out_idx] = max_val;
    indices[out_idx] = max_idx;
}

// Average pooling 2D forward
kernel void avgpool2d_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& in_h [[buffer(4)]],
    constant uint& in_w [[buffer(5)]],
    constant uint& out_h [[buffer(6)]],
    constant uint& out_w [[buffer(7)]],
    constant uint& kernel_h [[buffer(8)]],
    constant uint& kernel_w [[buffer(9)]],
    constant uint& stride_h [[buffer(10)]],
    constant uint& stride_w [[buffer(11)]],
    constant uint& pad_h [[buffer(12)]],
    constant uint& pad_w [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z / channels;
    uint c = gid.z % channels;
    uint oh = gid.y;
    uint ow = gid.x;

    if (b >= batch || oh >= out_h || ow >= out_w) return;

    int h_start = int(oh * stride_h) - int(pad_h);
    int w_start = int(ow * stride_w) - int(pad_w);
    int h_end = min(h_start + int(kernel_h), int(in_h) + int(pad_h));
    int w_end = min(w_start + int(kernel_w), int(in_w) + int(pad_w));
    int pool_size = (h_end - h_start) * (w_end - w_start);
    h_start = max(h_start, 0);
    w_start = max(w_start, 0);
    h_end = min(h_end, int(in_h));
    w_end = min(w_end, int(in_w));

    float sum = 0.0f;
    uint input_offset = b * channels * in_h * in_w + c * in_h * in_w;

    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            sum += input[input_offset + h * in_w + w];
        }
    }

    uint out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
    output[out_idx] = sum / float(pool_size);
}

// Global average pooling
kernel void global_avgpool(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& spatial_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint b = gid.y;
    uint c = gid.x;

    if (b >= batch || c >= channels) return;

    uint offset = b * channels * spatial_size + c * spatial_size;
    float sum = 0.0f;
    for (uint i = 0; i < spatial_size; i++) {
        sum += input[offset + i];
    }
    output[b * channels + c] = sum / float(spatial_size);
}
";

    #endregion

    #region Attention Operations

    /// <summary>
    /// Attention mechanism kernels for transformers.
    /// </summary>
    public const string AttentionKernels = CommonHeader + @"

// Scaled dot-product attention scores: scores = Q @ K^T / sqrt(d_k)
kernel void attention_scores(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z / heads;
    uint h = gid.z % heads;
    uint i = gid.y;
    uint j = gid.x;

    if (b >= batch || i >= seq_len || j >= seq_len) return;

    uint qk_offset = (b * heads + h) * seq_len * head_dim;
    uint score_offset = (b * heads + h) * seq_len * seq_len;

    float sum = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        sum += Q[qk_offset + i * head_dim + d] * K[qk_offset + j * head_dim + d];
    }

    scores[score_offset + i * seq_len + j] = sum * scale;
}

// Apply causal mask to attention scores
kernel void apply_causal_mask(
    device float* scores [[buffer(0)]],
    constant uint& batch_heads [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint bh = gid.z;
    uint i = gid.y;
    uint j = gid.x;

    if (bh >= batch_heads || i >= seq_len || j >= seq_len) return;

    if (j > i) {
        scores[bh * seq_len * seq_len + i * seq_len + j] = -INFINITY;
    }
}

// Attention weighted sum: output = softmax(scores) @ V
kernel void attention_output(
    device const float* attn_weights [[buffer(0)]],
    device const float* V [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z / heads;
    uint h = gid.z % heads;
    uint i = gid.y;
    uint d = gid.x;

    if (b >= batch || i >= seq_len || d >= head_dim) return;

    uint weight_offset = (b * heads + h) * seq_len * seq_len;
    uint v_offset = (b * heads + h) * seq_len * head_dim;
    uint out_offset = (b * heads + h) * seq_len * head_dim;

    float sum = 0.0f;
    for (uint j = 0; j < seq_len; j++) {
        sum += attn_weights[weight_offset + i * seq_len + j] * V[v_offset + j * head_dim + d];
    }

    output[out_offset + i * head_dim + d] = sum;
}

// Flash attention kernel (simplified version - compute block at a time)
kernel void flash_attention_block(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* lse [[buffer(4)]],
    constant uint& batch_heads [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& block_size [[buffer(9)]],
    threadgroup float* shared_qkv [[threadgroup(0)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]])
{
    uint bh = group_id.z;
    uint block_i = group_id.y;
    uint block_j = group_id.x;

    if (bh >= batch_heads) return;

    uint qkv_offset = bh * seq_len * head_dim;
    uint out_offset = bh * seq_len * head_dim;

    uint i_start = block_i * block_size;
    uint j_start = block_j * block_size;

    // Simplified flash attention - each thread handles one output position
    uint local_i = lid.y;
    uint local_d = lid.x;
    uint i = i_start + local_i;

    if (i >= seq_len || local_d >= head_dim) return;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_acc = 0.0f;

    for (uint j = j_start; j < min(j_start + block_size, seq_len); j++) {
        // Compute attention score for (i, j)
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += Q[qkv_offset + i * head_dim + d] * K[qkv_offset + j * head_dim + d];
        }
        score *= scale;

        // Online softmax update
        float new_max = max(max_score, score);
        float exp_diff = exp(max_score - new_max);
        float exp_score = exp(score - new_max);

        output_acc = output_acc * exp_diff + exp_score * V[qkv_offset + j * head_dim + local_d];
        sum_exp = sum_exp * exp_diff + exp_score;
        max_score = new_max;
    }

    // Atomic update to output (simplified - real implementation would be more complex)
    if (sum_exp > 0.0f) {
        output[out_offset + i * head_dim + local_d] = output_acc / sum_exp;
    }
}

// Rotary position embedding (RoPE)
kernel void rope_forward(
    device float* x [[buffer(0)]],
    device const float* cos_cache [[buffer(1)]],
    device const float* sin_cache [[buffer(2)]],
    constant uint& batch_heads [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint bh = gid.z;
    uint pos = gid.y;
    uint d = gid.x;

    if (bh >= batch_heads || pos >= seq_len || d >= head_dim / 2) return;

    uint offset = bh * seq_len * head_dim + pos * head_dim;

    float cos_val = cos_cache[pos * (head_dim / 2) + d];
    float sin_val = sin_cache[pos * (head_dim / 2) + d];

    float x0 = x[offset + d];
    float x1 = x[offset + d + head_dim / 2];

    x[offset + d] = x0 * cos_val - x1 * sin_val;
    x[offset + d + head_dim / 2] = x0 * sin_val + x1 * cos_val;
}
";

    #endregion

    #region Optimizer Operations

    /// <summary>
    /// Optimizer update kernels (SGD, Adam, etc.).
    /// </summary>
    public const string OptimizerKernels = CommonHeader + @"

// SGD update: param = param - lr * grad
kernel void sgd_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    constant float& lr [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        param[gid] -= lr * grad[gid];
    }
}

// SGD with momentum: v = momentum * v + grad; param = param - lr * v
kernel void sgd_momentum_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* velocity [[buffer(2)]],
    constant float& lr [[buffer(3)]],
    constant float& momentum [[buffer(4)]],
    constant uint& size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float v = momentum * velocity[gid] + grad[gid];
        velocity[gid] = v;
        param[gid] -= lr * v;
    }
}

// Adam update
kernel void adam_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& epsilon [[buffer(7)]],
    constant float& beta1_t [[buffer(8)]],
    constant float& beta2_t [[buffer(9)]],
    constant uint& size [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float g = grad[gid];

        // Update biased first moment estimate
        float m_new = beta1 * m[gid] + (1.0f - beta1) * g;
        m[gid] = m_new;

        // Update biased second moment estimate
        float v_new = beta2 * v[gid] + (1.0f - beta2) * g * g;
        v[gid] = v_new;

        // Bias correction
        float m_hat = m_new / (1.0f - beta1_t);
        float v_hat = v_new / (1.0f - beta2_t);

        // Update parameter
        param[gid] -= lr * m_hat / (sqrt(v_hat) + epsilon);
    }
}

// AdamW update (weight decay decoupled)
kernel void adamw_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& epsilon [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& beta1_t [[buffer(9)]],
    constant float& beta2_t [[buffer(10)]],
    constant uint& size [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float p = param[gid];
        float g = grad[gid];

        // Weight decay
        p -= lr * weight_decay * p;

        // Adam update
        float m_new = beta1 * m[gid] + (1.0f - beta1) * g;
        m[gid] = m_new;

        float v_new = beta2 * v[gid] + (1.0f - beta2) * g * g;
        v[gid] = v_new;

        float m_hat = m_new / (1.0f - beta1_t);
        float v_hat = v_new / (1.0f - beta2_t);

        param[gid] = p - lr * m_hat / (sqrt(v_hat) + epsilon);
    }
}

// RMSprop update
kernel void rmsprop_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* v [[buffer(2)]],
    constant float& lr [[buffer(3)]],
    constant float& alpha [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    constant uint& size [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float g = grad[gid];
        float v_new = alpha * v[gid] + (1.0f - alpha) * g * g;
        v[gid] = v_new;
        param[gid] -= lr * g / (sqrt(v_new) + epsilon);
    }
}

// Lion optimizer update
kernel void lion_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],
    constant float& lr [[buffer(3)]],
    constant float& beta1 [[buffer(4)]],
    constant float& beta2 [[buffer(5)]],
    constant float& weight_decay [[buffer(6)]],
    constant uint& size [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float p = param[gid];
        float g = grad[gid];
        float m_val = m[gid];

        // Compute update direction using interpolation
        float update = beta1 * m_val + (1.0f - beta1) * g;

        // Sign update
        float sign_update = (update > 0.0f) ? 1.0f : ((update < 0.0f) ? -1.0f : 0.0f);

        // Weight decay and update
        param[gid] = p - lr * (sign_update + weight_decay * p);

        // Update momentum
        m[gid] = beta2 * m_val + (1.0f - beta2) * g;
    }
}

// Gradient clipping by norm
kernel void clip_grad_norm(
    device float* grad [[buffer(0)]],
    constant float& clip_coef [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size && clip_coef < 1.0f) {
        grad[gid] *= clip_coef;
    }
}

// Gradient clipping by value
kernel void clip_grad_value(
    device float* grad [[buffer(0)]],
    constant float& clip_value [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        grad[gid] = clamp_value(grad[gid], -clip_value, clip_value);
    }
}
";

    #endregion

    #region Loss Functions

    /// <summary>
    /// Loss function kernels.
    /// </summary>
    public const string LossKernels = CommonHeader + @"

// Mean Squared Error (MSE) loss
kernel void mse_loss(
    device const float* predictions [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float diff = predictions[gid] - targets[gid];
        output[gid] = diff * diff;
    }
}

// MSE gradient
kernel void mse_backward(
    device const float* predictions [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* grad [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        grad[gid] = scale * 2.0f * (predictions[gid] - targets[gid]);
    }
}

// Binary Cross Entropy loss
kernel void bce_loss(
    device const float* predictions [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float p = clamp_value(predictions[gid], EPSILON, 1.0f - EPSILON);
        float t = targets[gid];
        output[gid] = -(t * log(p) + (1.0f - t) * log(1.0f - p));
    }
}

// BCE gradient
kernel void bce_backward(
    device const float* predictions [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* grad [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float p = clamp_value(predictions[gid], EPSILON, 1.0f - EPSILON);
        float t = targets[gid];
        grad[gid] = scale * ((p - t) / (p * (1.0f - p)));
    }
}

// Cross Entropy loss with softmax (combined for numerical stability)
kernel void cross_entropy_loss(
    device const float* logits [[buffer(0)]],
    device const int* targets [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < batch) {
        uint offset = gid * num_classes;
        int target = targets[gid];

        // Compute log-softmax
        float max_val = -INFINITY;
        for (uint c = 0; c < num_classes; c++) {
            max_val = max(max_val, logits[offset + c]);
        }

        float sum = 0.0f;
        for (uint c = 0; c < num_classes; c++) {
            sum += exp(logits[offset + c] - max_val);
        }

        float log_softmax = logits[offset + target] - max_val - log(sum);
        output[gid] = -log_softmax;
    }
}

// Cross entropy gradient
kernel void cross_entropy_backward(
    device const float* logits [[buffer(0)]],
    device const int* targets [[buffer(1)]],
    device float* grad [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& num_classes [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint b = gid.y;
    uint c = gid.x;

    if (b >= batch || c >= num_classes) return;

    uint offset = b * num_classes;
    int target = targets[b];

    // Compute softmax
    float max_val = -INFINITY;
    for (uint i = 0; i < num_classes; i++) {
        max_val = max(max_val, logits[offset + i]);
    }

    float sum = 0.0f;
    for (uint i = 0; i < num_classes; i++) {
        sum += exp(logits[offset + i] - max_val);
    }

    float softmax = exp(logits[offset + c] - max_val) / sum;
    float target_val = (c == uint(target)) ? 1.0f : 0.0f;
    grad[offset + c] = scale * (softmax - target_val);
}

// Huber loss (smooth L1)
kernel void huber_loss(
    device const float* predictions [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& delta [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float diff = predictions[gid] - targets[gid];
        float abs_diff = abs(diff);
        if (abs_diff <= delta) {
            output[gid] = 0.5f * diff * diff;
        } else {
            output[gid] = delta * (abs_diff - 0.5f * delta);
        }
    }
}

// Huber gradient
kernel void huber_backward(
    device const float* predictions [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* grad [[buffer(2)]],
    constant float& delta [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float diff = predictions[gid] - targets[gid];
        float abs_diff = abs(diff);
        if (abs_diff <= delta) {
            grad[gid] = scale * diff;
        } else {
            grad[gid] = scale * delta * ((diff > 0.0f) ? 1.0f : -1.0f);
        }
    }
}
";

    #endregion

    #region Comparison and Selection

    /// <summary>
    /// Comparison and selection operation kernels.
    /// </summary>
    public const string ComparisonKernels = CommonHeader + @"

// Greater than comparison
kernel void greater_than(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = A[gid] > B[gid] ? 1.0f : 0.0f;
    }
}

// Less than comparison
kernel void less_than(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = A[gid] < B[gid] ? 1.0f : 0.0f;
    }
}

// Equal comparison
kernel void equal(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = abs(A[gid] - B[gid]) < EPSILON ? 1.0f : 0.0f;
    }
}

// Where (conditional select): C = condition ? A : B
kernel void where_kernel(
    device const float* condition [[buffer(0)]],
    device const float* A [[buffer(1)]],
    device const float* B [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        C[gid] = condition[gid] != 0.0f ? A[gid] : B[gid];
    }
}

// Not equal to scalar
kernel void not_equal_scalar(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        B[gid] = abs(A[gid] - scalar) > EPSILON ? 1.0f : 0.0f;
    }
}

// TopK (simplified - finds top K values per row)
kernel void topk_values(
    device const float* input [[buffer(0)]],
    device float* values [[buffer(1)]],
    device int* indices [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return;

    uint offset = gid * size;

    // Simple selection sort for top K (not efficient for large K)
    // Production code would use a heap or parallel sort
    for (uint i = 0; i < k; i++) {
        float max_val = -INFINITY;
        int max_idx = -1;

        for (uint j = 0; j < size; j++) {
            float val = input[offset + j];
            bool already_selected = false;
            for (uint p = 0; p < i; p++) {
                if (indices[gid * k + p] == int(j)) {
                    already_selected = true;
                    break;
                }
            }
            if (!already_selected && val > max_val) {
                max_val = val;
                max_idx = int(j);
            }
        }

        values[gid * k + i] = max_val;
        indices[gid * k + i] = max_idx;
    }
}
";

    #endregion

    #region Random Number Generation

    /// <summary>
    /// Random number generation kernels.
    /// </summary>
    public const string RandomKernels = CommonHeader + @"

// Philox counter-based RNG (high quality, fast)
struct PhiloxState {
    uint4 counter;
    uint2 key;
};

inline uint mulhilo32(uint a, uint b, thread uint& hi) {
    ulong result = ulong(a) * ulong(b);
    hi = uint(result >> 32);
    return uint(result);
}

inline uint4 philox_round(uint4 counter, uint2 key) {
    uint hi0, hi1;
    uint lo0 = mulhilo32(0xD2511F53, counter.x, hi0);
    uint lo1 = mulhilo32(0xCD9E8D57, counter.z, hi1);

    return uint4(
        hi1 ^ counter.y ^ key.x,
        lo1,
        hi0 ^ counter.w ^ key.y,
        lo0
    );
}

inline uint4 philox4x32(uint4 counter, uint2 key) {
    counter = philox_round(counter, key);
    key.x += 0x9E3779B9;
    key.y += 0xBB67AE85;

    counter = philox_round(counter, key);
    key.x += 0x9E3779B9;
    key.y += 0xBB67AE85;

    counter = philox_round(counter, key);
    key.x += 0x9E3779B9;
    key.y += 0xBB67AE85;

    counter = philox_round(counter, key);
    key.x += 0x9E3779B9;
    key.y += 0xBB67AE85;

    return philox_round(counter, key);
}

inline float uint_to_uniform(uint x) {
    return float(x) * 2.3283064365386963e-10f; // x / 2^32
}

// Generate uniform random numbers [0, 1)
kernel void random_uniform(
    device float* output [[buffer(0)]],
    constant uint& seed [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        uint4 counter = uint4(gid, 0, 0, 0);
        uint2 key = uint2(seed, 0xDEADBEEF);
        uint4 result = philox4x32(counter, key);
        output[gid] = uint_to_uniform(result.x);
    }
}

// Generate normal random numbers using Box-Muller transform
kernel void random_normal(
    device float* output [[buffer(0)]],
    constant uint& seed [[buffer(1)]],
    constant float& mean [[buffer(2)]],
    constant float& std [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        uint pair_idx = gid / 2;
        bool is_second = (gid % 2) == 1;

        uint4 counter = uint4(pair_idx, 0, 0, 0);
        uint2 key = uint2(seed, 0xDEADBEEF);
        uint4 result = philox4x32(counter, key);

        float u1 = uint_to_uniform(result.x);
        float u2 = uint_to_uniform(result.y);

        // Avoid log(0)
        u1 = max(u1, 1e-7f);

        float radius = sqrt(-2.0f * log(u1));
        float theta = 2.0f * PI * u2;

        float z = is_second ? radius * sin(theta) : radius * cos(theta);
        output[gid] = mean + std * z;
    }
}

// Dropout mask generation
kernel void dropout_mask(
    device uchar* mask [[buffer(0)]],
    constant uint& seed [[buffer(1)]],
    constant float& keep_prob [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        uint4 counter = uint4(gid, 0, 0, 0);
        uint2 key = uint2(seed, 0x12345678);
        uint4 result = philox4x32(counter, key);
        float rand = uint_to_uniform(result.x);
        mask[gid] = rand < keep_prob ? 1 : 0;
    }
}
";

    #endregion
}
