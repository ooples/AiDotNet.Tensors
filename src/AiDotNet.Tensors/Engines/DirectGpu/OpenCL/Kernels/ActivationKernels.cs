// Copyright (c) AiDotNet. All rights reserved.
// Activation function kernels for neural network layers.
// Works on ALL .NET versions including .NET Framework 4.6.2.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for common activation functions.
    /// </summary>
    internal static class ActivationKernels
    {
        /// <summary>
        /// Gets all activation kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// Fast exp4 — Cephes-style polynomial approximation for float4.
// ~0.01% relative error, 3-5x faster than hardware exp() on most GPUs.
// Uses range reduction: exp(x) = 2^n * exp(r) where r = x - n*ln2, |r| < ln2/2
// Then exp(r) via 6th-order minimax polynomial.
// ===========================================================================
inline float4 fast_exp4(float4 x) {
    // Clamp to avoid inf/nan
    x = clamp(x, (float4)(-87.3f), (float4)(88.7f));
    // Range reduction: n = round(x / ln2), r = x - n * ln2
    const float4 LOG2E = (float4)(1.44269504088896f);
    const float4 LN2_HI = (float4)(0.693359375f);
    const float4 LN2_LO = (float4)(-2.12194440e-4f);
    float4 n = rint(x * LOG2E);
    float4 r = x - n * LN2_HI - n * LN2_LO;
    // Polynomial: exp(r) ~ 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720
    float4 r2 = r * r;
    float4 p = (float4)(1.0f/720.0f);
    p = p * r + (float4)(1.0f/120.0f);
    p = p * r + (float4)(1.0f/24.0f);
    p = p * r + (float4)(1.0f/6.0f);
    p = p * r + (float4)(0.5f);
    p = p * r + (float4)(1.0f);
    p = p * r + (float4)(1.0f);
    // Reconstruct: exp(x) = 2^n * exp(r) via IEEE 754 exponent manipulation
    int4 ni = convert_int4(n);
    // ldexp(p, n) = p * 2^n
    p = p * as_float4((ni + 127) << 23);
    return p;
}

inline float fast_exp1(float x) {
    x = clamp(x, -87.3f, 88.7f);
    float n = rint(x * 1.44269504088896f);
    float r = x - n * 0.693359375f - n * (-2.12194440e-4f);
    float r2 = r * r;
    float p = 1.0f/720.0f;
    p = p * r + 1.0f/120.0f;
    p = p * r + 1.0f/24.0f;
    p = p * r + 1.0f/6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    int ni = convert_int(n);
    return p * as_float((ni + 127) << 23);
}

// ===========================================================================
// ACTIVATION KERNELS
// ===========================================================================

// ReLU: max(0, x)
__kernel void relu(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 x = vload4(idx, input);
        vstore4(fmax((float4)(0.0f), x), idx, output);
    } else {
        for (int i = idx4; i < size; i++) output[i] = fmax(0.0f, input[i]);
    }
}

// Leaky ReLU: x > 0 ? x : alpha * x
__kernel void leaky_relu(
    __global const float* input,
    __global float* output,
    const float alpha,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 x = vload4(idx, input);
        vstore4(select(x * alpha, x, isgreater(x, (float4)(0.0f))), idx, output);
    } else {
        for (int i = idx4; i < size; i++) { float x = input[i]; output[i] = x > 0.0f ? x : alpha * x; }
    }
}

// Sigmoid: 1 / (1 + exp(-x))
__kernel void sigmoid(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 x = vload4(idx, input);
        vstore4((float4)(1.0f) / ((float4)(1.0f) + fast_exp4(-x)), idx, output);
    } else {
        for (int i = idx4; i < size; i++) output[i] = 1.0f / (1.0f + fast_exp1(-input[i]));
    }
}

// Tanh: clamp to [-20, 20] to avoid NaN, then use hardware tanh
__kernel void tanh_activation(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 x = clamp(vload4(idx, input), (float4)(-20.0f), (float4)(20.0f));
        vstore4(tanh(x), idx, output);
    } else {
        for (int i = idx4; i < size; i++) output[i] = tanh(clamp(input[i], -20.0f, 20.0f));
    }
}

// GELU (Gaussian Error Linear Unit) - approximation
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__kernel void gelu(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    if (idx4 + 3 < size) {
        float4 x = vload4(idx, input);
        float4 x3 = x * x * x;
        float4 inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        vstore4(0.5f * x * ((float4)(1.0f) + tanh(inner)), idx, output);
    } else {
        for (int i = idx4; i < size; i++) {
            float x = input[i]; float x3 = x * x * x;
            output[i] = 0.5f * x * (1.0f + tanh(SQRT_2_OVER_PI * (x + COEFF * x3)));
        }
    }
}

// Swish: x * sigmoid(x) = x / (1 + exp(-x))
__kernel void swish(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 x = vload4(idx, input);
        vstore4(x / ((float4)(1.0f) + fast_exp4(-x)), idx, output);
    } else {
        for (int i = idx4; i < size; i++) { float x = input[i]; output[i] = x / (1.0f + fast_exp1(-x)); }
    }
}

// Softmax (per batch row) — multi-workgroup for large feature dimensions
// For features <= 4096: single workgroup per row (low overhead)
// For features > 4096: multiple workgroups per row with global max/sum scratch
__kernel void softmax(
    __global const float* input,
    __global float* output,
    const int batchSize,
    const int features,
    __local float* localBuf)
{
    const int batch = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);

    if (batch >= batchSize) return;

    __global const float* rowIn = input + batch * features;
    __global float* rowOut = output + batch * features;

    // Phase 1: Each thread finds max over its strided elements (float4 vectorized)
    float threadMax = -INFINITY;
    int f = lid * 4;
    for (; f + 3 < features; f += localSize * 4) {
        float4 v = vload4(0, rowIn + f);
        threadMax = fmax(threadMax, fmax(fmax(v.x, v.y), fmax(v.z, v.w)));
    }
    for (int i = f / 4 * 4 + lid % (localSize); i < features; i += localSize) {
        if (i < features) threadMax = fmax(threadMax, rowIn[i]);
    }

    // Reduce max across workgroup
    localBuf[lid] = threadMax;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) localBuf[lid] = fmax(localBuf[lid], localBuf[lid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rowMax = localBuf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Each thread computes exp and partial sum (float4 vectorized)
    float threadSum = 0.0f;
    f = lid * 4;
    for (; f + 3 < features; f += localSize * 4) {
        float4 v = vload4(0, rowIn + f);
        float4 e = fast_exp4(v - (float4)(rowMax));
        vstore4(e, 0, rowOut + f);
        threadSum += e.x + e.y + e.z + e.w;
    }
    for (int i = f / 4 * 4 + lid % (localSize); i < features; i += localSize) {
        if (i < features) { float e = fast_exp1(rowIn[i] - rowMax); rowOut[i] = e; threadSum += e; }
    }

    // Reduce sum across workgroup
    localBuf[lid] = threadSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) localBuf[lid] += localBuf[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rowSum = localBuf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: Normalize (float4 vectorized)
    float invSum = 1.0f / rowSum;
    f = lid * 4;
    for (; f + 3 < features; f += localSize * 4) {
        vstore4(vload4(0, rowOut + f) * invSum, 0, rowOut + f);
    }
    for (int i = f / 4 * 4 + lid % (localSize); i < features; i += localSize) {
        if (i < features) rowOut[i] *= invSum;
    }
}

// ===========================================================================
// ELEMENT-WISE OPERATIONS (float4-vectorized for 4x memory throughput)
// Each work item processes 4 floats via 128-bit loads/stores.
// Scalar tail handles sizes not divisible by 4.
// ===========================================================================

// Vector addition: C = A + B
__kernel void add_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 a = vload4(idx, A);
        float4 b = vload4(idx, B);
        vstore4(a + b, idx, C);
    } else {
        for (int i = idx4; i < size; i++) C[i] = A[i] + B[i];
    }
}

// Vector subtraction: C = A - B
__kernel void subtract_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 a = vload4(idx, A);
        float4 b = vload4(idx, B);
        vstore4(a - b, idx, C);
    } else {
        for (int i = idx4; i < size; i++) C[i] = A[i] - B[i];
    }
}

// Vector multiplication: C = A * B (element-wise)
__kernel void multiply_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 a = vload4(idx, A);
        float4 b = vload4(idx, B);
        vstore4(a * b, idx, C);
    } else {
        for (int i = idx4; i < size; i++) C[i] = A[i] * B[i];
    }
}

// Vector division: C = A / B (element-wise)
__kernel void divide_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 a = vload4(idx, A);
        float4 b = vload4(idx, B);
        vstore4(a / b, idx, C);
    } else {
        for (int i = idx4; i < size; i++) C[i] = A[i] / B[i];
    }
}

// Vector min: C = min(A, B)
__kernel void min_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 a = vload4(idx, A);
        float4 b = vload4(idx, B);
        vstore4(fmin(a, b), idx, C);
    } else {
        for (int i = idx4; i < size; i++) C[i] = fmin(A[i], B[i]);
    }
}

// Vector max: C = max(A, B)
__kernel void max_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 a = vload4(idx, A);
        float4 b = vload4(idx, B);
        vstore4(fmax(a, b), idx, C);
    } else {
        for (int i = idx4; i < size; i++) C[i] = fmax(A[i], B[i]);
    }
}

// Scalar multiplication: B = A * scalar
__kernel void scale_vector(
    __global const float* A,
    __global float* B,
    const float scalar,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 a = vload4(idx, A);
        vstore4(a * scalar, idx, B);
    } else {
        for (int i = idx4; i < size; i++) B[i] = A[i] * scalar;
    }
}

// Absolute value
__kernel void abs_vector(__global const float* A, __global float* B, const int size) {
    const int idx = get_global_id(0); const int idx4 = idx * 4;
    if (idx4 + 3 < size) { vstore4(fabs(vload4(idx, A)), idx, B); }
    else { for (int i = idx4; i < size; i++) B[i] = fabs(A[i]); }
}

// Exponential
__kernel void exp_vector(__global const float* A, __global float* B, const int size) {
    const int idx = get_global_id(0); const int idx4 = idx * 4;
    if (idx4 + 3 < size) { vstore4(exp(vload4(idx, A)), idx, B); }
    else { for (int i = idx4; i < size; i++) B[i] = exp(A[i]); }
}

// Natural log
__kernel void log_vector(__global const float* A, __global float* B, const int size) {
    const int idx = get_global_id(0); const int idx4 = idx * 4;
    if (idx4 + 3 < size) { vstore4(log(vload4(idx, A)), idx, B); }
    else { for (int i = idx4; i < size; i++) B[i] = log(A[i]); }
}

// Base-2 log
__kernel void log2_vector(__global const float* A, __global float* B, const int size) {
    const int idx = get_global_id(0); const int idx4 = idx * 4;
    if (idx4 + 3 < size) { vstore4(log2(vload4(idx, A)), idx, B); }
    else { for (int i = idx4; i < size; i++) B[i] = log2(A[i]); }
}

// Base-2 exp
__kernel void exp2_vector(__global const float* A, __global float* B, const int size) {
    const int idx = get_global_id(0); const int idx4 = idx * 4;
    if (idx4 + 3 < size) { vstore4(exp2(vload4(idx, A)), idx, B); }
    else { for (int i = idx4; i < size; i++) B[i] = exp2(A[i]); }
}

// Base-10 exp
__kernel void exp10_vector(__global const float* A, __global float* B, const int size) {
    const int idx = get_global_id(0); const int idx4 = idx * 4;
    if (idx4 + 3 < size) { float4 a = vload4(idx, A); vstore4(exp(a * (float4)(2.302585093f)), idx, B); }
    else { for (int i = idx4; i < size; i++) B[i] = exp(A[i] * 2.302585093f); }
}

// exp(x) - 1
__kernel void expm1_vector(__global const float* A, __global float* B, const int size) {
    const int idx = get_global_id(0); const int idx4 = idx * 4;
    if (idx4 + 3 < size) { vstore4(exp(vload4(idx, A)) - (float4)(1.0f), idx, B); }
    else { for (int i = idx4; i < size; i++) B[i] = exp(A[i]) - 1.0f; }
}

// log(1 + x)
__kernel void log1p_vector(__global const float* A, __global float* B, const int size) {
    const int idx = get_global_id(0); const int idx4 = idx * 4;
    if (idx4 + 3 < size) { vstore4(log((float4)(1.0f) + vload4(idx, A)), idx, B); }
    else { for (int i = idx4; i < size; i++) B[i] = log(1.0f + A[i]); }
}

// Square root
__kernel void sqrt_vector(__global const float* A, __global float* B, const int size) {
    const int idx = get_global_id(0); const int idx4 = idx * 4;
    if (idx4 + 3 < size) { vstore4(sqrt(vload4(idx, A)), idx, B); }
    else { for (int i = idx4; i < size; i++) B[i] = sqrt(A[i]); }
}

// Sign
__kernel void sign_vector(__global const float* A, __global float* B, const int size) {
    const int idx = get_global_id(0); const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 x = vload4(idx, A);
        vstore4(select(select((float4)(0.0f), (float4)(-1.0f), isless(x, (float4)(0.0f))), (float4)(1.0f), isgreater(x, (float4)(0.0f))), idx, B);
    } else { for (int i = idx4; i < size; i++) { float x = A[i]; B[i] = x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); } }
}

// Power with scalar exponent
__kernel void power_scalar(
    __global const float* A,
    __global float* B,
    const float exponent,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = pow(A[idx], exponent);
}

// ===========================================================================
// TRIGONOMETRIC OPERATIONS
// ===========================================================================

// Sine
__kernel void sin_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = sin(A[idx]);
}

// Cosine
__kernel void cos_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = cos(A[idx]);
}

// Tangent
__kernel void tan_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = tan(A[idx]);
}

// Arc sine
__kernel void asin_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = asin(A[idx]);
}

// Arc cosine
__kernel void acos_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = acos(A[idx]);
}

// Arc tangent
__kernel void atan_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = atan(A[idx]);
}

// ===========================================================================
// HYPERBOLIC OPERATIONS
// ===========================================================================

// Hyperbolic sine
__kernel void sinh_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = sinh(A[idx]);
}

// Hyperbolic cosine
__kernel void cosh_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = cosh(A[idx]);
}

// Inverse hyperbolic sine
__kernel void asinh_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = asinh(A[idx]);
}

// Inverse hyperbolic cosine
__kernel void acosh_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = acosh(A[idx]);
}

// Inverse hyperbolic tangent
__kernel void atanh_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = atanh(A[idx]);
}

// ===========================================================================
// ADDITIONAL UNARY OPERATIONS
// ===========================================================================

// Reciprocal: 1/x
__kernel void reciprocal_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = 1.0f / A[idx];
}

// Cube root
__kernel void cbrt_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = cbrt(A[idx]);
}

// Base-10 log
__kernel void log10_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = log10(A[idx]);
}

// Negate
__kernel void negate_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = -A[idx];
}

// Floor
__kernel void floor_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = floor(A[idx]);
}

// Ceiling
__kernel void ceil_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = ceil(A[idx]);
}

// Round
__kernel void round_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = round(A[idx]);
}

// Truncate
__kernel void trunc_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = trunc(A[idx]);
}

// ===========================================================================
// BROADCAST OPERATIONS
// ===========================================================================

// Bias addition with broadcast: C[i,j] = A[i,j] + bias[j]
// Broadcasts bias vector (N elements) across M rows
__kernel void bias_add(
    __global const float* A,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    if (row >= M || col >= N) return;

    const int idx = row * N + col;
    C[idx] = A[idx] + bias[col];
}

// Conv2D bias add in NCHW format: output[b,c,h,w] += bias[c]
// Memory layout: output is [batch, channels, height, width] in row-major order
__kernel void conv2d_bias_add(
    __global float* output,
    __global const float* bias,
    const int batch,
    const int channels,
    const int spatialSize)
{
    const int idx = get_global_id(0);
    const int totalSize = batch * channels * spatialSize;
    if (idx >= totalSize) return;
    const int channel = (idx / spatialSize) % channels;
    output[idx] += bias[channel];
}

// ===========================================================================
// ADDITIONAL FORWARD ACTIVATIONS
// ===========================================================================

// Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
__kernel void mish(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    // Numerically stable softplus
    float sp = x > 20.0f ? x : (x < -20.0f ? exp(x) : log(1.0f + exp(x)));
    output[idx] = x * tanh(sp);
}

// Softplus: ln(1 + exp(x))
__kernel void softplus(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    // Numerically stable softplus
    output[idx] = x > 20.0f ? x : (x < -20.0f ? exp(x) : log(1.0f + exp(x)));
}

// HardSwish: x * clamp(x + 3, 0, 6) / 6
__kernel void hardswish(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float inner = x + 3.0f;
    inner = fmax(0.0f, fmin(6.0f, inner));
    output[idx] = x * inner / 6.0f;
}

// SELU: scale * (x > 0 ? x : alpha * (exp(x) - 1))
// With scale = 1.0507 and alpha = 1.6733
__kernel void selu(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    const float scale = 1.0507009873554804934193349852946f;
    const float alpha = 1.6732632423543772848170429916717f;

    float x = input[idx];
    output[idx] = scale * (x > 0.0f ? x : alpha * (exp(x) - 1.0f));
}

// HardSigmoid: clamp((x + 3) / 6, 0, 1)
__kernel void hardsigmoid(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = fmax(0.0f, fmin(1.0f, (x + 3.0f) / 6.0f));
}

// HardTanh: clamp(x, -1, 1)
__kernel void hardtanh(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    output[idx] = fmax(-1.0f, fmin(1.0f, input[idx]));
}

// ===========================================================================
// ACTIVATION BACKWARD KERNELS
// ===========================================================================

// ReLU backward: grad * (x > 0 ? 1 : 0)
__kernel void relu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    gradInput[idx] = input[idx] > 0.0f ? gradOutput[idx] : 0.0f;
}

// Leaky ReLU backward: grad * (x > 0 ? 1 : alpha)
__kernel void leaky_relu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const float alpha,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    gradInput[idx] = gradOutput[idx] * (x > 0.0f ? 1.0f : alpha);
}

// Sigmoid backward: grad * sigmoid(x) * (1 - sigmoid(x))
__kernel void sigmoid_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float sig = 1.0f / (1.0f + exp(-input[idx]));
    gradInput[idx] = gradOutput[idx] * sig * (1.0f - sig);
}

// Tanh backward: grad * (1 - tanh(x)^2)
__kernel void tanh_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float t = tanh(input[idx]);
    gradInput[idx] = gradOutput[idx] * (1.0f - t * t);
}

// GELU backward (approximation)
__kernel void gelu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;

    float x = input[idx];
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    float t = tanh(inner);
    float sech2 = 1.0f - t * t;
    float dInner = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x2);

    float dgelu = 0.5f * (1.0f + t) + 0.5f * x * sech2 * dInner;
    gradInput[idx] = gradOutput[idx] * dgelu;
}

// Swish backward: grad * (swish(x) + sigmoid(x) * (1 - swish(x)))
__kernel void swish_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float sig = 1.0f / (1.0f + exp(-x));
    float swish_val = x * sig;
    float dswish = swish_val + sig * (1.0f - swish_val);
    gradInput[idx] = gradOutput[idx] * dswish;
}

// Mish backward
__kernel void mish_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    // Numerically stable softplus
    float sp = x > 20.0f ? x : (x < -20.0f ? exp(x) : log(1.0f + exp(x)));
    float tanh_sp = tanh(sp);
    float sig = 1.0f / (1.0f + exp(-x));
    float sech2_sp = 1.0f - tanh_sp * tanh_sp;
    float dmish = tanh_sp + x * sech2_sp * sig;
    gradInput[idx] = gradOutput[idx] * dmish;
}

// Softplus backward: grad * sigmoid(x)
__kernel void softplus_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float sig = 1.0f / (1.0f + exp(-x));
    gradInput[idx] = gradOutput[idx] * sig;
}

// HardSwish backward
__kernel void hardswish_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float grad;
    if (x <= -3.0f) {
        grad = 0.0f;
    } else if (x >= 3.0f) {
        grad = 1.0f;
    } else {
        grad = (2.0f * x + 3.0f) / 6.0f;
    }
    gradInput[idx] = gradOutput[idx] * grad;
}

// SELU backward
__kernel void selu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    const float scale = 1.0507009873554804934193349852946f;
    const float alpha = 1.6732632423543772848170429916717f;

    float x = input[idx];
    float grad = x > 0.0f ? scale : scale * alpha * exp(x);
    gradInput[idx] = gradOutput[idx] * grad;
}

// HardSigmoid backward
__kernel void hardsigmoid_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float grad = (x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f;
    gradInput[idx] = gradOutput[idx] * grad;
}

// HardTanh backward
__kernel void hardtanh_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float grad = (x > -1.0f && x < 1.0f) ? 1.0f : 0.0f;
    gradInput[idx] = gradOutput[idx] * grad;
}

// ============================================================================
// Online softmax: two-phase approach using ALL workgroups for large rows.
// Phase 1: each workgroup computes tile-local max + sum(exp(x-max)).
//          Results go to scratch[groupId] = {max, sum}.
//          The last workgroup (via atomic counter) computes the global result:
//          global_max, global_sum = correct_and_merge(all tile results)
// Phase 2: each work item computes output[i] = exp(input[i] - global_max) / global_sum
// ============================================================================

// Phase 1: compute per-tile max and sum, last workgroup merges all tiles
// scratch layout: [numGroups floats for max | numGroups floats for sum | 1 int counter | 1 float globalMax | 1 float globalSum]
__kernel void softmax_online_phase1(
    __global const float* input,
    __global float* scratch, // size: numGroups*2 + 3 (max[], sum[], counter, globalMax, globalSum)
    const int rowOffset,
    const int features,
    const int numGroups,
    __local float* localBuf)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);

    __global const float* row = input + rowOffset;

    // Each thread finds max over its strided elements within this tile
    int tileStart = gid * ((features + numGroups - 1) / numGroups);
    int tileEnd = min(tileStart + ((features + numGroups - 1) / numGroups), features);

    float threadMax = -INFINITY;
    for (int f = tileStart + lid; f < tileEnd; f += localSize) {
        threadMax = fmax(threadMax, row[f]);
    }

    // Reduce max within workgroup
    localBuf[lid] = threadMax;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = localSize >> 1; s > 0; s >>= 1) {
        if (lid < s) localBuf[lid] = fmax(localBuf[lid], localBuf[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float tileMax = localBuf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each thread computes partial exp sum
    float threadSum = 0.0f;
    for (int f = tileStart + lid; f < tileEnd; f += localSize) {
        threadSum += fast_exp1(row[f] - tileMax);
    }

    // Reduce sum within workgroup
    localBuf[lid] = threadSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = localSize >> 1; s > 0; s >>= 1) {
        if (lid < s) localBuf[lid] += localBuf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float tileSum = localBuf[0];

    // Thread 0 writes tile results
    if (lid == 0) {
        scratch[gid] = tileMax;
        scratch[numGroups + gid] = tileSum;

        // Atomic increment counter; last workgroup to finish does the merge
        __global int* counter = (__global int*)(scratch + numGroups * 2);
        int old = atomic_add(counter, 1);
        if (old == numGroups - 1) {
            // Last workgroup: merge all tile results
            float globalMax = scratch[0];
            for (int i = 1; i < numGroups; i++) globalMax = fmax(globalMax, scratch[i]);

            float globalSum = 0.0f;
            for (int i = 0; i < numGroups; i++) {
                // Correct each tile's sum to the global max: sum_i * exp(tileMax_i - globalMax)
                globalSum += scratch[numGroups + i] * fast_exp1(scratch[i] - globalMax);
            }

            // Store global results for phase 2
            __global float* globalResults = scratch + numGroups * 2;
            // globalResults[0] is the counter (int), skip it
            // Store at fixed offset after counter
            scratch[numGroups * 2 + 1] = globalMax;
            scratch[numGroups * 2 + 2] = globalSum;

            // Reset counter for potential reuse
            *counter = 0;
        }
    }
}

// Phase 2: normalize all elements using global max and sum (full GPU parallelism)
__kernel void softmax_online_phase2(
    __global const float* input,
    __global float* output,
    __global const float* scratch, // scratch[numGroups*2+1] = globalMax, scratch[numGroups*2+2] = globalSum
    const int rowOffset,
    const int features,
    const int numGroups)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;

    float globalMax = scratch[numGroups * 2 + 1];
    float invSum = 1.0f / scratch[numGroups * 2 + 2];

    if (idx4 + 3 < features) {
        float4 v = vload4(0, input + rowOffset + idx4);
        vstore4(fast_exp4(v - (float4)(globalMax)) * invSum, 0, output + rowOffset + idx4);
    } else {
        for (int i = idx4; i < features; i++)
            output[rowOffset + i] = fast_exp1(input[rowOffset + i] - globalMax) * invSum;
    }
}

// Softmax helper: B[offset + idx*4..+3] = exp(A[offset + idx*4..+3] - scalar)
// Full GPU parallelism for large-row softmax pass 2
__kernel void softmax_exp_sub(
    __global const float* A,
    __global float* B,
    const float scalar,
    const int offset,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 v = vload4(0, A + offset + idx4);
        vstore4(fast_exp4(v - (float4)(scalar)), 0, B + offset + idx4);
    } else {
        for (int i = idx4; i < size; i++) B[offset + i] = fast_exp1(A[offset + i] - scalar);
    }
}

// Softmax helper: B[offset + idx*4..+3] *= scalar
// Full GPU parallelism for large-row softmax pass 4
__kernel void softmax_div_scalar(
    __global float* B,
    const float scalar,
    const int offset,
    const int size)
{
    const int idx = get_global_id(0);
    const int idx4 = idx * 4;
    if (idx4 + 3 < size) {
        float4 v = vload4(0, B + offset + idx4);
        vstore4(v * scalar, 0, B + offset + idx4);
    } else {
        for (int i = idx4; i < size; i++) B[offset + i] *= scalar;
    }
}
";
        }

        /// <summary>
        /// Gets kernel names for compilation.
        /// </summary>
        public static string[] GetKernelNames()
        {
            return new string[]
            {
                // Activations (forward)
                "relu", "leaky_relu", "sigmoid", "tanh_activation",
                "gelu", "swish", "softmax", "softmax_exp_sub", "softmax_div_scalar",
                "softmax_online_phase1", "softmax_online_phase2",
                "mish", "softplus", "hardswish", "selu", "hardsigmoid", "hardtanh",
                // Activation backward kernels
                "relu_backward", "leaky_relu_backward", "sigmoid_backward", "tanh_backward",
                "gelu_backward", "swish_backward", "mish_backward", "softplus_backward",
                "hardswish_backward", "selu_backward", "hardsigmoid_backward", "hardtanh_backward",
                // Element-wise binary
                "add_vectors", "subtract_vectors", "multiply_vectors",
                "divide_vectors", "min_vectors", "max_vectors",
                // Scalar ops
                "scale_vector", "power_scalar",
                // Unary math
                "abs_vector", "exp_vector", "log_vector",
                "log2_vector", "exp2_vector", "exp10_vector",
                "expm1_vector", "log1p_vector", "sqrt_vector",
                "sign_vector",
                // Trigonometric
                "sin_vector", "cos_vector", "tan_vector",
                "asin_vector", "acos_vector", "atan_vector",
                // Hyperbolic
                "sinh_vector", "cosh_vector",
                "asinh_vector", "acosh_vector", "atanh_vector",
                // Additional unary
                "reciprocal_vector", "cbrt_vector", "log10_vector",
                "negate_vector", "floor_vector", "ceil_vector",
                "round_vector", "trunc_vector",
                // Broadcast operations
                "bias_add",
                // Conv2D operations
                "conv2d_bias_add"
            };
        }
    }
}
