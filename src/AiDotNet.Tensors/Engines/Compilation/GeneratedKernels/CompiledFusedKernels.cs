// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;

/// <summary>
/// Compile-time generated universal deep-learning fused kernels (CPU SIMD + CUDA C NVRTC source).
/// Bodies and CUDA kernel constants are emitted at compile time by <c>AiDotNet.Tensors.Generators.FusedKernelGenerator</c>.
/// </summary>
[FusedRowPatternSuite]
internal static partial class CompiledFusedKernels
{
    /// <summary>Fused SwiGLU over paired gate and up projections: <c>dst = swish(gate) * up</c>.</summary>
    [FusedElementwiseKernel("fused_swiglu_pair_f32", "swish(x0) * x1", inputCount: 2)]
    public static partial void SwiGluPair(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> destination);

    /// <summary>Fused SwiGLU backward w.r.t. gate: <c>dGate = gradOut * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))</c>.</summary>
    [FusedElementwiseKernel("fused_swiglu_grad_gate_f32", "x0 * x2 * (sigmoid(x1) * (1.0 + x1 * (1.0 - sigmoid(x1))))", inputCount: 3)]
    public static partial void SwiGluPairGradGate(ReadOnlySpan<float> gradOut, ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> destination);

    /// <summary>Fused SwiGLU backward w.r.t. up: <c>dUp = gradOut * swish(gate)</c>.</summary>
    [FusedElementwiseKernel("fused_swiglu_grad_up_f32", "x0 * swish(x1)", inputCount: 2)]
    public static partial void SwiGluPairGradUp(ReadOnlySpan<float> gradOut, ReadOnlySpan<float> gate, Span<float> destination);

    /// <summary>Fused GeGLU over paired gate and up projections: <c>dst = gelu(gate) * up</c>.</summary>
    [FusedElementwiseKernel("fused_geglu_pair_f32", "gelu(x0) * x1", inputCount: 2)]
    public static partial void GeGluPair(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> destination);

    /// <summary>Fused Add + GELU: <c>dst = gelu(a + b)</c>.</summary>
    [FusedElementwiseKernel("fused_add_gelu_f32", "gelu(x0 + x1)", inputCount: 2)]
    public static partial void AddGelu(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> destination);

    /// <summary>Fused Add + Swish/SiLU: <c>dst = swish(a + b)</c>.</summary>
    [FusedElementwiseKernel("fused_add_swish_f32", "swish(x0 + x1)", inputCount: 2)]
    public static partial void AddSwish(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> destination);

    /// <summary>Fused Add + ReLU: <c>dst = relu(a + b)</c>.</summary>
    [FusedElementwiseKernel("fused_add_relu_f32", "relu(x0 + x1)", inputCount: 2)]
    public static partial void AddRelu(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> destination);

    /// <summary>Fused scaled residual addition: <c>dst = fma(branch, scale, residual)</c>.</summary>
    [FusedElementwiseKernel("fused_residual_scale_add_f32", "fma(x0, s0, x1)", inputCount: 2, scalarCount: 1)]
    public static partial void ResidualScaleAdd(ReadOnlySpan<float> branch, ReadOnlySpan<float> residual, float scale, Span<float> destination);

    /// <summary>Fused affine scale-shift + Swish: <c>dst = swish(fma(input, scale, shift))</c>.</summary>
    [FusedElementwiseKernel("fused_affine_swish_f32", "swish(fma(x0, x1, x2))", inputCount: 3)]
    public static partial void AffineSwish(ReadOnlySpan<float> input, ReadOnlySpan<float> scale, ReadOnlySpan<float> shift, Span<float> destination);
}
