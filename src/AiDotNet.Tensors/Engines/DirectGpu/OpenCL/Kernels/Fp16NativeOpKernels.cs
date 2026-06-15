// Copyright (c) AiDotNet. All rights reserved.
// OpenCL FP16-NATIVE op kernels (Tensors #558): each kernel reads its activation DIRECTLY from a
// half buffer, up-casts to FP32 in-register for the math, and writes a half result — no separate
// FP32 buffer is materialized, so there is no convert transient (the failure mode that made the
// buffer-compression approach RAISE peak VRAM). Half storage uses the CORE OpenCL vload_half /
// vstore_half built-ins (NOT the optional cl_khr_fp16 extension), so they run on every OpenCL 1.0+
// device — half is the STORAGE format; the arithmetic is FP32.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// FP16-native elementwise / activation kernels for the OpenCL backend: GELU, ReLU and residual add
/// over half-stored activations, computed in FP32 in-register. Mirrors the CUDA <c>Fp16Gelu</c> pilot
/// (#558 layer 7) and the FP16-GEMM half-load convention.
/// </summary>
public static class Fp16NativeOpKernels
{
    public const string GeluKernelName = "fp16_gelu";
    public const string ReluKernelName = "fp16_relu";
    public const string AddKernelName = "fp16_add";
    public const string ConvertF32ToF16KernelName = "fp16_convert_f32_to_f16";
    public const string ConvertF16ToF32KernelName = "fp16_convert_f16_to_f32";

    public static string GetSource()
    {
        return @"
// half is the storage format; all math is FP32 (vload_half/vstore_half are CORE built-ins).

// GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))). The tanh argument is clamped to
// +/-20 so exp(2z) cannot overflow to Inf -> NaN (same guard as the CPU/SIMD GELU kernels).
__kernel void fp16_gelu(
    __global const half* input,
    __global half* output,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float x = vload_half(i, input);
    float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    z = clamp(z, -20.0f, 20.0f);
    float t = tanh(z);
    float y = 0.5f * x * (1.0f + t);
    vstore_half(y, i, output);
}

__kernel void fp16_relu(
    __global const half* input,
    __global half* output,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float x = vload_half(i, input);
    vstore_half(x > 0.0f ? x : 0.0f, i, output);
}

// Elementwise residual add: out = a + b (same shape), half in/out, FP32 accumulate.
__kernel void fp16_add(
    __global const half* a,
    __global const half* b,
    __global half* output,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float s = vload_half(i, a) + vload_half(i, b);
    vstore_half(s, i, output);
}

// FP32 -> FP16 (half storage) conversion. vstore_half rounds the float to IEEE-754 half and writes it,
// using ONLY the core vstore_half built-in (no cl_khr_fp16 arithmetic extension). This is the native
// counterpart of OpenClBackend.ConvertToFp16 so half-buffer round-trips work on devices (e.g. NVIDIA
// OpenCL) that expose vload_half/vstore_half but NOT the cl_khr_fp16 extension.
__kernel void fp16_convert_f32_to_f16(
    __global const float* input,
    __global half* output,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    vstore_half(input[i], i, output);
}

// FP16 (half storage) -> FP32 conversion via the core vload_half built-in.
__kernel void fp16_convert_f16_to_f32(
    __global const half* input,
    __global float* output,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    output[i] = vload_half(i, input);
}
";
    }

    public static string[] GetKernelNames() => new[] { GeluKernelName, ReluKernelName, AddKernelName, ConvertF32ToF16KernelName, ConvertF16ToF32KernelName };
}
