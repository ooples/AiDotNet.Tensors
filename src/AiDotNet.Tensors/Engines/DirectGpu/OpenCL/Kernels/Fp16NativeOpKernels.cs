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
    public const string SoftmaxKernelName = "fp16_softmax";
    public const string LayerNormKernelName = "fp16_layernorm";

    // Work-group size for the row-reduction kernels (softmax/layernorm): one work-group per row,
    // power-of-two so the tree reduction is exact. The dispatch launches rows*RowReduceLocalSize threads.
    public const int RowReduceLocalSize = 256;

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

// Row softmax over the last axis: ONE WORK-GROUP PER ROW (get_group_id(0) == row). FP32 max/sum
// reductions via local memory (numerically stable: subtract row max), half in/out. Local size must be
// RowReduceLocalSize (power of two) so the tree reduction is exact; tail threads (tid >= cols) reduce
// with the identity (-INF for max, 0 for sum).
__kernel void fp16_softmax(
    __global const half* input, __global half* output, const int rows, const int cols)
{
    int row = get_group_id(0);
    if (row >= rows) return;
    int tid = get_local_id(0);
    int bs = get_local_size(0);
    __global const half* in = input + (long)row * cols;
    __global half* out = output + (long)row * cols;
    __local float scratch[256];
    float m = -3.4e38f;
    for (int i = tid; i < cols; i += bs) { float v = vload_half(i, in); m = fmax(m, v); }
    scratch[tid] = m; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = bs >> 1; s > 0; s >>= 1) { if (tid < s) scratch[tid] = fmax(scratch[tid], scratch[tid + s]); barrier(CLK_LOCAL_MEM_FENCE); }
    float rowmax = scratch[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    for (int i = tid; i < cols; i += bs) sum += exp(vload_half(i, in) - rowmax);
    scratch[tid] = sum; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = bs >> 1; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
    float inv = 1.0f / scratch[0]; barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = tid; i < cols; i += bs) vstore_half(exp(vload_half(i, in) - rowmax) * inv, i, out);
}

// Row layernorm over the last axis with half gamma/beta: ONE WORK-GROUP PER ROW. FP32 mean/var via local
// memory, half in/out; writes the per-row FP32 mean + variance (the dispatch always supplies real buffers,
// allocating temporaries when the caller passes none). Population variance (/cols), eps inside rsqrt.
__kernel void fp16_layernorm(
    __global const half* input, __global const half* gamma, __global const half* beta,
    __global half* output, __global float* meanOut, __global float* varOut,
    const int rows, const int cols, const float eps)
{
    int row = get_group_id(0);
    if (row >= rows) return;
    int tid = get_local_id(0);
    int bs = get_local_size(0);
    __global const half* in = input + (long)row * cols;
    __global half* out = output + (long)row * cols;
    __local float scratch[256];
    float s = 0.0f;
    for (int i = tid; i < cols; i += bs) s += vload_half(i, in);
    scratch[tid] = s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = bs >> 1; st > 0; st >>= 1) { if (tid < st) scratch[tid] += scratch[tid + st]; barrier(CLK_LOCAL_MEM_FENCE); }
    float mean = scratch[0] / (float)cols; barrier(CLK_LOCAL_MEM_FENCE);
    float vv = 0.0f;
    for (int i = tid; i < cols; i += bs) { float d = vload_half(i, in) - mean; vv += d * d; }
    scratch[tid] = vv; barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = bs >> 1; st > 0; st >>= 1) { if (tid < st) scratch[tid] += scratch[tid + st]; barrier(CLK_LOCAL_MEM_FENCE); }
    float var = scratch[0] / (float)cols; barrier(CLK_LOCAL_MEM_FENCE);
    float invstd = rsqrt(var + eps);
    if (tid == 0) { meanOut[row] = mean; varOut[row] = var; }
    for (int i = tid; i < cols; i += bs) {
        float norm = (vload_half(i, in) - mean) * invstd;
        vstore_half(norm * vload_half(i, gamma) + vload_half(i, beta), i, out);
    }
}
";
    }

    public static string[] GetKernelNames() => new[] { GeluKernelName, ReluKernelName, AddKernelName, ConvertF32ToF16KernelName, ConvertF16ToF32KernelName, SoftmaxKernelName, LayerNormKernelName };
}
