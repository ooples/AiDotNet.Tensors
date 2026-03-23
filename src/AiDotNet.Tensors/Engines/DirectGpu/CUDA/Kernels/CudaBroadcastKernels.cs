namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Fused CUDA kernels for broadcast operations, scalar arithmetic, and
/// element-wise ops that don't fit the simple unary/binary pattern.
/// Uses grid-stride loops for maximum occupancy.
/// </summary>
public static class CudaBroadcastKernels
{
    public static string GetSource()
    {
        return @"
// ============================================================================
// Broadcast binary ops: a[outer, inner] op b[1, inner] → output[outer, inner]
// Last-axis broadcast (most common: bias add, channel scale, etc.)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void broadcast_add_last(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int outerSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;
    output[idx] = a[idx] + b[idx % innerSize];
}

extern ""C"" __global__ __launch_bounds__(256) void broadcast_sub_last(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int outerSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;
    output[idx] = a[idx] - b[idx % innerSize];
}

extern ""C"" __global__ __launch_bounds__(256) void broadcast_mul_last(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int outerSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;
    output[idx] = a[idx] * b[idx % innerSize];
}

extern ""C"" __global__ __launch_bounds__(256) void broadcast_div_last(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int outerSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;
    output[idx] = a[idx] / (b[idx % innerSize] + 1e-12f);
}

// First-axis broadcast: a[1, inner] op b[outer, inner]
extern ""C"" __global__ __launch_bounds__(256) void broadcast_add_first(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int outerSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;
    int outer = idx / innerSize;
    output[idx] = a[outer] + b[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void broadcast_mul_first(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int outerSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;
    int outer = idx / innerSize;
    output[idx] = a[outer] * b[idx];
}

// ============================================================================
// Scalar operations
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void add_scalar(
    const float* __restrict__ input, float* __restrict__ output,
    float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = input[idx] + scalar;
}

extern ""C"" __global__ __launch_bounds__(256) void sub_scalar(
    const float* __restrict__ input, float* __restrict__ output,
    float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = input[idx] - scalar;
}

extern ""C"" __global__ __launch_bounds__(256) void div_scalar(
    const float* __restrict__ input, float* __restrict__ output,
    float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = input[idx] / scalar;
}

extern ""C"" __global__ __launch_bounds__(256) void pow_scalar(
    const float* __restrict__ input, float* __restrict__ output,
    float exponent, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = powf(input[idx], exponent);
}

// ============================================================================
// Multi-tensor element-wise (add_many, multiply_many)
// Fused: output[i] = sum of all inputs[k][i]
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void add_many_2(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = a[idx] + b[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void add_many_3(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ c, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = a[idx] + b[idx] + c[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void add_many_4(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ c, const float* __restrict__ d,
    float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = a[idx] + b[idx] + c[idx] + d[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void multiply_many_3(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ c, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = a[idx] * b[idx] * c[idx];
}

// ============================================================================
// Frac, Clip, Reciprocal-sqrt, SinCos
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void frac_kernel(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = input[idx] - floorf(input[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void clip_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    float minVal, float maxVal, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = fminf(fmaxf(input[idx], minVal), maxVal);
}

extern ""C"" __global__ __launch_bounds__(256) void rsqrt_kernel(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = rsqrtf(input[idx] + 1e-12f);
}

extern ""C"" __global__ __launch_bounds__(256) void sincos_kernel(
    const float* __restrict__ input,
    float* __restrict__ sin_output, float* __restrict__ cos_output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    sincosf(input[idx], &sin_output[idx], &cos_output[idx]);
}

// ============================================================================
// Comparison ops
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void equals_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void not_equals_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = (a[idx] != b[idx]) ? 1.0f : 0.0f;
}
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            "broadcast_add_last",
            "broadcast_sub_last",
            "broadcast_mul_last",
            "broadcast_div_last",
            "broadcast_add_first",
            "broadcast_mul_first",
            "add_scalar",
            "sub_scalar",
            "div_scalar",
            "pow_scalar",
            "add_many_2",
            "add_many_3",
            "add_many_4",
            "multiply_many_3",
            "frac_kernel",
            "clip_kernel",
            "rsqrt_kernel",
            "sincos_kernel",
            "equals_kernel",
            "not_equals_kernel"
        ];
    }
}
