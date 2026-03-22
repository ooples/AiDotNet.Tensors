namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL broadcast, scalar, and element-wise utility kernels.
/// </summary>
public static class BroadcastKernels
{
    public static string GetSource()
    {
        return @"
__kernel void broadcast_add_last(__global const float* a, __global const float* b, __global float* output, int outerSize, int innerSize) {
    int idx = get_global_id(0); if (idx >= outerSize * innerSize) return;
    output[idx] = a[idx] + b[idx % innerSize];
}
__kernel void broadcast_sub_last(__global const float* a, __global const float* b, __global float* output, int outerSize, int innerSize) {
    int idx = get_global_id(0); if (idx >= outerSize * innerSize) return;
    output[idx] = a[idx] - b[idx % innerSize];
}
__kernel void broadcast_mul_last(__global const float* a, __global const float* b, __global float* output, int outerSize, int innerSize) {
    int idx = get_global_id(0); if (idx >= outerSize * innerSize) return;
    output[idx] = a[idx] * b[idx % innerSize];
}
__kernel void broadcast_div_last(__global const float* a, __global const float* b, __global float* output, int outerSize, int innerSize) {
    int idx = get_global_id(0); if (idx >= outerSize * innerSize) return;
    output[idx] = a[idx] / (b[idx % innerSize] + 1e-12f);
}
__kernel void broadcast_add_first(__global const float* a, __global const float* b, __global float* output, int outerSize, int innerSize) {
    int idx = get_global_id(0); if (idx >= outerSize * innerSize) return;
    output[idx] = a[idx / innerSize] + b[idx];
}
__kernel void broadcast_mul_first(__global const float* a, __global const float* b, __global float* output, int outerSize, int innerSize) {
    int idx = get_global_id(0); if (idx >= outerSize * innerSize) return;
    output[idx] = a[idx / innerSize] * b[idx];
}
__kernel void add_scalar(__global const float* input, __global float* output, float scalar, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = input[idx] + scalar;
}
__kernel void sub_scalar(__global const float* input, __global float* output, float scalar, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = input[idx] - scalar;
}
__kernel void div_scalar(__global const float* input, __global float* output, float scalar, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    float safe_scalar = (scalar == 0.0f) ? 1e-10f : scalar;
    output[idx] = input[idx] / safe_scalar;
}
__kernel void pow_scalar(__global const float* input, __global float* output, float exponent, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = pow(input[idx], exponent);
}
__kernel void frac_kernel(__global const float* input, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = input[idx] - floor(input[idx]);
}
__kernel void clip_kernel(__global const float* input, __global float* output, float minVal, float maxVal, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = fmin(fmax(input[idx], minVal), maxVal);
}
__kernel void rsqrt_kernel(__global const float* input, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = rsqrt(input[idx] + 1e-12f);
}
__kernel void equals_kernel(__global const float* a, __global const float* b, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
}
__kernel void not_equals_kernel(__global const float* a, __global const float* b, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = (a[idx] != b[idx]) ? 1.0f : 0.0f;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "broadcast_add_last", "broadcast_sub_last", "broadcast_mul_last", "broadcast_div_last",
            "broadcast_add_first", "broadcast_mul_first",
            "add_scalar", "sub_scalar", "div_scalar", "pow_scalar",
            "frac_kernel", "clip_kernel", "rsqrt_kernel",
            "equals_kernel", "not_equals_kernel"
        };
    }
}
