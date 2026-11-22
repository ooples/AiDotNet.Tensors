// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// Double-precision reductions and element-wise arithmetic. The program needs fp64, so a device without it fails to
/// compile it and the backend reports both unavailable.
/// </summary>
internal static class Float64Kernels
{
    public static string[] GetKernelNames() => new[]
    {
        "reduce_sum_f64", "reduce_dot_f64", "add_f64", "subtract_f64", "multiply_f64", "divide_f64",
        "negate_f64", "abs_f64", "sqrt_f64", "exp_f64", "log_f64", "tanh_f64",
        "sigmoid_f64", "relu_f64", "add_scalar_f64", "subtract_scalar_f64", "multiply_scalar_f64", "divide_scalar_f64",
    };

    public static string GetSource() => @"
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error double precision is not available on this device
#endif

// Ceil-halving tree over the work-group: the group size is not always a power of two, and a plain halving tree drops
// the last element of an odd level.
#define TREE_REDUCE_F64(scratch, localIdx, localSize) \
    for (int active = (localSize), stride = ((localSize) + 1) / 2; stride > 0; \
         active = stride, stride = (stride > 1) ? (stride + 1) / 2 : 0) { \
        if ((localIdx) < stride && (localIdx) + stride < active) scratch[(localIdx)] += scratch[(localIdx) + stride]; \
        barrier(CLK_LOCAL_MEM_FENCE); \
    }

// One partial sum per work-group.
__kernel void reduce_sum_f64(
    __global const double* input,
    __global double* output,
    __local double* scratch,
    const int size)
{
    const int globalIdx = get_global_id(0);
    const int localIdx = get_local_id(0);
    const int localSize = get_local_size(0);
    scratch[localIdx] = (globalIdx < size) ? input[globalIdx] : 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);
    TREE_REDUCE_F64(scratch, localIdx, localSize)
    if (localIdx == 0) output[get_group_id(0)] = scratch[0];
}

// One partial dot product per work-group.
__kernel void reduce_dot_f64(
    __global const double* a,
    __global const double* b,
    __global double* output,
    __local double* scratch,
    const int size)
{
    const int globalIdx = get_global_id(0);
    const int localIdx = get_local_id(0);
    const int localSize = get_local_size(0);
    scratch[localIdx] = (globalIdx < size) ? a[globalIdx] * b[globalIdx] : 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);
    TREE_REDUCE_F64(scratch, localIdx, localSize)
    if (localIdx == 0) output[get_group_id(0)] = scratch[0];
}

// One IEEE operation per element. The work size is rounded up to the group size, hence the bound check.
#define BINARY_F64(name, expression) \
__kernel void name(__global const double* a, __global const double* b, __global double* output, const int size) \
{ \
    const int i = get_global_id(0); \
    if (i < size) { const double x = a[i]; const double y = b[i]; output[i] = (expression); } \
}

BINARY_F64(add_f64, x + y)
BINARY_F64(subtract_f64, x - y)
BINARY_F64(multiply_f64, x * y)
BINARY_F64(divide_f64, x / y)

#define UNARY_F64(name, expression) \
__kernel void name(__global const double* a, __global double* output, const int size) \
{ \
    const int i = get_global_id(0); \
    if (i < size) { const double x = a[i]; output[i] = (expression); } \
}

UNARY_F64(negate_f64, -x)
UNARY_F64(abs_f64, fabs(x))
UNARY_F64(sqrt_f64, sqrt(x))
UNARY_F64(exp_f64, exp(x))
UNARY_F64(log_f64, log(x))
UNARY_F64(tanh_f64, tanh(x))
UNARY_F64(sigmoid_f64, 1.0 / (1.0 + exp(-x)))
UNARY_F64(relu_f64, x > 0.0 ? x : (x != x ? x : 0.0))

// The scalar is the right operand.
#define SCALAR_F64(name, expression) \
__kernel void name(__global const double* a, __global double* output, const double y, const int size) \
{ \
    const int i = get_global_id(0); \
    if (i < size) { const double x = a[i]; output[i] = (expression); } \
}

SCALAR_F64(add_scalar_f64, x + y)
SCALAR_F64(subtract_scalar_f64, x - y)
SCALAR_F64(multiply_scalar_f64, x * y)
SCALAR_F64(divide_scalar_f64, x / y)
";
}
