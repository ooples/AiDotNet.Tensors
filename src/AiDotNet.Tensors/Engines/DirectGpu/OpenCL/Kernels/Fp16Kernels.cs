// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for FP16 (half-precision) conversion operations.
// Uses cl_khr_fp16 extension for native half-precision support.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for FP16 (half-precision) conversion operations.
/// These kernels convert between FP32 (float) and FP16 (half) precision.
/// </summary>
public static class Fp16Kernels
{
    public static string GetSource()
    {
        // OpenCL FP16 support via cl_khr_fp16 extension
        // The half type is a 16-bit floating point format
        return @"
// Enable half-precision extension if available
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ============================================================================
// FP16 CONVERSION KERNELS
// ============================================================================

// Convert FP32 (float) array to FP16 (half) array
// input: float array of size 'size'
// output: half array of size 'size' (stored as ushort for compatibility)
__kernel void convert_fp32_to_fp16(
    __global const float* input,
    __global ushort* output,
    const int size)
{
    int idx = get_global_id(0);
    if (idx >= size) return;

    // Use vstore_half for portable FP32->FP16 conversion with round-to-nearest-even
    vstore_half(input[idx], idx, (__global half*)output);
}

// Convert FP16 (half) array to FP32 (float) array
// input: half array of size 'size' (stored as ushort for compatibility)
// output: float array of size 'size'
__kernel void convert_fp16_to_fp32(
    __global const ushort* input,
    __global float* output,
    const int size)
{
    int idx = get_global_id(0);
    if (idx >= size) return;

    // Use vload_half for portable FP16->FP32 conversion
    output[idx] = vload_half(idx, (__global const half*)input);
}

// Convert FP32 to FP16 with rounding mode control
// roundMode: 0 = round to nearest even (default), 1 = round toward zero, 2 = round down, 3 = round up
__kernel void convert_fp32_to_fp16_rounding(
    __global const float* input,
    __global ushort* output,
    const int size,
    const int roundMode)
{
    int idx = get_global_id(0);
    if (idx >= size) return;

    float val = input[idx];

    // OpenCL vstore_half supports different rounding modes
    switch (roundMode) {
        case 1:  // Round toward zero
            vstore_half_rtz(val, idx, (__global half*)output);
            break;
        case 2:  // Round down (toward negative infinity)
            vstore_half_rtn(val, idx, (__global half*)output);
            break;
        case 3:  // Round up (toward positive infinity)
            vstore_half_rtp(val, idx, (__global half*)output);
            break;
        default: // Round to nearest even
            vstore_half_rte(val, idx, (__global half*)output);
            break;
    }
}

// Vectorized FP32 to FP16 conversion (processes 2 elements per thread for better performance)
__kernel void convert_fp32_to_fp16_vec2(
    __global const float2* input,
    __global uint* output,
    const int size)
{
    int idx = get_global_id(0);
    int numPairs = size / 2;
    if (idx >= numPairs) return;

    float2 f2 = input[idx];

    // Convert both floats to half and pack into uint
    vstore_half2(f2, idx, (__global half*)output);
}

// Vectorized FP16 to FP32 conversion (processes 2 elements per thread for better performance)
__kernel void convert_fp16_to_fp32_vec2(
    __global const uint* input,
    __global float2* output,
    const int size)
{
    int idx = get_global_id(0);
    int numPairs = size / 2;
    if (idx >= numPairs) return;

    // Load two half values and convert to float2
    float2 f2 = vload_half2(idx, (__global const half*)input);
    output[idx] = f2;
}

// ============================================================================
// FP16 CONV im2col (#1650/#638): fused im2col + FP32->FP16 in TRANSPOSED [K,N] layout (K=channels*kernelH*kernelW,
// N=batch*outH*outW) so the conv becomes a plain NN GEMM out[outC,N] = weights[outC,K] . col[K,N] via
// GemmFp16In32fOut (the OpenCL HalfPrecisionGemmKernels, FP16 multiply / FP32 accumulate) — the industry conv path.
// ONE WORK-ITEM PER col ELEMENT (N*K items) → coalesced col[t]=col[k*N+n] writes via the CORE vstore_half built-in
// (round-to-nearest-even, matches convert_fp32_to_fp16 / ConvertToFp16 so the col + weights agree bit-for-bit).
// ============================================================================
__kernel void im2col_kn_fp16hw(
    __global const float* input,
    __global ushort* output,
    const int batch, const int channels, const int height, const int width,
    const int kernelH, const int kernelW, const int strideH, const int strideW,
    const int padH, const int padW, const int dilationH, const int dilationW,
    const int outH, const int outW)
{
    size_t t = get_global_id(0);
    long N = (long)batch * outH * outW;
    long K = (long)channels * kernelH * kernelW;
    if ((long)t >= N * K) return;
    int n = (int)((long)t % N);     // output position (fastest — coalesced)
    int k = (int)((long)t / N);     // unrolled (c,kh,kw) row
    int b = n / (outH * outW);
    int rem = n % (outH * outW);
    int oh = rem / outW;
    int ow = rem % outW;
    int kw = k % kernelW;
    int kh = (k / kernelW) % kernelH;
    int c  = k / (kernelW * kernelH);
    int ih = oh * strideH - padH + kh * dilationH;
    int iw = ow * strideW - padW + kw * dilationW;
    float val = (ih >= 0 && ih < height && iw >= 0 && iw < width)
        ? input[((b * channels + c) * height + ih) * width + iw] : 0.0f;
    vstore_half(val, t, (__global half*)output);
}
";
    }

    /// <summary>
    /// Gets the list of kernel names for compilation.
    /// </summary>
    public static string[] GetKernelNames()
    {
        return new[]
        {
            "im2col_kn_fp16hw",
            "convert_fp32_to_fp16",
            "convert_fp16_to_fp32",
            "convert_fp32_to_fp16_rounding",
            "convert_fp32_to_fp16_vec2",
            "convert_fp16_to_fp32_vec2"
        };
    }
}
