// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// Row-vector times matrix, y[N] = alpha * x[K] * B[K x N] (row-major B): a fully connected layer at batch 1. It
    /// reads every weight once and is bound by memory bandwidth, where a tiled GEMM spends its launch on an M of 1.
    /// Split-K: work item (n4, s) accumulates four adjacent columns over its slice of K, so a wavefront reads one
    /// contiguous run of each weight row; a second kernel sums the slices.
    /// </summary>
    internal static class GemvKernels
    {
        internal const string Partial = "gemv_rowvec_partial";
        internal const string Reduce = "gemv_rowvec_reduce";

        internal static string[] GetKernelNames() => new[] { Partial, Reduce };

        internal const string Source = @"
__kernel void gemv_rowvec_partial(__global const float* restrict x, __global const float* restrict b,
    __global float* restrict partial, const int N, const int K, const int kChunk)
{
    int n4 = (int)get_global_id(0);
    int s = (int)get_global_id(1);
    int col = n4 * 4;
    if (col >= N) return;
    int k0 = s * kChunk;
    int k1 = min(K, k0 + kChunk);
    float4 acc = (float4)(0.0f);
    for (int k = k0; k < k1; k++)
        acc += x[k] * vload4(0, b + (long)k * N + col);
    vstore4(acc, 0, partial + (long)s * N + col);
}

__kernel void gemv_rowvec_reduce(__global const float* restrict partial, __global float* restrict y, const int N,
    const int splits, const float alpha)
{
    int n = (int)get_global_id(0);
    if (n >= N) return;
    float sum = 0.0f;
    for (int s = 0; s < splits; s++) sum += partial[(long)s * N + n];
    y[n] = alpha * sum;
}
";
    }
}
