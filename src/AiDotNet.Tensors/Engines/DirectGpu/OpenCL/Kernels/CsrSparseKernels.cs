// Copyright (c) AiDotNet. All rights reserved.
// OpenCL CSR sparse matrix kernels for general sparsity patterns.
// Supports SpMM and GNN message passing operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for CSR (Compressed Sparse Row) sparse matrix operations
/// including SpMM and Graph Neural Network message passing.
/// </summary>
internal static class CsrSparseKernels
{
    public static string GetSource()
    {
        return @"
// ===========================================================================
// CSR SPARSE MATRIX - DENSE MATRIX MULTIPLICATION (SpMM)
// ===========================================================================

// CSR SpMM: C[M,N] = A[M,K] * B[K,N] where A is sparse (CSR format)
__kernel void csr_spmm(
    __global const float* restrict csrValues,      // [nnz] - non-zero values of A
    __global const int* restrict csrColIndices,    // [nnz] - column indices
    __global const int* restrict csrRowPointers,   // [M+1] - row pointers
    __global const float* restrict denseB,         // [K,N] - dense matrix B
    __global float* restrict output,               // [M,N] - output matrix C
    int M, int K, int N, int nnz)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row >= M || col >= N) return;

    int rowStart = csrRowPointers[row];
    int rowEnd = csrRowPointers[row + 1];

    float sum = 0.0f;
    for (int i = rowStart; i < rowEnd; i++)
    {
        int colA = csrColIndices[i];
        float valA = csrValues[i];
        sum += valA * denseB[colA * N + col];
    }

    output[row * N + col] = sum;
}

// CSR SpMM with fused bias addition
__kernel void csr_spmm_bias(
    __global const float* restrict csrValues,
    __global const int* restrict csrColIndices,
    __global const int* restrict csrRowPointers,
    __global const float* restrict denseB,
    __global const float* restrict bias,
    __global float* restrict output,
    int M, int K, int N, int nnz)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row >= M || col >= N) return;

    int rowStart = csrRowPointers[row];
    int rowEnd = csrRowPointers[row + 1];

    float sum = bias[col];
    for (int i = rowStart; i < rowEnd; i++)
    {
        int colA = csrColIndices[i];
        float valA = csrValues[i];
        sum += valA * denseB[colA * N + col];
    }

    output[row * N + col] = sum;
}

// CSR SpMM with fused bias and ReLU
__kernel void csr_spmm_bias_relu(
    __global const float* restrict csrValues,
    __global const int* restrict csrColIndices,
    __global const int* restrict csrRowPointers,
    __global const float* restrict denseB,
    __global const float* restrict bias,
    __global float* restrict output,
    int M, int K, int N, int nnz)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row >= M || col >= N) return;

    int rowStart = csrRowPointers[row];
    int rowEnd = csrRowPointers[row + 1];

    float sum = bias[col];
    for (int i = rowStart; i < rowEnd; i++)
    {
        int colA = csrColIndices[i];
        float valA = csrValues[i];
        sum += valA * denseB[colA * N + col];
    }

    output[row * N + col] = fmax(sum, 0.0f);
}

// ===========================================================================
// GRAPH NEURAL NETWORK MESSAGE PASSING OPERATIONS
// ===========================================================================

// Scatter-add for edge-based message passing
// For each edge (src -> tgt), atomically adds input[src, :] * edgeValue to output[tgt, :]
__kernel void scatter_add_edges(
    __global const float* restrict input,          // [numNodes, features]
    __global const int* restrict sourceIndices,    // [numEdges]
    __global const int* restrict targetIndices,    // [numEdges]
    __global const float* restrict edgeValues,     // [numEdges] or unused
    __global float* output,                        // [numNodes, features]
    int numNodes, int numEdges, int features,
    int hasEdgeValues)
{
    int edge = get_global_id(1);
    int feat = get_global_id(0);

    if (edge >= numEdges || feat >= features) return;

    int src = sourceIndices[edge];
    int tgt = targetIndices[edge];

    float value = input[src * features + feat];
    if (hasEdgeValues != 0)
    {
        value *= edgeValues[edge];
    }

    // OpenCL doesn't have atomicAdd for floats in all versions
    // Use atomic_xchg loop for portability.
    // NON-DETERMINISTIC (issue #382); see scatter_add_edges_deterministic below.
    __global volatile float* addr = &output[tgt * features + feat];
    float old_val, new_val;
    do {
        old_val = *addr;
        new_val = old_val + value;
    } while (atomic_cmpxchg((__global volatile int*)addr,
             as_int(old_val), as_int(new_val)) != as_int(old_val));
}

// scatter_add_edges — bit-deterministic variant (issue #382).
// One work-item per (tgt, feat) output cell scans edges in fixed ascending order.
__kernel void scatter_add_edges_deterministic(
    __global const float* restrict input,
    __global const int* restrict sourceIndices,
    __global const int* restrict targetIndices,
    __global const float* restrict edgeValues,
    __global float* output,
    int numNodes, int numEdges, int features,
    int hasEdgeValues)
{
    int tgt = get_global_id(1);
    int feat = get_global_id(0);
    if (tgt >= numNodes || feat >= features) return;

    float sum = 0.0f;
    for (int edge = 0; edge < numEdges; edge++) {
        if (targetIndices[edge] == tgt) {
            int src = sourceIndices[edge];
            float value = input[src * features + feat];
            if (hasEdgeValues != 0) value *= edgeValues[edge];
            sum += value;
        }
    }
    output[tgt * features + feat] += sum;
}

// Gather source features for each edge
__kernel void gather_source_features(
    __global const float* restrict input,          // [numNodes, features]
    __global const int* restrict sourceIndices,    // [numEdges]
    __global float* restrict output,               // [numEdges, features]
    int numEdges, int features)
{
    int edge = get_global_id(1);
    int feat = get_global_id(0);

    if (edge >= numEdges || feat >= features) return;

    int src = sourceIndices[edge];
    output[edge * features + feat] = input[src * features + feat];
}

// Gather target features for each edge
__kernel void gather_target_features(
    __global const float* restrict input,          // [numNodes, features]
    __global const int* restrict targetIndices,    // [numEdges]
    __global float* restrict output,               // [numEdges, features]
    int numEdges, int features)
{
    int edge = get_global_id(1);
    int feat = get_global_id(0);

    if (edge >= numEdges || feat >= features) return;

    int tgt = targetIndices[edge];
    output[edge * features + feat] = input[tgt * features + feat];
}

// Segment sum - sum items by segment ID
__kernel void segment_sum(
    __global const float* restrict input,          // [numItems, features]
    __global const int* restrict segmentIds,       // [numItems]
    __global float* output,                        // [numSegments, features]
    int numItems, int features)
{
    int item = get_global_id(1);
    int feat = get_global_id(0);

    if (item >= numItems || feat >= features) return;

    int segment = segmentIds[item];
    float value = input[item * features + feat];

    __global volatile float* addr = &output[segment * features + feat];
    float old_val, new_val;
    do {
        old_val = *addr;
        new_val = old_val + value;
    } while (atomic_cmpxchg((__global volatile int*)addr,
             as_int(old_val), as_int(new_val)) != as_int(old_val));
}

// ===========================================================================
// UTILITY OPERATIONS
// ===========================================================================

// Zero buffer
__kernel void zero_buffer(__global float* output, int size)
{
    int idx = get_global_id(0);
    if (idx < size) output[idx] = 0.0f;
}

// Initialize to negative infinity
__kernel void init_neg_inf(__global float* output, int size)
{
    int idx = get_global_id(0);
    if (idx < size) output[idx] = -3.402823466e+38f;
}

// Degree normalization for GCN
__kernel void degree_normalize(
    __global const float* restrict input,
    __global const float* restrict degrees,
    __global float* restrict output,
    int numNodes, int features, float epsilon)
{
    int node = get_global_id(1);
    int feat = get_global_id(0);

    if (node >= numNodes || feat >= features) return;

    float deg = degrees[node];
    float normFactor = rsqrt(deg + epsilon);
    output[node * features + feat] = input[node * features + feat] * normFactor;
}

// Symmetric degree normalization for edges
__kernel void symmetric_degree_normalize(
    __global const float* restrict edgeValues,
    __global const int* restrict sourceIndices,
    __global const int* restrict targetIndices,
    __global const float* restrict srcDegrees,
    __global const float* restrict tgtDegrees,
    __global float* restrict output,
    int numEdges, float epsilon)
{
    int edge = get_global_id(0);
    if (edge >= numEdges) return;

    int src = sourceIndices[edge];
    int tgt = targetIndices[edge];

    float srcDeg = srcDegrees[src];
    float tgtDeg = tgtDegrees[tgt];

    float normFactor = rsqrt(srcDeg + epsilon) * rsqrt(tgtDeg + epsilon);
    output[edge] = edgeValues[edge] * normFactor;
}

// SDDMM (sampled dense-dense matmul): for each pattern entry p, compute the (i,j) element of
// x . y^T restricted to the pattern -> output[p] = sum_k x[i,k] * y[j,k], where i=rowIndices[p]
// and j=colIndices[p]. One work-item per pattern non-zero. This is the pattern-preserving
// sparse-matmul backward's dA (x = grad, y = B).
__kernel void sddmm(
    __global const int* restrict rowIndices,
    __global const int* restrict colIndices,
    __global const float* restrict x,
    __global const float* restrict y,
    __global float* restrict output,
    int nnz, int innerK)
{
    int p = get_global_id(0);
    if (p >= nnz) return;

    int xoff = rowIndices[p] * innerK;
    int yoff = colIndices[p] * innerK;

    float sum = 0.0f;
    for (int k = 0; k < innerK; k++)
        sum += x[xoff + k] * y[yoff + k];

    output[p] = sum;
}

// Collaborative SDDMM: one WORKGROUP per pattern non-zero, 256 work-items share the innerK
// reduction via local memory tree reduce. Beats sddmm when innerK is large enough that a
// single work-item's serial reduction becomes the bottleneck (empirically innerK >= 64).
// One threadgroup per non-zero. Portable across OpenCL 1.2+ (uses barrier() + local memory,
// no cl_khr_subgroups extension required). Matches the Vulkan/Metal collaborative variants.
__kernel void sddmm_collab(
    __global const int* restrict rowIndices,
    __global const int* restrict colIndices,
    __global const float* restrict x,
    __global const float* restrict y,
    __global float* restrict output,
    int nnz, int innerK,
    __local float* partial)
{
    int p = get_group_id(0);
    int tid = get_local_id(0);

    // All 256 work-items must reach the barriers below. Compute partial for real-work
    // case, zero for the workgroup-count-rounding tail.
    float sum = 0.0f;
    if (p < nnz) {
        int xoff = rowIndices[p] * innerK;
        int yoff = colIndices[p] * innerK;
        for (int k = tid; k < innerK; k += 256) {
            sum += x[xoff + k] * y[yoff + k];
        }
    }
    partial[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Local-memory tree reduction: 256 -> 128 -> 64 -> ... -> 1.
    if (tid < 128) partial[tid] += partial[tid + 128]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <  64) partial[tid] += partial[tid +  64]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <  32) partial[tid] += partial[tid +  32]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <  16) partial[tid] += partial[tid +  16]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <   8) partial[tid] += partial[tid +   8]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <   4) partial[tid] += partial[tid +   4]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <   2) partial[tid] += partial[tid +   2]; barrier(CLK_LOCAL_MEM_FENCE);
    if (tid <   1) partial[tid] += partial[tid +   1];

    if (tid == 0 && p < nnz) {
        output[p] = partial[0];
    }
}
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            "csr_spmm",
            "sddmm",
            "sddmm_collab",
            "csr_spmm_bias",
            "csr_spmm_bias_relu",
            "scatter_add_edges",
            "scatter_add_edges_deterministic",
            "gather_source_features",
            "gather_target_features",
            "segment_sum",
            "zero_buffer",
            "init_neg_inf",
            "degree_normalize",
            "symmetric_degree_normalize"
        ];
    }
}
