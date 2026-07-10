// Copyright (c) AiDotNet. All rights reserved.
// Metal Shading Language (MSL) kernels for sparse ops (CSR SpMM + SDDMM).
// Mirrors the OpenCL csr_spmm / sddmm kernels bit-for-bit so the runtime-verified
// OpenCL correctness carries over to the Metal tier. Replaces the previous
// CPU download-compute-upload stub in MetalBackend.Sparse.CsrSpMM with a real
// GPU dispatch, and adds the SDDMM primitive the sparse-autograd backward needs.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    /// <summary>
    /// MSL sources for the Metal sparse compute library. Compiled at backend
    /// startup via <c>_shaderLibrary.CompileLibrary("Sparse", MetalSparseKernels.Source)</c>
    /// and dispatched through the standard <c>GetPipeline + DispatchThreadgroups</c>
    /// pattern used by the rest of MetalBackend.
    ///
    /// <para>The kernels use a flat 1D dispatch (one thread per output element)
    /// for parity with the Vulkan and OpenCL versions and to match the
    /// <c>Calculate1DDispatch</c> helper. Row/col decoding for SpMM happens
    /// inside the shader (<c>row = gid / N</c>, <c>col = gid % N</c>).</para>
    /// </summary>
    internal static class MetalSparseKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "sparse_csr_spmm",
            "sparse_sddmm",
            "sparse_sddmm_collab",
        };

        /// <summary>
        /// Threads-per-threadgroup used by the collaborative SDDMM kernel — kept at 256
        /// (same as <c>sparse_sddmm_collab</c>'s <c>threads_per_threadgroup</c>) so the
        /// host-side dispatch geometry matches the kernel's shared-memory reduction depth.
        /// </summary>
        public const int SddmmCollabThreadgroupSize = 256;

        public const string Source = @"
#include <metal_stdlib>
using namespace metal;

// CSR SpMM: C[M,N] = A[M,K] * B[K,N] where A is sparse (CSR format).
// One thread per output element (row, col); row = gid / N, col = gid - row * N.
// Mirrors OpenCL.Kernels.CsrSparseKernels.csr_spmm bit-for-bit.
kernel void sparse_csr_spmm(
    device const float* csrValues       [[buffer(0)]],
    device const int*   csrColIndices   [[buffer(1)]],
    device const int*   csrRowPointers  [[buffer(2)]],
    device const float* denseB          [[buffer(3)]],
    device float*       outC            [[buffer(4)]],
    constant int&       M               [[buffer(5)]],
    constant int&       K               [[buffer(6)]],
    constant int&       N               [[buffer(7)]],
    constant int&       nnz             [[buffer(8)]],
    uint gid                            [[thread_position_in_grid]])
{
    int total = M * N;
    if ((int)gid >= total) return;

    int row = (int)gid / N;
    int col = (int)gid - row * N;

    int rowStart = csrRowPointers[row];
    int rowEnd   = csrRowPointers[row + 1];

    float sum = 0.0f;
    for (int i = rowStart; i < rowEnd; i++)
    {
        int   colA = csrColIndices[i];
        float valA = csrValues[i];
        sum += valA * denseB[colA * N + col];
    }
    outC[gid] = sum;
}

// SDDMM: for each pattern non-zero p, output[p] = sum_k x[rowIndices[p], k] * y[colIndices[p], k].
// One thread per pattern non-zero. Used by the pattern-preserving sparse-matmul
// backward's dA. Mirrors OpenCL.Kernels.CsrSparseKernels.sddmm bit-for-bit.
kernel void sparse_sddmm(
    device const int*   rowIndices [[buffer(0)]],
    device const int*   colIndices [[buffer(1)]],
    device const float* x          [[buffer(2)]],
    device const float* y          [[buffer(3)]],
    device float*       outVals    [[buffer(4)]],
    constant int&       nnz        [[buffer(5)]],
    constant int&       innerK     [[buffer(6)]],
    uint gid                       [[thread_position_in_grid]])
{
    int p = (int)gid;
    if (p >= nnz) return;

    int xoff = rowIndices[p] * innerK;
    int yoff = colIndices[p] * innerK;

    float sum = 0.0f;
    for (int k = 0; k < innerK; k++)
    {
        sum += x[xoff + k] * y[yoff + k];
    }
    outVals[p] = sum;
}

// Collaborative SDDMM — one THREADGROUP per pattern non-zero, 256 threads share the innerK
// reduction via threadgroup memory (Metal's equivalent of GLSL shared memory) + a tree
// reduction. Beats sparse_sddmm when innerK is large enough that the base kernel's
// serial per-thread reduction becomes the bottleneck (empirically innerK >= 64 on
// Apple Silicon). Dispatch geometry: one threadgroup per non-zero, threadgroups.x = nnz,
// threadsPerThreadgroup.x = 256. Portable across all Metal versions (no simdgroup ops).
kernel void sparse_sddmm_collab(
    device const int*   rowIndices [[buffer(0)]],
    device const int*   colIndices [[buffer(1)]],
    device const float* x          [[buffer(2)]],
    device const float* y          [[buffer(3)]],
    device float*       outVals    [[buffer(4)]],
    constant int&       nnz        [[buffer(5)]],
    constant int&       innerK     [[buffer(6)]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint gid                       [[threadgroup_position_in_grid]],
    threadgroup float* partial     [[threadgroup(0)]])
{
    int p = (int)gid;

    // All 256 threads must reach the threadgroup_barrier calls below (Metal spec:
    // control flow must be uniform across the threadgroup at a barrier). We compute
    // the partial for the real-work case and zero the partial for the tail case.
    float sum = 0.0f;
    if (p < nnz) {
        int xoff = rowIndices[p] * innerK;
        int yoff = colIndices[p] * innerK;
        // Grid-strided walk over innerK; consecutive threads read consecutive memory.
        for (int k = (int)tid; k < innerK; k += 256) {
            sum += x[xoff + k] * y[yoff + k];
        }
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Shared-memory tree reduction: 256 -> 128 -> 64 -> ... -> 1.
    if (tid < 128u) partial[tid] += partial[tid + 128u]; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  64u) partial[tid] += partial[tid +  64u]; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  32u) partial[tid] += partial[tid +  32u]; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  16u) partial[tid] += partial[tid +  16u]; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <   8u) partial[tid] += partial[tid +   8u]; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <   4u) partial[tid] += partial[tid +   4u]; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <   2u) partial[tid] += partial[tid +   2u]; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <   1u) partial[tid] += partial[tid +   1u];

    if (tid == 0u && p < nnz) {
        outVals[p] = partial[0];
    }
}
";
    }
}
