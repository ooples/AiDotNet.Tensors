// Copyright (c) AiDotNet. All rights reserved.
// GLSL compute kernels for Vulkan sparse ops (CSR SpMM + SDDMM).
// Mirrors the OpenCL csr_spmm / sddmm kernels bit-for-bit so the runtime-verified
// OpenCL correctness carries over to the Vulkan tier.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// GLSL compute-shader source for Vulkan sparse ops. Compiled to SPIR-V at
/// runtime via <see cref="VulkanGlslCompiler"/> and dispatched through the
/// existing <c>GlslQuintOp</c> pipeline machinery.
///
/// <para>The kernels use a flat 1D dispatch (one work-item per output element)
/// because the shared Vulkan dispatch path is 1D
/// (<c>vkCmdDispatch(cmdBuffer, workgroupCount, 1, 1)</c>). Row/col decoding
/// happens inside the shader (row = id / N, col = id % N for SpMM).</para>
/// </summary>
internal static class VulkanSparseKernels
{
    /// <summary>
    /// CSR SpMM — <c>C[M,N] = A[M,K] · B[K,N]</c> where A is sparse (CSR).
    /// One work-item per output element (row, col). Mirrors
    /// <c>OpenCL.Kernels.CsrSparseKernels.csr_spmm</c>.
    /// </summary>
    public const string CsrSpMM = @"#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer CsrValues { float csrValues[]; };
layout(std430, binding = 1) readonly buffer CsrColIdx { int csrColIndices[]; };
layout(std430, binding = 2) readonly buffer CsrRowPtr { int csrRowPointers[]; };
layout(std430, binding = 3) readonly buffer DenseB    { float denseB[]; };
layout(std430, binding = 4) writeonly buffer Output   { float outC[]; };

layout(push_constant) uniform PC {
    uint M;
    uint K;
    uint N;
    uint nnz;
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total = pc.M * pc.N;
    if (gid >= total) return;

    uint row = gid / pc.N;
    uint col = gid - row * pc.N;

    int rowStart = csrRowPointers[row];
    int rowEnd   = csrRowPointers[row + 1u];

    float sum = 0.0;
    for (int i = rowStart; i < rowEnd; i++) {
        int colA  = csrColIndices[i];
        float vA  = csrValues[i];
        sum += vA * denseB[uint(colA) * pc.N + col];
    }

    outC[gid] = sum;
}
";

    /// <summary>
    /// Threads-per-workgroup used by the collaborative SDDMM kernel. Kept identical to
    /// <c>VulkanKernels.WorkgroupSize</c> so the framework's <c>CalculateWorkgroupCount</c>
    /// helper (which divides <c>dispatchSize</c> by 256) produces the workgroup-per-nnz
    /// launch geometry the collaborative kernel expects.
    /// </summary>
    public const int SddmmCollabWorkgroupSize = 256;

    /// <summary>
    /// SDDMM — for each pattern entry p, <c>output[p] = Σ_k x[rowIndices[p], k] · y[colIndices[p], k]</c>.
    /// One work-item per pattern non-zero. Used by the pattern-preserving
    /// sparse-matmul backward's dA. Mirrors <c>OpenCL.Kernels.CsrSparseKernels.sddmm</c>.
    /// </summary>
    public const string Sddmm = @"#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer RowIdx { int rowIndices[]; };
layout(std430, binding = 1) readonly buffer ColIdx { int colIndices[]; };
layout(std430, binding = 2) readonly buffer X      { float x[]; };
layout(std430, binding = 3) readonly buffer Y      { float y[]; };
layout(std430, binding = 4) writeonly buffer Out   { float outVals[]; };

layout(push_constant) uniform PC {
    uint nnz;
    uint innerK;
} pc;

void main() {
    uint p = gl_GlobalInvocationID.x;
    if (p >= pc.nnz) return;

    uint xoff = uint(rowIndices[p]) * pc.innerK;
    uint yoff = uint(colIndices[p]) * pc.innerK;

    float sum = 0.0;
    for (uint k = 0u; k < pc.innerK; k++) {
        sum += x[xoff + k] * y[yoff + k];
    }

    outVals[p] = sum;
}
";

    /// <summary>
    /// Subgroup-collaborative SDDMM — one WORKGROUP per pattern non-zero, 256 threads share
    /// the innerK reduction via shared-memory tree reduce. Beats the base (thread-per-nnz)
    /// kernel when <c>innerK</c> is large enough that a single thread scanning innerK dot-products
    /// serially becomes the bottleneck (empirically innerK ≥ 64 on modern GPUs — the exact
    /// crossover shape of stock cuSPARSE's csrsddmm heuristics). Portable across all Vulkan
    /// versions (uses shared memory + barrier(), no subgroup extensions required).
    ///
    /// <para>Dispatch geometry: pass <c>dispatchSize = nnz * SddmmCollabWorkgroupSize</c> so the
    /// framework's ceil(dispatchSize/256) yields exactly nnz workgroups. Kernel checks
    /// <c>p &lt; nnz</c> to skip the workgroup-count-rounding tail.</para>
    /// </summary>
    public const string SddmmCollab = @"#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer RowIdx { int rowIndices[]; };
layout(std430, binding = 1) readonly buffer ColIdx { int colIndices[]; };
layout(std430, binding = 2) readonly buffer X      { float x[]; };
layout(std430, binding = 3) readonly buffer Y      { float y[]; };
layout(std430, binding = 4) writeonly buffer Out   { float outVals[]; };

layout(push_constant) uniform PC {
    uint nnz;
    uint innerK;
} pc;

shared float partial[256];

void main() {
    uint p = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    // Guard against the workgroup-count-rounding tail — dispatchSize is nnz*256 rounded up
    // to whole workgroups, so the last workgroup may not correspond to a real non-zero.
    // All 256 threads must reach the barriers below, so we zero the partial and fall through.
    float sum = 0.0;
    if (p < pc.nnz) {
        uint xoff = uint(rowIndices[p]) * pc.innerK;
        uint yoff = uint(colIndices[p]) * pc.innerK;

        // Grid-strided accumulation over innerK — each thread walks every 256th element.
        // Coalesced reads inside the FMA loop (consecutive tids read consecutive x/y bytes).
        for (uint k = tid; k < pc.innerK; k += 256u) {
            sum += x[xoff + k] * y[yoff + k];
        }
    }
    partial[tid] = sum;
    barrier();

    // Shared-memory tree reduction: 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1.
    // barrier() between each halving because writes on step N are read on step N+1.
    if (tid < 128u) partial[tid] += partial[tid + 128u]; barrier();
    if (tid <  64u) partial[tid] += partial[tid +  64u]; barrier();
    if (tid <  32u) partial[tid] += partial[tid +  32u]; barrier();
    if (tid <  16u) partial[tid] += partial[tid +  16u]; barrier();
    if (tid <   8u) partial[tid] += partial[tid +   8u]; barrier();
    if (tid <   4u) partial[tid] += partial[tid +   4u]; barrier();
    if (tid <   2u) partial[tid] += partial[tid +   2u]; barrier();
    if (tid <   1u) partial[tid] += partial[tid +   1u];

    if (tid == 0u && p < pc.nnz) {
        outVals[p] = partial[0];
    }
}
";
}
