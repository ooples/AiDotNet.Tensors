// Copyright (c) AiDotNet. All rights reserved.
// Vulkan sparse compute pipeline (CSR SpMM + SDDMM) — mirrors the runtime-verified
// OpenCL sparse backend so all Vulkan-capable devices (mobile / cross-vendor / MoltenVK)
// get the same tape-aware sparse autograd acceleration.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    /// <summary>
    /// CSR · dense → dense: <c>C[M,N] = A[M,K] · B[K,N]</c> with A supplied as CSR
    /// (values / colIndices / rowPointers). All five buffers are GPU-resident.
    /// Bindings match the shader layout in <see cref="VulkanSparseKernels.CsrSpMM"/>
    /// (0=values, 1=colIdx, 2=rowPtr, 3=denseB, 4=output). Push constants supply
    /// M/K/N/nnz. Dispatch is 1D with one work-item per output element (row, col);
    /// the shader decodes row/col from <c>gl_GlobalInvocationID.x</c>.
    /// </summary>
    public void CsrSpMM(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (M <= 0 || N <= 0) return;

        // one work-item per output element (row, col) ∈ [0, M) × [0, N)
        long total = (long)M * N;
        if (total > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(M),
                $"CSR SpMM output too large for a single Vulkan dispatch (M*N = {total}). " +
                "Tile the problem or route to a vendor SpMM backend.");

        uint[] pc = { (uint)M, (uint)K, (uint)N, (uint)nnz };
        GlslQuintOp(
            VulkanSparseKernels.CsrSpMM,
            csrValues, csrColIndices, csrRowPointers, denseB, output,
            dispatchSize: (int)total,
            pushConstants: pc,
            pushConstantSize: sizeof(uint) * 4u);
    }

    /// <summary>
    /// Empirically-picked crossover: below this innerK, the thread-per-nnz base kernel
    /// wins because each thread's serial dot-product fits in a few cycles and the
    /// collaborative kernel's 256-thread tree reduction is pure overhead. Above this,
    /// the reduction latency dominates and the collaborative kernel wins. 64 matches the
    /// typical attention head_dim boundary — smaller innerK is usually per-token features.
    /// </summary>
    private const int SddmmCollabInnerKThreshold = 64;

    /// <summary>
    /// SDDMM: <c>output[p] = Σ_k x[rowIndices[p], k] · y[colIndices[p], k]</c> for each
    /// pattern non-zero p ∈ [0, nnz). All five buffers are GPU-resident. Bindings match
    /// <see cref="VulkanSparseKernels.Sddmm"/> (0=rowIdx, 1=colIdx, 2=x, 3=y, 4=output).
    ///
    /// <para><b>Adaptive dispatch (beyond-industry-standard):</b> for large <c>innerK</c>
    /// (typical attention head_dim ≥ 64), routes to
    /// <see cref="VulkanSparseKernels.SddmmCollab"/> — a workgroup-per-nnz kernel where
    /// 256 threads collaborate on the innerK reduction via shared-memory tree reduce.
    /// Reduces per-nnz latency from O(innerK) to O(log₂ 256) + O(innerK/256). Stock
    /// cuSPARSE / rocSPARSE's csrsddmm uses fixed thread-per-nnz for all shapes.</para>
    ///
    /// <para>Below the threshold, falls back to the base thread-per-nnz kernel where the
    /// serial dot-product finishes before the collaborative kernel's reduction barriers pay off.</para>
    /// </summary>
    public void CsrSddmm(
        IGpuBuffer rowIndices,
        IGpuBuffer colIndices,
        IGpuBuffer x,
        IGpuBuffer y,
        IGpuBuffer output,
        int nnz, int innerK)
    {
        if (nnz <= 0) return;

        uint[] pc = { (uint)nnz, (uint)innerK };

        if (innerK >= SddmmCollabInnerKThreshold)
        {
            // Workgroup-per-nnz: pass dispatchSize = nnz * 256 so the framework's
            // CalculateWorkgroupCount(dispatchSize) = ceil(dispatchSize/256) yields exactly
            // nnz workgroups (with the kernel guarding against the rounding tail).
            long dispatch = (long)nnz * VulkanSparseKernels.SddmmCollabWorkgroupSize;
            if (dispatch > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(nnz),
                    $"Collaborative SDDMM dispatch overflow (nnz*256 = {dispatch}); fall back to base kernel.");
            GlslQuintOp(
                VulkanSparseKernels.SddmmCollab,
                rowIndices, colIndices, x, y, output,
                dispatchSize: (int)dispatch,
                pushConstants: pc,
                pushConstantSize: sizeof(uint) * 2u);
        }
        else
        {
            GlslQuintOp(
                VulkanSparseKernels.Sddmm,
                rowIndices, colIndices, x, y, output,
                dispatchSize: nnz,
                pushConstants: pc,
                pushConstantSize: sizeof(uint) * 2u);
        }
    }
}
