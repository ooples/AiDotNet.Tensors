#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxSparseGraphCompletionStatus
{
    MissingDirectPtx,
    ImplementedDirectPtx
}

internal sealed record DirectPtxSparseGraphCompletionEntry(
    string Operation,
    string Surface,
    DirectPtxSparseGraphCompletionStatus Status);

/// <summary>
/// Machine-readable #852 completion gate. A PR must not be opened until every
/// entry is ImplementedDirectPtx; benchmark promotion is tracked separately.
/// </summary>
internal static class DirectPtxSparseGraphCompletionLedger
{
    internal static IReadOnlyList<DirectPtxSparseGraphCompletionEntry> All { get; } = Create();
    internal static IReadOnlyList<DirectPtxSparseGraphCompletionEntry> Missing =>
        All.Where(entry => entry.Status != DirectPtxSparseGraphCompletionStatus.ImplementedDirectPtx).ToArray();
    internal static bool IsComplete => Missing.Count == 0;

    internal static void RequireComplete()
    {
        if (!IsComplete)
            throw new InvalidOperationException(
                $"Issue #852 is not PR-ready: {Missing.Count} sparse/graph operations still lack direct PTX.");
    }

    private static IReadOnlyList<DirectPtxSparseGraphCompletionEntry> Create()
    {
        var entries = new List<DirectPtxSparseGraphCompletionEntry>();
        void Add(string surface, DirectPtxSparseGraphCompletionStatus status, params string[] names)
        {
            foreach (string name in names) entries.Add(new(name, surface, status));
        }

        Add("CudaSparseKernels", DirectPtxSparseGraphCompletionStatus.ImplementedDirectPtx,
            "csr_spmm", "csr_spmm_vec4", "sddmm", "csr_spmm_warp", "csr_spmm_bias",
            "csr_spmm_double", "csr_spmm_bias_relu", "gather_source_features",
            "gather_target_features", "scatter_add_edges_deterministic", "segment_sum",
            "segment_sum_deterministic", "segment_mean", "segment_mean_deterministic", "segment_max",
            "csr_spmm_backward_b", "csr_spmm_backward_b_deterministic", "csr_spmm_backward_values",
            "zero_buffer", "init_neg_inf", "degree_normalize", "symmetric_degree_normalize",
            "scatter_add_edges");
        Add("CudaSnnKernels", DirectPtxSparseGraphCompletionStatus.ImplementedDirectPtx,
            "enforce_2x4_sparsity", "decompress_2x4_sparse", "sparse_gemm_2x4",
            "sparse_gemm_bias_relu", "sparse_2_4_matmul_baseline");
        Add("CudaNeuralNetKernels", DirectPtxSparseGraphCompletionStatus.ImplementedDirectPtx,
            "scatter_add", "scatter_add_deterministic", "scatter_add_accumulate_deterministic",
            "scatter_add_batched", "scatter_add_batched_deterministic",
            "scatter_mean_accumulate", "scatter_mean_accumulate_deterministic", "scatter_mean_normalize");
        Add("CudaInstantNgpKernels", DirectPtxSparseGraphCompletionStatus.ImplementedDirectPtx,
            "resident_scatter_max_argmax_rows", "resident_scatter_add_rows",
            "resident_scatter_add_backward_rows", "resident_scatter_mean_backward_rows");
        Add("CudaBackend.CSR", DirectPtxSparseGraphCompletionStatus.ImplementedDirectPtx,
            "csr_segmented_max", "csr_segmented_min", "csr_segmented_stddev");
        Add("CudaSnnKernels", DirectPtxSparseGraphCompletionStatus.MissingDirectPtx,
            "sparse_2_4_matmul_mma_sp");
        Add("CudaNeuralNetKernels", DirectPtxSparseGraphCompletionStatus.MissingDirectPtx,
            "scatter_max");
        Add("CudaInstantNgpKernels", DirectPtxSparseGraphCompletionStatus.MissingDirectPtx,
            "resident_uniform_mesh_laplacian",
            "resident_scatter_mean_rows_counts",
            "resident_scatter_softmax_rows", "resident_scatter_max_backward_rows",
            "resident_scatter_softmax_backward_rows");
        Add("CudaCapsuleKernels", DirectPtxSparseGraphCompletionStatus.MissingDirectPtx,
            "squash", "squash_backward", "capsule_predictions", "capsule_transform",
            "capsule_weighted_sum", "capsule_agreement");
        Add("CudaMeshPoolKernels", DirectPtxSparseGraphCompletionStatus.MissingDirectPtx,
            "mesh_pool_compute_scores", "mesh_pool_gather", "mesh_pool_backward",
            "mesh_pool_backward_deterministic", "mesh_pool_importance_backward", "mesh_pool_zero_grad",
            "mesh_pool_softmax_find_max", "mesh_pool_softmax_final_max", "mesh_pool_softmax_exp_sum",
            "mesh_pool_softmax_final_sum", "mesh_pool_softmax_normalize", "mesh_pool_softmax_scores",
            "mesh_pool_weighted_gather", "mesh_pool_weighted_backward",
            "mesh_pool_weighted_backward_deterministic", "mesh_pool_scores_backward");
        Add("CudaOptimizerKernels", DirectPtxSparseGraphCompletionStatus.MissingDirectPtx,
            "sparse_sgd_update", "sparse_sgd_momentum_update", "sparse_adam_update",
            "sparse_adamw_update", "sparse_rmsprop_update", "sparse_adagrad_update", "sparse_nag_update",
            "sparse_adadelta_update", "sparse_amsgrad_update", "sparse_adamax_update",
            "sparse_lion_update", "sparse_nadam_update", "sparse_ftrl_update", "sparse_proximal_l1_update");
        Add("ISparseEngine", DirectPtxSparseGraphCompletionStatus.MissingDirectPtx,
            "SpMV", "SpMVTranspose", "SpMM", "SpSpMM", "AddSparseDense", "MultiplySparseDense",
            "SparseGather", "SparseScatter", "SparseScatterAdd", "SparseToDense", "DenseToSparse",
            "Coalesce", "SparseTranspose", "SparseMatMul", "SparseMatMulPatternPreserving", "SparseAddMM",
            "SparseSampledAddMM", "SparseSpGeMM", "SparseSum", "SparseMean", "SparseSoftmax",
            "SparseLogSoftmax");
        Add("DirectGpuTensorEngine", DirectPtxSparseGraphCompletionStatus.MissingDirectPtx,
            "FusedSparseLinear", "ScatterMean", "ScatterAddBackward", "Gather", "ScatterReduce");
        return entries;
    }
}
#endif
