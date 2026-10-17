using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxSparseCoverageStatus
{
    ExistingBackend,
    ExperimentalDirectPtx,
    PromotedDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxSparseCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxSparseCoverageStatus Status);

/// <summary>
/// Executable issue-#852 inventory: every CUDA sparse, CSR, graph-message and segmented entry point, and every NVRTC
/// kernel in <c>CudaSparseKernels</c>, has one explicit direct-PTX lane. "kernel only" means the NVRTC kernel is
/// compiled but no backend method dispatches it, so it needs an entry point before a PTX port can be wired.
/// </summary>
internal static class DirectPtxSparseCoverageManifest
{
    private const string Csr = "CSR: int32 row pointers [M+1], int32 column indices [nnz], values [nnz]; dense row-major [K,N] and [M,N]";
    private const string Edges = "edge list: int32 source/target [E]; node features row-major [V,F]";
    private const string Segments = "CSR segments: int32 row pointers [M+1]; values or rows in segment order";
    private const string CsrFloatIndex = "CSR: fp32 row pointers [M+1] and column indices [nnz] holding integral values; dense row-major input [K,N] and output [M,N]";

    internal static IReadOnlyList<DirectPtxSparseCoverageCell> All { get; } =
    [
        Experimental("CudaBackend.CsrSpMMVec4", "NVRTC csr_spmm_vec4", "output = A_csr * B, 4 columns per thread", Csr, "FP32",
            "v1 SM86 lanes-per-row exact (M,K,N) cells: (2708,2708,64) (19717,500,128) (10000,10000,32) (512,512,64)"),
        Planned("CudaBackend.CsrSpMM", "NVRTC csr_spmm", "output = A_csr * B, one thread per output", Csr, "FP32",
            "covered by the lanes-per-row family for N % 4 == 0; N % 4 != 0 needs a scalar-tail family"),
        Planned("CudaBackend.CsrSpMMWarp", "NVRTC csr_spmm_warp", "output = A_csr * B, one warp per row", Csr, "FP32",
            "covered by the lanes-per-row family for N % 4 == 0; N % 4 != 0 needs a scalar-tail family"),
        Planned("CudaBackend.CsrSpMMDouble", "NVRTC csr_spmm_double", "output = A_csr * B", Csr, "FP64",
            "lanes-per-row-fp64x2-families"),
        Planned("CudaBackend.CsrSpMMBias", "NVRTC csr_spmm_bias", "output = A_csr * B + bias[N]", Csr + "; bias [N]", "FP32",
            "lanes-per-row family with a fused bias epilogue"),
        Planned("CsrSpMMBiasRelu", "NVRTC csr_spmm_bias_relu (kernel only)", "output = relu(A_csr * B + bias[N])", Csr + "; bias [N]", "FP32",
            "blocked: needs a backend entry point, then lanes-per-row family with a bias + ReLU epilogue"),
        Planned("CudaBackend.CsrSddmm", "NVRTC sddmm", "out[p] = x[row(p),:] . y[col(p),:] at the nonzeros", "COO-like int32 row/col [nnz]; x [M,K], y [N,K] row-major", "FP32",
            "warp-per-nonzero dot families with baked K"),
        Planned("CudaBackend.ScatterAddEdges", "NVRTC scatter_add_edges / scatter_add_edges_deterministic", "out[target[e],:] += w[e] * x[source[e],:]", Edges, "FP32",
            "atomic-row-scatter families; deterministic mode as a CSR-by-target segmented gather"),
        Planned("GatherSourceFeatures", "NVRTC gather_source_features (kernel only)", "out[e,:] = x[source[e],:]", Edges, "FP32",
            "blocked: needs a backend entry point; then the #844 warp-row gather family applies"),
        Planned("GatherTargetFeatures", "NVRTC gather_target_features (kernel only)", "out[e,:] = x[target[e],:]", Edges, "FP32",
            "blocked: needs a backend entry point; then the #844 warp-row gather family applies"),
        Planned("SegmentSum", "NVRTC segment_sum / segment_sum_deterministic (kernel only)", "out[s,:] = sum of rows in segment s", Segments, "FP32",
            "blocked: needs a backend entry point, then segmented-row-reduction families (no atomics: one segment per lane group)"),
        Planned("SegmentMean", "NVRTC segment_mean / segment_mean_deterministic (kernel only)", "out[s,:] = mean of rows in segment s", Segments, "FP32",
            "blocked: needs a backend entry point, then segmented-row-reduction families"),
        Planned("SegmentMax", "NVRTC segment_max (kernel only)", "out[s,:] = max of rows in segment s", Segments, "FP32",
            "blocked: needs a backend entry point, then segmented-row-reduction families"),
        Experimental("CudaBackend.CsrSegmentedMax", "NVRTC csr_segmented_max",
            "output[i,j] = max over the row's nonzeros p of input[col_idx[p], j]; 0 for an empty row", CsrFloatIndex, "FP32",
            "v1 SM86 lanes-per-row exact (M,K,N) cells, the CSR SpMM family's: (2708,2708,64) (19717,500,128) (10000,10000,32) (512,512,64)"),
        Experimental("CudaBackend.CsrSegmentedMin", "NVRTC csr_segmented_min",
            "output[i,j] = min over the row's nonzeros p of input[col_idx[p], j]; 0 for an empty row", CsrFloatIndex, "FP32",
            "v1 SM86 lanes-per-row exact (M,K,N) cells, the CSR SpMM family's: (2708,2708,64) (19717,500,128) (10000,10000,32) (512,512,64)"),
        Planned("CudaBackend.CsrSegmentedStdDev", "NVRTC csr_segmented_stddev", "standard deviation of each CSR row's values", Segments, "FP32",
            "segmented-value-reduction families (two-pass or Welford, fixed order)"),
        Planned("CsrSpMMBackwardB", "NVRTC csr_spmm_backward_b / csr_spmm_backward_b_deterministic (kernel only)", "dB = A_csr^T * dOut", Csr, "FP32",
            "blocked: needs a backend entry point; then a transposed-CSR lanes-per-row family (deterministic, no atomics)"),
        Planned("CsrSpMMBackwardValues", "NVRTC csr_spmm_backward_values (kernel only)", "dValues[p] = dOut[row(p),:] . B[col(p),:]", Csr, "FP32",
            "blocked: needs a backend entry point; then the SDDMM family applies"),
        Planned("DegreeNormalize", "NVRTC degree_normalize / symmetric_degree_normalize (kernel only)", "scale CSR values by 1/deg or 1/sqrt(deg_i deg_j)", Segments, "FP32",
            "blocked: needs a backend entry point; then a fused prologue of the SpMM family"),
        Planned("SparseUtilities", "NVRTC zero_buffer / init_neg_inf", "buffer initialization for the scatter and max kernels", "contiguous vector", "FP32",
            "not ported separately: the PTX families write every output element and need no initialization pass"),
    ];

    private static readonly IReadOnlyDictionary<string, DirectPtxSparseCoverageCell> ByApi =
        All.ToDictionary(cell => cell.Api, StringComparer.Ordinal);

    internal static DirectPtxSparseCoverageCell Get(string api) =>
        ByApi.TryGetValue(api, out DirectPtxSparseCoverageCell? cell)
            ? cell
            : throw new KeyNotFoundException($"No direct-PTX sparse coverage cell for '{api}'.");

    private static DirectPtxSparseCoverageCell Experimental(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxSparseCoverageStatus.ExperimentalDirectPtx);

    private static DirectPtxSparseCoverageCell Planned(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxSparseCoverageStatus.PlannedDirectPtx);
}
