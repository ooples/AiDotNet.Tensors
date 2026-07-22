#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxGatherCoverageStatus
{
    ExistingBackend,
    ExperimentalDirectPtx,
    PromotedDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxGatherCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxGatherCoverageStatus Status);

/// <summary>
/// Executable issue-#844 inventory. Every CUDA embedding, gather/scatter,
/// indexing, segment, and sparse-update boundary has one explicit direct-PTX
/// lane. Shape and layout families remain closed sets: a planned row must be
/// split into exact physical ABI cells before it can be promoted.
/// </summary>
internal static class DirectPtxGatherCoverageManifest
{
    private const string Rows = "canonical contiguous row-major [rows,feature]";
    private const string Indexed = "canonical contiguous source/index/output rows";

    internal static IReadOnlyList<DirectPtxGatherCoverageCell> All { get; } =
    [
        Experimental("CudaBackend.Gather", "NVRTC embedding_forward", "output[i,:] = source[index[i],:]", Indexed, "FP32 rows; INT32 indices", "v1 Ampere warp-row F64/F128 exact-shape cells"),
        Planned("CudaBackend.IndexSelect", "NVRTC index_select", "select rows by index along an axis", Indexed, "FP32 rows; INT32 indices", "warp-row-index-select-exact-shape-families"),
        Planned("CudaBackend.EmbeddingBackward", "NVRTC embedding_backward", "scatter-add gradient rows into the table", Indexed, "FP32 rows; INT32 indices", "atomic-scatter-add-row-families"),
        Planned("CudaBackend.GatherBackward", "NVRTC gather_backward", "scatter-add output gradient to source rows", Indexed, "FP32 rows; INT32 indices", "atomic-scatter-add-row-families"),
        Planned("CudaBackend.ScatterRows", "NVRTC scatter_rows", "output[index[i],:] = source[i,:]", Indexed, "FP32 rows; INT32 indices", "warp-row-scatter-exact-shape-families"),
        Planned("CudaBackend.ScatterAddRows", "NVRTC scatter_add_rows", "output[index[i],:] += source[i,:]", Indexed, "FP32 rows; INT32 indices", "atomic-scatter-add-row-families"),
        Planned("CudaBackend.ScatterReduce", "NVRTC scatter_reduce", "reduce source into output by index and mode", "canonical contiguous source/index/output segments", "FP32 rows; INT32 indices", "segmented-scatter-reduce-mode-families"),
        Planned("CudaBackend.GatherNd", "NVRTC gather_nd", "gather by multi-dimensional coordinate", "canonical contiguous coordinate/output rows", "FP32 rows; INT32 coordinates", "baked-rank-coordinate-gather-families"),
        Planned("CudaBackend.GatherAxis", "NVRTC gather_axis", "gather along an arbitrary axis", "canonical contiguous input/index/output with baked axis stride", "FP32 rows; INT32 indices", "baked-axis-stride-gather-families"),
        Planned("CudaBackend.TakeAlongAxis", "NVRTC take_along_axis", "elementwise index along an axis", Rows, "FP32 rows; INT32 indices", "warp-row-take-along-axis-families"),
        Planned("CudaBackend.OneHot", "NVRTC one_hot", "index to baked-width one-hot rows", Indexed, "INT32 indices; FP32 output", "baked-width-one-hot-families"),
        Planned("CudaBackend.SegmentSum", "NVRTC segment_sum", "sum source rows within contiguous segments", "canonical contiguous source/segment-offset rows", "FP32 rows; INT32 offsets", "segmented-scan-offset-families"),
        Planned("CudaBackend.SegmentMean", "NVRTC segment_mean", "mean source rows within contiguous segments", "canonical contiguous source/segment-offset rows", "FP32 rows; INT32 offsets", "segmented-scan-offset-families"),
        Planned("CudaBackend.SparseUpdateRows", "NVRTC sparse_update_rows", "apply a sparse row update set to a dense table", Indexed, "FP32 rows; INT32 indices", "atomic-or-exclusive-sparse-update-families"),
        Planned("CudaBackend.MaskedSelect", "NVRTC masked_select", "compact rows where a boolean mask is set", "canonical contiguous input/mask/output rows", "FP32 rows; boolean mask", "prefix-sum-compaction-families"),
        Planned("DirectGpuEngine.Gather", "array upload + CudaBackend.Gather + download", "public array embedding gather", Indexed, "generic public; CUDA FP32/INT32", "public-array-gather-routing"),
        Planned("DirectGpuTensorEngine.TensorGather", "resident CudaBackend.Gather or CPU fallback", "public tensor gather over arbitrary axis", "logical strided tensor; canonical contiguous last-axis fast path", "generic public; CUDA FP32/INT32", "public-tensor-gather-routing-and-materialization"),
        Planned("DirectGpuTensorEngine.IEngine.GatherInto", "resident route, uploaded backend route, or CPU fallback", "gather into existing destination", "logical arbitrary axis; canonical contiguous admitted view", "generic public; CUDA FP32/INT32", "gather-into-routing-and-layout-proof"),
        Planned("DirectGpuTensorEngine.TensorEmbedding", "CudaBackend.Gather", "public embedding lookup", Indexed, "generic public; CUDA FP32/INT32", "public-embedding-routing"),
        Planned("DirectGpuTensorEngine.TensorEmbeddingBackward", "CudaBackend.EmbeddingBackward", "public embedding table gradient", Indexed, "generic public; CUDA FP32/INT32", "public-embedding-backward-routing"),
        Planned("DirectGpuTensorEngine.TensorScatter", "CudaBackend.ScatterRows or CPU fallback", "public tensor scatter over an axis", "logical arbitrary axis; contiguous scattered-axis admitted view", "generic public; CUDA FP32/INT32", "public-scatter-routing-and-axis-materialization"),
        Planned("DirectGpuTensorEngine.TensorIndexSelect", "CudaBackend.IndexSelect or CPU fallback", "public tensor row selection", "logical arbitrary axis; contiguous selected-axis admitted view", "generic public; CUDA FP32/INT32", "public-index-select-routing"),
    ];

    private static readonly IReadOnlyDictionary<string, DirectPtxGatherCoverageCell> ByApi =
        All.ToDictionary(cell => cell.Api, StringComparer.Ordinal);

    internal static DirectPtxGatherCoverageCell Get(string api) =>
        ByApi.TryGetValue(api, out DirectPtxGatherCoverageCell? cell)
            ? cell
            : throw new KeyNotFoundException($"No direct-PTX gather coverage cell for '{api}'.");

    private static DirectPtxGatherCoverageCell Experimental(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxGatherCoverageStatus.ExperimentalDirectPtx);

    private static DirectPtxGatherCoverageCell Planned(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxGatherCoverageStatus.PlannedDirectPtx);
}
#endif
