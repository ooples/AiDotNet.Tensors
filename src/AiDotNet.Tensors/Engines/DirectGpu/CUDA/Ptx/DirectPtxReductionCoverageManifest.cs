using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxReductionCoverageStatus
{
    ExistingBackend,
    ExperimentalDirectPtx,
    PromotedDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxReductionCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxReductionCoverageStatus Status);

/// <summary>
/// Executable issue-#843 inventory. Every CUDA axis reduction, scan,
/// arg-reduction, sort, TopK, and histogram boundary has one explicit
/// direct-PTX lane. Shape and layout families remain closed sets: a planned row
/// must be split into exact physical ABI cells before it can be promoted.
/// </summary>
internal static class DirectPtxReductionCoverageManifest
{
    private const string Rows = "canonical contiguous row-major [rows,columns]";
    private const string Vector = "canonical contiguous vector";

    internal static IReadOnlyList<DirectPtxReductionCoverageCell> All { get; } =
    [
        Experimental("CudaBackend.SumAxis", "NVRTC sum_axis", "sum over last axis to [rows]", Rows, "FP32", "v1 Ampere warp-row C64/C128 exact-shape cells"),
        Experimental("CudaBackend.MeanAxis", "NVRTC mean_axis", "arithmetic mean over last axis", Rows, "FP32", "v1 Ampere warp-row exact-shape cells; reciprocal baked, no divide"),
        Experimental("CudaBackend.MaxAxis", "NVRTC max_axis", "maximum over last axis", Rows, "FP32", "v1 Ampere warp-row exact-shape cells; max.f32 quieting matches fmaxf"),
        Planned("CudaBackend.MinAxis", "none - no backend op and no min_axis kernel exist", "minimum over last axis", Rows, "FP32", "blocked: the direct-PTX min operator is implemented and tested, but there is no CudaBackend.MinAxis to route it through - the backend op has to be written first"),
        Planned("CudaBackend.ProductAxis", "NVRTC product_axis", "product over last axis", Rows, "FP32", "warp-row-product-families"),
        Planned("CudaBackend.VarianceAxis", "NVRTC variance_axis", "population variance over last axis", Rows, "FP32", "warp-row-two-pass-variance-families"),
        Planned("CudaBackend.StdAxis", "NVRTC std_axis", "standard deviation over last axis", Rows, "FP32", "warp-row-std-families"),
        Planned("CudaBackend.NormAxis", "NVRTC norm_axis", "L2 norm over last axis", Rows, "FP32", "warp-row-norm-families"),
        Planned("CudaBackend.CumSumAxis", "NVRTC cumsum_axis", "inclusive prefix sum along inner axis", Rows, "FP32", "warp-scan-inclusive-families"),
        Experimental("CudaBackend.NormalizeL2", "NVRTC normalize_l2", "row L2 normalization (fused reduce+rescale, eps=1e-12)", Rows, "FP32", "v1 Ampere warp-row C64/C128 exact-shape cells (fused, no materialized norm)"),
        Planned("CudaBackend.ReduceSumOfSquares", "NVRTC reduce_sum_of_squares", "scalar sum of squares", Vector, "FP32", "single-vector-sum-of-squares-size-families; the row-wise sum-of-squares operator is implemented but does NOT fit this cell - this op folds a whole vector to one scalar, a different physical contract needing a grid-wide reduction"),
        Planned("CudaBackend.ReduceMaxMagnitude", "NVRTC reduce_max_magnitude", "scalar max |x|", Vector, "FP32", "single-vector-max-magnitude-families"),
        Planned("CudaBackend.Sum", "NVRTC segmented reduce_sum", "scalar full-tensor sum", Vector, "FP32", "two-stage-grid-reduce-families"),
        Planned("CudaBackend.ArgMaxAxis", "NVRTC argmax_axis", "index of maximum over last axis", Rows, "FP32 values; INT32 indices", "warp-row-arg-reduction-families"),
        Planned("CudaBackend.ArgMinAxis", "NVRTC argmin_axis", "index of minimum over last axis", Rows, "FP32 values; INT32 indices", "warp-row-arg-reduction-families"),
        Planned("CudaBackend.TopK", "NVRTC topk", "top-k values and indices over last axis", Rows, "FP32 values; INT32 indices", "bounded-k-bitonic-or-threshold-families"),
        Planned("CudaBackend.SortAxis", "NVRTC sort_axis", "sort values over last axis", Rows, "FP32 values; INT32 indices", "bitonic-power-of-two-length-families"),
        Planned("CudaBackend.Histogram", "NVRTC histogram", "binned counts over baked bin edges", Vector, "FP32 input; INT32 counts", "baked-bin-count-shared-privatized-families"),
        Planned("CudaBackend.CumMaxAxis", "NVRTC cummax_axis", "inclusive prefix maximum along inner axis", Rows, "FP32", "warp-scan-prefix-max-families"),
        Planned("CudaBackend.ReduceSumBackward", "NVRTC reduce_sum_backward", "broadcast upstream gradient over reduced axis", Rows, "FP32", "broadcast-backward-row-families"),
        Planned("CudaBackend.ReduceMeanBackward", "NVRTC reduce_mean_backward", "broadcast gradient/reduceSize", Rows, "FP32", "broadcast-scaled-backward-row-families"),
        Planned("CudaBackend.ReduceMaxBackward", "NVRTC reduce_max_backward", "route gradient to the arg-max element", Rows, "FP32 gradient; FP32 max cache", "selective-backward-row-families"),
        Planned("CudaBackend.SegmentSum", "NVRTC segment_sum", "sum within contiguous segments", "canonical contiguous source/segment-offset rows", "FP32 plus integer offsets", "segmented-scan-offset-layout-families"),
        Planned("DirectGpuEngine.SumAxis", "array upload + CudaBackend.SumAxis + download", "public array axis sum", Rows, "generic public; CUDA FP32", "public-array-sum-axis-routing"),
        Planned("DirectGpuTensorEngine.TensorSum", "resident CudaBackend.SumAxis or CPU fallback", "public tensor sum over arbitrary axis", "logical strided tensor; canonical contiguous last-axis fast path", "generic public; CUDA FP32", "public-tensor-sum-routing-and-materialization"),
        Planned("DirectGpuTensorEngine.IEngine.SumInto", "resident route, uploaded backend route, or CPU fallback", "sum into existing destination", "logical arbitrary axis; canonical contiguous admitted view", "generic public; CUDA FP32", "sum-into-routing-and-layout-proof"),
        Planned("DirectGpuTensorEngine.TensorArgMax", "CudaBackend.ArgMaxAxis or CPU fallback", "public tensor arg-max over an axis", "logical arbitrary axis; contiguous reduced-axis admitted view", "generic public; CUDA FP32/INT32", "public-argmax-routing-and-axis-materialization"),
        Planned("DirectGpuTensorEngine.TensorCumSum", "CudaBackend.CumSumAxis or CPU fallback", "public tensor inclusive prefix sum", "logical arbitrary axis; contiguous scanned-axis admitted view", "generic public; CUDA FP32", "public-cumsum-routing"),
    ];

    private static readonly IReadOnlyDictionary<string, DirectPtxReductionCoverageCell> ByApi =
        All.ToDictionary(cell => cell.Api, StringComparer.Ordinal);

    internal static DirectPtxReductionCoverageCell Get(string api) =>
        ByApi.TryGetValue(api, out DirectPtxReductionCoverageCell? cell)
            ? cell
            : throw new KeyNotFoundException($"No direct-PTX reduction coverage cell for '{api}'.");

    private static DirectPtxReductionCoverageCell Experimental(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxReductionCoverageStatus.ExperimentalDirectPtx);

    private static DirectPtxReductionCoverageCell Planned(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxReductionCoverageStatus.PlannedDirectPtx);
}
