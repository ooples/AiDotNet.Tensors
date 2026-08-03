using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxSoftmaxCoverageStatus
{
    ExistingBackend,
    ExperimentalDirectPtx,
    PromotedDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxSoftmaxCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxSoftmaxCoverageStatus Status);

/// <summary>
/// Executable issue-#840 inventory. Every CUDA softmax, log-sum-exp, masking,
/// and sparse/scatter softmax boundary has one explicit direct-PTX lane.
/// Shape and layout families remain closed sets: a planned row must be split
/// into exact physical ABI cells before it can be promoted.
/// </summary>
internal static class DirectPtxSoftmaxCoverageManifest
{
    private const string Rows = "canonical contiguous row-major [rows,columns]";
    private const string Vector = "canonical contiguous vector";

    internal static IReadOnlyList<DirectPtxSoftmaxCoverageCell> All { get; } =
    [
        Experimental("CudaBackend.Softmax", "direct PTX or NVRTC softmax", "stable last-axis softmax", Rows, "FP32", "v1 Ampere warp-row C64/C128 exact-shape cells"),
        Planned("CudaBackend.SoftmaxBackward", "NVRTC softmax_backward", "y*(grad-sum(grad*y))", Rows, "FP32", "warp-row-backward-exact-shape-families"),
        Planned("CudaBackend.Fp16Softmax", "NVRTC fp16_softmax_native", "stable last-axis softmax with FP32 accumulation", Rows, "FP16 I/O; FP32 accumulator", "warp-row-fp16-forward-families"),
        Planned("CudaBackend.Fp16SoftmaxBackward", "NVRTC fp16_softmax_backward_native", "FP16 softmax output gradient", Rows, "FP16 I/O; FP32 accumulator", "warp-row-fp16-backward-families"),
        Planned("CudaBackend.SoftmaxRows", "NVRTC softmax_rows", "stable row softmax", Rows, "FP32", "unified-softmax-forward-routing"),
        Planned("CudaBackend.LogSoftmax", "resident softmax-variant NVRTC log_softmax", "x-max-log(sum(exp(x-max)))", Rows, "FP32", "warp-row-log-softmax-families"),
        Planned("CudaBackend.GumbelSoftmax", "resident softmax-variant NVRTC gumbel_softmax", "softmax((logits+gumbel)/temperature)", Rows, "FP32", "deterministic-counter-rng-gumbel-families"),
        Planned("CudaBackend.Sparsemax", "resident softmax-variant NVRTC sparsemax", "probability-simplex projection", Rows, "FP32", "bounded-row-sort-or-threshold-families"),
        Planned("CudaBackend.TaylorSoftmax", "resident softmax-variant NVRTC taylor_softmax", "Taylor exponential normalization", Rows, "FP32", "warp-row-taylor-families"),
        Planned("CudaBackend.SphericalSoftmax", "resident softmax-variant NVRTC spherical_softmax", "spherical normalization followed by softmax", Rows, "FP32", "warp-row-spherical-families"),
        Planned("CudaBackend.ReduceLogSumExp", "NVRTC reduce_logsumexp and reduce_logsumexp_deterministic", "scalar log-sum-exp using supplied maximum", Vector, "FP32", "single-vector-logsumexp-size-and-determinism-families"),
        Planned("CudaBackend.LogSumExpAxis", "NVRTC logsumexp_axis", "stable row log-sum-exp", Rows, "FP32", "warp-row-logsumexp-families"),
        Planned("CudaBackend.LogSumExpBackward", "NVRTC logsumexp_backward", "grad*exp(input-lse)", Rows, "FP32", "logsumexp-backward-row-families"),
        Planned("CudaBackend.TriangularMask", "NVRTC triangular_mask", "fill above or below a compile-time diagonal", Rows, "FP32", "baked-diagonal-mask-shape-families"),
        Planned("CudaBackend.MaskedFillKernel", "NVRTC masked_fill_kernel", "select input or fill value from mask", Vector, "FP32 input/output/mask", "baked-fill-vector-families"),
        Planned("CudaBackend.MaskedFillBackward", "NVRTC masked_fill_backward", "zero gradient at masked elements", Vector, "FP32 gradient/mask", "masked-fill-backward-vector-families"),
        Planned("CudaBackend.ScatterSoftmaxRows", "NVRTC scatter_softmax_rows", "indexed/scatter row softmax", "canonical contiguous source/index/output rows", "FP32 plus integer indices", "scatter-softmax-index-layout-families"),
        Planned("CudaBackend.ScatterSoftmaxBackwardRows", "NVRTC scatter_softmax_backward_rows", "indexed/scatter softmax gradient", "canonical contiguous gradient/output/index rows", "FP32 plus integer indices", "scatter-softmax-backward-layout-families"),
        Planned("CudaBackend.HierarchicalSoftmaxPaths", "NVRTC hsoftmax_paths", "tree-path probability product", "canonical contiguous [rows,tree-depth/classes]", "FP32", "hierarchical-path-depth-families"),
        Planned("CudaBackend.FusedGemmBiasActivation.Softmax", "GEMM+bias temporary followed by Softmax", "softmax(gemm(input,weight)+bias)", Rows, "FP32", "fused-linear-softmax-no-global-intermediate-families"),
        Planned("CudaFusedKernels.scaled_softmax", "resident NVRTC scaled_softmax", "softmax(scale*x)", Rows, "FP32", "baked-scale-warp-row-families"),
        Planned("DirectGpuEngine.Softmax", "array upload + CudaBackend.Softmax + download", "public array row-softmax", Rows, "generic public; CUDA FP32", "public-array-softmax-routing"),
        Planned("DirectGpuTensorEngine.Softmax", "resident FP16/FP32 backend route or CPU fallback", "public tensor softmax on arbitrary axis", "logical strided tensor; canonical contiguous last-axis fast path", "generic public; CUDA FP16/FP32", "public-tensor-softmax-routing-and-materialization"),
        Planned("DirectGpuTensorEngine.TrySoftmaxResidentInto", "resident CudaBackend.Softmax", "capture-safe softmax into caller-owned tensor", Rows, "FP32", "resident-softmax-into-routing"),
        Planned("DirectGpuTensorEngine.IEngine.SoftmaxInto", "resident route, uploaded backend route, or CPU fallback", "softmax into existing destination", "logical arbitrary axis; canonical contiguous admitted view", "generic public; CUDA FP32", "softmax-into-routing-and-layout-proof"),
        Planned("DirectGpuTensorEngine.IEngine.LogSoftmaxInto", "CudaBackend.Softmax temporary followed by Log", "log-softmax into existing destination", "logical arbitrary axis; canonical contiguous admitted view", "generic public; CUDA FP32", "direct-logsoftmax-into-no-probability-temporary"),
        Planned("DirectGpuTensorEngine.IEngine.SoftmaxBackward", "FP16/native or FP32 CudaBackend.SoftmaxBackward", "public softmax input gradient", "logical arbitrary axis; canonical contiguous last-axis fast path", "generic public; CUDA FP16/FP32", "public-softmax-backward-routing"),
        Planned("DirectGpuTensorEngine.SoftmaxGpu", "CudaBackend.Softmax", "explicit GPU tensor softmax", Rows, "generic public; CUDA FP32", "explicit-gpu-softmax-routing"),
        Planned("DirectGpuTensorEngine.SoftmaxBackwardGpu", "CudaBackend.SoftmaxBackward", "explicit GPU tensor softmax gradient", Rows, "generic public; CUDA FP32", "explicit-gpu-softmax-backward-routing"),
        Planned("DirectGpuTensorEngine.TensorLogSoftmax", "IEngine TensorLogSoftmax route", "public tensor log-softmax", "logical arbitrary axis; canonical contiguous admitted view", "generic public; CUDA FP32", "public-logsoftmax-routing"),
        Planned("DirectGpuTensorEngine.TensorMaskedFill<bool>", "CudaBackend.MaskedFillKernel", "public boolean-mask fill", "canonical contiguous tensor and boolean mask", "generic public; CUDA FP32 mask representation", "public-bool-masked-fill-routing"),
        Planned("DirectGpuTensorEngine.TensorMaskedFill<Bit>", "CudaBackend.MaskedFillKernel", "public bit-mask fill", "canonical contiguous tensor and bit mask", "generic public; CUDA FP32 mask representation", "public-bit-masked-fill-routing"),
        Planned("DirectGpuTensorEngine.TensorLogSumExp", "CudaBackend.LogSumExpAxis or CPU fallback", "public stable axis log-sum-exp with keepDims", "logical arbitrary axis; contiguous reduced-axis admitted view", "generic public; CUDA FP32", "public-logsumexp-routing-and-axis-materialization"),
        Planned("DirectGpuTensorEngine.IEngine.TensorTriangularMask", "CudaBackend.TriangularMask or CPU fallback", "public upper/lower triangular fill", "canonical contiguous square row-major", "generic public; CUDA FP32", "public-triangular-mask-routing"),
        Planned("DirectGpuTensorEngine.IEngine.TensorSoftmax", "currently CPU fallback for some tape/loss routes", "public tensor softmax compatibility route", "logical arbitrary axis", "generic public", "resident-public-tensor-softmax-routing"),
        Planned("DirectGpuTensorEngine.IEngine.TensorSoftmaxBackward", "CudaBackend.SoftmaxBackward or CPU fallback", "public tensor softmax gradient compatibility route", "logical arbitrary axis; canonical contiguous admitted view", "generic public; CUDA FP32", "resident-public-softmax-backward-routing"),
        Planned("DirectGpuTensorEngine.TensorSoftmaxRows", "CudaBackend.SoftmaxRows", "public row-softmax helper", Rows, "generic public; CUDA FP32", "public-softmax-rows-routing"),
        Planned("DirectGpuTensorEngine.GumbelSoftmaxBackward (inherited)", "inherited CpuEngine derivative route", "temperature-scaled softmax gradient", "logical arbitrary axis; canonical contiguous admitted view", "generic public; CUDA FP32", "public-gumbel-softmax-backward-routing"),
        Planned("DirectGpuTensorEngine.GumbelSoftmax (inherited)", "inherited CpuEngine route", "public Gumbel-softmax including hard/soft mode", "logical arbitrary axis", "generic public", "public-gumbel-softmax-gpu-routing"),
        Planned("DirectGpuTensorEngine.Sparsemax (inherited)", "inherited CpuEngine route", "public probability-simplex projection", "logical arbitrary axis", "generic public", "public-sparsemax-gpu-routing"),
        Planned("DirectGpuTensorEngine.SparsemaxBackward", "resident composite or CpuEngine fallback", "gradient over sparsemax support set", "logical arbitrary axis; canonical contiguous admitted view", "generic public; CUDA FP32", "public-sparsemax-backward-routing"),
        Planned("DirectGpuTensorEngine.TaylorSoftmax (inherited)", "inherited CpuEngine route", "public order-parameterized Taylor softmax", "logical arbitrary axis", "generic public", "public-taylor-softmax-gpu-routing"),
        Planned("DirectGpuTensorEngine.TaylorSoftmaxBackward", "resident composite or CpuEngine fallback", "Taylor-normalization gradient", "logical arbitrary axis; canonical contiguous admitted view", "generic public; CUDA FP32", "public-taylor-softmax-backward-routing"),
        Planned("DirectGpuTensorEngine.SphericalSoftmax (inherited)", "inherited CpuEngine route", "public spherical softmax", "logical arbitrary axis", "generic public", "public-spherical-softmax-gpu-routing"),
        Planned("DirectGpuTensorEngine.SphericalSoftmaxBackward", "resident composite or CpuEngine fallback", "spherical-normalization gradient", "logical arbitrary axis; canonical contiguous admitted view", "generic public; CUDA FP32", "public-spherical-softmax-backward-routing"),
        Planned("DirectGpuTensorEngine.ScatterSoftmax", "scatter rows backend or CPU fallback", "public indexed/scatter normalization", "source/index/output layout families", "generic public; CUDA FP32 plus integer indices", "public-scatter-softmax-routing"),
        Planned("DirectGpuTensorEngine.ScatterSoftmaxBackward", "scatter backward rows backend or CPU fallback", "public indexed/scatter normalization gradient", "gradient/output/index layout families", "generic public; CUDA FP32 plus integer indices", "public-scatter-softmax-backward-routing"),
        Planned("DirectGpuTensorEngine.FusedHierarchicalSoftmax", "HierarchicalSoftmaxPaths plus resident weights", "public tree-path logits-to-class probabilities", "canonical contiguous [rows,nodes/classes]", "generic public; CUDA FP32", "fused-hierarchical-softmax-routing"),
        Planned("DirectGpuTensorEngine.TensorLogSumExpScalar", "inherited scalar-reduction overload", "public all-element log-sum-exp", Vector, "generic public; CUDA FP32", "scalar-logsumexp-routing"),
        Planned("DirectGpuTensorEngine.TensorMaskedFill<bool[]>", "inherited host-array mask route", "public host-boolean-array masked fill", "canonical tensor plus host boolean mask", "generic public", "materialize-canonical-mask-then-direct-masked-fill"),
        Planned("ActivationHandler.softmax-family", "engine-routed softmax/log/sparse/Taylor/spherical/Gumbel handlers", "activation forward and registered backward routes", "inherits tensor operation layout", "generic public", "frontend-routing-to-assigned-direct-ptx-cells"),
        Planned("Distributions.SoftmaxTransform", "public CPU transform over simplex coordinates", "K-to-K+1 softmax distribution transform", "canonical vector transform layout", "generic public", "distribution-transform-routing-or-explicit-cpu-only-decision"),
        Planned("IEngine.Softmax(Vector)", "TensorPrimitivesHelper/CpuEngine vector route", "public vector softmax", Vector, "generic public", "canonical-vector-direct-softmax-routing"),
        Existing("ISparseEngine.SparseSoftmax/SparseLogSoftmax", "CPU sparse-tensor operations", "sparse storage normalization", "sparse coordinate/value layouts", "generic sparse", "owned by #852 general sparse PTX child; explicitly excluded from dense #840 implementation"),
        Existing("NestedOps.Softmax/LogSoftmax", "nested CPU tensor operations", "ragged nested-tensor normalization", "nested/ragged storage", "generic nested", "requires nested layout materialization contract; not admitted by dense row ABI"),
        Existing("CuDnnSoftmax.ForwardGpu", "cuDNN accurate channel-mode softmax", "resident library softmax competitor", "contiguous NCHW interpreted as [rows,columns,1,1]", "FP16/FP32", "benchmark competitor and possible fallback lane; direct PTX remains separate"),
        Existing("CuDnnSoftmax.BackwardGpu", "cuDNN accurate channel-mode softmax backward", "resident library softmax gradient competitor", "contiguous NCHW interpreted as [rows,columns,1,1]", "FP16/FP32", "backward benchmark competitor and fallback lane"),
        Planned("CudaMeshPoolKernels.softmax-family", "five-stage large-edge or single-block NVRTC mesh-pool softmax", "normalize per-edge mesh-pooling scores", "mesh edge/group contiguous workspaces", "FP32", "fused-mesh-pool-probability-normalization-families"),
        Existing("CudaAttentionKernels.OnlineSoftmax", "resident NVRTC fused attention online recurrence", "streaming attention softmax and optional LSE", "tiled attention Q/K/V layouts", "FP32", "owned by #833; shared saved-LSE and numerical ABI"),
        Existing("CudaDecodeAndPagedAttentionKernels.OnlineSoftmax", "resident NVRTC flash-decode and paged-attention recurrences", "streaming decode probability normalization", "dense or block-table K/V layouts", "FP32", "owned by #833; shared online-softmax ABI"),
        Existing("PtxAttentionSoftmax32Kernel", "direct PTX attention score softmax", "stable S32 attention softmax with optional causal mask", "exact contiguous attention score rows", "FP32", "shared-attention-ABI; retain as standalone validation oracle"),
        Existing("PtxOnlineFusedAttention128x64Kernel.OnlineSoftmax", "direct PTX online softmax fused with QK/PV", "streaming online softmax recurrence", "exact tiled attention Q/K/V layouts", "FP16 I/O; FP32 accumulator", "owned by #833; canonical no-score/probability-VRAM blueprint")
    ];

    internal static DirectPtxSoftmaxCoverageCell Get(string api) =>
        All.FirstOrDefault(cell => string.Equals(cell.Api, api, StringComparison.Ordinal)) ??
        throw new KeyNotFoundException(
            $"No #840 softmax coverage cell is assigned to '{api}'.");

    private static DirectPtxSoftmaxCoverageCell Planned(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxSoftmaxCoverageStatus.PlannedDirectPtx);

    private static DirectPtxSoftmaxCoverageCell Existing(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxSoftmaxCoverageStatus.ExistingBackend);

    private static DirectPtxSoftmaxCoverageCell Experimental(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx);
}
