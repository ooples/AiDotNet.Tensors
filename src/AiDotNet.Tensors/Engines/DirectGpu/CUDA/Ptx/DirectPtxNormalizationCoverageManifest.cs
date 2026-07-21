#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxNormalizationCoverageStatus
{
    ExistingBackend,
    ExperimentalDirectPtx,
    PromotedDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxNormalizationCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxNormalizationCoverageStatus Status);

/// <summary>
/// Executable issue-#838 inventory. Every public normalization route and every
/// CUDA normalization/fusion kernel family has an explicit PTX assignment.
/// A planned row is not an implicit promise of generic dynamic PTX: each
/// assignment must still be split into exact physical shape/dtype cells.
/// </summary>
internal static class DirectPtxNormalizationCoverageManifest
{
    private const string RowMajor = "canonical contiguous row-major rows x normalized-size";
    private const string Nchw = "canonical contiguous NCHW [batch,channels,spatial]";

    internal static IReadOnlyList<DirectPtxNormalizationCoverageCell> All { get; } =
    [
        Planned("CudaBackend.BatchNorm", "cuDNN Spatial or NVRTC batchnorm_forward", "training/inference per-channel normalization plus affine and saved/running statistics", Nchw, "FP32", "batchnorm-forward-training/inference-shape-families"),
        Planned("CudaBackend.TryFusedBatchNormActivation", "NVRTC batchnorm_{relu,gelu,sigmoid,tanh}", "inference BatchNorm plus selected activation", Nchw, "FP32", "batchnorm-inference-affine-activation-families"),
        Planned("CudaBackend.BatchNormBackward", "NVRTC batchnorm_backward", "gradInput/gradGamma/gradBeta from saved statistics", Nchw, "FP32", "batchnorm-backward-deterministic-reduction-families"),
        Planned("CudaBackend.LayerNorm", "NVRTC layernorm_forward", "LayerNorm affine plus saved mean/inv-variance", RowMajor, "FP32", "layernorm-forward-size-families"),
        Planned("CudaBackend.LayerNormBackward", "NVRTC layernorm_backward + layernorm_grad_params", "gradInput plus cross-row affine gradients", RowMajor, "FP32", "layernorm-backward-and-param-gradient-families"),
        Planned("CudaBackend.GroupNorm", "NVRTC groupnorm_forward", "per-batch-group normalization plus channel affine", Nchw, "FP32", "groupnorm-forward-group-size-families"),
        Planned("CudaBackend.InstanceNorm", "NVRTC instancenorm_forward", "per-batch-channel normalization plus affine", Nchw, "FP32", "instancenorm-forward-spatial-families"),
        Planned("CudaBackend.InstanceNormBackward", "NVRTC deterministic/nondeterministic instance-normalization reductions", "gradInput/gradGamma/gradBeta", Nchw, "FP32", "instancenorm-backward-determinism-families"),
        Planned("CudaBackend.RmsNorm", "NVRTC rmsnorm_forward", "RMS normalization plus affine and saved RMS", RowMajor, "FP32", "rmsnorm-forward-size-families"),
        Planned("CudaBackend.RmsNormBackward", "NVRTC rmsnorm_backward + rmsnorm_grad_gamma", "gradInput and affine gradient", RowMajor, "FP32", "rmsnorm-backward-size-families"),
        Promoted("CudaBackend.FusedResidualBiasLayerNormGeluD64", "direct PTX or Add+BiasAdd+NVRTC layernorm_gelu", "gelu_tanh(layernorm(input+residual+bias,gamma,beta))", "exact row-major [rows,64], three exact vectors[64]", "FP32", "v1 Ampere warp-row W8: rows 2048/8192 promoted; rows 256 measured but performance-gated"),
        Direct("PtxFusedResidualRmsNormD64Kernel", "direct PTX residual+RMSNorm experiment", "rmsnorm(input+residual,gamma) plus saved RMS", "contiguous [rows,64] and vector[64]", "FP32", "residual-rmsnorm-d64-forward-family"),
        Planned("CudaBackend.Fp16LayerNorm", "NVRTC fp16_layernorm_native", "FP16 I/O with FP32 statistics and FP16 affine", RowMajor, "FP16", "layernorm-fp16-forward-size-families"),
        Planned("CudaBackend.Fp16GroupNormSwish", "NVRTC fp16_groupnorm_swish", "GroupNorm affine plus Swish without FP32 round-trip", Nchw, "FP16/FP32 affine", "groupnorm-swish-fp16-group-size-families"),
        Planned("CudaBackend.Fp16LayerNormBackward", "NVRTC fp16_layernorm_backward_native", "FP16 gradInput from saved FP32 statistics", RowMajor, "FP16/FP32 stats", "layernorm-fp16-backward-size-families"),
        Planned("CudaBackend.Fp16LayerNormGradParams", "NVRTC fp16_layernorm_grad_params_native", "FP32 gamma/beta gradients", RowMajor, "FP16/FP32", "layernorm-fp16-param-gradient-families"),
        Planned("CudaBackend.NormAxis", "NVRTC norm_axis", "row L2 norm statistic", RowMajor, "FP32", "row-l2-statistic-size-families"),
        Planned("CudaBackend.NormBackward", "NVRTC norm_backward", "gradient of row L2 norm", RowMajor, "FP32", "row-l2-backward-size-families"),
        Planned("CudaBackend.NormalizeL2", "NVRTC normalize_l2", "row-wise unit-L2 normalization", RowMajor, "FP32", "row-l2-normalize-size-families"),
        Planned("CudaBackend.NormalizeRowsFused", "NVRTC normalize_rows_fused", "per-row L2 reduce and normalize", RowMajor, "FP32", "spectral-row-normalize-size-families"),
        Planned("CudaBackend.ReduceNormL2", "NVRTC reduce_norm_l2", "single-buffer L2 statistic", "contiguous vector", "FP32", "vector-l2-reduction-size-families"),
        Planned("DirectGpuTensorEngine.FusedBatchNorm", "engine BatchNorm with optional activation fusion", "public BatchNorm training/inference contract", Nchw, "generic public; GPU FP32", "public-batchnorm-forward-routing"),
        Planned("DirectGpuTensorEngine.FusedBatchNormGpu", "resident CudaBackend.BatchNorm", "resident BatchNorm plus saved statistics", Nchw, "FP32", "resident-batchnorm-forward-routing"),
        Planned("DirectGpuTensorEngine.LayerNormGpu", "resident CudaBackend.LayerNorm", "resident LayerNorm plus saved statistics", RowMajor, "FP32", "resident-layernorm-forward-routing"),
        Planned("DirectGpuTensorEngine.LayerNormBackwardGpu", "resident CudaBackend.LayerNormBackward", "resident LayerNorm input/affine gradients", RowMajor, "FP32", "resident-layernorm-backward-routing"),
        Planned("DirectGpuTensorEngine.BatchNormBackwardGpu", "resident CudaBackend.BatchNormBackward", "resident BatchNorm input/affine gradients", Nchw, "FP32", "resident-batchnorm-backward-routing"),
        Planned("IEngine.BatchNorm", "DirectGpuTensorEngine GPU route or CPU fallback", "public BatchNorm and true variance outputs", Nchw, "generic", "public-batchnorm-api-family"),
        Planned("IEngine.BatchNormBackward", "DirectGpuTensorEngine GPU route or CPU fallback", "public BatchNorm gradients", Nchw, "generic", "public-batchnorm-backward-api-family"),
        Planned("IEngine.LayerNorm", "DirectGpuTensorEngine FP16/FP32 GPU route or CPU fallback", "public last-axis LayerNorm and statistics", RowMajor, "generic", "public-layernorm-api-family"),
        Planned("IEngine.LayerNormBackward", "DirectGpuTensorEngine FP16/FP32 GPU route or CPU fallback", "public LayerNorm gradients", RowMajor, "generic", "public-layernorm-backward-api-family"),
        Planned("IEngine.GroupNorm", "DirectGpuTensorEngine GPU forward or CPU fallback", "public grouped-channel normalization", Nchw, "generic", "public-groupnorm-api-family"),
        Planned("IEngine.GroupNormBackward", "established CPU/composed fallback", "public GroupNorm gradients", Nchw, "generic", "public-groupnorm-backward-api-family"),
        Planned("IEngine.InstanceNorm", "DirectGpuTensorEngine GPU forward or CPU fallback", "public per-instance/channel normalization", Nchw, "generic", "public-instancenorm-api-family"),
        Planned("IEngine.InstanceNormBackward", "DirectGpuTensorEngine GPU route or CPU fallback", "public InstanceNorm gradients", Nchw, "generic", "public-instancenorm-backward-api-family"),
        Planned("IEngine.RMSNorm", "DirectGpuTensorEngine GPU route or CPU fallback", "public RMSNorm plus saved RMS", RowMajor, "generic", "public-rmsnorm-api-family"),
        Planned("IEngine.RMSNormBackward", "DirectGpuTensorEngine GPU route or CPU fallback", "public RMSNorm input/affine gradients", RowMajor, "generic", "public-rmsnorm-backward-api-family"),
        Existing("CudaFusedKernels.layernorm_{relu,gelu}", "resident NVRTC fusion kernels", "LayerNorm affine plus activation", RowMajor, "FP32", "layernorm-activation-size-families"),
        Existing("CudaFusedKernels.residual_layernorm", "resident NVRTC fusion kernel", "LayerNorm(input+residual) plus affine", RowMajor, "FP32", "residual-layernorm-size-families"),
        Existing("CudaFusedKernels.residual_batchnorm_relu", "resident NVRTC fusion kernel", "inference BatchNorm affine plus residual and ReLU", Nchw, "FP32", "residual-batchnorm-relu-shape-families"),
        Existing("CudaFusedConvolutionKernels.conv2d_batchnorm_*", "resident NVRTC folded-BatchNorm convolution kernels", "Conv2D plus folded BatchNorm and optional activation", "canonical NCHW convolution tensors", "FP32", "conv-folded-batchnorm-activation-shape-families"),
        Existing("CudaFusedConvolutionKernels.depthwise_conv2d_batchnorm_relu", "resident NVRTC folded-BatchNorm depthwise kernel", "depthwise Conv2D plus folded BatchNorm and ReLU", "canonical NCHW depthwise tensors", "FP32", "depthwise-folded-batchnorm-relu-shape-families")
    ];

    internal static DirectPtxNormalizationCoverageCell Get(string api) =>
        All.FirstOrDefault(cell => string.Equals(cell.Api, api, StringComparison.Ordinal)) ??
        throw new KeyNotFoundException($"No #838 normalization coverage cell is assigned to '{api}'.");

    private static DirectPtxNormalizationCoverageCell Planned(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxNormalizationCoverageStatus.PlannedDirectPtx);

    private static DirectPtxNormalizationCoverageCell Existing(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxNormalizationCoverageStatus.ExistingBackend);

    private static DirectPtxNormalizationCoverageCell Direct(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxNormalizationCoverageStatus.ExperimentalDirectPtx);

    private static DirectPtxNormalizationCoverageCell Promoted(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxNormalizationCoverageStatus.PromotedDirectPtx);
}
#endif
