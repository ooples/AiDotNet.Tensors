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
/// Direct rows identify a routed, exact physical specialization. Shapes not
/// listed by that specialization fail closed to the established backend.
/// </summary>
internal static class DirectPtxNormalizationCoverageManifest
{
    private const string RowMajor = "canonical contiguous row-major rows x normalized-size";
    private const string Nchw = "canonical contiguous NCHW [batch,channels,spatial]";

    internal static IReadOnlyList<DirectPtxNormalizationCoverageCell> All { get; } =
    [
        Direct("CudaBackend.BatchNorm", "cuDNN Spatial or NVRTC batchnorm_forward", "training/inference affine normalization with saved/running statistics", Nchw, "FP32", "Ampere B8/C64/S8 direct training+inference"),
        Direct("CudaBackend.TryFusedBatchNormActivation", "NVRTC batchnorm_{relu,gelu,sigmoid,tanh}", "inference BatchNorm plus selected activation", Nchw, "FP32", "Ampere B8/C64/S8 direct activation family"),
        Direct("CudaBackend.BatchNormBackward", "NVRTC batchnorm_backward", "gradInput/gradGamma/gradBeta", Nchw, "FP32", "Ampere B8/C64/S8 direct backward"),
        Direct("CudaBackend.LayerNorm", "NVRTC layernorm_forward", "affine plus saved mean/inverse variance", RowMajor, "FP32", "Ampere D64 rows 256/2048/8192"),
        Direct("CudaBackend.LayerNormBackward", "NVRTC layernorm_backward + layernorm_grad_params", "gradInput plus deterministic affine gradients", RowMajor, "FP32", "Ampere D64 rows 256/2048/8192"),
        Direct("CudaBackend.GroupNorm", "NVRTC groupnorm_forward", "group statistics plus channel affine", Nchw, "FP32", "Ampere B32/C64/G8/S8"),
        Direct("DirectGpuTensorEngine.GroupNormBackward", "24-buffer composed GPU fallback", "gradInput plus deterministic channel gradients from true variance", Nchw, "FP32", "Ampere B32/C64/G8/S8 two-kernel no-temp path"),
        Direct("CudaBackend.InstanceNorm", "NVRTC instancenorm_forward", "instance/channel statistics plus affine", Nchw, "FP32", "Ampere B32/C64/S64"),
        Direct("CudaBackend.InstanceNormBackward", "NVRTC multi-stage deterministic/nondeterministic kernels", "gradInput/gradGamma/gradBeta", Nchw, "FP32", "Ampere B32/C64/S64 two-kernel no-temp path"),
        Direct("CudaBackend.RmsNorm", "NVRTC rmsnorm_forward", "RMS normalization plus affine and saved RMS", RowMajor, "FP32", "Ampere D64 rows 256/2048/8192"),
        Direct("CudaBackend.RmsNormBackward", "NVRTC rmsnorm_backward + rmsnorm_grad_gamma", "gradInput plus deterministic gamma gradient", RowMajor, "FP32", "Ampere D64 rows 256/2048/8192"),
        Direct("CudaBackend.FusedResidualBiasLayerNormGeluD64", "Add+BiasAdd+NVRTC layernorm_gelu", "gelu_tanh(layernorm(input+residual+bias,gamma,beta))", "exact row-major [rows,64]", "FP32", "Ampere rows 256/2048/8192; all performance-gated pending current proof"),
        Direct("PtxFusedResidualRmsNormD64Kernel", "direct PTX residual+RMSNorm experiment", "rmsnorm(input+residual,gamma) plus saved RMS", "contiguous [rows,64]", "FP32", "Ampere D64 residual RMSNorm"),
        Direct("CudaBackend.Fp16LayerNorm", "NVRTC fp16_layernorm_native", "FP16 affine output with FP32 mean/true variance", RowMajor, "FP16", "Ampere D64 rows 256/2048/8192"),
        Direct("CudaBackend.Fp16GroupNormSwish", "NVRTC fp16_groupnorm_swish", "GroupNorm affine plus Swish", Nchw, "FP16/FP32 affine", "Ampere B32/C64/G8/S8"),
        Direct("CudaBackend.Fp16LayerNormBackward", "NVRTC fp16_layernorm_backward_native", "FP16 gradInput from FP32 saved statistics", RowMajor, "FP16/FP32 stats", "Ampere D64 rows 256/2048/8192"),
        Direct("CudaBackend.Fp16LayerNormGradParams", "NVRTC fp16_layernorm_grad_params_native", "deterministic FP32 accumulation and FP16 outputs", RowMajor, "FP16/FP32", "Ampere D64 rows 256/2048/8192"),
        Direct("CudaBackend.NormAxis", "NVRTC norm_axis", "row L2 statistic", RowMajor, "FP32", "Ampere D64 rows 256/2048/8192"),
        Direct("CudaBackend.NormBackward", "NVRTC norm_backward", "row L2 gradient", RowMajor, "FP32", "Ampere D64 rows 256/2048/8192"),
        Direct("CudaBackend.NormalizeL2", "NVRTC normalize_l2", "row unit-L2 normalization", RowMajor, "FP32", "Ampere D64 rows 256/2048/8192"),
        Direct("CudaBackend.NormalizeRowsFused", "NVRTC normalize_rows_fused", "row L2 reduce and normalize", RowMajor, "FP32", "Ampere D64 rows 256/2048/8192"),
        Direct("CudaBackend.ReduceNormL2", "NVRTC atomic/deterministic scalar reduction", "deterministic whole-buffer sum of squares", "contiguous vector", "FP32", "Ampere sizes 16384/131072/524288"),
        Direct("PtxChannelNormalizationD64Kernel.ResidualBatchNormRelu", "NVRTC residual_batchnorm_relu", "relu(batchnorm(input)+residual)", Nchw, "FP32", "Ampere B8/C64/S8"),
        Direct("DirectGpuTensorEngine.GroupNormSwishInto", "GroupNorm scratch plus Swish", "fused GroupNorm affine plus Swish", Nchw, "FP32", "Ampere B32/C64/G8/S8 one-launch"),
        Direct("DirectGpuTensorEngine.AddGroupNormInto", "Add scratch plus GroupNorm", "fused GroupNorm(left+right)", Nchw, "FP32", "Ampere B32/C64/G8/S8 one-launch"),
        Direct("DirectGpuTensorEngine.FusedBatchNorm", "engine BatchNorm with optional activation", "public BatchNorm contract", Nchw, "generic public; GPU FP32", "routes exact direct BatchNorm cells"),
        Direct("DirectGpuTensorEngine.FusedBatchNormGpu", "resident CudaBackend.BatchNorm", "resident BatchNorm with saved statistics", Nchw, "FP32", "routes exact direct BatchNorm cells"),
        Direct("DirectGpuTensorEngine.BatchNormInference", "resident inference BatchNorm", "running-stat affine normalization", Nchw, "generic public; GPU FP32", "routes exact direct inference cell"),
        Direct("DirectGpuTensorEngine.BatchNormAffine", "BatchNormInference route", "provided-stat affine normalization", Nchw, "generic public; GPU FP32", "routes exact direct inference cell"),
        Direct("DirectGpuTensorEngine.LayerNormGpu", "resident CudaBackend.LayerNorm", "resident LayerNorm with saved statistics", RowMajor, "FP32", "routes exact D64 direct cells"),
        Direct("DirectGpuTensorEngine.LayerNormBackwardGpu", "resident CudaBackend.LayerNormBackward", "resident input/affine gradients", RowMajor, "FP32", "routes exact D64 direct cells"),
        Direct("DirectGpuTensorEngine.BatchNormBackwardGpu", "resident CudaBackend.BatchNormBackward", "resident BatchNorm gradients", Nchw, "FP32", "routes exact direct backward cell"),
        Direct("DirectGpuTensorEngine.TensorLayerNorm", "public tensor LayerNorm route", "last-axis affine normalization", RowMajor, "generic public; GPU FP32/FP16", "routes exact D64 direct cells"),
        Direct("DirectGpuTensorEngine.NativeNormalizeRows", "NormalizeRowsFused", "public row L2 normalization", RowMajor, "generic public; GPU FP32", "routes exact D64 direct cells"),
        Direct("IEngine.BatchNorm", "DirectGpuTensorEngine or CPU fallback", "public BatchNorm and true variance", Nchw, "generic", "exact CUDA cell assigned"),
        Direct("IEngine.BatchNormBackward", "DirectGpuTensorEngine or CPU fallback", "public BatchNorm gradients", Nchw, "generic", "exact CUDA cell assigned"),
        Direct("IEngine.LayerNorm", "DirectGpuTensorEngine or CPU fallback", "public last-axis LayerNorm", RowMajor, "generic", "exact CUDA FP32/FP16 cells assigned"),
        Direct("IEngine.LayerNormBackward", "DirectGpuTensorEngine or CPU fallback", "public LayerNorm gradients", RowMajor, "generic", "exact CUDA FP32/FP16 cells assigned"),
        Direct("IEngine.GroupNorm", "DirectGpuTensorEngine or CPU fallback", "public grouped-channel normalization", Nchw, "generic", "exact CUDA cell assigned"),
        Direct("IEngine.GroupNormSwishInto", "resident or legacy composed GPU route", "public fused GroupNorm+Swish", Nchw, "generic", "exact CUDA fused cell assigned"),
        Direct("IEngine.AddGroupNormInto", "legacy Add+GroupNorm route", "public fused add then GroupNorm", Nchw, "generic", "exact CUDA fused cell assigned"),
        Direct("IEngine.GroupNormBackward", "composed GPU/CPU fallback", "public GroupNorm gradients", Nchw, "generic", "exact CUDA two-kernel cell assigned"),
        Direct("IEngine.InstanceNorm", "DirectGpuTensorEngine or CPU fallback", "public instance normalization", Nchw, "generic", "exact CUDA cell assigned"),
        Direct("IEngine.InstanceNormBackward", "DirectGpuTensorEngine or CPU fallback", "public InstanceNorm gradients", Nchw, "generic", "exact CUDA two-kernel cell assigned"),
        Direct("IEngine.RMSNorm", "DirectGpuTensorEngine or CPU fallback", "public RMSNorm", RowMajor, "generic", "exact CUDA D64 cells assigned"),
        Direct("IEngine.RMSNormBackward", "DirectGpuTensorEngine or CPU fallback", "public RMSNorm gradients", RowMajor, "generic", "exact CUDA D64 cells assigned"),
        Existing("CudaFusedKernels.layernorm_{relu,gelu}", "registry-only NVRTC fusion kernels", "LayerNorm affine plus activation", RowMajor, "FP32", "no separate public execution surface; residual+bias+LayerNorm+GELU direct superset assigned"),
        Existing("CudaFusedKernels.residual_layernorm", "registry-only NVRTC fusion kernel", "LayerNorm(input+residual) plus affine", RowMajor, "FP32", "no separate public execution surface; residual normalization direct family assigned"),
        Existing("CudaFusedConvolutionKernels.conv2d_batchnorm_*", "folded convolution kernels", "Conv2D with compile-time folded BatchNorm", "canonical NCHW convolution tensors", "FP32", "convolution-owned: no runtime normalization/statistics ABI"),
        Existing("CudaFusedConvolutionKernels.depthwise_conv2d_batchnorm_relu", "folded depthwise convolution kernel", "depthwise Conv2D with folded BatchNorm", "canonical NCHW depthwise tensors", "FP32", "convolution-owned: no runtime normalization/statistics ABI")
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
