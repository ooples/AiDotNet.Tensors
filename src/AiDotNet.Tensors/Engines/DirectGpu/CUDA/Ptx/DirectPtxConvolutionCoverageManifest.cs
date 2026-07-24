using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxConvolutionCoverageStatus
{
    BaselineOnly,

    /// <summary>
    /// A direct-PTX specialization is reachable from the public API through a real dispatch
    /// hook (e.g. <c>TryDirectPtxFusedConv2DBiasRelu</c>), gated off by default.
    /// </summary>
    ExperimentalDirectPtx,

    /// <summary>
    /// An SM86 kernel exists and its content-addressed cubin is released and SASS-audited,
    /// but NOTHING routes the public API to it yet: the API still takes the baseline path in
    /// every case. Distinct from <see cref="ExperimentalDirectPtx"/>, which has a dispatch
    /// hook, and from <see cref="PlannedDirectPtx"/>, for which no kernel exists.
    /// </summary>
    KernelReleasedNotRouted,

    PlannedDirectPtx
}

internal sealed record DirectPtxConvolutionCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    DirectPtxConvolutionCoverageStatus Status,
    string DirectPtxAssignment);

/// <summary>
/// Executable issue-#841 inventory. Entries remain explicit even when several
/// public overloads share one backend primitive, preventing a future PR from
/// silently declaring the entire family ported after one specialization.
/// </summary>
internal static class DirectPtxConvolutionCoverageManifest
{
    private static DirectPtxConvolutionCoverageCell Planned(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes,
            DirectPtxConvolutionCoverageStatus.PlannedDirectPtx, assignment);

    // Each cell below has an SM86 kernel verified on-device against an fp64 CPU oracle, a
    // released content-addressed cubin (PTX<->cubin identity + nvdisasm zero-spill audited),
    // and a Deferred PtxParityRegistry entry. It is deliberately NOT ExperimentalDirectPtx:
    // no dispatch hook routes the public API to these kernels, so the API still takes the
    // baseline path 100% of the time. Perf evidence (>=1.10x median vs the strongest cuDNN
    // competitor, P95, three clean runs) is also still outstanding.
    private static DirectPtxConvolutionCoverageCell Released(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes,
            DirectPtxConvolutionCoverageStatus.KernelReleasedNotRouted, assignment);

    internal static IReadOnlyList<DirectPtxConvolutionCoverageCell> All { get; } =
    [
        new("DirectGpuTensorEngine.FusedConv2D",
            "cuDNN or CUDA Conv2D + NVRTC bias + activation",
            "inference convolution with optional bias and activation",
            "NCHW input/output, OIHW weights", "generic public; CUDA FP32",
            DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            "v1 exact N1/C64/H16/W16/K64 1x1 FP32 bias+ReLU; all other contracts fall back"),
        new("CudaBackend.TryDirectPtxFusedConv2DBiasRelu",
            "new direct Driver-API PTX route with established fallback owned by caller",
            "same v1 fused inference contract", "exact contiguous NCHW/OIHW", "FP32",
            DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            "hand-emitted sm_86 pointer-only specialization, disabled by default"),
        Released("IEngine.Conv1D", "CUDA Conv1D routed through Conv2D", "1D forward", "NCL/OIL", "generic/FP32", "direct forward families by channel/spatial bucket"),
        Released("IEngine.Conv1DBackwardInput", "CUDA direct backward-input kernel", "1D input gradient", "NCL/OIL", "generic/FP32", "deterministic backward-input specializations"),
        Released("IEngine.Conv1DBackwardKernel", "CUDA direct backward-weight kernel", "1D weight gradient", "NCL/OIL", "generic/FP32", "deterministic backward-weight specializations"),
        Released("IEngine.Conv2D", "cuDNN, Winograd, tiled NVRTC, or direct NVRTC", "2D forward", "NCHW/OIHW plus public format overload", "generic/FP32", "1x1, direct-tiled, and Tensor-Core families"),
        Released("IEngine.Conv2DBackwardInput", "cuDNN or CUDA direct backward-input", "2D input gradient", "NCHW/OIHW", "generic/FP32", "deterministic direct/tiled backward-input"),
        Released("IEngine.Conv2DBackwardKernel", "cuDNN or CUDA direct backward-weight", "2D weight gradient", "NCHW/OIHW", "generic/FP32", "deterministic direct/tiled backward-weight"),
        Released("DirectGpuTensorEngine.Conv2DBackwardBiasGpu", "CUDA reduction", "2D bias gradient", "NCHW to vector", "FP32", "fuse with backward-weight where beneficial"),
        Released("IEngine.Conv3D", "CUDA direct 3D kernel", "3D forward", "NCDHW/OIDHW", "generic/FP32", "3D direct/tiled families"),
        Released("IEngine.Conv3DBackwardInput", "CPU fallback/current backend composition", "3D input gradient", "NCDHW/OIDHW", "generic", "CUDA direct backward-input"),
        Released("IEngine.Conv3DBackwardKernel", "CPU fallback/current backend composition", "3D weight gradient", "NCDHW/OIDHW", "generic", "CUDA deterministic backward-weight"),
        Released("IEngine.DepthwiseConv1D", "reshape to CUDA depthwise Conv2D", "depthwise 1D forward", "NCL/channel multiplier", "generic/FP32", "native 1D depthwise PTX"),
        Released("IEngine.DepthwiseConv1DBackwardInput", "CPU/current backend path", "depthwise 1D input gradient", "NCL", "generic", "native 1D backward-input PTX"),
        Released("IEngine.DepthwiseConv1DBackwardKernel", "CPU/current backend path", "depthwise 1D weight gradient", "OIL", "generic", "native 1D backward-weight PTX"),
        Released("IEngine.DepthwiseConv2D", "CUDA NVRTC depthwise kernel", "depthwise 2D forward", "NCHW/depthwise OIHW", "generic/FP32", "direct/tiled bias-activation families"),
        Released("IEngine.DepthwiseConv2DBackwardInput", "CUDA depthwise backward-input", "depthwise 2D input gradient", "NCHW", "generic/FP32", "deterministic backward-input PTX"),
        Released("IEngine.DepthwiseConv2DBackwardKernel", "CUDA depthwise backward-weight", "depthwise 2D weight gradient", "OIHW", "generic/FP32", "deterministic backward-weight PTX"),
        Released("IEngine.ConvTranspose2D", "CUDA NVRTC transposed convolution", "2D transposed forward", "NCHW/IOHW", "generic/FP32", "direct/tiled transposed families"),
        Released("IEngine.ConvTranspose2DBackwardInput", "CUDA direct transposed backward-input", "transposed input gradient", "NCHW/IOHW", "generic/FP32", "deterministic backward-input PTX"),
        Released("IEngine.ConvTranspose2DBackwardKernel", "CUDA direct transposed backward-weight", "transposed weight gradient", "NCHW/IOHW", "generic/FP32", "deterministic backward-weight PTX"),
        Released("IEngine.ConvTranspose3D", "CPU/current backend fallback", "3D transposed forward", "NCDHW/IODHW", "generic", "CUDA direct/tiled transposed PTX"),
        Released("IEngine.ConvTranspose3DBackwardInput", "CPU/current backend fallback", "3D transposed input gradient", "NCDHW/IODHW", "generic", "CUDA deterministic backward-input PTX"),
        Released("IEngine.ConvTranspose3DBackwardKernel", "CPU/current backend fallback", "3D transposed weight gradient", "NCDHW/IODHW", "generic", "CUDA deterministic backward-weight PTX"),
        Released("IEngine.DeformableConv2D", "CUDA NVRTC DCNv2", "offset/mask deformable forward", "NCHW plus offsets/mask", "generic/FP32", "fused sampling+convolution PTX"),
        Released("IEngine.DeformableConv2DBackwardInput", "CUDA NVRTC DCNv2 backward", "input gradient", "NCHW plus offsets/mask", "generic/FP32", "deterministic gradient PTX"),
        Released("IEngine.DeformableConv2DBackwardKernel", "CUDA NVRTC DCNv2 backward", "weight gradient", "NCHW plus offsets/mask", "generic/FP32", "deterministic gradient PTX"),
        Released("IEngine.DeformableConv2DBackwardOffset", "CUDA NVRTC DCNv2 backward", "offset gradient", "NCHW plus offsets/mask", "generic/FP32", "fused sampling derivative PTX"),
        Released("IEngine.DeformableConv2DBackwardMask", "CUDA NVRTC DCNv2 backward", "mask gradient", "NCHW plus offsets/mask", "generic/FP32", "fused mask-gradient PTX"),
        Released("IEngine.DeformableConv2DGrouped", "CUDA grouped deformable kernel", "grouped/deform-group forward", "NCHW plus grouped offsets/mask", "generic/FP32", "group-specialized PTX"),
        Released("IEngine.DeformableConv2DGroupedBackward", "CUDA grouped deformable backward family", "all grouped gradients", "NCHW plus grouped offsets/mask", "generic/FP32", "four deterministic grouped backward families"),
        Released("IEngine.LocallyConnectedConv2D", "CUDA NVRTC locally-connected kernel", "unshared spatial weights forward", "NCHW plus per-position OIHW", "generic/FP32", "position-tiled PTX"),
        Released("IEngine.LocallyConnectedConv2DBackwardInput", "CUDA NVRTC locally-connected backward", "input gradient", "NCHW plus per-position weights", "generic/FP32", "deterministic backward-input PTX"),
        Released("IEngine.LocallyConnectedConv2DBackwardWeights", "CUDA NVRTC locally-connected backward", "weight gradient", "per-position OIHW", "generic/FP32", "deterministic backward-weight PTX"),
        Released("IEngine.LocallyConnectedConv2DBackwardBias", "CUDA reduction", "bias gradient", "NCHW to per-position bias", "generic/FP32", "fused backward reduction PTX"),
        Released("IEngine.FusedConv3D", "CUDA Conv3D plus bias/activation kernels", "3D inference epilogue fusion", "NCDHW/OIDHW", "generic/FP32", "bias/norm/activation epilogue families"),
        Released("IEngine.FusedConvTranspose2D", "CUDA transposed convolution plus epilogue", "transposed inference fusion", "NCHW/IOHW", "generic/FP32", "bias/norm/activation epilogue families"),
        Released("IEngine.Unfold", "CUDA im2col/unfold NVRTC kernels", "materialized patch extraction", "NCHW to matrix/windows", "generic/FP32", "eliminate materialization in fused conv; standalone PTX remains explicit"),
        Released("CudaBackend.Conv2dDirectFp16Hw", "NVRTC half-weight direct convolution", "FP32 input rounded to FP16, FP32 accumulation/output", "NCHW/OIHW-half", "mixed FP16/FP32", "architecture-specific half/Tensor-Core family"),
        Released("CudaBackend.Im2colKNFp16", "NVRTC fused im2col+FP16 conversion", "FP16 Tensor-Core GEMM preparation", "NCHW to KxN half", "FP16/FP32", "fuse producer into PTX convolution tiles"),
        Released("CudaBackend.UnfoldKNFp16FromFp16", "NVRTC FP16 im2col", "FP16 patch extraction", "NCHW-half to KxN-half", "FP16", "fuse producer into PTX convolution tiles")
    ];

    internal static DirectPtxConvolutionCoverageCell Get(string api)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(api, nameof(api));
        foreach (DirectPtxConvolutionCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException(
            $"Convolution API '{api}' is not assigned in the #841 coverage manifest.");
    }
}
