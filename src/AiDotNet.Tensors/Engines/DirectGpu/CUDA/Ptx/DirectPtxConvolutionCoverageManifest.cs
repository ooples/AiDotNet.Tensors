using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxConvolutionCoverageStatus
{
    BaselineOnly,
    ExperimentalDirectPtx,
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
        Planned("IEngine.Conv1D", "CUDA Conv1D routed through Conv2D", "1D forward", "NCL/OIL", "generic/FP32", "direct forward families by channel/spatial bucket"),
        Planned("IEngine.Conv1DBackwardInput", "CUDA direct backward-input kernel", "1D input gradient", "NCL/OIL", "generic/FP32", "deterministic backward-input specializations"),
        Planned("IEngine.Conv1DBackwardKernel", "CUDA direct backward-weight kernel", "1D weight gradient", "NCL/OIL", "generic/FP32", "deterministic backward-weight specializations"),
        Planned("IEngine.Conv2D", "cuDNN, Winograd, tiled NVRTC, or direct NVRTC", "2D forward", "NCHW/OIHW plus public format overload", "generic/FP32", "1x1, direct-tiled, and Tensor-Core families"),
        Planned("IEngine.Conv2DBackwardInput", "cuDNN or CUDA direct backward-input", "2D input gradient", "NCHW/OIHW", "generic/FP32", "deterministic direct/tiled backward-input"),
        Planned("IEngine.Conv2DBackwardKernel", "cuDNN or CUDA direct backward-weight", "2D weight gradient", "NCHW/OIHW", "generic/FP32", "deterministic direct/tiled backward-weight"),
        new("DirectGpuTensorEngine.Conv2DBackwardBiasGpu",
            "CUDA reduction with established fallback owned by caller",
            "2D bias gradient", "NCHW grad-output to per-channel vector", "generic public; CUDA FP32",
            DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            "v1 exact N1/K64/H16/W16 FP32 warp-per-channel deterministic reduction; all else falls back"),
        new("CudaBackend.TryDirectPtxConv2DBackwardBias",
            "new direct Driver-API PTX route with established fallback owned by caller",
            "same v1 bias-gradient contract", "exact contiguous NCHW to vector", "FP32",
            DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            "hand-emitted sm_86 pointer-only warp-reduction specialization, disabled by default"),
        Planned("IEngine.Conv3D", "CUDA direct 3D kernel", "3D forward", "NCDHW/OIDHW", "generic/FP32", "3D direct/tiled families"),
        Planned("IEngine.Conv3DBackwardInput", "CPU fallback/current backend composition", "3D input gradient", "NCDHW/OIDHW", "generic", "CUDA direct backward-input"),
        Planned("IEngine.Conv3DBackwardKernel", "CPU fallback/current backend composition", "3D weight gradient", "NCDHW/OIDHW", "generic", "CUDA deterministic backward-weight"),
        Planned("IEngine.DepthwiseConv1D", "reshape to CUDA depthwise Conv2D", "depthwise 1D forward", "NCL/channel multiplier", "generic/FP32", "native 1D depthwise PTX"),
        Planned("IEngine.DepthwiseConv1DBackwardInput", "CPU/current backend path", "depthwise 1D input gradient", "NCL", "generic", "native 1D backward-input PTX"),
        Planned("IEngine.DepthwiseConv1DBackwardKernel", "CPU/current backend path", "depthwise 1D weight gradient", "OIL", "generic", "native 1D backward-weight PTX"),
        new("IEngine.DepthwiseConv2D",
            "CUDA NVRTC depthwise kernel with established fallback owned by caller",
            "depthwise 2D forward", "NCHW input/output, depthwise OIHW weights",
            "generic public; CUDA FP32",
            DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            "v1 exact N1/C64/H16/W16 3x3 stride1 pad1 FP32; all other contracts fall back"),
        new("CudaBackend.TryDirectPtxDepthwiseConv2D3x3",
            "new direct Driver-API PTX route with established fallback owned by caller",
            "same v1 depthwise contract", "exact contiguous NCHW/depthwise OIHW", "FP32",
            DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            "hand-emitted sm_86 pointer-only specialization, disabled by default"),
        new("IEngine.DepthwiseConv2DBackwardInput",
            "CUDA depthwise backward-input with established fallback owned by caller",
            "depthwise 2D input gradient", "NCHW grad-out/in, depthwise OIHW weights",
            "generic public; CUDA FP32",
            DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            "v1 exact N1/C64/H16/W16 3x3 stride1 pad1 FP32 (transpose of forward); all else falls back"),
        new("CudaBackend.TryDirectPtxDepthwiseConv2D3x3BackwardInput",
            "new direct Driver-API PTX route with established fallback owned by caller",
            "same v1 depthwise backward-input contract", "exact contiguous NCHW/depthwise OIHW", "FP32",
            DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            "hand-emitted sm_86 pointer-only specialization, disabled by default"),
        new("IEngine.DepthwiseConv2DBackwardKernel",
            "CUDA depthwise backward-weight with established fallback owned by caller",
            "depthwise 2D weight gradient", "NCHW grad-out/input, depthwise OIHW grad-weight",
            "generic public; CUDA FP32",
            DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            "v1 exact N1/C64/H16/W16 3x3 stride1 pad1 FP32, warp-per-channel deterministic reduction; all else falls back"),
        new("CudaBackend.TryDirectPtxDepthwiseConv2D3x3BackwardWeight",
            "new direct Driver-API PTX route with established fallback owned by caller",
            "same v1 depthwise backward-weight contract", "exact contiguous NCHW/depthwise OIHW", "FP32",
            DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            "hand-emitted sm_86 pointer-only warp-reduction specialization, disabled by default"),
        Planned("IEngine.ConvTranspose2D", "CUDA NVRTC transposed convolution", "2D transposed forward", "NCHW/IOHW", "generic/FP32", "direct/tiled transposed families"),
        Planned("IEngine.ConvTranspose2DBackwardInput", "CUDA direct transposed backward-input", "transposed input gradient", "NCHW/IOHW", "generic/FP32", "deterministic backward-input PTX"),
        Planned("IEngine.ConvTranspose2DBackwardKernel", "CUDA direct transposed backward-weight", "transposed weight gradient", "NCHW/IOHW", "generic/FP32", "deterministic backward-weight PTX"),
        Planned("IEngine.ConvTranspose3D", "CPU/current backend fallback", "3D transposed forward", "NCDHW/IODHW", "generic", "CUDA direct/tiled transposed PTX"),
        Planned("IEngine.ConvTranspose3DBackwardInput", "CPU/current backend fallback", "3D transposed input gradient", "NCDHW/IODHW", "generic", "CUDA deterministic backward-input PTX"),
        Planned("IEngine.ConvTranspose3DBackwardKernel", "CPU/current backend fallback", "3D transposed weight gradient", "NCDHW/IODHW", "generic", "CUDA deterministic backward-weight PTX"),
        Planned("IEngine.DeformableConv2D", "CUDA NVRTC DCNv2", "offset/mask deformable forward", "NCHW plus offsets/mask", "generic/FP32", "fused sampling+convolution PTX"),
        Planned("IEngine.DeformableConv2DBackwardInput", "CUDA NVRTC DCNv2 backward", "input gradient", "NCHW plus offsets/mask", "generic/FP32", "deterministic gradient PTX"),
        Planned("IEngine.DeformableConv2DBackwardKernel", "CUDA NVRTC DCNv2 backward", "weight gradient", "NCHW plus offsets/mask", "generic/FP32", "deterministic gradient PTX"),
        Planned("IEngine.DeformableConv2DBackwardOffset", "CUDA NVRTC DCNv2 backward", "offset gradient", "NCHW plus offsets/mask", "generic/FP32", "fused sampling derivative PTX"),
        Planned("IEngine.DeformableConv2DBackwardMask", "CUDA NVRTC DCNv2 backward", "mask gradient", "NCHW plus offsets/mask", "generic/FP32", "fused mask-gradient PTX"),
        Planned("IEngine.DeformableConv2DGrouped", "CUDA grouped deformable kernel", "grouped/deform-group forward", "NCHW plus grouped offsets/mask", "generic/FP32", "group-specialized PTX"),
        Planned("IEngine.DeformableConv2DGroupedBackward", "CUDA grouped deformable backward family", "all grouped gradients", "NCHW plus grouped offsets/mask", "generic/FP32", "four deterministic grouped backward families"),
        Planned("IEngine.LocallyConnectedConv2D", "CUDA NVRTC locally-connected kernel", "unshared spatial weights forward", "NCHW plus per-position OIHW", "generic/FP32", "position-tiled PTX"),
        Planned("IEngine.LocallyConnectedConv2DBackwardInput", "CUDA NVRTC locally-connected backward", "input gradient", "NCHW plus per-position weights", "generic/FP32", "deterministic backward-input PTX"),
        Planned("IEngine.LocallyConnectedConv2DBackwardWeights", "CUDA NVRTC locally-connected backward", "weight gradient", "per-position OIHW", "generic/FP32", "deterministic backward-weight PTX"),
        Planned("IEngine.LocallyConnectedConv2DBackwardBias", "CUDA reduction", "bias gradient", "NCHW to per-position bias", "generic/FP32", "fused backward reduction PTX"),
        Planned("IEngine.FusedConv3D", "CUDA Conv3D plus bias/activation kernels", "3D inference epilogue fusion", "NCDHW/OIDHW", "generic/FP32", "bias/norm/activation epilogue families"),
        Planned("IEngine.FusedConvTranspose2D", "CUDA transposed convolution plus epilogue", "transposed inference fusion", "NCHW/IOHW", "generic/FP32", "bias/norm/activation epilogue families"),
        Planned("IEngine.Unfold", "CUDA im2col/unfold NVRTC kernels", "materialized patch extraction", "NCHW to matrix/windows", "generic/FP32", "eliminate materialization in fused conv; standalone PTX remains explicit"),
        Planned("CudaBackend.Conv2dDirectFp16Hw", "NVRTC half-weight direct convolution", "FP32 input rounded to FP16, FP32 accumulation/output", "NCHW/OIHW-half", "mixed FP16/FP32", "architecture-specific half/Tensor-Core family"),
        Planned("CudaBackend.Im2colKNFp16", "NVRTC fused im2col+FP16 conversion", "FP16 Tensor-Core GEMM preparation", "NCHW to KxN half", "FP16/FP32", "fuse producer into PTX convolution tiles"),
        Planned("CudaBackend.UnfoldKNFp16FromFp16", "NVRTC FP16 im2col", "FP16 patch extraction", "NCHW-half to KxN-half", "FP16", "fuse producer into PTX convolution tiles")
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
