using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxPoolingCoverageStatus
{
    ExistingBackend,
    ExperimentalDirectPtx,
    PromotedDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxPoolingCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxPoolingCoverageStatus Status);

/// <summary>
/// Executable issue-#842 inventory. Every CUDA pooling, interpolation, padding,
/// grid-sample, and spatial-transform boundary has one explicit direct-PTX lane.
/// Shape families remain closed sets: a planned row must be split into exact
/// physical ABI cells before it can be promoted.
/// </summary>
internal static class DirectPtxPoolingCoverageManifest
{
    private const string Bchw = "canonical contiguous NCHW flattened to [batch*channels, H*W]";
    private const string Windowed = "canonical contiguous NCHW with baked window/stride";

    internal static IReadOnlyList<DirectPtxPoolingCoverageCell> All { get; } =
    [
        Experimental("CudaBackend.GlobalAvgPool2D", "NVRTC global_avgpool2d", "mean over H*W per channel", Bchw, "FP32", "v1 Ampere warp-row S64/S128 exact-shape cells"),
        Planned("CudaBackend.GlobalMaxPool2D+indices", "NVRTC global_maxpool2d with saveIndices=1", "max over H*W per channel plus arg-max indices", Bchw, "FP32 values; INT32 indices", "blocked-by-design: reducing (value,index) pairs needs a paired warp shuffle, so the value-only lane must never be used when indices are requested"),
        Experimental("CudaBackend.GlobalMaxPool2D", "NVRTC global_maxpool2d (one serial thread per plane)", "max over H*W per channel", Bchw, "FP32", "v1 Ampere warp-per-plane exact-bucket cells; VALUE PATH ONLY - the saveIndices arg-max path stays on the established kernel and is tracked separately"),
        Planned("CudaBackend.AdaptiveAvgPool2D", "NVRTC adaptive_avgpool2d", "average pool to a baked output grid", Windowed, "FP32", "baked-output-grid-families"),
        Planned("CudaBackend.AvgPool2D", "NVRTC avgpool2d", "windowed average pool with baked kernel/stride", Windowed, "FP32", "baked-window-stride-avg-families"),
        Planned("CudaBackend.MaxPool2D", "NVRTC maxpool2d", "windowed max pool with argmax indices", Windowed, "FP32; INT32 indices", "baked-window-stride-max-families"),
        Planned("CudaBackend.GlobalAvgPool2DBackward", "NVRTC global_avgpool2d_backward", "broadcast gradient / (H*W)", Bchw, "FP32", "broadcast-scaled-backward-families"),
        Planned("CudaBackend.MaxPool2DBackward", "NVRTC maxpool2d_backward", "route gradient to argmax positions", Windowed, "FP32; INT32 indices", "selective-backward-families"),
        Planned("CudaBackend.UpsampleNearest2D", "none - no backend op exists", "nearest-neighbor spatial upsample", Windowed, "FP32", "blocked: needs a public backend op first, then baked-scale-nearest-families"),
        Planned("CudaBackend.UpsampleBilinear2D", "none - no backend op exists", "bilinear spatial upsample", Windowed, "FP32", "blocked: needs a public backend op first, then baked-scale-bilinear-families"),
        Planned("CudaBackend.Interpolate", "none - no backend op exists", "generic spatial interpolation", Windowed, "FP32", "blocked: needs a public backend op first, then baked-mode-scale-families"),
        Planned("CudaBackend.Pad2D", "NVRTC pad2d", "constant/reflect/replicate spatial padding", Windowed, "FP32", "baked-pad-mode-extent-families"),
        Planned("CudaBackend.GridSample2D", "none - no backend op exists", "bilinear sampling at normalized grid coords", "canonical contiguous NCHW input + [N,Ho,Wo,2] grid", "FP32", "blocked: needs a public backend op first, then bilinear-grid-sample-families"),
        Planned("CudaBackend.RoiAlign", "none - no backend op exists", "region-of-interest aligned pooling", "canonical contiguous NCHW + [K,5] rois", "FP32", "blocked: needs a public backend op first, then baked-output-bins-roi-families"),
        Planned("CudaBackend.SpatialTransform", "none - no backend op exists", "affine-grid spatial warp", "canonical contiguous NCHW + [N,2,3] theta", "FP32", "blocked: needs a public backend op first, then affine-grid-sample-families"),
        Planned("CudaBackend.PixelShuffle", "NVRTC pixel_shuffle", "sub-pixel channel-to-space rearrange", Windowed, "FP32", "baked-upscale-factor-families"),
        Planned("DirectGpuEngine.GlobalAvgPool2D", "array upload + CudaBackend.GlobalAvgPool2D + download", "public array global average pool", Bchw, "generic public; CUDA FP32", "public-array-gap-routing"),
        Planned("DirectGpuTensorEngine.TensorGlobalAvgPool", "resident CudaBackend.GlobalAvgPool2D or CPU fallback", "public tensor global average pool", "logical NCHW; canonical contiguous fast path", "generic public; CUDA FP32", "public-tensor-gap-routing-and-materialization"),
        Planned("DirectGpuTensorEngine.TensorAvgPool2D", "CudaBackend.AvgPool2D or CPU fallback", "public windowed average pool", "logical NCHW; canonical contiguous admitted view", "generic public; CUDA FP32", "public-avgpool-routing"),
        Planned("DirectGpuTensorEngine.TensorMaxPool2D", "CudaBackend.MaxPool2D or CPU fallback", "public windowed max pool", "logical NCHW; canonical contiguous admitted view", "generic public; CUDA FP32/INT32", "public-maxpool-routing"),
        Planned("DirectGpuTensorEngine.TensorInterpolate", "CudaBackend.Interpolate or CPU fallback", "public spatial interpolation", "logical NCHW; canonical contiguous admitted view", "generic public; CUDA FP32", "public-interpolate-routing"),
    ];

    private static readonly IReadOnlyDictionary<string, DirectPtxPoolingCoverageCell> ByApi =
        All.ToDictionary(cell => cell.Api, StringComparer.Ordinal);

    internal static DirectPtxPoolingCoverageCell Get(string api) =>
        ByApi.TryGetValue(api, out DirectPtxPoolingCoverageCell? cell)
            ? cell
            : throw new KeyNotFoundException($"No direct-PTX pooling coverage cell for '{api}'.");

    private static DirectPtxPoolingCoverageCell Experimental(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxPoolingCoverageStatus.ExperimentalDirectPtx);

    private static DirectPtxPoolingCoverageCell Planned(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxPoolingCoverageStatus.PlannedDirectPtx);
}
