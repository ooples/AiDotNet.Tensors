#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxVisionCoverageStatus
{
    ExperimentalDirectPtx
}

internal sealed record DirectPtxVisionCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    DirectPtxVisionCoverageStatus Status,
    string DirectPtxAssignment);

/// <summary>
/// Executable issue-#851 inventory. Experimental means a direct-PTX emitter
/// and production fallback route exist, not that the specialization has passed
/// the still-pending resident-GPU promotion gate.
/// </summary>
internal static class DirectPtxVisionCoverageManifest
{
    private static DirectPtxVisionCoverageCell Ptx(
        string api, string existing, string semantics, string layout,
        string assignment) =>
        new(api, existing, semantics, layout, "FP32",
            DirectPtxVisionCoverageStatus.ExperimentalDirectPtx, assignment);

    internal static IReadOnlyList<DirectPtxVisionCoverageCell> All { get; } =
    [
        Ptx("IEngine.BoxIou -> IDetectionBackend.BoxIou", "NVRTC detection_box_iou",
            "pairwise XYXY IoU; clamped degenerate area; zero non-positive union",
            "boxes [N,4] + [M,4] -> row-major [N,M]",
            "pointer-only SM86 v1 for 256x256, 1024x256, 1024x1024, 4096x256"),
        Ptx("IEngine.GeneralizedBoxIou -> IDetectionBackend.GeneralizedBoxIou", "NVRTC detection_generalized_box_iou",
            "pairwise GIoU", "canonical XYXY -> row-major [N,M]",
            "baked metric and exact BoxIoU shape matrix"),
        Ptx("IEngine.DistanceBoxIou -> IDetectionBackend.DistanceBoxIou", "NVRTC detection_distance_box_iou",
            "pairwise DIoU", "canonical XYXY -> row-major [N,M]",
            "baked metric and exact BoxIoU shape matrix"),
        Ptx("IEngine.CompleteBoxIou -> IDetectionBackend.CompleteBoxIou", "NVRTC detection_complete_box_iou",
            "pairwise CIoU", "canonical XYXY -> row-major [N,M]",
            "baked metric and exact BoxIoU shape matrix"),
        Ptx("IEngine.BoxArea -> IDetectionBackend.BoxArea", "NVRTC detection_box_area",
            "clamped XYXY area", "[N,4] -> [N]",
            "exact N=256/1024/4096 vector-load specialization"),
        Ptx("IEngine.BoxConvert -> IDetectionBackend.BoxConvert", "NVRTC detection_box_convert",
            "XYXY/XYWH/CXCYWH conversion", "[N,4] -> [N,4]",
            "all nine baked format pairs for N=256/1024/4096"),
        Ptx("IEngine.BoxIouBackward -> IDetectionBackend.IouFamilyBackward(v0)", "two NVRTC deterministic reductions",
            "IoU gradients for both box owners",
            "grad [N,M], boxes A/B -> grad A/B",
            "separate atomics-free owner-A and owner-B exact PTX modules"),
        Ptx("IEngine.GeneralizedBoxIouBackward -> IDetectionBackend.IouFamilyBackward(v1)", "two NVRTC deterministic reductions",
            "GIoU gradients for both box owners", "grad [N,M], boxes A/B -> grad A/B",
            "separate atomics-free owner-A and owner-B exact PTX modules"),
        Ptx("IEngine.DistanceBoxIouBackward -> IDetectionBackend.IouFamilyBackward(v2)", "two NVRTC deterministic reductions",
            "DIoU gradients for both box owners", "grad [N,M], boxes A/B -> grad A/B",
            "separate atomics-free owner-A and owner-B exact PTX modules"),
        Ptx("IEngine.CompleteBoxIouBackward -> IDetectionBackend.IouFamilyBackward(v3)", "two NVRTC deterministic reductions",
            "CIoU gradients for both box owners", "grad [N,M], boxes A/B -> grad A/B",
            "separate atomics-free owner-A and owner-B exact PTX modules"),
        Ptx("IDirectGpuBackend.PairwiseIou", "NVRTC parity210_pairwise_iou",
            "self-pairwise XYXY IoU", "[N,4] -> [N,N]",
            "routes admitted N through the BoxIoU module with read-only input alias"),
        Ptx("IEngine.TensorIoULoss -> IDirectGpuBackend.IoULoss", "NVRTC iou_loss", "aligned 1-IoU",
            "two [N,4] inputs -> [N]", "exact N=256/1024/4096"),
        Ptx("IEngine.TensorGIoULoss -> IDirectGpuBackend.GIoULoss", "NVRTC giou_loss", "aligned 1-GIoU",
            "two [N,4] inputs -> [N]", "exact N=256/1024/4096"),
        Ptx("IEngine.TensorDIoULoss -> IDirectGpuBackend.DIoULoss", "NVRTC diou_loss", "aligned 1-DIoU",
            "two [N,4] inputs -> [N]", "exact N=256/1024/4096"),
        Ptx("IEngine.TensorCIoULoss -> IDirectGpuBackend.CIoULoss", "NVRTC ciou_loss", "aligned 1-CIoU",
            "two [N,4] inputs -> [N]", "exact N=256/1024/4096"),
        Ptx("IDirectGpuBackend.IoULossBackward", "NVRTC analytical backward",
            "aligned IoU-loss gradient", "grad [N] + boxes -> grad [N,4]",
            "deterministic register-resident v1"),
        Ptx("IDirectGpuBackend.GIoULossBackward", "NVRTC analytical backward",
            "aligned GIoU-loss gradient", "grad [N] + boxes -> grad [N,4]",
            "deterministic register-resident v1"),
        Ptx("IDirectGpuBackend.DIoULossBackward", "NVRTC analytical backward",
            "aligned DIoU-loss gradient", "grad [N] + boxes -> grad [N,4]",
            "deterministic register-resident v1"),
        Ptx("IDirectGpuBackend.CIoULossBackward", "NVRTC analytical backward",
            "aligned CIoU-loss gradient", "grad [N] + boxes -> grad [N,4]",
            "deterministic register-resident v1"),
        Ptx("IEngine.Nms -> INmsBackend.Nms", "NVRTC resident_nms",
            "stable score ordering, lower-index ties, optional class isolation",
            "boxes/scores/classes/scratch/output/count",
            "single-controller deterministic N=256/1024, threshold=0.5 and batched mode baked"),
        Ptx("IEngine.BatchedNms -> INmsBackend.Nms", "class-aware INmsBackend route",
            "public class-aware stable NMS", "canonical boxes/scores/class IDs",
            "same resident direct-PTX NMS with baked batched flag"),
        Ptx("IEngine.MasksToBoxes -> IDirectGpuBackend.MasksToBoxes", "NVRTC parity210_masks_to_boxes",
            "tight bounds of nonzero pixels; empty mask -> zero box",
            "dense row-major [N,H,W] -> [N,4]", "exact 256x28x28 and 64x64x64 reductions"),
        Ptx("IEngine.RoIAlign -> IRoiBackend.RoIAlign", "NVRTC roi_align", "bilinear sampled pooling",
            "NCHW [1,256,56,56] + [256,5] -> [256,256,7,7]",
            "sampling=2, scale=0.25, aligned semantics baked"),
        Ptx("IEngine.RoIPool -> IRoiBackend.RoIPool", "NVRTC roi_pool", "quantized-bin max pooling",
            "NCHW [1,256,56,56] + [256,5] -> [256,256,7,7]",
            "pooled geometry and scale=0.25 baked"),
        Ptx("IEngine.PsRoIAlign -> IRoiBackend.PsRoIAlign", "NVRTC ps_roi_align",
            "position-sensitive bilinear pooling",
            "NCHW [1,196,56,56] + [256,5] -> [256,4,7,7]",
            "channel mapping, sampling=2, and scale=0.25 baked"),
        Ptx("IEngine.PsRoIPool -> IRoiBackend.PsRoIPool", "NVRTC ps_roi_pool",
            "position-sensitive average pooling",
            "NCHW [1,196,56,56] + [256,5] -> [256,4,7,7]",
            "channel mapping and scale=0.25 baked"),
        Ptx("IEngine.TensorCross -> IDirectGpuBackend.Cross3", "NVRTC parity210_cross3",
            "cross product along an extent-three axis",
            "dense row-major [outer,3,inner]", "exact (256,1), (1024,1), (256,64)"),
        Ptx("IEngine.TensorMeshgrid(x,y) tuple overload", "CpuEngine tuple materialization",
            "two-vector xy coordinate grids",
            "vectors [N0]/[N1] -> two row-major dense grids",
            "explicit DirectGpu route to the atomic two-module PTX pair"),
        Ptx("IEngine.TensorMeshgrid(Tensor<T>[],indexing) two-vector overload", "Repeat/Tile composition",
            "two-vector ij/xy coordinate grids",
            "vectors [N0]/[N1] -> two row-major dense grids",
            "direct load/store broadcast for 256x256 and 1024x256")
    ];

    internal static DirectPtxVisionCoverageCell Get(string api)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(api);
        foreach (DirectPtxVisionCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException(
            $"Vision API '{api}' is not assigned in the #851 coverage manifest.");
    }
}
#endif
