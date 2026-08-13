// Copyright (c) AiDotNet. All rights reserved.
// PTX transfer tracking. Which CUDA kernels have a PTX replacement, and how far
// along it is — the ledger that answers "when can we delete the CUDA kernels?".
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Tests.Engines.PtxParity;

public enum PtxTransferStatus
{
    /// <summary>
    /// A PTX kernel targeting this CUDA kernel exists but is not yet promoted
    /// (fails closed behind the release gate / awaiting benchmark evidence).
    /// The CUDA kernel must stay.
    /// </summary>
    PtxInProgress,

    /// <summary>
    /// One or more exact dispatch cells have cleared parity and the release
    /// gate, but the CUDA kernel still serves unsupported shapes, architectures,
    /// or phases. The validated cells remain promoted; the CUDA kernel must stay.
    /// </summary>
    PtxValidatedPartialCoverage,

    /// <summary>
    /// A PTX kernel has cleared parity + the release gate and is promoted as the
    /// default path. The CUDA kernel is now redundant and may be deleted.
    /// </summary>
    PtxPromotedReplaced,

    /// <summary>
    /// Deliberately excluded from the transfer denominator: an infra/utility
    /// kernel that is not a compute op, a genuine gap not covered by any epic
    /// #833 child, or a kernel a category issue will subsume without a dedicated
    /// port entry. Excluded kernels do not block <c>FullTransferComplete</c>; the
    /// <c>Note</c> states the disposition.
    /// </summary>
    NotPlanned
}

/// <summary>One CUDA kernel's PTX-replacement record.</summary>
public sealed record CudaToPtxEntry(
    string CudaKernel,
    string PtxKernel,
    PtxTransferStatus Status,
    string Note);

/// <summary>
/// The explicit record of every CUDA kernel that has a PTX replacement effort.
/// Kernels absent from this ledger have not been started (they are counted as
/// remaining by <see cref="CudaToPtxTransferTests"/>). Full transfer is complete
/// — and the CUDA kernels are deletable — only when every kernel in
/// <see cref="CudaKernelCensus"/> appears here as
/// <see cref="PtxTransferStatus.PtxPromotedReplaced"/>.
/// </summary>
public static class CudaToPtxTransferLedger
{
    public static IReadOnlyList<CudaToPtxEntry> Entries { get; } = new[]
    {
        new CudaToPtxEntry("sum_axis", "PtxFusedRowReduceF32Kernel", PtxTransferStatus.PtxInProgress,
            "row-sum reduction (#843); PTX kernel on agent/direct-ptx-reduction-843, fails closed until 3 clean >=1.10x runs."),
        new CudaToPtxEntry("reduce_sum", "PtxFusedRowReduceF32Kernel", PtxTransferStatus.PtxInProgress,
            "same reduction family (#843); shares the row-sum PTX kernel."),
        new CudaToPtxEntry("softmax", "PtxFusedSoftmaxF32Kernel", PtxTransferStatus.PtxInProgress,
            "row-softmax (#840); PTX kernel on agent/direct-ptx-softmax-840, fails closed until promotion evidence."),
        new CudaToPtxEntry("softmax_rows", "PtxFusedSoftmaxF32Kernel", PtxTransferStatus.PtxInProgress,
            "same softmax family (#840); shares the row-softmax PTX kernel."),
        new CudaToPtxEntry("rmsnorm_forward", "PtxFusedResidualRmsNormD64Kernel", PtxTransferStatus.PtxInProgress,
            "fused residual RMSNorm; PTX kernel exists but has no public route wired yet (see parity registry)."),
        new CudaToPtxEntry("parity211_cholesky", "PtxRegisterCholesky4x4F32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #853 exact lower FP32 4x4 batch family; NVRTC stays until correctness/performance/spill promotion."),
        new CudaToPtxEntry("parity211_lu_factor", "PtxRegisterSolver4x4F32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #853 exact pivoted FP32 4x4 batch family; NVRTC remains the established fallback."),
        new CudaToPtxEntry("parity211_qr_reduced", "PtxRegisterSolver4x4F32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #853 exact reduced FP32 4x4 batch family; NVRTC remains the established fallback."),
        new CudaToPtxEntry("parity211_eigh", "PtxRegisterSolver4x4F32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #853 exact upper-triangle FP32 4x4 batch family; NVRTC remains the established fallback."),
        new CudaToPtxEntry("dropout_mask", "PtxFusedPhiloxDropoutF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 fuses versioned Philox mask production into dropout forward; disabled until three clean benchmark/Nsight runs."),
        new CudaToPtxEntry("dropout_forward", "PtxFusedPhiloxDropoutF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 exact-shape FP32 fused dropout forward; established CUDA kernels remain the fail-closed fallback."),
        new CudaToPtxEntry("dropout_backward", "PtxDropoutBackwardF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 exact-shape saved-mask float4 backward; established CUDA kernel remains the fail-closed fallback."),
        new CudaToPtxEntry("stateless_dropout_mask", "PtxPhiloxFillF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 exact-shape Philox drop-threshold mask; established CUDA kernel remains the fail-closed fallback."),
        new CudaToPtxEntry("generate_random_uniform", "PtxPhiloxFillF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 exact-shape Philox uniform fill; disabled until parity and benchmark evidence are complete."),
        new CudaToPtxEntry("generate_random_normal", "PtxPhiloxFillF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 exact-shape paired-word Philox Box-Muller fill; established CUDA kernel remains the fallback."),
        new CudaToPtxEntry("gaussian_noise", "PtxPhiloxFillF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 routes Gaussian-noise generation to the versioned normal-fill specialization."),
        new CudaToPtxEntry("gumbel_softmax", "PtxFusedGumbelSoftmax32F32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 fuses Philox Gumbel perturbation and one-warp softmax without a global noise or perturbed-logit intermediate."),
        new CudaToPtxEntry("importance_sampling", "PtxFusedImportanceSampling64F32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 stages each ray once and fuses Philox stratification, CDF traversal, and interpolation without a global CDF."),
        new CudaToPtxEntry("bias_dropout", "PtxFusedBiasPhiloxDropout256F32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 generates the saved Philox mask inside the bias-add consumer and removes the previous temporary device buffer."),
        new CudaToPtxEntry("fused_ddim_step", "PtxFusedDdimStepF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 ports the currently advertised deterministic DDIM API; it does not invent an eta/noise semantic absent from that API."),
        new CudaToPtxEntry("rrelu", "PtxFusedPhiloxRreluF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 fuses Philox slope generation into the public training RReLU consumer; PtxRreluF32Kernel also ports the explicit saved-noise backend contract."),
        new CudaToPtxEntry("rrelu_backward", "PtxRreluF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 ports the exact-shape saved-noise RReLU backward kernel; established CUDA remains the fail-closed fallback."),
        new CudaToPtxEntry("rglru_scan_forward", "PtxFusedRgLruScan128x256Kernel", PtxTransferStatus.PtxValidatedPartialCoverage,
            "exact RG-LRU [1,128,256]/SM86 forward (#846) cleared parity, three clean >=1.10x runs, tails, and Nsight zero-spill gates; the generic CUDA kernel remains required for other shapes and architectures."),

        // Vision/detection/ROI/geometry family (#851). All entries are routed,
        // disabled by default, and remain in progress until the resident-GPU
        // correctness, spill, and >=1.10x promotion evidence is attached.
        new CudaToPtxEntry("detection_box_iou", "PtxFusedPairwiseBoxIouF32Kernel", PtxTransferStatus.PtxInProgress,
            "exact SM86 FP32 pairwise BoxIoU specializations (#851)."),
        new CudaToPtxEntry("detection_generalized_box_iou", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "baked pairwise GIoU specialization family (#851)."),
        new CudaToPtxEntry("detection_distance_box_iou", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "baked pairwise DIoU specialization family (#851)."),
        new CudaToPtxEntry("detection_complete_box_iou", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "baked pairwise CIoU specialization family (#851)."),
        new CudaToPtxEntry("detection_box_area", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "exact contiguous BoxArea specialization family (#851)."),
        new CudaToPtxEntry("detection_box_convert", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "all nine baked XYXY/XYWH/CXCYWH conversion pairs (#851)."),
        new CudaToPtxEntry("detection_iou_backward_a", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "atomics-free owner-A IoU-family backward specializations (#851)."),
        new CudaToPtxEntry("detection_iou_backward_b", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "atomics-free owner-B IoU-family backward specializations (#851)."),
        new CudaToPtxEntry("parity210_pairwise_iou", "PtxFusedPairwiseBoxIouF32Kernel", PtxTransferStatus.PtxInProgress,
            "self-pairwise route reuses the exact BoxIoU PTX module (#851)."),
        new CudaToPtxEntry("iou_loss", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "aligned IoU-loss forward specialization family (#851)."),
        new CudaToPtxEntry("giou_loss", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "aligned GIoU-loss forward specialization family (#851)."),
        new CudaToPtxEntry("diou_loss", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "aligned DIoU-loss forward specialization family (#851)."),
        new CudaToPtxEntry("ciou_loss", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "aligned CIoU-loss forward specialization family (#851)."),
        new CudaToPtxEntry("iou_loss_backward", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "aligned IoU-loss backward specialization family (#851)."),
        new CudaToPtxEntry("giou_loss_backward", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "aligned GIoU-loss backward specialization family (#851)."),
        new CudaToPtxEntry("diou_loss_backward", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "aligned DIoU-loss backward specialization family (#851)."),
        new CudaToPtxEntry("ciou_loss_backward", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "aligned CIoU-loss backward specialization family (#851)."),
        new CudaToPtxEntry("resident_nms", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "fused deterministic selection, geometry, suppression, and compaction (#851)."),
        new CudaToPtxEntry("parity210_masks_to_boxes", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "exact dense mask-to-box reduction specializations (#851)."),
        new CudaToPtxEntry("roi_align", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "baked RoIAlign geometry/sampling specialization (#851)."),
        new CudaToPtxEntry("roi_pool", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "baked RoIPool geometry/reduction specialization (#851)."),
        new CudaToPtxEntry("ps_roi_align", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "baked position-sensitive RoIAlign specialization (#851)."),
        new CudaToPtxEntry("ps_roi_pool", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "baked position-sensitive RoIPool specialization (#851)."),
        new CudaToPtxEntry("parity210_cross3", "PtxVisionKernel", PtxTransferStatus.PtxInProgress,
            "exact contiguous extent-three cross-product specializations (#851)."),

        // --- Triaged anomalies: the 4 census kernels that mapped to no epic
        // #833 child during the full 888-kernel cross-reference. Recorded so the
        // tracker's disposition of every kernel is explicit.
        new CudaToPtxEntry("resident_mode", "(none)", PtxTransferStatus.NotPlanned,
            "GPU-residency infra/utility kernel, not a compute op; excluded from the PTX transfer."),
        new CudaToPtxEntry("squash", "(none)", PtxTransferStatus.NotPlanned,
            "capsule-network squash activation; reassigned to #839 (pointwise/activation) scope — no standalone transfer entry."),
        new CudaToPtxEntry("squash_backward", "(none)", PtxTransferStatus.NotPlanned,
            "squash activation backward; reassigned to #839 (pointwise/activation) scope."),
    };

    public static IEnumerable<CudaToPtxEntry> NotPlanned =>
        Entries.Where(e => e.Status == PtxTransferStatus.NotPlanned);

    public static IEnumerable<CudaToPtxEntry> Replaced =>
        Entries.Where(e => e.Status == PtxTransferStatus.PtxPromotedReplaced);

    public static IEnumerable<CudaToPtxEntry> ValidatedPartial =>
        Entries.Where(e => e.Status == PtxTransferStatus.PtxValidatedPartialCoverage);

    public static IEnumerable<CudaToPtxEntry> InProgress =>
        Entries.Where(e => e.Status == PtxTransferStatus.PtxInProgress);
}
#endif
