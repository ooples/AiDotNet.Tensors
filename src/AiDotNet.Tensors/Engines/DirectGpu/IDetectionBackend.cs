// Copyright (c) AiDotNet. All rights reserved.
// Secondary interface for backends that ship native vision-detection
// kernels (BoxIoU family, BoxArea, BoxConvert, MasksToBoxes).
// DirectGpuTensorEngine type-tests against this interface when
// dispatching; backends that don't implement it transparently fall
// through to the CpuEngine path via inheritance.

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Optional capability interface for GPU backends that ship native
    /// kernels for the vision detection ops added by Issue #217.
    /// </summary>
    /// <remarks>
    /// Each method takes raw <see cref="IGpuBuffer"/> handles so callers
    /// (typically <see cref="DirectGpuTensorEngine"/>) can plumb their own
    /// activation cache and uploads. Output buffers are caller-allocated.
    /// All box buffers carry float[..., 4] data in <see cref="BoxFormat.XYXY"/>.
    /// </remarks>
    public interface IDetectionBackend
    {
        /// <summary>
        /// Pairwise IoU between two box sets.
        /// <para><c>boxesA</c> = N×4, <c>boxesB</c> = M×4, <c>output</c> = N×M.</para>
        /// </summary>
        void BoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m);

        /// <summary>
        /// Pairwise generalized IoU (Rezatofighi 2019). Same shapes as
        /// <see cref="BoxIou"/>; output in <c>(-1, 1]</c>.
        /// </summary>
        void GeneralizedBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m);

        /// <summary>
        /// Pairwise distance IoU (Zheng 2020). Same shapes as
        /// <see cref="BoxIou"/>.
        /// </summary>
        void DistanceBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m);

        /// <summary>
        /// Pairwise complete IoU (Zheng 2020) — DIoU plus aspect-ratio
        /// penalty. Same shapes as <see cref="BoxIou"/>.
        /// </summary>
        void CompleteBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m);

        /// <summary>
        /// Per-box area for a flat box buffer. <c>boxes</c> = N×4 (xyxy);
        /// <c>output</c> = N. Negative widths/heights clamp to 0 to match
        /// torchvision's <c>box_area</c>.
        /// </summary>
        void BoxArea(IGpuBuffer boxes, IGpuBuffer output, int n);

        /// <summary>
        /// Box-format conversion. Source and destination encodings are
        /// identified by integer codes matching <see cref="BoxFormat"/>:
        /// 0 = XYXY, 1 = XYWH, 2 = CXCYWH. <c>boxes</c> = N×4;
        /// <c>output</c> = N×4.
        /// </summary>
        void BoxConvert(IGpuBuffer boxes, IGpuBuffer output, int n, int fromFormat, int toFormat);

        /// <summary>
        /// Backward kernel for the pairwise IoU family. Implemented as
        /// two independent passes (atomics-free, portable to WebGPU
        /// which has no <c>atomic&lt;f32&gt;</c>):
        /// <list type="bullet">
        ///   <item><c>detection_iou_backward_a</c> — one thread per row
        ///     of <c>A</c>, iterates <c>j = 0..M</c>, writes the four
        ///     coord gradients for <c>gradA[i]</c> directly.</item>
        ///   <item><c>detection_iou_backward_b</c> — symmetric for
        ///     <c>gradB</c>.</item>
        /// </list>
        /// <paramref name="variant"/> is an int code matching the CPU
        /// enum: 0 = IoU, 1 = GIoU, 2 = DIoU, 3 = CIoU. The CIoU variant
        /// treats α as stop-gradient (Zheng 2020 / torchvision).
        /// <para>Shapes: gradOutput = N×M; boxesA = N×4; boxesB = M×4;
        /// gradA = N×4; gradB = M×4.</para>
        /// <para>The kernel writes final gradients directly — callers do
        /// NOT need to pre-zero <paramref name="gradA"/> or <paramref name="gradB"/>.</para>
        /// </summary>
        void IouFamilyBackward(
            IGpuBuffer gradOutput, IGpuBuffer boxesA, IGpuBuffer boxesB,
            IGpuBuffer gradA, IGpuBuffer gradB,
            int n, int m, int variant);
    }
}
