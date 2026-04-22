// Copyright (c) AiDotNet. All rights reserved.
// Vulkan launcher shims for the vision detection kernels (Issue #217).
// Mirrors VulkanBackend.Parity210.cs — pipelines cached by GLSL source hash
// via GetOrCreateGlslPipeline, so first call compiles SPIR-V, subsequent
// dispatches are O(1) lookups.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IDetectionBackend
{
    private const uint DetectionPushNm = 8u;       // 2 ints (n, m)
    private const uint DetectionPushN = 4u;        // 1 int (n)
    private const uint DetectionPushNFromTo = 12u; // 3 ints (n, fromFormat, toFormat)

    private void DispatchPairwiseIou(string shader,
        IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
    {
        if (n <= 0 || m <= 0) return;
        int total = n * m;
        var pc = new uint[] { (uint)n, (uint)m };
        GlslBinaryOp(shader, boxesA, boxesB, output, total, pc, DetectionPushNm);
    }

    public void BoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou(VulkanDetectionKernels.BoxIou, boxesA, boxesB, output, n, m);

    public void GeneralizedBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou(VulkanDetectionKernels.GeneralizedBoxIou, boxesA, boxesB, output, n, m);

    public void DistanceBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou(VulkanDetectionKernels.DistanceBoxIou, boxesA, boxesB, output, n, m);

    public void CompleteBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou(VulkanDetectionKernels.CompleteBoxIou, boxesA, boxesB, output, n, m);

    public void BoxArea(IGpuBuffer boxes, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        var pc = new uint[] { (uint)n };
        GlslUnaryOp(VulkanDetectionKernels.BoxArea, boxes, output, n, pc, DetectionPushN);
    }

    public void BoxConvert(IGpuBuffer boxes, IGpuBuffer output, int n, int fromFormat, int toFormat)
    {
        if (n <= 0) return;
        if ((uint)fromFormat > 2 || (uint)toFormat > 2)
            throw new ArgumentException(
                $"fromFormat/toFormat must be 0/1/2; got {fromFormat}, {toFormat}.");
        var pc = new uint[] { (uint)n, (uint)fromFormat, (uint)toFormat };
        GlslUnaryOp(VulkanDetectionKernels.BoxConvert, boxes, output, n, pc, DetectionPushNFromTo);
    }

    // -----------------------------------------------------------------------
    // IoU family backward — 4 SSBO bindings, 3 push-constant ints.
    // Dispatched as two separate pipelines (A-side N threads, B-side M).
    // -----------------------------------------------------------------------
    private const uint DetectionPushNMVariant = 12u;   // 3 ints

    public void IouFamilyBackward(
        IGpuBuffer gradOutput, IGpuBuffer boxesA, IGpuBuffer boxesB,
        IGpuBuffer gradA, IGpuBuffer gradB,
        int n, int m, int variant)
    {
        // Don't short-circuit on n==0 or m==0: the kernels iterate the
        // "other" dim with a for-loop, so when M==0 the A-side kernel
        // writes zero sums for each of N rows (correct), and symmetric
        // for N==0. Skipping the dispatch would leak whatever stale data
        // was in the pooled gradA/gradB buffers into autodiff. Only bail
        // when BOTH dims are zero (nothing to write).
        if (n <= 0 && m <= 0) return;
        var pc = new uint[] { (uint)n, (uint)m, (uint)variant };
        if (n > 0)
            GlslQuadOp(VulkanDetectionKernels.IouBackwardA,
                gradOutput, boxesA, boxesB, gradA, n, pc, DetectionPushNMVariant);
        if (m > 0)
            GlslQuadOp(VulkanDetectionKernels.IouBackwardB,
                gradOutput, boxesA, boxesB, gradB, m, pc, DetectionPushNMVariant);
    }
}
