// Copyright (c) AiDotNet. All rights reserved.
// PTX-vs-CUDA-vs-CPU parity scaffold. Per-kernel coverage decisions.
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Tests.Engines.PtxParity;

/// <summary>How a direct-PTX kernel is covered by the parity scaffold.</summary>
public enum PtxParityStatus
{
    /// <summary>
    /// A three-way parity spec exists: the op runs with the direct-PTX gate off
    /// (existing CUDA kernel), on (direct-PTX kernel), and on the CPU fp64
    /// oracle; PTX==CPU and CUDA==CPU are asserted independently.
    /// </summary>
    ThreeWayParity,

    /// <summary>
    /// Coverage is intentionally deferred with a stated reason (e.g. the kernel
    /// backs a multi-tensor fp16 attention path whose Tensor-level entry point
    /// needs bespoke input synthesis). Deferral is an explicit, auditable
    /// decision — NOT a silent gap.
    /// </summary>
    Deferred
}

/// <summary>One kernel's coverage decision.</summary>
public sealed record PtxParitySpec(
    string KernelTypeName,
    PtxParityStatus Status,
    string BackingPublicOp,
    string Note);

/// <summary>
/// The explicit coverage decision for every direct-PTX kernel. The coverage
/// audit (<see cref="PtxKernelCoverageTests"/>) fails if any kernel in
/// <see cref="PtxKernelInventory"/> has no entry here, so a new kernel forces a
/// decision (a real parity spec, or a documented deferral) — never a silent gap.
///
/// As the softmax (#840) and reduction (#843) kernels merge, their entries move
/// to <see cref="PtxParityStatus.ThreeWayParity"/> with a runnable spec.
/// </summary>
public static class PtxParityRegistry
{
    public static IReadOnlyList<PtxParitySpec> Specs { get; } = new[]
    {
        new PtxParitySpec("PtxFusedResidualRmsNormD64Kernel", PtxParityStatus.Deferred,
            "fused residual + RMSNorm (D=64)",
            "backend method has no public op route on main (only the CUDA RmsNorm path is wired), " +
            "and its opt-in is captured at backend construction with no call-time experiment override, " +
            "so a toggle-based three-way spec is not yet possible. Wire a public route + experiment " +
            "override first (mirroring softmax/reduction) to convert to ThreeWayParity."),

        new PtxParitySpec("PtxOnlineFusedAttention128x64Kernel", PtxParityStatus.Deferred,
            "online fused attention",
            "fp16 Q/K/V + softmax-stats side output; needs bespoke fp16 input synthesis and a flash-attention oracle."),
        new PtxParitySpec("PtxAttentionSoftmax32Kernel", PtxParityStatus.Deferred,
            "attention softmax (S=32)",
            "sub-kernel of the attention path; covered transitively by the attention spec once added."),
        new PtxParitySpec("PtxWmmaFusedAttention32x16Kernel", PtxParityStatus.Deferred,
            "wmma fused attention (32x16)",
            "Tensor-Core fp16 path; TF32/fp16 accumulation oracle differs from strict fp32, needs a dedicated tolerance."),
        new PtxParitySpec("PtxWmmaBatchedQkKernel", PtxParityStatus.Deferred,
            "wmma batched Q·Kᵀ",
            "Tensor-Core fp16 GEMM fragment; same fp16-accumulation oracle question as the wmma attention kernel."),
        new PtxParitySpec("PtxFusedDecodeAttentionD64Kernel", PtxParityStatus.Deferred,
            "fused decode attention (D=64)",
            "single-token decode over a KV cache; needs cache-state input synthesis."),
        new PtxParitySpec("PtxFusedPagedPrefillAttentionD64Kernel", PtxParityStatus.Deferred,
            "fused paged-prefill attention (D=64)",
            "paged KV block table input; needs page-table synthesis."),
        new PtxParitySpec("PtxFusedAttentionBackwardD64Kernel", PtxParityStatus.Deferred,
            "fused attention backward (D=64)",
            "gradient kernel; oracle is the backward pass, covered by the tape-gradient parity harness once wired."),
        new PtxParitySpec("PtxFlashAttentionBackwardD64Kernel", PtxParityStatus.Deferred,
            "flash attention backward (D=64)",
            "gradient kernel with bias; same backward-oracle wiring as the fused attention backward kernel."),
        new PtxParitySpec("PtxFusedQkvRopeCacheD64Kernel", PtxParityStatus.Deferred,
            "fused QKV + RoPE + KV-cache write (#858)",
            "multi-output (Q + K/V cache) with baked RoPE tables; needs a dedicated QKV/RoPE/cache oracle."),

        // Issue #841 convolution golden-slice cells. Each is an exact-shape experimental
        // SM86 emitter, GPU-correctness verified against a high-precision CPU oracle
        // (DirectPtxConvolutionTests DriverOnly* facts). Deferred — NOT promoted: the
        // strongest-cuDNN/PyTorch promotion benchmark (3 clean processes, >=1.10x median,
        // p95 <=+10%, Nsight spill evidence) and the toggle-based CUDA-vs-PTX-vs-oracle
        // three-way spec are pending an idle-GPU evidence window (see
        // docs/research/2026-07-22-direct-ptx-convolution-blueprint.md). Winning cells
        // convert to ThreeWayParity with runnable specs as the evidence campaign lands.
        new PtxParitySpec("PtxFusedConv2DNchwK1Kernel", PtxParityStatus.Deferred,
            "conv2d 1x1 + bias + ReLU (foundation cell)",
            "golden slice N1/C64/H16/W16/K64; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv2DNchwK1BackwardInputF32Kernel", PtxParityStatus.Deferred,
            "conv2d 1x1 backward-input", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv2DNchwK1BackwardWeightF32Kernel", PtxParityStatus.Deferred,
            "conv2d 1x1 backward-weight", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv2DBackwardBiasF32Kernel", PtxParityStatus.Deferred,
            "conv2d backward-bias", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv2DFp16K1NchwF32Kernel", PtxParityStatus.Deferred,
            "conv2d 1x1 FP16 storage", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv2DNchw3x3ForwardF32Kernel", PtxParityStatus.Deferred,
            "conv2d 3x3 forward", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv2DNchw3x3BackwardInputF32Kernel", PtxParityStatus.Deferred,
            "conv2d 3x3 backward-input", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv2DNchw3x3BackwardWeightF32Kernel", PtxParityStatus.Deferred,
            "conv2d 3x3 backward-weight", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxFusedConv2DNchw3x3BiasReluF32Kernel", PtxParityStatus.Deferred,
            "conv2d 3x3 + bias + ReLU (fused)", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv1DNclForwardF32Kernel", PtxParityStatus.Deferred,
            "conv1d forward", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv1DNclBackwardInputF32Kernel", PtxParityStatus.Deferred,
            "conv1d backward-input", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv1DNclBackwardWeightF32Kernel", PtxParityStatus.Deferred,
            "conv1d backward-weight", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv3DNcdhw3x3x3ForwardF32Kernel", PtxParityStatus.Deferred,
            "conv3d 3x3x3 forward", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv3DNcdhw3x3x3BackwardInputF32Kernel", PtxParityStatus.Deferred,
            "conv3d 3x3x3 backward-input", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel", PtxParityStatus.Deferred,
            "conv3d 3x3x3 backward-weight", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel", PtxParityStatus.Deferred,
            "conv3d 3x3x3 + bias + ReLU (fused)", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConvTranspose2DNchw3x3ForwardF32Kernel", PtxParityStatus.Deferred,
            "convtranspose2d 3x3 forward", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConvTranspose2DNchw3x3BackwardInputF32Kernel", PtxParityStatus.Deferred,
            "convtranspose2d 3x3 backward-input", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel", PtxParityStatus.Deferred,
            "convtranspose2d 3x3 backward-weight", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel", PtxParityStatus.Deferred,
            "convtranspose2d 3x3 + bias + ReLU (fused)", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConvTranspose3DNcdhw3x3x3ForwardF32Kernel", PtxParityStatus.Deferred,
            "convtranspose3d 3x3x3 forward", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConvTranspose3DNcdhw3x3x3BackwardInputF32Kernel", PtxParityStatus.Deferred,
            "convtranspose3d 3x3x3 backward-input", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxConvTranspose3DNcdhw3x3x3BackwardWeightF32Kernel", PtxParityStatus.Deferred,
            "convtranspose3d 3x3x3 backward-weight", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDepthwiseConv2D3x3BackwardInputF32Kernel", PtxParityStatus.Deferred,
            "depthwise conv2d 3x3 backward-input", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDepthwiseConv2D3x3BackwardWeightF32Kernel", PtxParityStatus.Deferred,
            "depthwise conv2d 3x3 backward-weight", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxFusedDepthwiseConv2D3x3F32Kernel", PtxParityStatus.Deferred,
            "depthwise conv2d 3x3 fused forward", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDepthwiseConv1DNcl3ForwardF32Kernel", PtxParityStatus.Deferred,
            "depthwise conv1d forward", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDepthwiseConv1DNcl3BackwardInputF32Kernel", PtxParityStatus.Deferred,
            "depthwise conv1d backward-input", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel", PtxParityStatus.Deferred,
            "depthwise conv1d backward-weight", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxLocallyConnectedConv2DNchw3x3F32Kernel", PtxParityStatus.Deferred,
            "locally-connected conv2d forward", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel", PtxParityStatus.Deferred,
            "locally-connected conv2d backward-input", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel", PtxParityStatus.Deferred,
            "locally-connected conv2d backward-weight", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxLocallyConnectedConv2DNchwBackwardBiasF32Kernel", PtxParityStatus.Deferred,
            "locally-connected conv2d backward-bias", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDeformableConv2DNchw3x3F32Kernel", PtxParityStatus.Deferred,
            "deformable conv2d (DCNv2) forward", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel", PtxParityStatus.Deferred,
            "deformable conv2d backward-weight", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel", PtxParityStatus.Deferred,
            "deformable conv2d backward-mask", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel", PtxParityStatus.Deferred,
            "deformable conv2d backward-offset", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDeformableConv2DNchw3x3BackwardInputF32Kernel", PtxParityStatus.Deferred,
            "deformable conv2d backward-input", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDeformableConv2DGroupedNchw3x3F32Kernel", PtxParityStatus.Deferred,
            "grouped deformable conv2d forward", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDeformableConv2DGroupedNchw3x3BackwardWeightF32Kernel", PtxParityStatus.Deferred,
            "grouped deformable conv2d backward-weight", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDeformableConv2DGroupedNchw3x3BackwardMaskF32Kernel", PtxParityStatus.Deferred,
            "grouped deformable conv2d backward-mask", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDeformableConv2DGroupedNchw3x3BackwardOffsetF32Kernel", PtxParityStatus.Deferred,
            "grouped deformable conv2d backward-offset", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxDeformableConv2DGroupedNchw3x3BackwardInputF32Kernel", PtxParityStatus.Deferred,
            "grouped deformable conv2d backward-input", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxUnfoldIm2ColNchw3x3F32Kernel", PtxParityStatus.Deferred,
            "unfold / im2col (FP32)", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxIm2colKNFp16Nchw3x3Kernel", PtxParityStatus.Deferred,
            "im2col FP32->FP16 (KxN)", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),
        new PtxParitySpec("PtxUnfoldKNFp16FromFp16Nchw3x3Kernel", PtxParityStatus.Deferred,
            "unfold FP16->FP16 (KxN)", "golden slice; competitive benchmark + three-way spec pending idle-GPU evidence window."),

        // ResNet-class promotion-track cells (realistic compute-bound shapes).
        new PtxParitySpec("PtxConv2DNchwK1ResnetC64H56ForwardF32Kernel", PtxParityStatus.Deferred,
            "conv2d 1x1 + bias + ReLU ResNet-class N32/C64/H56/W56/K64",
            "v1 thread-per-output baseline; shared-mem tiled-GEMM optimization + strongest-cuDNN/cuBLASLt promotion benchmark pending idle-GPU window (see blueprint ResNet campaign)."),
    };

    private static readonly Dictionary<string, PtxParitySpec> ByKernel =
        Specs.ToDictionary(s => s.KernelTypeName, StringComparer.Ordinal);

    public static bool TryGet(string kernelTypeName, out PtxParitySpec spec) =>
        ByKernel.TryGetValue(kernelTypeName, out spec!);

    public static IEnumerable<PtxParitySpec> ThreeWay =>
        Specs.Where(s => s.Status == PtxParityStatus.ThreeWayParity);
}
#endif
