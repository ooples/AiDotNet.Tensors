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
        new PtxParitySpec("PtxFusedLinearGeluM1Kernel", PtxParityStatus.Deferred,
            "FusedLinearGELUTransposedM1 (#836)",
            "driver-only PTX-vs-fp64-oracle and public-route tests exist, but the output-major weight " +
            "contract has no equivalent existing-CUDA public route in the parity harness yet; add that " +
            "layout-explicit baseline before classifying this as three-way parity."),
        new PtxParitySpec("PtxFusedLinearTiledKernel", PtxParityStatus.Deferred,
            "general-M fused linear + bias + activation (#836)",
            "driver-only PTX-vs-fp64-oracle coverage exists for None, ReLU, and GELU. The tiled route is " +
            "not production-promoted until its resident three-way championship matrix passes, so the " +
            "current-CUDA arm remains intentionally outside the parity runner."),
        new PtxParitySpec("PtxDenseVectorKernel", PtxParityStatus.Deferred,
            "dot and outer products (#836)",
            "direct PTX driver/public-route oracle coverage exists; the current-CUDA comparison and " +
            "resident release-evidence matrix must pass before promotion to three-way parity."),
        new PtxParitySpec("PtxFusedLoRAKernel", PtxParityStatus.Deferred,
            "FusedLoRAForward (#836)",
            "direct PTX oracle and public-route coverage exists; the current NVRTC and PyTorch GPU " +
            "championship arms remain required before production promotion."),
        new PtxParitySpec("PtxFusedLinearBackwardKernel", PtxParityStatus.Deferred,
            "fused linear activation backward family (#836)",
            "direct PTX covers dInput, dWeight, and dBias without a global masked-gradient tensor; " +
            "the five-activation current-CUDA championship matrix is still required for promotion."),
        new PtxParitySpec("PtxBatchedVectorKernel", PtxParityStatus.Deferred,
            "batched dot and batched outer products (#836)",
            "direct PTX oracle/public-route coverage exists; the current-CUDA three-way evidence " +
            "matrix remains required before production promotion."),
        new PtxParitySpec("PtxStridedDotKernel", PtxParityStatus.Deferred,
            "explicitly strided dot product (#836)",
            "direct PTX bakes offset/step and the valid interval; current-CUDA three-way evidence " +
            "remains required before production promotion."),
        new PtxParitySpec("PtxFusedLinearCrossEntropyKernel", PtxParityStatus.Deferred,
            "fused linear cross entropy, index and dense targets (#836)",
            "direct PTX keeps logits register-only; current-CUDA and PyTorch championship evidence " +
            "remains required before production promotion."),
        new PtxParitySpec("PtxFp16GemmKernel", PtxParityStatus.Deferred,
            "FP16/BF16 GEMM, fanout, conversion, and transpose-free backward (#836)",
            "direct PTX exact-shape correctness baseline exists; Tensor-Core championship " +
            "specializations and three-way evidence remain required before production promotion."),
        new PtxParitySpec("PtxGemmKernel", PtxParityStatus.Deferred,
            "Gemm, GemmAsync, and MatMul (#836)",
            "driver and backend oracle coverage exists; strongest-cuBLASLt/PyTorch three-way evidence is pending."),
        new PtxParitySpec("PtxMatMulTransposedKernel", PtxParityStatus.Deferred,
            "MatMulTransposed (#836)",
            "transpose-B layout is covered by a driver oracle; production evidence remains pending."),
        new PtxParitySpec("PtxFusedGemmBiasKernel", PtxParityStatus.Deferred,
            "GemmBias and fused activation epilogues (#836)",
            "fused driver oracle coverage exists; current CUDA and PyTorch championship arms remain pending."),
        new PtxParitySpec("PtxBatchedGemmKernel", PtxParityStatus.Deferred,
            "strided batched GEMM (#836)",
            "driver oracle coverage exists; the resident three-way performance matrix remains pending."),
        new PtxParitySpec("PtxBatchedGemmFanoutKernel", PtxParityStatus.Deferred,
            "FP32 pointer-array fanout GEMM (#836)",
            "pointer-array driver oracle coverage exists; established-backend parity evidence remains pending."),
        new PtxParitySpec("PtxBatchedGemmExFanoutKernel", PtxParityStatus.Deferred,
            "FP16 pointer-array fanout GEMM (#836)",
            "mixed-precision driver oracle coverage exists; Tensor-Core and established-backend evidence remains pending."),
        new PtxParitySpec("PtxGemmFp16Kernel", PtxParityStatus.Deferred,
            "FP16-input FP32-output GEMM (#836)",
            "conversion and driver oracle coverage exists; Tensor-Core championship evidence remains pending."),
        new PtxParitySpec("PtxHgemmKernel", PtxParityStatus.Deferred,
            "FP16-input FP16-output Hgemm (#836)",
            "driver oracle coverage exists; Tensor-Core championship evidence remains pending."),
        new PtxParitySpec("PtxGemmFp16TransposedKernel", PtxParityStatus.Deferred,
            "FP16 transpose-B GEMM (#836)",
            "driver backward-oracle coverage exists; established-backend parity evidence remains pending."),
        new PtxParitySpec("PtxGemmFp16ContractMKernel", PtxParityStatus.Deferred,
            "FP16 contract-M GEMM (#836)",
            "driver dWeight-oracle coverage exists; established-backend parity evidence remains pending."),
        new PtxParitySpec("PtxGemmContractMKernel", PtxParityStatus.Deferred,
            "FP32 contract-M GEMM (#836)",
            "driver dWeight-oracle coverage exists; established-backend parity evidence remains pending."),
        new PtxParitySpec("PtxBiasGradientKernel", PtxParityStatus.Deferred,
            "linear bias gradient (#836)",
            "driver column-reduction oracle coverage exists; public three-way parity remains pending."),
        new PtxParitySpec("PtxLinearActivationBackwardKernel", PtxParityStatus.Deferred,
            "linear activation gradient (#836)",
            "preactivation and saved-output driver oracles exist; public three-way parity remains pending."),
        new PtxParitySpec("PtxDotProductKernel", PtxParityStatus.Deferred,
            "dot product (#836)",
            "ISA-correct CTA reduction has a driver oracle; strongest-library evidence remains pending."),
        new PtxParitySpec("PtxOuterProductKernel", PtxParityStatus.Deferred,
            "outer product (#836)",
            "pointer-only driver oracle coverage exists; strongest-library evidence remains pending."),
        new PtxParitySpec("PtxFusedLoRAForwardKernel", PtxParityStatus.Deferred,
            "output-major fused LoRA forward (#836)",
            "single-launch shared-Z driver oracle coverage exists; layout-equivalent competitor evidence remains pending."),
        new PtxParitySpec("PtxFusedLoRAForwardStandardKernel", PtxParityStatus.Deferred,
            "standard-layout fused LoRA forward (#836)",
            "multi-rank driver and backend oracle coverage exists; production timing evidence remains pending."),
        new PtxParitySpec("PtxSoftmaxCrossEntropyIndexKernel", PtxParityStatus.Deferred,
            "fused index-target softmax cross entropy (#836)",
            "stable loss/gradient driver oracle coverage exists; public fused three-way parity remains pending."),
        new PtxParitySpec("PtxSoftmaxCrossEntropyDenseKernel", PtxParityStatus.Deferred,
            "fused dense-target softmax cross entropy (#836)",
            "stable loss/gradient driver oracle coverage exists; public fused three-way parity remains pending."),
    };

    private static readonly Dictionary<string, PtxParitySpec> ByKernel =
        Specs.ToDictionary(s => s.KernelTypeName, StringComparer.Ordinal);

    public static bool TryGet(string kernelTypeName, out PtxParitySpec spec) =>
        ByKernel.TryGetValue(kernelTypeName, out spec!);

    public static IEnumerable<PtxParitySpec> ThreeWay =>
        Specs.Where(s => s.Status == PtxParityStatus.ThreeWayParity);
}
#endif
