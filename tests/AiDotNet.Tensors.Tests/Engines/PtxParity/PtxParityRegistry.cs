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

        // Issue #854 specialized-scientific / ANN / hypercomplex / hyperbolic / quantum / Instant-NGP
        // kernels. Each has a GPU-gated DriverOnly CPU-fp64-oracle parity test, an emitter structure
        // test, and a backend dispatch test in DirectPtxScientificTests. The three-way gate-toggle
        // parity spec in this harness is deferred pending the scientific parity harness (mirrors the
        // attention entries above); every op fails closed and is unpromoted until GPU-validated.
        new PtxParitySpec("PtxComplexMultiplyKernel", PtxParityStatus.Deferred, "complex multiply (#854)", ScientificNote),
        new PtxParitySpec("PtxComplexConjugateKernel", PtxParityStatus.Deferred, "complex conjugate (#854)", ScientificNote),
        new PtxParitySpec("PtxComplexMagnitudeKernel", PtxParityStatus.Deferred, "complex magnitude (#854)", ScientificNote),
        new PtxParitySpec("PtxComplexPhaseKernel", PtxParityStatus.Deferred, "complex phase / atan2 (#854)", ScientificNote),
        new PtxParitySpec("PtxComplexMatVecKernel", PtxParityStatus.Deferred, "complex mat-vec (#854)", ScientificNote),
        new PtxParitySpec("PtxOctonionAddKernel", PtxParityStatus.Deferred, "octonion add (#854)", ScientificNote),
        new PtxParitySpec("PtxOctonionMultiplyKernel", PtxParityStatus.Deferred, "octonion multiply (#854)", ScientificNote),
        new PtxParitySpec("PtxMobiusAddKernel", PtxParityStatus.Deferred, "mobius add (#854)", ScientificNote),
        new PtxParitySpec("PtxPoincareDistanceKernel", PtxParityStatus.Deferred, "poincare distance (#854)", ScientificNote),
        new PtxParitySpec("PtxPoincareProjectKernel", PtxParityStatus.Deferred, "poincare project (#854)", ScientificNote),
        new PtxParitySpec("PtxPoincareExpMapKernel", PtxParityStatus.Deferred, "poincare exp-map (#854)", ScientificNote),
        new PtxParitySpec("PtxRbfForwardKernel", PtxParityStatus.Deferred, "rbf forward (#854)", ScientificNote),
        new PtxParitySpec("PtxPairwiseDistanceKernel", PtxParityStatus.Deferred, "pairwise distance L2/squared (#854)", ScientificNote),
        new PtxParitySpec("PtxCosineSimilarityKernel", PtxParityStatus.Deferred, "cosine similarity (#854)", ScientificNote),
        new PtxParitySpec("PtxQuantumMeasurementKernel", PtxParityStatus.Deferred, "quantum measurement (#854)", ScientificNote),
        new PtxParitySpec("PtxQuantumRotationKernel", PtxParityStatus.Deferred, "quantum rotation (#854)", ScientificNote),
        new PtxParitySpec("PtxMeasurementForwardKernel", PtxParityStatus.Deferred, "measurement forward (#854)", ScientificNote),
        new PtxParitySpec("PtxNormalizeProbabilitiesKernel", PtxParityStatus.Deferred, "normalize probabilities (#854)", ScientificNote),
        new PtxParitySpec("PtxSphericalHarmonicsKernel", PtxParityStatus.Deferred, "spherical harmonics (#854)", ScientificNote),
        new PtxParitySpec("PtxSphericalHarmonicsBackwardKernel", PtxParityStatus.Deferred, "spherical harmonics backward (#854)", ScientificNote),
        new PtxParitySpec("PtxSphericalSoftmaxKernel", PtxParityStatus.Deferred, "spherical softmax (#854)", ScientificNote),
        new PtxParitySpec("PtxCapsuleContractionKernel", PtxParityStatus.Deferred, "capsule predictions/transform (#854)", ScientificNote),
        new PtxParitySpec("PtxCapsuleWeightedSumKernel", PtxParityStatus.Deferred, "capsule weighted sum (#854)", ScientificNote),
        new PtxParitySpec("PtxCapsuleAgreementKernel", PtxParityStatus.Deferred, "capsule agreement (#854)", ScientificNote),
        new PtxParitySpec("PtxAnnComputeDistancesKernel", PtxParityStatus.Deferred, "ann compute distances (#854)", ScientificNote),
        new PtxParitySpec("PtxAnnPqDistanceTablesKernel", PtxParityStatus.Deferred, "ann pq distance tables (#854)", ScientificNote),
        new PtxParitySpec("PtxAnnIvfAssignKernel", PtxParityStatus.Deferred, "ann ivf assign (#854)", ScientificNote),
        new PtxParitySpec("PtxAnnPqAdcScanKernel", PtxParityStatus.Deferred, "ann pq adc scan (#854)", ScientificNote),
        new PtxParitySpec("PtxInstantNgpHashEncodeKernel", PtxParityStatus.Deferred, "instant-ngp hash encode (#854)", ScientificNote),
        new PtxParitySpec("PtxInstantNgpHashEncodeBackwardKernel", PtxParityStatus.Deferred, "instant-ngp hash encode backward (#854)", ScientificNote),
        new PtxParitySpec("PtxMeshLaplacianKernel", PtxParityStatus.Deferred, "uniform mesh laplacian (#854)", ScientificNote),
    };

    private const string ScientificNote =
        "issue #854 direct-PTX kernel; a GPU-gated DriverOnly CPU-fp64-oracle parity test, an emitter " +
        "structure test, and a backend dispatch test exist in DirectPtxScientificTests. The three-way " +
        "gate-toggle parity spec in this harness is deferred pending the scientific parity harness; the " +
        "op fails closed and stays unpromoted until GPU-validated.";

    private static readonly Dictionary<string, PtxParitySpec> ByKernel =
        Specs.ToDictionary(s => s.KernelTypeName, StringComparer.Ordinal);

    public static bool TryGet(string kernelTypeName, out PtxParitySpec spec) =>
        ByKernel.TryGetValue(kernelTypeName, out spec!);

    public static IEnumerable<PtxParitySpec> ThreeWay =>
        Specs.Where(s => s.Status == PtxParityStatus.ThreeWayParity);
}
#endif
