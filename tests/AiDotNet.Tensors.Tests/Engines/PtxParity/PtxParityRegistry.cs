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
        new PtxParitySpec("PtxRowNormalizationD64Kernel", PtxParityStatus.Deferred,
            "row normalization family (D=64, #838)",
            "the 16-operation family has direct GPU-vs-CPU correctness coverage, including forward, " +
            "backward, parameter-gradient, fp16, L2, and atomic experimental variants, but the parity " +
            "scaffold does not yet provide an equivalent gate-off CUDA leg for every specialization. " +
            "Keep the family deferred and unpromoted until that three-way matrix and the competitive " +
            "performance gates are complete."),

        new PtxParitySpec("PtxChannelNormalizationD64Kernel", PtxParityStatus.Deferred,
            "channel normalization family (64-value units, #838)",
            "the 17-operation BatchNorm, GroupNorm, InstanceNorm, activation, residual, backward, and " +
            "parameter-gradient family has direct GPU-vs-CPU correctness coverage, but no single " +
            "gate-off CUDA route exercises identical baked shapes and multi-output semantics for every " +
            "specialization. Keep it deferred and unpromoted until the full three-way matrix exists."),

        new PtxParitySpec("PtxFusedResidualBiasLayerNormGeluD64Kernel", PtxParityStatus.Deferred,
            "fused residual + bias + LayerNorm + GELU (D=64, #838)",
            "reachable only through the internal TryDirectPtx* entry point; like the RMSNorm sibling it " +
            "has no public op route on main, so a gate-off leg has nothing to compare against. Wire a " +
            "public route plus a call-time experiment override first (mirroring softmax/reduction), " +
            "then the four fused stages need a single fused fp64 oracle rather than a stage-by-stage " +
            "reference, which rounds differently."),

        new PtxParitySpec("PtxFusedLinearGeluM1Kernel", PtxParityStatus.Deferred,
            "fused decode linear + GELU, fp32 M=1 (#836) — CudaBackend.FusedLinearGELUTransposedM1",
            "has a public route, but its tests compare the PTX result against a CPU reference only, so " +
            "the gate-off CUDA==CPU leg is unproven. The op fuses matmul + bias + GELU, so a three-way " +
            "spec must compare against the same fused CPU oracle on both legs rather than the " +
            "three-kernel fallback sequence, which rounds differently."),

        new PtxParitySpec("PtxFusedLinearGeluFp16M1Kernel", PtxParityStatus.Deferred,
            "fused decode linear + GELU, fp16 weights M=1 (#837)",
            "fp16 weight operand; its harness needs System.Half, which does not exist on net471, so the " +
            "spec is net-core-only. Same missing gate-off leg as the fp32 variant, plus an fp16 " +
            "accumulation oracle question that needs a dedicated tolerance."),

        new PtxParitySpec("PtxFusedLinearGeluFp16M16Kernel", PtxParityStatus.Deferred,
            "fused decode linear + GELU, fp16 weights M=16 (#837)",
            "the M=16 tile of the fp16 decode-linear family; same fp16 oracle and missing gate-off leg " +
            "as the M=1 variant, and its larger tile also needs an occupancy assertion before promotion."),

        new PtxParitySpec("PtxFusedLinearGeluW8A8M1Kernel", PtxParityStatus.Deferred,
            "fused decode linear + GELU, W8A8 M=1 (#837)",
            "int8 weights and activations with per-tensor activation scale and per-column weight scales. " +
            "A three-way spec needs a quantization-aware oracle (dequantize in fp64, then fuse) rather " +
            "than a direct float comparison, so it is deferred until that oracle exists."),

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

        new PtxParitySpec("PtxFusedConv2DNchwK1Kernel", PtxParityStatus.Deferred,
            "fused Conv2D + bias + ReLU, exact N1/C64/H16/W16/K64 1x1 fp32 (#841) — " +
            "CudaBackend.TryDirectPtxFusedConv2DBiasRelu",
            "the first #841 convolution specialization is reachable only through the internal " +
            "TryDirectPtxFusedConv2DBiasRelu experiment gate (disabled by default), so a toggle-based " +
            "gate-off CUDA==CPU leg has nothing to compare against for the identical baked contract yet. " +
            "It also stays unpromoted until the #841 resource-and-performance proof is attached: a " +
            "three-independent-run apples-to-apples benchmark clearing the >=1.10x median / P95<=+10% gate " +
            "against the strongest eligible cuDNN/PyTorch competitor, plus Nsight zero-spill SASS evidence. " +
            "Keep deferred until that three-way matrix and the competitive gates exist."),

        new PtxParitySpec("PtxConv2DNchwK1TiledKernel", PtxParityStatus.Deferred,
            "shared-memory tiled 1x1 Conv2D+bias+ReLU GEMM, realistic ResNet shapes (#841)",
            "the tiled-GEMM specialization staged ahead of the #841 GPU measurement window: it kills the " +
            "~100x redundant global traffic of the naive golden slice by reusing shared weight/input tiles, " +
            "targeting the realistic ResNet-class 1x1 projections where cuDNN is strongest. Its device " +
            "correctness (<= 2e-4 vs the fp64 oracle), register/occupancy budget, tile-size sweep, and the " +
            ">=1.10x-vs-cuDNN performance gate are all pending GPU verification; the emitter is drafted, not " +
            "yet measured. Keep deferred and unpromoted until the three-way matrix and competitive gates pass."),
    };

    private static readonly Dictionary<string, PtxParitySpec> ByKernel =
        Specs.ToDictionary(s => s.KernelTypeName, StringComparer.Ordinal);

    public static bool TryGet(string kernelTypeName, out PtxParitySpec spec) =>
        ByKernel.TryGetValue(kernelTypeName, out spec!);

    public static IEnumerable<PtxParitySpec> ThreeWay =>
        Specs.Where(s => s.Status == PtxParityStatus.ThreeWayParity);
}
#endif
