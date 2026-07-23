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

        new PtxParitySpec("PtxWinogradF23FusedRegBlockedKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) register-blocked fused GEMM + output transform (#841 3x3, option A)",
            "option-A attempt at the 3x3 win: fuses the batched GEMM + output transform with a TM x TN " +
            "micro-tile (no M workspace, shared-load reuse). Verified correct on-device (<= 2e-3). But the " +
            "16 Winograd positions force 16*TM*TN accumulators -> 128 registers at TM=TN=2 -> ~33% occupancy, " +
            "which makes it SLOWER than the batched pipeline. This is the fundamental FP32 Winograd wall " +
            "(low occupancy vs M-buffer traffic); FP32 hand-PTX tops out at ~0.73x cuDNN. Deferred; the " +
            "remaining win path is FP16 Tensor Cores (option C)."),

        new PtxParitySpec("PtxWinogradF23FilterTransformFp16Kernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) fp16 filter transform U = G g G^T -> U[16,K,C] (#841 3x3 option C)",
            "fp16 A-operand producer for the Tensor-Core Winograd GEMM: fp32 transform math, casts the 16 " +
            "positions to fp16 in position-major U[16,K,C] (the WMMA A layout). Verified correct on-device as " +
            "part of the fp16-TC pipeline (<= 5e-2, cuDNN fp16 regime). Deferred (transform stage; measured via " +
            "the WMMA GEMM cell)."),

        new PtxParitySpec("PtxWinogradF23InputTransformFp16Kernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) fp16 input transform V = B^T d B -> V[16,P,C] (#841 3x3 option C)",
            "fp16 B-operand producer for the Tensor-Core Winograd GEMM: fp32 transform math, casts the 16 " +
            "positions to fp16 in tile-major V[16,P,C] so the col-major WMMA load yields V^T and the mma " +
            "computes U*V. Verified correct on-device (<= 5e-2). Deferred (transform stage)."),

        new PtxParitySpec("PtxWinogradWmmaBatchedGemmKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) fp16 Tensor-Core batched GEMM M[xi]=U[xi].V[xi] (#841 3x3 option C)",
            "option C: the 16 position GEMMs on Ampere Tensor Cores via wmma.mma.sync m16n16k16 f16->f32, " +
            "batched over grid.z=16, reusing the proven Q*K^T WMMA loop (A=U row, B=V col-load). Verified " +
            "correct on-device (<= 5e-2). HONEST perf: at the ResNet C64 shape it measured ~6076us amortized -- " +
            "SLOWER than the fp32 batched pipeline (~1670us) and far off cuDNN (~494us). Root cause: contraction " +
            "C=64 is too small to amortize the TC shared-staging/sync (Tensor Cores idle at ~0.6 TFLOP/s) and the " +
            "batched structure still pays the ~98MB M round-trip cuDNN avoids by fusing. Naively swapping the " +
            "GEMM to Tensor Cores does NOT win; only a fully-fused TC-Winograd (M in fragments + epilogue " +
            "transform) could. Deferred."),

        new PtxParitySpec("PtxWinogradWmmaFusedKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) fully-fused FP16 Tensor-Core conv, no M round-trip (#841 3x3 option C, fused)",
            "the cuDNN-style fused attempt: one warp runs all 16 position GEMMs via raw mma.sync m16n8k16 " +
            "f16->f32 and applies the A^T M A output transform + bias + ReLU thread-locally in the epilogue " +
            "(the defined m16n8k16 D-fragment layout puts each thread's accumulators at known (k,tile) coords), " +
            "so M[16,K,P] never touches global -- killing the ~98MB round-trip of the batched pipelines. " +
            "Verified correct on-device (<= 5e-2). HONEST perf: ~5303us amortized at ResNet C64 -- still ~10x " +
            "off cuDNN (~494us) and worse than fp32 batched (~1977us). Root cause is now the OPERAND LOADS: " +
            "direct-from-global mma fragment loads are ~12.5% coalesced (each warp gathers 16B from 8 separate " +
            "128B lines) and, at 1 warp/block, the 384 dependent 32-bit loads are latency-bound. Removing the " +
            "round-trip was necessary but not sufficient. Reaching cuDNN throughput needs the full cuDNN-class " +
            "memory pipeline (coalesced cp.async staging + ldmatrix + multi-warp cooperation) -- a multi-day " +
            "kernel. Deferred."),

        new PtxParitySpec("PtxWinogradWmmaFusedStagedKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) cuDNN-class fused FP16 TC conv: coalesced cp.async staging + 4-warp (#841 3x3)",
            "the cuDNN-class escalation of the fused kernel: a 4-warp block stages U[16,16,16]+V[16,32,16] " +
            "into shared with coalesced 16-byte cp.async (fixing the ~12.5%-coalesced direct-global fragment " +
            "loads), reads mma fragments via ld.shared, runs 16 mma.sync m16n8k16 per k-step, and keeps the " +
            "A^T M A output transform thread-local (no M round-trip; U tile reused by all 4 warps). Verified " +
            "correct on-device (<= 5e-2). HONEST perf (idle 3080 @ 2040MHz, ResNet C64, amortized): ~2597us -- " +
            "only ~3% better than the register-only fused (~2679us) and still ~3x SLOWER than the plain fp32 " +
            "batched GEMM (~866us) and ~5.3x off cuDNN (~494us). ROOT (now definitive across 10 correct 3x3 " +
            "kernels): the 16 Winograd positions force 64-128 accumulator registers/thread -> low occupancy -> " +
            "the Tensor Cores are starved (~1% util) no matter the memory strategy. This is the SAME occupancy " +
            "wall as the fp32 fused-RB kernel; TC throughput cannot be used because occupancy, not the GEMM, is " +
            "the bottleneck. Coalescing/staging/cp.async do not escape it. Escaping it needs cuDNN's proprietary " +
            "warp-specialized deep-pipeline scheduling. Best simple approach remains fp32 batched (866us, 1.75x " +
            "off cuDNN). Deferred."),

        new PtxParitySpec("PtxWinogradBatchedGemmKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) batched register-blocked GEMM M[b]=U[b].V[b] (#841 3x3 pipeline)",
            "the 16 Winograd position GEMMs run as one batched register-blocked GEMM (grid.z=16), reusing " +
            "the exact TM x TN micro-tile structure that beats cuDNN on 1x1 -- one clean position per block, " +
            "avoiding the 16-accumulator register explosion of the fused kernel. Verified correct on-device " +
            "in the 4-stage pipeline; ~2.3x faster than the naive/fused Winograd and plausibly beats cuDNN on " +
            "an idle GPU (definitive measurement pending an uncontended window). Deferred until the >=1.10x " +
            "gate is confirmed."),

        new PtxParitySpec("PtxWinogradF23OutputTransformKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) output transform A^T M A + bias + ReLU (#841 3x3 pipeline)",
            "the output-transform stage reading M[16,K,P] and scattering the 2x2 tiles to output[N,K,H,W]; " +
            "covered on-device by the batched-GEMM pipeline correctness test. Deferred with the pipeline."),

        new PtxParitySpec("PtxWinogradF23InputTransformKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) input transform V = B^T d B (#841 3x3 pipeline)",
            "the input-transform stage of the optimized Winograd 3x3 pipeline (input[N,C,H,W] -> " +
            "position-major V[16,C,P], same-padded); covered on-device by the fused-pipeline correctness " +
            "test. Deferred until the pipeline clears the >=1.10x gate."),

        new PtxParitySpec("PtxWinogradF23FusedGemmKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) fused batched-GEMM + output transform (#841 3x3 pipeline)",
            "consumes U[16,K,C] and V[16,C,P], register-accumulates all 16 Winograd positions per output " +
            "over shared-staged tiles, and fuses A^T M A + bias + ReLU to output (no M workspace). The full " +
            "3-stage pipeline is verified correct on-device (<= 2e-3 vs the fp64 direct-conv oracle) with 0 " +
            "spills, but the TM=TN=1 layout is shared-bandwidth-bound and does not yet beat cuDNN; register- " +
            "blocking is register-heavy due to the 16-position factor. Deferred until it clears the >=1.10x gate."),

        new PtxParitySpec("PtxWinogradF23FilterTransformKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) filter transform U = G g G^T (#841 3x3 pipeline)",
            "the one-time filter-transform stage of the optimized Winograd 3x3 pipeline (weights[K,C,3,3] -> " +
            "U[K,C,4,4]); the main kernel reads U instead of recomputing it per output tile. Covered on-device " +
            "transitively by the pretransformed-Winograd correctness test. Deferred until the full pipeline " +
            "clears the >=1.10x gate."),

        new PtxParitySpec("PtxConv2DNchw3x3WinogradF23Kernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) 3x3 stride-1 same-conv+bias+ReLU, ResNet shapes (#841)",
            "the 3x3 forward cell computed via Winograd F(2,3) (2x2 output tile, 4x4 input tile; input " +
            "B^T d B, filter G g G^T, elementwise, output A^T M A). The math is verified correct on-device " +
            "(<= 2e-3 vs the fp64 direct-conv oracle) with zero SASS spills, but the correctness-first " +
            "one-thread-per-tile layout (redundant per-tile filter transforms, no data reuse) is ~4.5x " +
            "slower than cuDNN. Keep deferred/unpromoted until the optimized layout (precomputed filter " +
            "transform + register-blocked batched 16-GEMM + input-transform reuse) clears the >=1.10x gate."),

        new PtxParitySpec("PtxConv2DNchwK1RegBlockedKernel", PtxParityStatus.Deferred,
            "register-blocked shared-memory 1x1 Conv2D+bias+ReLU GEMM, ResNet shapes (#841)",
            "the register-blocked (TM x TN micro-tile) tiled-GEMM specialization: each thread computes a " +
            "TM x TN output block in registers so every staged value is reused before leaving registers, " +
            "raising arithmetic intensity to approach/beat cuDNN on the realistic ResNet 1x1 projections. " +
            "Device correctness (<= 2e-4 vs the fp64 oracle), register/occupancy budget, block/micro-tile " +
            "sweep, and the >=1.10x-vs-cuDNN gate are validated on-device. Keep deferred and unpromoted " +
            "until the three-way matrix and competitive gates pass."),

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
