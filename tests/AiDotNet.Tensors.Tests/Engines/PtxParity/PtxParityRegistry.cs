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
        new PtxParitySpec("PtxFusedPhiloxDropoutF32Kernel", PtxParityStatus.Deferred,
            "fused Philox inverted-dropout forward + saved mask (#849)",
            "the public route, CPU Philox oracle, established CUDA peer, and exact-shape harness are wired, " +
            "but three-way device parity is intentionally deferred until exclusive access to the admitted SM86 GPU; " +
            "non-GPU tests cover the emitter, published Philox vector, admission, and fallback contracts."),
        new PtxParitySpec("PtxPhiloxFillF32Kernel", PtxParityStatus.Deferred,
            "Philox uniform, normal, Bernoulli, and stateless drop-threshold fills (#849)",
            "the public CUDA routes and exact-shape emitters are wired behind the same fail-closed SM86 gate; " +
            "three-way device parity is deferred until exclusive access to the admitted GPU. Non-GPU tests " +
            "cover the versioned Philox rounds, exact ABI, range transforms, Box-Muller structure, opposite " +
            "mask semantics, and architecture rejection."),
        new PtxParitySpec("PtxDropoutBackwardF32Kernel", PtxParityStatus.Deferred,
            "saved-mask dropout backward (#849)",
            "the public CUDA route is wired behind the fail-closed SM86 gate; device parity is deferred " +
            "until the admitted GPU is available. Static tests prove the exact pointer-only ABI, float4 " +
            "dataflow, lack of stride/tail branches, and unmeasured-architecture rejection."),
        new PtxParitySpec("PtxFusedGumbelSoftmax32F32Kernel", PtxParityStatus.Deferred,
            "Gumbel-softmax over an exact contiguous 32-class last axis (#849)",
            "the public DirectGpuTensorEngine route now reaches the fused backend kernel and fails closed for " +
            "unadmitted shapes/SMs. Device parity and distribution checks await the admitted GPU; static tests " +
            "prove the fixed warp reduction, versioned Philox rounds, no global intermediates, and exact ABI."),
        new PtxParitySpec("PtxFusedImportanceSampling64F32Kernel", PtxParityStatus.Deferred,
            "NeRF importance sampling for exact 64-coarse/64-fine layouts (#849)",
            "the public IEngine route already reaches the CUDA capability and now dispatches direct PTX for " +
            "admitted shapes. Device distribution/oracle parity awaits the SM86 GPU; static tests prove one-time " +
            "coarse loads, shared layout, fully unrolled predicated CDF traversal, no tail branch, and exact fallback."),
        new PtxParitySpec("PtxFusedBiasPhiloxDropout256F32Kernel", PtxParityStatus.Deferred,
            "bias-add plus Philox inverted dropout for an exact 256-column layout (#849)",
            "the public FusedBiasDropout path now invokes the optional fused-random capability before allocating " +
            "the established temporary. Device parity awaits SM86 access; static tests prove the float4 input/bias " +
            "transactions, fused mask/output stores, repeated-bias address mapping, and pointer-only exact ABI."),
        new PtxParitySpec("PtxFusedDdimStepF32Kernel", PtxParityStatus.Deferred,
            "currently advertised deterministic fused DDIM update (#849)",
            "the public fused-advanced CUDA route now attempts exact-shape direct PTX first. Device parity " +
            "awaits SM86 access; static tests prove host-collapsed schedule coefficients, two float4 reads, " +
            "one output write, no intermediate allocation, and no stride/tail branch."),
        new PtxParitySpec("PtxPhiloxCategorical32F32Kernel", PtxParityStatus.Deferred,
            "one-hot categorical tensor sampling over an exact 32-class last axis (#849)",
            "the new public CPU oracle and DirectGpuTensorEngine route are wired; admitted CUDA shapes use " +
            "a one-warp direct-PTX CDF scan. Device distribution parity awaits SM86 access; static tests " +
            "prove the prefix scan, one Philox draw per row, no global CDF/index, and exact ABI."),
        new PtxParitySpec("PtxGumbelSoftmaxBackward32F32Kernel", PtxParityStatus.Deferred,
            "Gumbel-softmax backward over an exact 32-class last axis (#849)",
            "the public backward route dispatches this direct specialization before the composed fallback. " +
            "Device parity awaits SM86 access; static tests prove the one-warp Jacobian reduction, inverse-" +
            "temperature epilogue, no global reduction temporary, and exact pointer-only ABI."),
        new PtxParitySpec("PtxFusedPhiloxRreluF32Kernel", PtxParityStatus.Deferred,
            "Philox slope generation fused into training RReLU with a public saved-noise output (#849)",
            "the public TensorRReLU route attempts this specialization before the two-launch fallback. " +
            "Device parity awaits SM86 access; static tests prove float4 Philox generation, one input read, " +
            "only required saved-noise/output writes, exact pointer-only ABI, and no tail/layout path."),
        new PtxParitySpec("PtxRreluF32Kernel", PtxParityStatus.Deferred,
            "saved-noise RReLU forward and backward CUDA-kernel ports (#849)",
            "the CudaBackend forward/backward methods dispatch exact direct PTX before NVRTC fallback. " +
            "Device parity awaits SM86 access; static tests prove float4 dataflow, fixed extents, no global " +
            "intermediate, and unmeasured-architecture rejection."),
    };

    private static readonly Dictionary<string, PtxParitySpec> ByKernel =
        Specs.ToDictionary(s => s.KernelTypeName, StringComparer.Ordinal);

    public static bool TryGet(string kernelTypeName, out PtxParitySpec spec) =>
        ByKernel.TryGetValue(kernelTypeName, out spec!);

    public static IEnumerable<PtxParitySpec> ThreeWay =>
        Specs.Where(s => s.Status == PtxParityStatus.ThreeWayParity);
}
#endif
