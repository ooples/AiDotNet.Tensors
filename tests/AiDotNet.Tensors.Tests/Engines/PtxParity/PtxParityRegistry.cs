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

        new PtxParitySpec("PtxFusedComplexUnaryF32Kernel", PtxParityStatus.Deferred,
            "complex conjugate and magnitude, fp32 (#850) - CudaBackend.ComplexConjugate, CudaBackend.ComplexMagnitude",
            "one module per operator. Conjugate is a sign-bit flip, so its spec can be bit-exact and " +
            "must include NaN payloads and both signed zeros. Magnitude can be bit-exact too, but only " +
            "because the emitter deliberately leaves the multiply-add UNFUSED: an fma would be more " +
            "accurate than sqrtf(re*re + im*im) and would therefore disagree with the reference, so " +
            "the spec must assert equality rather than a tolerance to keep that property honest."),

        new PtxParitySpec("PtxFusedComplexMultiplyF32Kernel", PtxParityStatus.Deferred,
            "interleaved-complex multiply, fp32 (#850)",
            "structurally ready for a three-way spec — it is the first direct-PTX kernel with both a " +
            "public op route (CudaBackend.ComplexMultiply) and a call-time experiment override — but " +
            "the kernel is admitted only on exact SM86, and its issue (#850) explicitly holds GPU " +
            "correctness back to the admitted release machine. Converts to ThreeWayParity when the " +
            "fp64-oracle run over the four exact pair counts lands; until then the route stays disabled " +
            "and every shape unpromoted."),

        new PtxParitySpec("PtxSplitComplexUnaryF32Kernel", PtxParityStatus.Deferred,
            "split-buffer complex magnitude and magnitude-squared, fp32 (#850) - CudaBackend.SplitComplexMagnitude, CudaBackend.SplitComplexMagnitudeSquared",
            "one module per operator over the four exact element counts. Both can be bit-exact because " +
            "the emitter leaves the multiply-add UNFUSED to match sqrtf(re*re + im*im) / the reference " +
            "power sum; an fma would be more accurate and would disagree, so the spec must assert " +
            "equality rather than a tolerance. Converts to ThreeWayParity when the SM86 fp64-oracle run " +
            "lands; until then the shapes stay unpromoted and fail closed."),

        new PtxParitySpec("PtxSplitComplexBinaryF32Kernel", PtxParityStatus.Deferred,
            "split-buffer complex multiply, add, and cross-spectral, fp32 (#850) - CudaBackend.SplitComplexMultiply, CudaBackend.SplitComplexAdd, CudaBackend.SplitComplexCrossSpectral",
            "one module per operator over the four exact element counts. Multiply forms ar*br-ai*bi and " +
            "ar*bi+ai*br, and cross-spectral (a*conj(b)) forms xr*yr+xi*yi and xi*yr-xr*yi, both with the " +
            "same multiply-then-fma contraction the interleaved multiply kernel uses (the reference's " +
            "default fused evaluation); add is two add.rn lanes. Converts to ThreeWayParity when the " +
            "SM86 fp64-oracle run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxSplitComplexConjugateF32Kernel", PtxParityStatus.Deferred,
            "split-buffer complex conjugate, fp32 (#850) - CudaBackend.SplitComplexConjugate",
            "one module over the four exact element counts. The real lane is copied and the imaginary " +
            "lane is a neg.f32 sign-bit flip, so the spec is bit-exact and must include NaN payloads and " +
            "both signed zeros. Converts to ThreeWayParity when the SM86 run lands; until then unpromoted."),

        new PtxParitySpec("PtxComplexInterleaveF32Kernel", PtxParityStatus.Deferred,
            "complex interleave and deinterleave layout bridges, fp32 (#850) - CudaBackend.InterleaveComplex, CudaBackend.DeinterleaveComplex",
            "one module per direction over the four exact element counts. Both directions are pure data " +
            "movement (a v2 transaction on the interleaved side, two scalar transactions on the split " +
            "side), so the spec is bit-exact including NaN payloads and signed zeros. Converts to " +
            "ThreeWayParity when the SM86 run lands; until then the shapes stay unpromoted and fail closed."),

        new PtxParitySpec("PtxSplitComplexScaleF32Kernel", PtxParityStatus.Deferred,
            "split-buffer complex real-scalar scale, fp32 (#850) - CudaBackend.SplitComplexScale",
            "one module over the four exact element counts; the scalar is a per-launch .param .f32. Each " +
            "lane is a single mul.rn, so the spec is bit-exact with the reference x*scalar. Converts to " +
            "ThreeWayParity when the SM86 run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxSplitComplexPhaseF32Kernel", PtxParityStatus.Deferred,
            "split-buffer complex phase, fp32 (#850) - CudaBackend.SplitComplexPhase",
            "one module over the four exact element counts. PTX has no atan2 primitive, so the angle is a " +
            "minimax atan (~1e-4) plus quadrant folding; unlike the other split operators its spec is " +
            "TOLERANCE-based, not bit-exact. Converts to ThreeWayParity (with tolerance) when the SM86 " +
            "run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxSplitComplexFromPolarF32Kernel", PtxParityStatus.Deferred,
            "split-buffer polar-to-Cartesian, fp32 (#850) - CudaBackend.SplitComplexFromPolar",
            "one module over the four exact element counts using cos.approx/sin.approx, so its spec is " +
            "TOLERANCE-based, not bit-exact. Converts to ThreeWayParity (with tolerance) when the SM86 " +
            "run lands; until then unpromoted and fail-closed."),

        new PtxParitySpec("PtxApplyMelFilterbankF32Kernel", PtxParityStatus.Deferred,
            "mel filterbank application, fp32 (#850) - CudaBackend.ApplyMelFilterbank",
            "thread-per-(frame,mel) fma reduction over the frequency axis, matching the reference's " +
            "fused sum. The spec is a fp64-oracle comparison over exact (frames,freqs,mels) shapes on " +
            "SM86; converts to ThreeWayParity when that run lands. Until then unpromoted and fail-closed."),

        new PtxParitySpec("PtxBitReversePermutationF32Kernel", PtxParityStatus.Deferred,
            "FFT bit-reversal permutation, fp32 (#850) - CudaBackend.FFT (stage 1 of radix-2 DIT)",
            "one module per power-of-two length; the log2(n)-bit-reversed index is a single brev.b32 " +
            "shifted right by 32-log2(n), and the lower thread of each pair performs the in-place swap. " +
            "It is pure data movement, so the spec is bit-exact including NaN payloads and signed zeros. " +
            "Converts to ThreeWayParity when the SM86 fp64-oracle FFT run lands; until then unpromoted."),
        new PtxParitySpec("PtxFftButterflyF32Kernel", PtxParityStatus.Deferred,
            "FFT radix-2 butterfly stage, fp32 (#850) - CudaBackend.FFT (stage 2 of radix-2 DIT)",
            "one module per (length, stage stride); each thread owns one butterfly wing and applies a " +
            "cos.approx/sin.approx twiddle, so - unlike the pure-movement bit-reverse stage - its spec is " +
            "TOLERANCE-based, not bit-exact. The full transform launches this stage log2(n) times with " +
            "doubling strides after the bit-reverse pass. Converts to ThreeWayParity (with tolerance) when " +
            "the SM86 fp64-oracle FFT run lands; until then unpromoted and fail-closed."),

        new PtxParitySpec("PtxRfftPostprocessF32Kernel", PtxParityStatus.Deferred,
            "RFFT positive-frequency extraction, fp32 (#850) - CudaBackend.RFFT",
            "one module per power-of-two length; copies the first n/2+1 bins of the full complex spectrum " +
            "into the packed real-FFT output. It is pure data movement, so the spec is bit-exact including " +
            "NaN payloads and signed zeros; a single guard drops lanes past the (non-power-of-two) output " +
            "length. Converts to ThreeWayParity when the SM86 fp64-oracle RFFT run lands; until then unpromoted."),
        new PtxParitySpec("PtxIrfftPreprocessF32Kernel", PtxParityStatus.Deferred,
            "IRFFT Hermitian reconstruction, fp32 (#850) - CudaBackend.IRFFT (stage 1)",
            "one module per power-of-two length; the lower half is copied and the upper half is filled by " +
            "conjugate symmetry (fullReal[i]=inReal[n-i], fullImag[i]=-inImag[n-i]) via a neg.f32 sign-bit " +
            "flip, so the spec is bit-exact including NaN payloads and signed zeros. Converts to " +
            "ThreeWayParity when the SM86 fp64-oracle IRFFT run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxScaleInverseF32Kernel", PtxParityStatus.Deferred,
            "inverse-FFT normalization, fp32 (#850) - CudaBackend.IRFFT (final scale)",
            "one module per element count; multiplies both split lanes in place by a per-launch .param .f32 " +
            "reciprocal transform length (1/n). Each lane is a single mul.rn, so the spec is bit-exact " +
            "against x*scale. Also normalizes batched inverse transforms (batch*n elements scaled by 1/n). " +
            "Converts to ThreeWayParity when the SM86 run lands; until then unpromoted and fail-closed."),

        new PtxParitySpec("PtxBatchedBitReverseF32Kernel", PtxParityStatus.Deferred,
            "batched FFT bit-reversal permutation, fp32 (#850) - CudaBackend.BatchedFFT (stage 1)",
            "one module per (length, batch); the batch index is gridDim.y and offsets into each row at b*n, " +
            "then the same brev.b32 guarded swap the single-transform kernel uses. It is pure data movement, " +
            "so the spec is bit-exact including NaN payloads and signed zeros. Converts to ThreeWayParity " +
            "when the SM86 fp64-oracle batched-FFT run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxBatchedFftButterflyF32Kernel", PtxParityStatus.Deferred,
            "batched FFT radix-2 butterfly stage, fp32 (#850) - CudaBackend.BatchedFFT (stage 2)",
            "one module per (length, batch, stage stride); the batch index is gridDim.y (baseOffset=b*n) and " +
            "each thread applies a cos.approx/sin.approx twiddle to one wing, so its spec is TOLERANCE-based, " +
            "not bit-exact. A batched transform launches this stage log2(n) times with doubling strides after " +
            "the batched bit-reverse pass. Converts to ThreeWayParity (with tolerance) when the SM86 run lands."),
    };

    private static readonly Dictionary<string, PtxParitySpec> ByKernel =
        Specs.ToDictionary(s => s.KernelTypeName, StringComparer.Ordinal);

    public static bool TryGet(string kernelTypeName, out PtxParitySpec spec) =>
        ByKernel.TryGetValue(kernelTypeName, out spec!);

    public static IEnumerable<PtxParitySpec> ThreeWay =>
        Specs.Where(s => s.Status == PtxParityStatus.ThreeWayParity);
}
#endif
