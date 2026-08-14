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
        new PtxParitySpec("PtxFusedIndexSelectF32Kernel", PtxParityStatus.Deferred,
            "index select, fp32 (#844) - CudaBackend.IndexSelect",
            "a permuted copy, so its three-way spec can be bit-exact with no tolerance. The spec must " +
            "cover one thing the gather spec does not: this op's index buffer holds FLOAT values and " +
            "the reference applies a truncating (int) cast, so the legs have to be driven with " +
            "non-integral index values too (3.7 must select row 3) to prove the conversion is numeric " +
            "and not a bit reinterpretation."),

        new PtxParitySpec("PtxFusedGatherF32Kernel", PtxParityStatus.Deferred,
            "embedding gather, fp32 (#844) — CudaBackend.Gather",
            "has a public route, but its tests cover the emitter and the PTX-vs-CPU result only, so the " +
            "gate-off CUDA==CPU leg is unproven. Converts to ThreeWayParity by mirroring " +
            "BackendRowSum_ThreeWay_CudaAndPtxBothMatchCpuOracle over Gather — gather is exact " +
            "(a pure permuted copy, no accumulation), so the oracle comparison can be bit-exact."),
        new PtxParitySpec("PtxSoftmaxKernel", PtxParityStatus.ThreeWayParity,
            "CudaBackend.Softmax / SoftmaxRows",
            "Backend_Softmax_ThreeWayParityAndAudit runs the incumbent CUDA route and direct PTX on " +
            "identical inputs, proves the selected path with the dispatch counter, and checks both against " +
            "the same double-precision softmax oracle."),
        new PtxParitySpec("PtxLogSoftmaxKernel", PtxParityStatus.ThreeWayParity,
            "CudaBackend.LogSoftmax",
            "Backend_SoftmaxVariants_ThreeWayParity compares incumbent CUDA and direct PTX independently " +
            "against the same double-precision log-sum-exp interpretation."),
        new PtxParitySpec("PtxTaylorSoftmaxKernel", PtxParityStatus.ThreeWayParity,
            "CudaBackend.TaylorSoftmax",
            "Backend_SoftmaxVariants_ThreeWayParity compares both GPU routes against the identical " +
            "1+x+x^2/2 double-precision normalization oracle."),
        new PtxParitySpec("PtxSparsemaxKernel", PtxParityStatus.ThreeWayParity,
            "CudaBackend.Sparsemax",
            "Backend_SoftmaxVariants_ThreeWayParity compares both GPU routes against the sorted closed-form " +
            "simplex projection and verifies non-negativity, exact zeros, and row sums."),
        new PtxParitySpec("PtxSoftmaxBackwardKernel", PtxParityStatus.ThreeWayParity,
            "CudaBackend.SoftmaxBackward",
            "Backend_SoftmaxBackwardReductionAndMasking_ThreeWayParity checks incumbent CUDA and direct PTX " +
            "against the same double-precision Jacobian-vector product."),
        new PtxParitySpec("PtxLogSumExpKernel", PtxParityStatus.ThreeWayParity,
            "CudaBackend.LogSumExpAxis",
            "Backend_SoftmaxBackwardReductionAndMasking_ThreeWayParity checks both GPU routes against a " +
            "stable double-precision log-sum-exp oracle."),
        new PtxParitySpec("PtxLogSumExpBackwardKernel", PtxParityStatus.ThreeWayParity,
            "CudaBackend.LogSumExpBackward",
            "Backend_SoftmaxBackwardReductionAndMasking_ThreeWayParity supplies the same CPU-derived log " +
            "partition to both routes, compares both against the derivative oracle, and separately verifies " +
            "that direct PTX consumes a deliberately changed supplied partition."),
        new PtxParitySpec("PtxMaskedFillKernel", PtxParityStatus.ThreeWayParity,
            "CudaBackend.MaskedFillKernel",
            "Backend_SoftmaxBackwardReductionAndMasking_ThreeWayParity compares incumbent CUDA and direct PTX " +
            "bit-for-bit against the elementwise mask/select contract."),
        new PtxParitySpec("PtxMaskedFillBackwardKernel", PtxParityStatus.ThreeWayParity,
            "CudaBackend.MaskedFillBackward",
            "Backend_SoftmaxBackwardReductionAndMasking_ThreeWayParity compares incumbent CUDA and direct PTX " +
            "bit-for-bit against the masked-gradient contract."),
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

        new PtxParitySpec("PtxFftColsBitReverseF32Kernel", PtxParityStatus.Deferred,
            "column FFT bit-reversal, fp32 (#850) - CudaBackend.FFT2D (column pass, stage 1)",
            "one module per (height,width); permutes each column of a row-major matrix in place along the " +
            "column stride (width) via a brev.b32 guarded swap on the row index. The 2D FFT's row pass reuses " +
            "the contiguous batched kernels; this is the strided column pass. It is pure data movement, so " +
            "the spec is bit-exact including NaN payloads and signed zeros. Converts to ThreeWayParity when " +
            "the SM86 fp64-oracle FFT2D run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxFftColsButterflyF32Kernel", PtxParityStatus.Deferred,
            "column FFT radix-2 butterfly stage, fp32 (#850) - CudaBackend.FFT2D (column pass, stage 2)",
            "one module per (height,width,stage stride); transforms each column along the column stride " +
            "(width) with a cos.approx/sin.approx twiddle, so its spec is TOLERANCE-based, not bit-exact. A 2D " +
            "FFT launches this stage log2(height) times with doubling strides after the column bit-reverse " +
            "pass. Converts to ThreeWayParity (with tolerance) when the SM86 run lands; until then unpromoted."),

        new PtxParitySpec("PtxBatchedFftColsBitReverseF32Kernel", PtxParityStatus.Deferred,
            "batched column FFT bit-reversal, fp32 (#850) - CudaBackend.BatchedFFT2D (column pass, stage 1)",
            "one module per (height,width,images); adds an image dimension (gridDim.y, imgOffset=img*height*width) " +
            "to the strided column bit-reverse so columns never cross image boundaries. The batched 2D FFT's row " +
            "pass is one contiguous batched FFT over all images*height rows; this is the per-image column pass. It " +
            "is pure data movement, so the spec is bit-exact including NaN payloads and signed zeros. Converts to " +
            "ThreeWayParity when the SM86 fp64-oracle batched-FFT2D run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxBatchedFftColsButterflyF32Kernel", PtxParityStatus.Deferred,
            "batched column FFT radix-2 butterfly stage, fp32 (#850) - CudaBackend.BatchedFFT2D (column pass, stage 2)",
            "one module per (height,width,images,stage stride); adds the image offset (gridDim.y) to the strided " +
            "column butterfly. Each thread applies a cos.approx/sin.approx twiddle, so its spec is TOLERANCE-based, " +
            "not bit-exact. A batched 2D FFT launches this stage log2(height) times with doubling strides after the " +
            "batched column bit-reverse pass. Converts to ThreeWayParity (with tolerance) when the SM86 run lands."),

        new PtxParitySpec("PtxOverlapAddF32Kernel", PtxParityStatus.Deferred,
            "ISTFT overlap-add, fp32 (#850) - CudaBackend.IstftFromSpectrum (reconstruction)",
            "one module per (numFrames,nFft,hopLength,outputLength); each thread owns one output sample and " +
            "loops over the frames, accumulating frames[frame*nFft+localIdx]*window[localIdx] with fma.rn for " +
            "every frame whose support covers the sample. The fma matches the reference's fused sum, so its spec " +
            "is TOLERANCE-based against the fp64 oracle. Converts to ThreeWayParity (with tolerance) when the " +
            "SM86 run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxWindowSumSquaresF32Kernel", PtxParityStatus.Deferred,
            "ISTFT window normalization, fp32 (#850) - CudaBackend.IstftFromSpectrum (normalizer)",
            "one module per (nFft,hopLength,outputLength); each thread owns one output sample and loops over the " +
            "derived numFrames=(outputLength-nFft)/hop+1 frames, accumulating window[localIdx]^2 with fma.rn for " +
            "every covering frame. The fma matches the reference's fused sum, so its spec is TOLERANCE-based " +
            "against the fp64 oracle. Converts to ThreeWayParity (with tolerance) when the SM86 run lands."),

        new PtxParitySpec("PtxApplyWindowF32Kernel", PtxParityStatus.Deferred,
            "window application, fp32 (#850) - CudaBackend.ApplyWindow",
            "one module per element count; output[i]=input[i]*window[i] with a single mul.rn, so the spec is " +
            "bit-exact against the reference. Converts to ThreeWayParity when the SM86 run lands; until then " +
            "unpromoted and fail-closed."),
        new PtxParitySpec("PtxPowerToDbF32Kernel", PtxParityStatus.Deferred,
            "power-to-decibel conversion, fp32 (#850) - CudaBackend.PowerToDb",
            "one module per element count; PTX has no log10, so the base-10 log is lg2.approx scaled by " +
            "10*log10(2) with refValue/minDb per-launch .param .f32. Its spec is TOLERANCE-based, not bit-exact. " +
            "Converts to ThreeWayParity (with tolerance) when the SM86 run lands; until then unpromoted."),
        new PtxParitySpec("PtxDbToPowerF32Kernel", PtxParityStatus.Deferred,
            "decibel-to-power conversion, fp32 (#850) - CudaBackend.DbToPower",
            "one module per element count; PTX has no pow, so pow(10,db/10) is ex2.approx of db*log2(10)/10 with " +
            "refValue per-launch .param .f32. Its spec is TOLERANCE-based, not bit-exact. Converts to " +
            "ThreeWayParity (with tolerance) when the SM86 run lands; until then unpromoted and fail-closed."),

        new PtxParitySpec("PtxStftMagPhaseF32Kernel", PtxParityStatus.Deferred,
            "direct windowed STFT magnitude/phase, fp32 (#850) - CudaBackend.StftMagPhase",
            "one module per (batch,Lp,nFft,hop,numFrames,numFreqs); each thread evaluates a windowed length-nFft " +
            "DFT for one output bin with cos.approx/sin.approx twiddles and fma accumulation, then emits " +
            "sqrt(re^2+im^2) and a minimax atan2 phase. Its spec is TOLERANCE-based, not bit-exact. Converts to " +
            "ThreeWayParity (with tolerance) when the SM86 fp64-oracle run lands; until then unpromoted."),
        new PtxParitySpec("PtxBuildSpectrumF32Kernel", PtxParityStatus.Deferred,
            "magnitude/phase to Hermitian spectrum, fp32 (#850) - CudaBackend.BuildSpectrum",
            "one module per (batch,numFreqs,numFrames,nFft); each thread zeroes the nFft bins, fills the first " +
            "numFreqs from cos.approx/sin.approx polar reconstruction, then mirrors by conjugate symmetry. Its " +
            "spec is TOLERANCE-based, not bit-exact. Converts to ThreeWayParity (with tolerance) when the SM86 " +
            "run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxPhaseVocoderF32Kernel", PtxParityStatus.Deferred,
            "phase-vocoder time scaling, fp32 (#850) - CudaBackend.PhaseVocoder",
            "one module per (leading,nFramesV,nFreqV,outFrames); each thread resamples one frequency channel " +
            "along time by a rate .param .f32, lerping the magnitude and accumulating the cvt.rni-wrapped phase " +
            "advance. Its spec is TOLERANCE-based, not bit-exact. Converts to ThreeWayParity (with tolerance) " +
            "when the SM86 run lands; until then unpromoted and fail-closed."),

        new PtxParitySpec("PtxAmplitudeToDbF32Kernel", PtxParityStatus.Deferred,
            "amplitude-to-decibel conversion, fp32 (#850) - CudaBackend.AmplitudeToDb",
            "one module per (length, clipTopDb); the log step of a log-mel/log-magnitude spectrogram. PTX has " +
            "no log10, so db is lg2.approx(max(input,minAmp)) scaled by 20*log10(2), optionally floored at " +
            "topDbFloor. minAmp/topDbFloor are .param .f32 and clipTopDb is baked. Its spec is TOLERANCE-based, " +
            "not bit-exact. Converts to ThreeWayParity (with tolerance) when the SM86 run lands."),
        new PtxParitySpec("PtxMfccLog1pF32Kernel", PtxParityStatus.Deferred,
            "MFCC log1p compression, fp32 (#850) - CudaBackend.MfccLog1p",
            "one module per element count; output=log1p(input)=ln(1+input) via lg2.approx(1+x) scaled by " +
            "ln(2). Its spec is TOLERANCE-based, not bit-exact, and the tolerance is ABSOLUTE: relative error is " +
            "unbounded as x approaches zero. Converts to ThreeWayParity (with tolerance) " +
            "when the SM86 run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxNormalizeRowsFusedF32Kernel", PtxParityStatus.Deferred,
            "fused per-row L2 normalization, fp32 (#850) - CudaBackend.NormalizeRowsFused",
            "one module per (rows,cols); one 256-thread block per row grid-strides the columns to a fma " +
            "sum-of-squares, tree-reduces it in static shared memory under bar.sync barriers, and scales the " +
            "row by rsqrt.approx of the total (zero-norm rows pass through as zero). The rsqrt.approx makes its " +
            "spec TOLERANCE-based. Converts to ThreeWayParity (with tolerance) when the SM86 run lands."),
        new PtxParitySpec("PtxCavityBounceInplaceF32Kernel", PtxParityStatus.Deferred,
            "cavity-bounce post-IFFT nonlinearity, fp32 (#850) - CudaBackend.CavityBounceInplace",
            "one module per element count; the fused post-IFFT step workReal=tanh(clamp(real*invN,-20,20)), " +
            "workImag=0, with invN a per-launch .param .f32. The tanh.approx makes its spec TOLERANCE-based. " +
            "Converts to ThreeWayParity (with tolerance) when the SM86 run lands; until then unpromoted."),
        new PtxParitySpec("PtxWidebandLogBinPoolF32Kernel", PtxParityStatus.Deferred,
            "wideband logarithmic magnitude bin pooling, fp32 (#850) - CudaBackend.WidebandLogBinPool",
            "one module per (totalSegBatch,fftSize,numBins,usable); each thread averages magBuf over a " +
            "quadratic log-spaced bin range and writes log1p(avg). The integer floor bin edges and " +
            "lg2.approx log make its spec TOLERANCE-based. Converts to ThreeWayParity (with tolerance) when " +
            "the SM86 run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxPacPhaseBinMiF32Kernel", PtxParityStatus.Deferred,
            "phase-amplitude coupling modulation index, fp32 (#850) - CudaBackend.PacPhaseBinMi",
            "one module per (batch,numSamples,numGammaBands,gammaIdx); one 18-thread block per batch row bins " +
            "gamma amplitude by theta phase (each thread owns one bin, order-fixed deterministic scan into " +
            "static shared memory), then thread 0 reduces to the Tort modulation index (normalized KL-divergence " +
            "from uniform) using lg2.approx for the natural log. Its spec is TOLERANCE-based against the fp64 " +
            "oracle. Converts to ThreeWayParity (with tolerance) when the SM86 run lands; until then unpromoted."),
        new PtxParitySpec("PtxBispectrumGatherF32Kernel", PtxParityStatus.Deferred,
            "third-order spectral cumulant (bispectrum), fp32 (#850) - CudaBackend.BispectrumGather",
            "one module per (maxF1,maxF2,specLength); each thread forms B(f1,f2)=X(f1)*X(f2)*conj(X(f1+f2)) " +
            "as a triple complex product with the multiply-then-fma contraction (nvcc's default fused form), " +
            "so its spec is TOLERANCE-based against the fp64 oracle. Converts to ThreeWayParity (with tolerance) " +
            "when the SM86 run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxTrispectrumGatherF32Kernel", PtxParityStatus.Deferred,
            "fourth-order spectral cumulant (trispectrum), fp32 (#850) - CudaBackend.TrispectrumGather",
            "one module per (maxF1,maxF2,maxF3,specLength); each thread forms " +
            "T(f1,f2,f3)=X(f1)*X(f2)*X(f3)*conj(X(f1+f2+f3)) as a quadruple complex product with the " +
            "multiply-then-fma contraction, so its spec is TOLERANCE-based against the fp64 oracle. Converts to " +
            "ThreeWayParity (with tolerance) when the SM86 run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxAnalyticSignalMaskF32Kernel", PtxParityStatus.Deferred,
            "analytic-signal (Hilbert) frequency mask, fp32 (#850) - CudaBackend.AnalyticSignalMask",
            "one module per (batch,fftSize,binLow,binHigh); the frequency-domain Hilbert step, multiplying the " +
            "split spectrum by a per-bin gain (0 negative / 2 positive / 1 DC and Nyquist) zeroed outside the " +
            "pass band. The gains are exact powers of two applied with one mul per lane, so the spec is " +
            "bit-exact including NaN payloads and signed zeros. Converts to ThreeWayParity when the SM86 run lands."),
        new PtxParitySpec("PtxIstftNormalizeF32Kernel", PtxParityStatus.Deferred,
            "ISTFT window-sum normalization, fp32 (#850) - CudaBackend.IstftNormalize",
            "one module per element count; the final ISTFT stage dividing result by windowSum in place, guarded " +
            "against tiny denominators with a div.rn matching the reference and a predicated selp. Its spec is " +
            "bit-exact. Converts to ThreeWayParity when the SM86 run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxAudioResampleF32Kernel", PtxParityStatus.Deferred,
            "polyphase Hann-windowed sinc resampling, fp32 (#850) - IEngine.Resample",
            "one module per (leading,inLen,outLen,up,down,halfWidth); each thread resamples one output sample " +
            "by accumulating sinc((idx-srcIdx)*cutoff)*hann(k)*input over the tap window and normalizing by the " +
            "window sum. Each sin/cos is argument-reduced to [-pi,pi] before sin.approx/cos.approx, so its spec " +
            "is TOLERANCE-based, not bit-exact. Converts to ThreeWayParity (with tolerance) when the SM86 run lands."),
        new PtxParitySpec("PtxFftRollF32Kernel", PtxParityStatus.Deferred,
            "batched contiguous fft roll, fp32 (#850) - Fft.FftShift / Fft.IFftShift",
            "one module per (dim,shift,batch); output[b,i]=input[b,(i-shift) mod dim] along the last axis. " +
            "fftshift bakes shift=floor(dim/2), ifftshift bakes ceil(dim/2); the same kernel serves both " +
            "directions and both the real and complex-interleaved axes. It is pure data movement, so the spec " +
            "is bit-exact including NaN payloads and signed zeros. Converts to ThreeWayParity when the SM86 run lands."),
        new PtxParitySpec("PtxFftFreqF32Kernel", PtxParityStatus.Deferred,
            "fft sample-frequency generation, fp32 (#850) - Fft.FftFreq / Fft.RFftFreq",
            "one module per (n, op); Full writes the n signed DFT bins [0..split-1, split-n..-1]*scale and " +
            "Real writes the n/2+1 non-negative bins [0..n/2]*scale, with scale=1/(d*n) a per-launch .param " +
            ".f32. Each output is an integer index cast to fp32 and scaled, while Fft.FftFreq/RFftFreq accumulate " +
            "the bin in double, so the spec is TOLERANCE-based rather than bit-exact. Converts to " +
            "ThreeWayParity (with tolerance) when the SM86 run lands; until then unpromoted."),
        new PtxParitySpec("PtxMelFilterbankApplyF32Kernel", PtxParityStatus.Deferred,
            "segmented mel filterbank application, fp32 (#850) - CudaBackend.MelFilterbankApply",
            "one module per (totalSegBatch,specBins,melBins); each thread owns one (seg,mel) output and reduces " +
            "over the spectral bins with fma.rn, guarded past totalSegBatch*melBins. The fma matches the " +
            "reference's fused sum, so its spec is TOLERANCE-based against the fp64 oracle. Converts to " +
            "ThreeWayParity (with tolerance) when the SM86 run lands; until then unpromoted and fail-closed."),
        new PtxParitySpec("PtxMaskedSoftmaxKernel", PtxParityStatus.Deferred,
            "driver-only fused masked-fill + softmax candidate",
            "DriverOnlyMaskedSoftmax_MatchesComposedOracle checks the direct-PTX result against the stable " +
            "double-precision composed oracle. The candidate has no CudaBackend/public dispatch route, so it " +
            "cannot yet run the required incumbent-vs-direct-PTX-vs-CPU parity test and remains unpromoted. " +
            "Wire a backend route and its incumbent composed path before converting this entry to ThreeWayParity."),
        new PtxParitySpec("PtxMaskedSoftmaxBackwardKernel", PtxParityStatus.Deferred,
            "driver-only fused masked-softmax backward candidate",
            "DriverOnlyMaskedSoftmax_MatchesComposedOracle checks the direct-PTX gradient against the same " +
            "double-precision Jacobian-vector-product oracle with the gradient mask applied. The candidate has " +
            "no CudaBackend/public dispatch route, so it cannot yet run the required three-way parity test and " +
            "remains unpromoted. Wire that route and incumbent composition before promoting it."),

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
        new PtxParitySpec("PtxRegisterCholesky4x4F32Kernel", PtxParityStatus.Deferred,
            "Linalg.CholeskyEx lower FP32 batched 4x4 (#853)",
            "a dedicated opt-in DriverOnly structural/parity matrix is checked in; promotion to three-way " +
            "coverage waits for execution on the admitted SM86 device and attached failure-info evidence."),
        new PtxParitySpec("PtxRegisterSolver4x4F32Kernel", PtxParityStatus.Deferred,
            "dense decomposition/solve forward and backward FP32 batched 4x4 family (#853)",
            "the operation-parameterized DriverOnly matrix and public routing are checked in; three-way " +
            "classification waits for SM86 execution, operation-specific oracle evidence, and the release gate."),
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
        new PtxParitySpec("PtxFusedRgLruScan128x256Kernel", PtxParityStatus.ThreeWayParity,
            "RG-LRU scan forward [1,128,256] (#846) — CudaBackend.RgLruScanForward",
            "DirectPtxRecurrentTests.DriverRgLru_MatchesDoubleOracleAndRoutesDirectPtxRepeatedly " +
            "runs the exact SM86 direct route and incumbent NVRTC kernel against an independent fp64 " +
            "full-sequence recurrence oracle at 2e-5, and proves direct route entry over repeated launches."),

        new PtxParitySpec("PtxFusedPairwiseBoxIouF32Kernel", PtxParityStatus.Deferred,
            "pairwise BoxIoU (#851)",
            "the dedicated SM86 test and benchmark harness compare direct PTX and the established CUDA route " +
            "against an fp64 oracle, but the driver-only matrix cannot run on CPU-only CI; retain the explicit " +
            "deferral until resident-GPU evidence is attached."),
        new PtxParitySpec("PtxVisionKernel", PtxParityStatus.Deferred,
            "vision/detection/ROI/geometry specialization family (#851)",
            "all 120 closed specializations pass static ABI/emitter validation and have driver-only direct-PTX " +
            "versus established-CUDA checks; the operation-specific fp64 resident-GPU oracle matrix remains " +
            "deferred until the required SM86 hardware run."),
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

        new PtxParitySpec("PtxWinogradWmmaPipelinedKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) software-pipelined fully-fused FP16 TC conv — 1.16x off cuDNN (#841 3x3, best)",
            "the all-K fusion + cuDNN-style software pipelining: the contraction C is split into 16-channel " +
            "chunks and the input transform is double-buffered against the GEMM (transform chunk N+1's V into one " +
            "shared buffer while the Tensor Cores accumulate chunk N from the other), so the transform's high-" +
            "latency ~14%-coalesced global loads overlap the mma instead of running as a serial phase. Verified " +
            "correct on-device (<= 5e-2, 2-chunk and 4-chunk). HONEST perf (idle 3080 @ 2040MHz, ResNet C64, " +
            "amortized, same window): 356us vs cuDNN fp16 3x3+ReLU 307us = 1.16x off -- the culmination of the " +
            "3x3 arc 6300us (11x) -> 356us (1.16x). regs=57 (transform + GEMM phases coexist -> ~67% occupancy, " +
            "which caps the pipeline gain at ~6%). Best 3x3 kernel; near cuDNN parity with hand-PTX. K<=64. Deferred."),

        new PtxParitySpec("PtxWinogradWmmaFusedAllKKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) fully-fused FP16 TC conv, V computed once per tile-block (#841 3x3, best fused)",
            "the correct cuDNN-style fusion: grid.y=1 so one 16-warp block owns 8 tiles across ALL K channels and " +
            "computes V = B^T d B exactly once into shared (phase 1); each warp then runs K/16 mma m-subtiles " +
            "reusing the shared B(V) fragment while streaming A(U) (phase 2); M is exchanged through shared -- " +
            "overlapping the now-dead V region -- and the A^T M A + bias + ReLU output transform runs in the " +
            "epilogue (phase 3). No global V, no round-trip, no redundant transform (unlike the grid.y=4 fully-" +
            "fused kernel). Verified correct on-device (<= 5e-2). HONEST perf (idle 3080 @ 2040MHz, ResNet C64, " +
            "amortized): ~527us -- best of all fused variants (2.3x faster than the redundant-V fusion's 1188us), " +
            "edges the staged pipeline (543us), and is 1.67x off cuDNN fp16+ReLU (314us idle, same window; the " +
            "campaign started at 11x / 6300us). Remaining gap is cuDNN's phase pipelining (overlap V-compute with " +
            "GEMM) + interior-tile fast path. K<=64 (M-exchange fits shared). Deferred."),

        new PtxParitySpec("PtxWinogradWmmaFullyFusedKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) fully-fused FP16 TC conv: input transform + GEMM + output transform, 1 kernel (#841 3x3)",
            "cuDNN-structure single kernel: phase 1 computes V = B^T d B straight into shared (no global V, no " +
            "51MB round-trip), phase 2 runs the per-warp position mma reading B from shared V + A from precomputed " +
            "fp16 U, phase 3 exchanges M in shared and applies A^T M A + bias + ReLU. Verified correct on-device " +
            "(<= 5e-2). HONEST perf (idle 3080 @ 2040MHz, ResNet C64): ~1195us -- SLOWER than the separate-kernel " +
            "STAGED pipeline (~554us) and coop (~613us). Root cause found by measurement: grid.y=K/16=4 makes each " +
            "channel-block recompute V for the same 8 tiles -> 4x redundant input-transform work, which outweighs " +
            "eliminating the V round-trip. The correct fusion must process ALL K channels per tile-block (V computed " +
            "once); this kernel proves fusion alone is not the win. Best 3x3 to date = STAGED 554us, 1.8x off cuDNN " +
            "fp16+ReLU (308us idle, same window) -- down from 11x. Deferred."),

        new PtxParitySpec("PtxWinogradWmmaCoopKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) cooperative FP16 TC conv: one position per warp, M via shared (#841 3x3)",
            "attacks the occupancy question head-on: 16 warps, one Winograd position each (4 accumulators/thread " +
            "-> full occupancy, 40 regs, 0 spills), 16 position results exchanged through shared (8KB, no global M " +
            "round-trip), combined A^T M A epilogue. Verified correct on-device (<= 5e-2). KEY DIAGNOSTIC: this " +
            "kernel proved occupancy was NOT the bottleneck -- fully occupied yet as slow as the low-occupancy " +
            "fused kernels. Isolating the pipeline stages then revealed the real wall was the fp16 input transform " +
            "(uncoalesced P-major stores, 5651us) not the GEMM (981us). After the transform coalescing fix the " +
            "full coop pipeline dropped ~4.4x (6300us -> 1437us, contended) and beats the fp32 batched pipeline. " +
            "Deferred pending an uncontended idle-vs-cuDNN measurement."),

        new PtxParitySpec("PtxWinogradWmmaCoopBlockedKernel", PtxParityStatus.Deferred,
            "Winograd F(2,3) cooperative FP16 TC conv with mma-level A-fragment reuse (#841 3x3)",
            "coop kernel + register blocking: each warp computes 16ch x 32tiles reusing one loaded A=U fragment " +
            "across 4 N-subtiles (SASS confirmed loads-per-mma dropped from 6:1 to 3:1). Verified correct " +
            "on-device (<= 5e-2). Second key diagnostic: halving the load ratio did NOT speed the pipeline up, " +
            "which (with the isolation data) pinned the bottleneck on the input transform rather than GEMM memory. " +
            "Retained as the load-amortization lesson for the post-transform-fix GEMM tuning. Deferred."),

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

        new PtxParitySpec("PtxUnfold2DFp16FromFp16Kernel", PtxParityStatus.Deferred,
            "FP16 im2col from FP16 input (UnfoldKNFp16FromFp16) direct-PTX (#841 FP16 family)",
            "columns_fp16[n,c*KH*KW+kh*KW+kw,oh*OW+ow] = input_fp16[n,c,oh*s+kh-pad,ow*s+kw-pad] -- a pure " +
            "half-to-half patch gather preparing the KxN operand for a Tensor-Core GEMM. Thread-per-output with " +
            "consecutive spatial index -> coalesced fp16 reads + stores. Exact half copy on-device. Deferred."),

        new PtxParitySpec("PtxUnfold2DFp16Kernel", PtxParityStatus.Deferred,
            "Fused im2col + FP16 conversion (Im2colKNFp16) direct-PTX (#841 FP16 family)",
            "columns_fp16[n,c*KH*KW+kh*KW+kw,oh*OW+ow] = fp16(input[n,c,oh*s+kh-pad,ow*s+kw-pad]); prepares the " +
            "KxN half operand for a Tensor-Core GEMM. Thread-per-output with consecutive spatial index -> coalesced " +
            "input reads + fp16 column stores at stride 1. Verified on-device (<= 1e-3 vs fp16-rounded CPU reference). " +
            "Deferred."),

        new PtxParitySpec("PtxConv2DDirectFp16Kernel", PtxParityStatus.Deferred,
            "Half-weight direct Conv2D (Conv2dDirectFp16Hw) direct-PTX (#841 FP16 family)",
            "out[n,k,oh,ow] = relu(bias[k] + sum_{c,kh,kw} f16(W[k,c,kh,kw])*f16(in[n,c,oh*s+kh-pad,ow*s+kw-pad])) " +
            "with FP32 accumulation; general kernel/stride/padding. FP16 OIHW weights, FP32 input rounded to FP16 " +
            "before the multiply. Thread-per-output with consecutive ow -> coalesced in/out; fp16 weights broadcast. " +
            "Verified on-device (<= 3e-3 vs fp16-rounded CPU reference). Deferred."),

        new PtxParitySpec("PtxUnfold2DKernel", PtxParityStatus.Deferred,
            "Unfold / im2col patch extraction direct-PTX (#841 unfold family)",
            "columns[n,c*KH*KW+kh*KW+kw,oh*OW+ow] = input[n,c,oh*s+kh-pad,ow*s+kw-pad] (0 outside padded input); " +
            "general kernel/stride/padding. One thread per output element with consecutive output-spatial index " +
            "(ow-fast) -> coalesced input reads + column stores at stride 1. Exact (<= 1e-5 vs CPU reference). " +
            "Deferred."),

        new PtxParitySpec("PtxDeformableConv2DBackwardOffsetKernel", PtxParityStatus.Deferred,
            "Deformable Conv2D backward-offset (DCNv2 offset gradient via bilinear derivative) direct-PTX (#841)",
            "for each (n,pos,oh,ow) writes both dy/dx offset gradients: dOff_y = mask*sum gradOut*W*(wx0*(v10-v00)+" +
            "wx1*(v11-v01)), dOff_x = mask*sum gradOut*W*(wy0*(v01-v00)+wy1*(v11-v10)) over the four zero-padded " +
            "input corners. One thread per offset position, consecutive ow -> coalesced. Verified correct on-device " +
            "(<= 3e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxDeformableConv2DBackwardWeightKernel", PtxParityStatus.Deferred,
            "Deformable Conv2D backward-weight (DCNv2 weight gradient) direct-PTX (#841 deformable family)",
            "dW[k,c,pos] = sum_{n,oh,ow} gradOut[n,k,oh,ow]*mask[n,pos,oh,ow]*bilinear(input[n,c]; py, px). One " +
            "thread per weight element loops over batch+output-spatial (bounds-guarded ceil-div grid since " +
            "K*C*KH*KW is small), reusing the forward 4-corner bilinear. Verified correct on-device (<= 3e-3 vs " +
            "fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxDeformableConv2DGroupedForwardKernel", PtxParityStatus.Deferred,
            "Grouped Deformable Conv2D forward (DCNv2 deform_groups>1) direct-PTX (#841 grouped-deformable family)",
            "input channels partitioned into dg deformable groups, each with its own offset/mask field; for channel c " +
            "the group g=c/(C/dg) selects offset[n,g*2*taps+2*pos(+1),oh,ow] and mask[n,g*taps+pos,oh,ow]. " +
            "out[n,k,oh,ow] = bias[k] + sum W[k,c,pos]*mask_g*bilinear(input[n,c]; py_g, px_g), zero-padded 4-corner. " +
            "One thread per output, consecutive ow -> coalesced. dg=1 reproduces the single-group kernel. Verified " +
            "correct on-device (<= 3e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxFusedConv3DKernel", PtxParityStatus.Deferred,
            "Fused Conv3D inference epilogue (conv + bias + per-channel scale + ReLU) direct-PTX (#841 fused family)",
            "out[n,k,od,oh,ow] = relu(scale[k]*(bias[k] + sum_{c,kd,kh,kw} W*in[...])); the bias, per-output-channel " +
            "scale, and ReLU epilogue fold into the accumulator before the store with no intermediate materialization. " +
            "One thread per output element, consecutive ow -> coalesced NCDHW reads/stores, bounds-guarded grid. " +
            "Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxFusedConvTranspose2DKernel", PtxParityStatus.Deferred,
            "Fused ConvTranspose2D inference epilogue (transposed conv + bias + per-channel scale + ReLU) direct-PTX (#841 fused family)",
            "out[n,co,oh,ow] = relu(scale[co]*(bias[co] + sum input[n,ci,(oh+pad-kh)/s,(ow+pad-kw)/s]*W[ci,co,kh,kw])); " +
            "IOHW weights, transpose-gather with valid-index checks, scale+ReLU epilogue folded before the store. One " +
            "thread per output element, consecutive ow coalesced, bounds-guarded grid. Verified correct on-device " +
            "(<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxDeformableConv2DGroupedBackwardOffsetKernel", PtxParityStatus.Deferred,
            "Grouped Deformable Conv2D backward-offset (deform_groups>1) direct-PTX (#841 grouped-deformable family)",
            "one thread per (n,g,pos,oh,ow) writes dOff_y/dOff_x = mask_g*sum_{c in group g, k} gradOut*W*bilinear-" +
            "derivative into dOff[n,g*2*taps+2*pos(+1),oh,ow]; the channel reduction runs only over the deformable " +
            "group's channels c in [g*C/dg, (g+1)*C/dg). Bilinear derivatives over 4 zero-padded corners, bounds-" +
            "guarded grid. Verified correct on-device (<= 3e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxDeformableConv2DGroupedBackwardMaskKernel", PtxParityStatus.Deferred,
            "Grouped Deformable Conv2D backward-mask (deform_groups>1) direct-PTX (#841 grouped-deformable family)",
            "dMask[n,g,pos,oh,ow] = sum_{c in group g, k} gradOut[n,k,oh,ow]*W[k,c,pos]*bilinear(input[n,c]; py_g, " +
            "px_g); the channel reduction is restricted to the deformable group. One thread per grouped mask element, " +
            "consecutive ow -> coalesced, zero-padded 4-corner bilinear, bounds-guarded grid. Verified correct " +
            "on-device (<= 3e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxDeformableConv2DGroupedBackwardInputKernel", PtxParityStatus.Deferred,
            "Grouped Deformable Conv2D backward-input (deform_groups>1, atomic scatter) direct-PTX (#841 grouped-deformable family)",
            "dInput[n,c,yy,xx] += (sum_k gradOut*W)*mask_g*corner_weight scattered to the four bilinear corners via " +
            "red.global.add.f32, with group g=c/(C/dg) selecting the offset/mask field. gradInput zero-initialized, " +
            "one thread per (n,c,oh,ow), bounds-guarded grid. Verified correct on-device (<= 3e-3 vs fp64 CPU " +
            "reference). Deferred."),

        new PtxParitySpec("PtxDeformableConv2DGroupedBackwardWeightKernel", PtxParityStatus.Deferred,
            "Grouped Deformable Conv2D backward-weight (deform_groups>1) direct-PTX (#841 grouped-deformable family)",
            "dW[k,c,pos] = sum_{n,oh,ow} gradOut[n,k,oh,ow]*mask_g*bilinear(input[n,c]; py_g, px_g), group g=c/(C/dg) " +
            "selecting the offset/mask field. One thread per weight element reduces over batch+output-spatial with a " +
            "bounds-guarded grid. Verified correct on-device (<= 3e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxDeformableConv2DBackwardInputKernel", PtxParityStatus.Deferred,
            "Deformable Conv2D backward-input (DCNv2 input gradient via atomic scatter) direct-PTX (#841 deformable family)",
            "dInput[n,c,yy,xx] += (sum_k gradOut[n,k,oh,ow]*W[k,c,pos])*mask[n,pos,oh,ow]*corner_weight(yy,xx). One " +
            "thread per (n,c,oh,ow) loops the taps and scatters each sample point's contribution to its four bilinear " +
            "input corners via red.global.add.f32 (gradInput zero-initialized), since many (oh,ow,pos) alias onto the " +
            "same input pixel. Bounds-guarded ceil-div grid. Verified correct on-device (<= 3e-3 vs fp64 CPU " +
            "reference). Deferred."),

        new PtxParitySpec("PtxDeformableConv2DBackwardMaskKernel", PtxParityStatus.Deferred,
            "Deformable Conv2D backward-mask (DCNv2 modulation gradient) direct-PTX (#841 deformable family)",
            "dMask[n,pos,oh,ow] = sum_{c,k} gradOut[n,k,oh,ow]*W[k,c,pos]*bilinear(input[n,c]; py, px); reuses the " +
            "forward zero-padded 4-corner bilinear at the learned offset positions. One thread per mask element, " +
            "consecutive ow -> coalesced offset reads. Verified correct on-device (<= 3e-3 vs fp64 CPU reference). " +
            "Deferred."),

        new PtxParitySpec("PtxDeformableConv2DKernel", PtxParityStatus.Deferred,
            "Deformable Conv2D forward (DCNv2, bilinear sampling + offsets + mask) direct-PTX (#841 deformable family)",
            "out[n,k,oh,ow] = bias[k] + sum W[k,c,kh,kw]*mask[n,pos,oh,ow]*bilinear(input[n,c]; oh*s+kh-pad+offY, " +
            "ow*s+kw-pad+offX); zero-padded 4-corner bilinear at the learned offset positions with per-tap " +
            "modulation (DCNv2, single deformable group). Thread-per-output with consecutive ow -> coalesced " +
            "offset/mask reads. Verified correct on-device (<= 3e-3 vs fp64 bilinear CPU reference). Deferred."),

        new PtxParitySpec("PtxConv2DBackwardBiasKernel", PtxParityStatus.Deferred,
            "Conv2D backward-bias direct-PTX coalesced reduction (#841 backward family)",
            "gradBias[k] = sum over batch+spatial of gradOutput[b,k,h,w]. One block per output channel " +
            "reduces the B x H x W slice with consecutive threads reading consecutive spatial elements (the " +
            "contiguous NCHW axis -> coalesced loads, the same thread-to-memory lesson from the 3x3 kernel) " +
            "and a shared tree reduction. Replaces the CPU-download reduction in " +
            "DirectGpuTensorEngine.Conv2DBackwardBiasGpu. Verified correct on-device (<= 2e-3 vs fp64 CPU " +
            "reduction, non-power-of-2 spatial). Deferred (experimental-pending-gpu-evidence)."),

        new PtxParitySpec("PtxConvTranspose2DBackwardInputKernel", PtxParityStatus.Deferred,
            "ConvTranspose2D backward-input direct-PTX (#841 transposed family)",
            "dInput[n,ci,ih,iw] = sum over (co,kh,kw) valid of gradOut[n,co,ih*s-pad+kh,iw*s-pad+kw]*W[ci,co,kh,kw] " +
            "(IOHW) -- a regular-conv-style correlation of gradOut with the transposed weights. Thread-per-output, " +
            "consecutive iw -> coalesced gradOut reads + dInput stores, weights broadcast. Verified correct on-device " +
            "(<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxConvTranspose2DBackwardWeightKernel", PtxParityStatus.Deferred,
            "ConvTranspose2D backward-weight direct-PTX coalesced reduction (#841 transposed family)",
            "dW[ci,co,kh,kw] = sum_{n,ih,iw} input[n,ci,ih,iw]*gradOut[n,co,ih*s-pad+kh,iw*s-pad+kw] (IOHW). One block " +
            "per (ci,co) reduces N x H x W into the KH*KW taps with coalesced input reads (reused across taps) and a " +
            "shared tree reduce per tap (KH*KW<=25). Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxConvTranspose3DBackwardInputKernel", PtxParityStatus.Deferred,
            "ConvTranspose3D backward-input direct-PTX (#841 transposed-3D family)",
            "dInput[n,ci,id,ih,iw] = sum over (co,kd,kh,kw) valid of gradOut[n,co,id*s-pad+kd,ih*s-pad+kh," +
            "iw*s-pad+kw]*W[ci,co,kd,kh,kw] (IODHW) -- 3D correlation of gradOut with the transposed weights. " +
            "Thread-per-output, consecutive iw -> coalesced gradOut reads + dInput stores. Verified correct " +
            "on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxConvTranspose3DBackwardWeightKernel", PtxParityStatus.Deferred,
            "ConvTranspose3D backward-weight direct-PTX coalesced reduction (#841 transposed-3D family)",
            "dW[ci,co,kd,kh,kw] = sum_{n,id,ih,iw} input[n,ci,id,ih,iw]*gradOut[n,co,id*s-pad+kd,ih*s-pad+kh," +
            "iw*s-pad+kw]. One block per (ci,co) reduces N x D x H x W into the KD*KH*KW taps with coalesced input " +
            "reads (reused across taps) + shared tree reduce (<=27 taps). Verified correct on-device (<= 3e-3). Deferred."),

        new PtxParitySpec("PtxConvTranspose3DKernel", PtxParityStatus.Deferred,
            "ConvTranspose3D forward + bias + ReLU direct-PTX (#841 transposed-3D family)",
            "out[n,co,od,oh,ow] = relu(bias[co] + sum over (ci,kd,kh,kw) valid of input[n,ci,(od+pad-kd)/s," +
            "(oh+pad-kh)/s,(ow+pad-kw)/s]*W[ci,co,kd,kh,kw]); IODHW weights, general kernel/stride/padding/output-" +
            "padding. 3D transpose-gather run as a forward op; thread-per-output with consecutive ow -> coalesced " +
            "in/out at stride 1. Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxConvTranspose2DKernel", PtxParityStatus.Deferred,
            "ConvTranspose2D forward + bias + ReLU direct-PTX (#841 transposed family)",
            "out[n,co,oh,ow] = relu(bias[co] + sum over (ci,kh,kw) valid of input[n,ci,(oh+pad-kh)/s,(ow+pad-kw)/s]" +
            " * W[ci,co,kh,kw]); IOHW weights, general kernel/stride/padding/output-padding. The transpose-gather " +
            "pattern (same shape as a regular conv's backward-input) run as a forward op; thread-per-output with " +
            "consecutive ow -> coalesced input reads + output stores at stride 1, weights broadcast across the warp. " +
            "Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxConv1DBackwardInputKernel", PtxParityStatus.Deferred,
            "Native Conv1D backward-input direct-PTX (#841 Conv1D family)",
            "dX[n,c,il] = sum_{k,kl} W[k,c,kl]*gradOut[n,k,(il+pad-kl)/stride] (valid divisible/in-range terms). " +
            "Thread-per-output, consecutive il -> coalesced gradOut reads + dX stores; weights broadcast. General " +
            "kernel length/stride/padding. Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxConv1DBackwardWeightKernel", PtxParityStatus.Deferred,
            "Native Conv1D backward-weight direct-PTX coalesced reduction (#841 Conv1D family)",
            "dW[k,c,kl] = sum_{n,ol} input[n,c,ol*stride+kl-pad]*gradOut[n,k,ol]. One block per (k,c) reduces the " +
            "N x OL contraction into KL taps with coalesced spatial reads (gradOut reused across taps) and a shared " +
            "tree reduce per tap. KL <= 11. Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxConv3DBackwardInputKernel", PtxParityStatus.Deferred,
            "Native Conv3D backward-input direct-PTX (#841 Conv3D family)",
            "dInput[n,c,id,ih,iw] = sum over (k,kd,kh,kw) valid of W[k,c,kd,kh,kw]*gradOut[n,k,(id+pad-kd)/s," +
            "(ih+pad-kh)/s,(iw+pad-kw)/s]. Thread-per-output, consecutive iw -> coalesced gradOut reads + dInput " +
            "stores, weights broadcast. Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxConv3DBackwardWeightKernel", PtxParityStatus.Deferred,
            "Native Conv3D backward-weight direct-PTX coalesced reduction (#841 Conv3D family)",
            "dW[k,c,kd,kh,kw] = sum_{n,od,oh,ow} input[n,c,od*s+kd-pad,oh*s+kh-pad,ow*s+kw-pad]*gradOut[n,k,od,oh,ow]. " +
            "One block per (k,c) reduces N x OD x OH x OW into the KD*KH*KW taps with coalesced gradOut reads (reused " +
            "across taps) and a shared tree reduce per tap (KD*KH*KW<=27). Verified correct on-device (<= 3e-3). Deferred."),

        new PtxParitySpec("PtxLocallyConnected2DBackwardInputKernel", PtxParityStatus.Deferred,
            "LocallyConnected2D backward-input direct-PTX (#841 locally-connected family)",
            "dInput[n,c,ih,iw] = sum over (k,kh,kw) valid of W[oh,ow,k,c,kh,kw]*gradOut[n,k,(ih+pad-kh)/s,(iw+pad-kw)/s] " +
            "(per-position weights). Thread-per-output, consecutive iw -> coalesced gradOut reads + dInput stores. " +
            "Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxLocallyConnected2DBackwardWeightKernel", PtxParityStatus.Deferred,
            "LocallyConnected2D backward-weight direct-PTX (#841 locally-connected family)",
            "each weight is used by one output position, so dW[oh,ow,k,c,kh,kw] = sum_n input[n,c,oh*s+kh-pad," +
            "ow*s+kw-pad]*gradOut[n,k,oh,ow]. One thread per weight loops over N (no block reduction). Verified " +
            "correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxLocallyConnected2DKernel", PtxParityStatus.Deferred,
            "LocallyConnected2D forward (unshared per-position weights) direct-PTX (#841 locally-connected family)",
            "out[n,k,oh,ow] = relu(bias[k,oh,ow] + sum_{c,kh,kw} W[oh,ow,k,c,kh,kw]*in[n,c,oh*s+kh-pad,ow*s+kw-pad]); " +
            "each output position has its own filter (weights [OH,OW,K,C,KH,KW]). Thread-per-output with consecutive " +
            "ow -> coalesced in/out. Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxLocallyConnected2DBackwardBiasKernel", PtxParityStatus.Deferred,
            "LocallyConnected2D backward-bias direct-PTX (#841 locally-connected family)",
            "per-position bias so dBias[k,oh,ow] = sum_n gradOut[n,k,oh,ow]. One thread per output-bias element " +
            "loops over N; consecutive threads (ow) coalesce the gradOut reads. Verified correct on-device (<= 2e-3). " +
            "Deferred."),

        new PtxParitySpec("PtxConv3DKernel", PtxParityStatus.Deferred,
            "Native Conv3D forward + bias + ReLU direct-PTX (#841 Conv3D family)",
            "out[n,k,od,oh,ow] = relu(bias[k] + sum_{c,kd,kh,kw} W[k,c,kd,kh,kw]*in[n,c,od*s+kd-pad,oh*s+kh-pad," +
            "ow*s+kw-pad]); general kernel extent/stride/padding. Thread-per-output with consecutive ow -> " +
            "coalesced innermost input reads and output stores at stride 1 (contiguous NCDHW axis); weights broadcast. " +
            "Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxConv1DKernel", PtxParityStatus.Deferred,
            "Native Conv1D forward + bias + ReLU direct-PTX (#841 Conv1D family)",
            "out[n,k,ol] = relu(bias[k] + sum_{c,kl} W[k,c,kl]*in[n,c,ol*stride+kl-pad]); general kernel " +
            "length/stride/padding. One thread per output element with consecutive ol -> coalesced input reads " +
            "and output stores at stride 1 (contiguous NCL axis). Native 1D instead of routing through Conv2D. " +
            "Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxDepthwiseConv1DForwardKernel", PtxParityStatus.Deferred,
            "Native depthwise Conv1D forward (channel-multiplier 1) direct-PTX (#841 depthwise family)",
            "out[n,c,ol] = sum_kl in[n,c,ol*stride+kl-pad]*W[c,kl]; each channel convolves with its own length-KL " +
            "filter. One thread per output element, consecutive ol -> coalesced NCL reads/stores at stride 1. Native " +
            "1D depthwise instead of reshaping through Conv2D. Verified correct on-device (<= 2e-3 vs fp64 CPU " +
            "reference). Deferred."),

        new PtxParitySpec("PtxDepthwiseConv1DBackwardInputKernel", PtxParityStatus.Deferred,
            "Native depthwise Conv1D backward-input direct-PTX (#841 depthwise family)",
            "dInput[n,c,il] = sum_kl gradOut[n,c,(il+pad-kl)/stride]*W[c,kl] over taps where (il+pad-kl) is a " +
            "non-negative in-range multiple of stride (transpose-gather of the forward correlation). One thread per " +
            "input element, il-contiguous. Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxDepthwiseConv1DBackwardWeightKernel", PtxParityStatus.Deferred,
            "Native depthwise Conv1D backward-weight direct-PTX (#841 depthwise family)",
            "dW[c,kl] = sum_{n,ol} gradOut[n,c,ol]*in[n,c,ol*stride+kl-pad]. One thread per weight element (C*KL is " +
            "small) reduces over batch+output-spatial with a bounds-guarded ceil-div grid. Verified correct on-device " +
            "(<= 2e-3 vs fp64 CPU reference). Deferred."),

        new PtxParitySpec("PtxDepthwiseConv2D3x3Kernel", PtxParityStatus.Deferred,
            "Depthwise Conv2D 3x3 forward + bias + ReLU direct-PTX (#841 depthwise family)",
            "channel-multiplier-1 depthwise forward: out[n,c,oh,ow] = relu(bias[c] + sum_{r,s} " +
            "W[c,r,s]*in[n,c,oh+r-1,ow+s-1]). No channel reduction (memory-bound), so one thread per output " +
            "with consecutive threads owning consecutive ow (contiguous NCHW axis) coalesces the in/out accesses " +
            "and broadcasts the 9 depthwise weights across the warp. Verified correct on-device (<= 2e-3). Deferred."),

        new PtxParitySpec("PtxDepthwiseConv2D3x3BackwardInputKernel", PtxParityStatus.Deferred,
            "Depthwise Conv2D 3x3 backward-input direct-PTX (#841 depthwise family)",
            "dX[n,c,ih,iw] = sum_{r,s} W[c,r,s]*gradOut[n,c,ih-r+1,iw-s+1] (per channel). Thread-per-output, " +
            "consecutive iw -> coalesced gradOut reads + dX stores. Verified correct on-device (<= 2e-3). Deferred."),

        new PtxParitySpec("PtxDepthwiseConv2D3x3BackwardWeightKernel", PtxParityStatus.Deferred,
            "Depthwise Conv2D 3x3 backward-weight direct-PTX coalesced reduction (#841 depthwise family)",
            "dW[c,r,s] = sum_{n,oh,ow} input[n,c,oh+r-1,ow+s-1]*gradOut[n,c,oh,ow] (per channel). One block per " +
            "channel reduces N x H x W into the 9 taps with coalesced spatial reads and a shared tree reduce. " +
            "Verified correct on-device (<= 2e-3). Deferred."),

        new PtxParitySpec("PtxConv2DBackwardWeight3x3Kernel", PtxParityStatus.Deferred,
            "Conv2D backward-weight 3x3 direct-PTX coalesced reduction (#841 backward family)",
            "dW[k,c,r,s] = sum_{n,oh,ow} input[n,c,oh+r-1,ow+s-1] * gradOut[n,k,oh,ow]. One block per (k,c) " +
            "pair reduces the N x H x W contraction into the 9 filter taps: consecutive threads walk the " +
            "flattened (n,oh,ow) index so the gradOut reads coalesce, the gradOut value is read once and reused " +
            "across all 9 overlapping (cached) input taps, and a shared tree reduction combines the taps. " +
            "Verified correct on-device (<= 2e-3 vs fp64 CPU reference, non-power-of-2 spatial). Deferred."),

        new PtxParitySpec("PtxConv2DBackwardInput3x3Kernel", PtxParityStatus.Deferred,
            "Conv2D backward-input 3x3 direct-PTX transpose-gather (#841 backward family)",
            "dX[n,c,ih,iw] = sum_{k,r,s} W[k,c,r,s] * gradOut[n,k,ih-r+1,iw-s+1] (flip/transpose of the forward " +
            "correlation). One thread per input-gradient element with consecutive threads owning consecutive iw " +
            "(contiguous NCHW axis) so gradOut reads (ow=iw-s+1) and dX stores coalesce and the weight reads " +
            "broadcast across the warp. Verified correct on-device (<= 2e-3 vs fp64 CPU reference). Deferred."),

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
