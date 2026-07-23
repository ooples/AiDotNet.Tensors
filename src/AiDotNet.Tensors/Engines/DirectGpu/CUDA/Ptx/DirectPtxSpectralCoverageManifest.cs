using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxSpectralCoverageStatus
{
    ExistingBackend,
    ExperimentalDirectPtx,
    PromotedDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxSpectralCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxSpectralCoverageStatus Status);

/// <summary>
/// Executable issue-#850 inventory. Every discovered FFT, audio/spectral, and
/// complex-number boundary has an explicit direct-PTX assignment. Planned
/// entries are not promotion claims and must be split into exact ABI cells.
/// </summary>
internal static class DirectPtxSpectralCoverageManifest
{
    private const string Split = "canonical split real/imag contiguous vectors";
    private const string Interleaved = "canonical interleaved [pair,real-imag]";
    private const string Frames = "canonical contiguous frames/frequency-major audio tensors";

    internal static IReadOnlyList<DirectPtxSpectralCoverageCell> All { get; } =
    [
        Experimental("CudaBackend.ComplexMultiply", "NVRTC complex_multiply", "pairwise complex product", Interleaved, "FP32", "v1 SM86 pairwise-v2 exact-count cells"),
        Experimental("CudaBackend.ComplexConjugate", "NVRTC complex_conjugate", "complex conjugate", Interleaved, "FP32", "v1 Ampere pairwise-v2 exact-pair cells; neg.f32 sign flip preserves NaN payloads and signed zeros"),
        Experimental("CudaBackend.ComplexMagnitude", "NVRTC complex_magnitude", "complex magnitude", Interleaved, "FP32", "v1 Ampere pairwise-v2 exact-pair cells; unfused mul-mul-add then sqrt.rn to match sqrtf rounding"),
        Experimental("CudaBackend.InterleaveComplex", "NVRTC interleave_complex", "split-to-interleaved conversion", Split, "FP32", "v1 Ampere exact-count cells (PtxComplexInterleaveF32Kernel); pure bit-exact copy, v2 store to the interleaved pair"),
        Experimental("CudaBackend.DeinterleaveComplex", "NVRTC deinterleave_complex", "interleaved-to-split conversion", Interleaved, "FP32", "v1 Ampere exact-count cells (PtxComplexInterleaveF32Kernel); pure bit-exact copy, v2 load from the interleaved pair"),
        Experimental("CudaBackend.SplitComplexMultiply", "NVRTC split_complex_multiply", "pairwise split complex product", Split, "FP32", "v1 Ampere split exact-count cells (PtxSplitComplexBinaryF32Kernel); multiply-then-fma contraction matching the interleaved multiply and the reference fused form"),
        Experimental("CudaBackend.SplitComplexConjugate", "NVRTC split_complex_conjugate", "split complex conjugate", Split, "FP32", "v1 Ampere split exact-count cells (PtxSplitComplexConjugateF32Kernel); real lane copied, imag neg.f32 sign flip preserves NaN payloads and signed zeros"),
        Experimental("CudaBackend.SplitComplexMagnitude", "NVRTC split_complex_magnitude", "split magnitude", Split, "FP32", "v1 Ampere split exact-count cells (PtxSplitComplexUnaryF32Kernel); unfused mul-mul-add then sqrt.rn to match sqrtf rounding"),
        Experimental("CudaBackend.SplitComplexMagnitudeSquared", "NVRTC split_complex_magnitude_squared", "split power", Split, "FP32", "v1 Ampere split exact-count cells (PtxSplitComplexUnaryF32Kernel); unfused mul-mul-add power sum"),
        Experimental("CudaBackend.SplitComplexPhase", "NVRTC split_complex_phase", "split phase", Split, "FP32", "v1 Ampere split exact-count cells (PtxSplitComplexPhaseF32Kernel); minimax atan2 (~1e-4), tolerance-based parity"),
        Experimental("CudaBackend.SplitComplexFromPolar", "NVRTC split_complex_from_polar", "polar to split Cartesian", Split, "FP32", "v1 Ampere split exact-count cells (PtxSplitComplexFromPolarF32Kernel); cos.approx/sin.approx, tolerance-based parity"),
        Experimental("CudaBackend.SplitComplexScale", "NVRTC split_complex_scale", "complex scalar scale", Split, "FP32", "v1 Ampere split exact-count cells (PtxSplitComplexScaleF32Kernel); one mul.rn per lane, bit-exact; scalar via .param .f32"),
        Experimental("CudaBackend.SplitComplexAdd", "NVRTC split_complex_add", "complex add", Split, "FP32", "v1 Ampere split exact-count cells (PtxSplitComplexBinaryF32Kernel); two add.rn lanes"),
        Experimental("CudaBackend.SplitComplexCrossSpectral", "NVRTC split_complex_cross_spectral", "cross spectral product", Split, "FP32", "v1 Ampere split exact-count cells (PtxSplitComplexBinaryF32Kernel CrossSpectral); a*conj(b) with the multiply-then-fma contraction matching the reference fused form"),
        Experimental("CudaBackend.FFT", "NVRTC radix-2 stages", "1D complex FFT/IFFT", Split, "FP32", "v1 Ampere radix-2 DIT: PtxBitReversePermutationF32Kernel (brev.b32 guarded swap, bit-exact) then log2(n) PtxFftButterflyF32Kernel stages (cos.approx/sin.approx twiddles, tolerance-based)"),
        Experimental("CudaBackend.BatchedFFT", "NVRTC batched radix-2 stages", "batched 1D complex FFT/IFFT", Split, "FP32", "v1 Ampere batched radix-2 DIT: PtxBatchedBitReverseF32Kernel then log2(n) PtxBatchedFftButterflyF32Kernel stages, batch via gridDim.y (baseOffset=b*n); bit-exact reorder, tolerance-based twiddles"),
        Experimental("CudaBackend.FFT2D", "NVRTC row/column radix-2 stages", "2D complex FFT/IFFT", Split, "FP32", "v1 Ampere separable radix-2 DIT: contiguous row pass reuses the batched FFT kernels (batch=height, n=width), then the strided column pass PtxFftColsBitReverseF32Kernel (bit-exact) + log2(height) PtxFftColsButterflyF32Kernel stages (column stride=width, tolerance-based twiddles)"),
        Experimental("CudaBackend.BatchedFFT2D", "NVRTC batched FFT2D", "batched 2D complex FFT/IFFT", Split, "FP32", "v1 Ampere separable radix-2 DIT over images: contiguous row pass reuses the batched FFT kernels (batch=images*height, n=width), then the per-image strided column pass PtxBatchedFftColsBitReverseF32Kernel (bit-exact) + log2(height) PtxBatchedFftColsButterflyF32Kernel stages (image offset via gridDim.y, tolerance-based twiddles)"),
        Experimental("CudaBackend.RFFT", "NVRTC complex FFT plus extraction", "real-to-positive-frequency FFT", Split, "FP32", "v1 Ampere: radix-2 FFT then PtxRfftPostprocessF32Kernel copies the n/2+1 positive-frequency bins, bit-exact"),
        Experimental("CudaBackend.IRFFT", "NVRTC reconstruction plus IFFT", "positive-frequency-to-real IFFT", Split, "FP32", "v1 Ampere: PtxIrfftPreprocessF32Kernel expands Hermitian symmetry (neg.f32 conjugate, bit-exact) then radix-2 inverse FFT then PtxScaleInverseF32Kernel by 1/n (one mul.rn per lane, bit-exact)"),
        Experimental("CudaBackend.StftMagPhase", "NVRTC direct STFT mag/phase", "windowed STFT magnitude and phase", Frames, "FP32", "v1 Ampere direct windowed DFT per (b,k,frame) (PtxStftMagPhaseF32Kernel); cos.approx/sin.approx twiddle, fma accumulation, minimax atan2 phase; tolerance-based"),
        Experimental("CudaBackend.BuildSpectrum", "NVRTC build_spectrum", "magnitude/phase to Hermitian spectrum", Frames, "FP32", "v1 Ampere thread-per-(b,frame) (PtxBuildSpectrumF32Kernel); zero nFft bins, cos.approx/sin.approx polar fill of numFreqs bins, conjugate mirror; tolerance-based"),
        Experimental("CudaBackend.IstftFromSpectrum", "NVRTC ISTFT overlap-add", "spectrum inverse transform and overlap-add", Frames, "FP32", "v1 Ampere: PtxOverlapAddF32Kernel windows and overlap-adds the synthesis frames (fma accumulation matching the fused reference) normalized by PtxWindowSumSquaresF32Kernel; both tolerance-based, thread-per-output-sample frame loops"),
        Experimental("CudaBackend.PhaseVocoder", "NVRTC phase_vocoder", "rate-adjusted magnitude/phase", Frames, "FP32", "v1 Ampere thread-per-(leading,freq) time resample (PtxPhaseVocoderF32Kernel); fma magnitude lerp and cvt.rni wrapped phase accumulation with rate .param .f32; tolerance-based"),
        Experimental("CudaBackend.ApplyMelFilterbank", "NVRTC mel filterbank", "power spectrum to mel bands", Frames, "FP32", "v1 Ampere exact-shape cells (PtxApplyMelFilterbankF32Kernel); thread-per-(frame,mel) fma reduction over freqs"),
        Experimental("CudaBackend.ApplyWindow", "NVRTC apply_window", "windowed frame elementwise product", Frames, "FP32", "v1 Ampere exact-count cells (PtxApplyWindowF32Kernel); one mul.rn per element, bit-exact input*window"),
        Experimental("CudaBackend.PowerToDb", "NVRTC power_to_db", "power spectrum to decibels", Frames, "FP32", "v1 Ampere exact-count cells (PtxPowerToDbF32Kernel); lg2.approx scaled by 10*log10(2) with refValue/minDb .param .f32, tolerance-based"),
        Experimental("CudaBackend.DbToPower", "NVRTC db_to_power", "decibels to power spectrum", Frames, "FP32", "v1 Ampere exact-count cells (PtxDbToPowerF32Kernel); ex2.approx of db*log2(10)/10 with refValue .param .f32, tolerance-based"),
        Experimental("CudaBackend.MelFilterbankApply", "NVRTC spectral mel apply", "segmented power-to-mel", Frames, "FP32", "v1 Ampere thread-per-(seg,mel) guarded fma reduction over specBins (PtxMelFilterbankApplyF32Kernel); matches the perf-path mel_filterbank_apply, tolerance-based"),
        Planned("CudaBackend.SpectralFilter", "FFT2D multiply IFFT2D pipeline", "real spatial spectral filtering", Split, "FP32", "fft-filter-ifft-fusion-cells"),
        Planned("DirectGpuTensorEngine.FFT", "resident CUDA FFT or CPU fallback", "public complex FFT", Split, "generic public; CUDA FP32", "public-fft-routing"),
        Planned("DirectGpuTensorEngine.IFFT", "resident CUDA IFFT or CPU fallback", "public complex IFFT", Split, "generic public; CUDA FP32", "public-ifft-routing"),
        Planned("IEngine.STFT", "CudaBackend.StftMagPhase or CPU", "public short-time Fourier transform", Frames, "generic public; CUDA FP32", "public-stft-routing"),
        Planned("IEngine.ISTFT", "CudaBackend ISTFT stages or CPU", "public inverse STFT", Frames, "generic public; CUDA FP32", "public-istft-routing"),
        Planned("IEngine.MelSpectrogram", "STFT plus mel/filter/log pipeline", "public mel spectrogram", Frames, "generic public; CUDA FP32", "stft-power-mel-log-fusion-cells"),
        Planned("IEngine.GriffinLim", "iterative STFT/ISTFT composition", "phase reconstruction", Frames, "generic public; CUDA FP32", "griffin-lim-step-cells"),
        Planned("IEngine.Resample", "CUDA audio resample", "polyphase waveform resample", "canonical contiguous waveform", "generic public; CUDA FP32", "resample-ratio-cells"),
        Planned("IEngine.NativeComplexMultiply", "native/split complex dispatch", "public native complex product", Interleaved, "Complex<FP32/FP64>", "public-complex-product-routing"),
        Planned("IEngine.NativeSpectralFilter", "FFT multiply IFFT composition", "public 2D spectral filter", Split, "generic public; CUDA FP32", "public-spectral-filter-routing")
    ];

    private static readonly IReadOnlyDictionary<string, DirectPtxSpectralCoverageCell> ByApi =
        All.ToDictionary(cell => cell.Api, StringComparer.Ordinal);

    internal static DirectPtxSpectralCoverageCell Get(string api) =>
        ByApi.TryGetValue(api, out DirectPtxSpectralCoverageCell? cell)
            ? cell
            : throw new KeyNotFoundException($"No direct-PTX spectral coverage cell for '{api}'.");

    private static DirectPtxSpectralCoverageCell Experimental(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment,
            DirectPtxSpectralCoverageStatus.ExperimentalDirectPtx);

    private static DirectPtxSpectralCoverageCell Planned(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment,
            DirectPtxSpectralCoverageStatus.PlannedDirectPtx);
}
