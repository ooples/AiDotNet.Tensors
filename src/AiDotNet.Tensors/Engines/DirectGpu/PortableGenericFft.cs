// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IGenericFftBackend"/> for every backend without CUDA's dedicated module (#962, #961, #963-#965), built
/// from that backend's own radix-2 <see cref="IDirectGpuBackend.BatchedFFT"/> and the shared batch kernels. Powers of
/// two run radix-2 directly; other lengths run Bluestein on the device: chirp, pre-multiply and zero-pad to a power
/// of two m >= 2n - 1, convolve with the chirp filter through three length-m FFTs, post-multiply. Arithmetic is
/// float32; 16-bit storage converts at the load/store boundary. Forward is unscaled, inverse scaled by 1/n.
/// </summary>
internal static class PortableGenericFft
{
    public const int MaximumLength = 1 << 29;
    public const int MaximumBatch = 65535;

    /// <summary>Launches a shared batch kernel by name: (kernel, thread count, arguments in kernel order).</summary>
    public delegate void Launcher(string kernel, int threads, params object[] args);

    /// <summary>Every kernel the transforms and storage conversions launch.</summary>
    internal static readonly string[] KernelNames =
    {
        "bx_fft_to_storage", "bx_fft_from_storage", "bx_fft_chirp", "bx_fft_bluestein_pre", "bx_fft_cmul", "bx_fft_bluestein_post",
        "bx_fft_bit_reverse", "bx_fft_butterfly",
    };

    /// <summary>
    /// A batched radix-2 FFT of power-of-two length <paramref name="n"/> on the shared kernels (#965), for a backend
    /// whose own <see cref="IDirectGpuBackend.BatchedFFT"/> is not one: a bit-reversal permute into the output (scaled
    /// by 1/n for the inverse), then log2(n) butterfly stages in place. Input and output may be the same buffers.
    /// </summary>
    public static void BatchedRadix2(IDirectGpuBackend backend, Launcher launch, IGpuBuffer inputReal, IGpuBuffer inputImag,
        IGpuBuffer outputReal, IGpuBuffer outputImag, int batch, int n, bool inverse)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (launch is null) throw new ArgumentNullException(nameof(launch));
        if (batch < 1) throw new ArgumentOutOfRangeException(nameof(batch));
        if (n < 1 || (n & (n - 1)) != 0 || n > MaximumLength) throw new ArgumentOutOfRangeException(nameof(n), "n must be a power of two.");
        int count = checked(batch * n);
        foreach (var (buffer, name) in new[] { (inputReal, nameof(inputReal)), (inputImag, nameof(inputImag)),
                     (outputReal, nameof(outputReal)), (outputImag, nameof(outputImag)) })
        {
            if (buffer is null) throw new ArgumentNullException(name);
            if (buffer.Size < count) throw new ArgumentException($"{name} holds {buffer.Size} values, fewer than {batch} x {n}.", name);
        }
        // The permute reads every input before any output position is final, so it cannot run in place.
        if (ReferenceEquals(inputReal, outputReal) || ReferenceEquals(inputImag, outputImag)
            || ReferenceEquals(inputReal, outputImag) || ReferenceEquals(inputImag, outputReal))
        {
            using var re = backend.AllocateBuffer(count);
            using var im = backend.AllocateBuffer(count);
            backend.Copy(inputReal, re, count);
            backend.Copy(inputImag, im, count);
            BatchedRadix2(backend, launch, re, im, outputReal, outputImag, batch, n, inverse);
            return;
        }
        int log2n = 0;
        while ((1 << log2n) < n) log2n++;
        launch("bx_fft_bit_reverse", count, inputReal, inputImag, outputReal, outputImag, batch, n, log2n, inverse ? 1f / n : 1f);
        for (int span = 1; span < n; span <<= 1)
            launch("bx_fft_butterfly", count / 2, outputReal, outputImag, batch, n, span, inverse ? 1 : 0);
    }
    public static bool Supports(FftElementType type)
    {
        _ = type.ByteSize();
        return type is FftElementType.Float32 or FftElementType.Float16 or FftElementType.BFloat16;
    }

    private static int StorageMode(FftElementType type) => type == FftElementType.Float16 ? 1 : 2;

    public static void ToStorage(IDirectGpuBackend backend, Launcher launch, IGpuBuffer source, IGpuBuffer destination,
        int count, FftElementType type)
    {
        ValidateCount(count);
        if (type == FftElementType.Float32) { backend.Copy(source, destination, count); return; }
        launch("bx_fft_to_storage", (count + 1) / 2, source, destination, count, StorageMode(type));
    }

    public static void FromStorage(IDirectGpuBackend backend, Launcher launch, IGpuBuffer source, IGpuBuffer destination,
        int count, FftElementType type)
    {
        ValidateCount(count);
        if (type == FftElementType.Float32) { backend.Copy(source, destination, count); return; }
        launch("bx_fft_from_storage", count, source, destination, count, StorageMode(type));
    }

    public static void Launch(IDirectGpuBackend backend, Launcher launch, IGpuBuffer real, IGpuBuffer imaginary,
        int batch, int n, bool inverse, FftElementType type)
    {
        if (real is null) throw new ArgumentNullException(nameof(real));
        if (imaginary is null) throw new ArgumentNullException(nameof(imaginary));
        if (batch < 1 || batch > MaximumBatch) throw new ArgumentOutOfRangeException(nameof(batch));
        if (n < 1 || n > MaximumLength) throw new ArgumentOutOfRangeException(nameof(n));
        if (!Supports(type)) throw new NotSupportedException($"{type} FFT storage is not supported.");
        int count = checked(batch * n);
        long needed = (long)count * type.ByteSize();
        if (real.SizeInBytes < needed || imaginary.SizeInBytes < needed)
            throw new ArgumentException($"FFT buffers need {needed} bytes for {batch} x {n} {type} values.");

        if (type == FftElementType.Float32)
        {
            LaunchFloat(backend, launch, real, imaginary, batch, n, inverse);
            return;
        }
        using var re = backend.AllocateBuffer(count);
        using var im = backend.AllocateBuffer(count);
        FromStorage(backend, launch, real, re, count, type);
        FromStorage(backend, launch, imaginary, im, count, type);
        LaunchFloat(backend, launch, re, im, batch, n, inverse);
        ToStorage(backend, launch, re, real, count, type);
        ToStorage(backend, launch, im, imaginary, count, type);
    }

    private static void LaunchFloat(IDirectGpuBackend backend, Launcher launch, IGpuBuffer re, IGpuBuffer im,
        int batch, int n, bool inverse)
    {
        if (n == 1) return; // a length-1 DFT, and its 1/1-scaled inverse, is the identity
        int count = checked(batch * n);
        if ((n & (n - 1)) == 0)
        {
            using var outRe = backend.AllocateBuffer(count);
            using var outIm = backend.AllocateBuffer(count);
            backend.BatchedFFT(re, im, outRe, outIm, batch, n, inverse);
            backend.Copy(outRe, re, count);
            backend.Copy(outIm, im, count);
            return;
        }

        int m = 1;
        while (m < 2 * n - 1) m <<= 1;
        int padded = checked(batch * m);
        using var chirpRe = backend.AllocateBuffer(n);
        using var chirpIm = backend.AllocateBuffer(n);
        using var filterRe = backend.AllocateBuffer(m);
        using var filterIm = backend.AllocateBuffer(m);
        using var filterFreqRe = backend.AllocateBuffer(m);
        using var filterFreqIm = backend.AllocateBuffer(m);
        using var padRe = backend.AllocateBuffer(padded);
        using var padIm = backend.AllocateBuffer(padded);
        using var freqRe = backend.AllocateBuffer(padded);
        using var freqIm = backend.AllocateBuffer(padded);

        launch("bx_fft_chirp", m, chirpRe, chirpIm, filterRe, filterIm, n, m, inverse ? 1 : 0);
        backend.BatchedFFT(filterRe, filterIm, filterFreqRe, filterFreqIm, 1, m, false);
        launch("bx_fft_bluestein_pre", padded, re, im, chirpRe, chirpIm, padRe, padIm, batch, n, m);
        backend.BatchedFFT(padRe, padIm, freqRe, freqIm, batch, m, false);
        launch("bx_fft_cmul", padded, freqRe, freqIm, filterFreqRe, filterFreqIm, batch, m);
        backend.BatchedFFT(freqRe, freqIm, padRe, padIm, batch, m, true);
        launch("bx_fft_bluestein_post", count, padRe, padIm, chirpRe, chirpIm, re, im, batch, n, m, inverse ? 1f / n : 1f);
    }

    /// <summary>
    /// <see cref="IFftBackend.LaunchFft"/> over the generic transform: deinterleave the re/im pairs, transform, and
    /// interleave back into the same buffer, all with device copies.
    /// </summary>
    public static void LaunchInterleaved(IDirectGpuBackend backend, IGenericFftBackend fft, IGpuBuffer buffer,
        int batchCount, int n, bool inverse)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (batchCount < 1) throw new ArgumentOutOfRangeException(nameof(batchCount));
        if (n < 1 || (n & (n - 1)) != 0) throw new ArgumentOutOfRangeException(nameof(n), "IFftBackend transforms power-of-two lengths.");
        int count = checked(batchCount * n);
        using var re = backend.AllocateBuffer(count);
        using var im = backend.AllocateBuffer(count);
        backend.StridedGather(buffer, re, 0, 2, count);
        backend.StridedGather(buffer, im, 1, 2, count);
        fft.LaunchFftGeneric(re, im, batchCount, n, inverse, FftElementType.Float32);
        backend.StridedScatter(re, buffer, 0, 2, count);
        backend.StridedScatter(im, buffer, 1, 2, count);
    }

    private static void ValidateCount(int count)
    {
        if (count < 1) throw new ArgumentOutOfRangeException(nameof(count));
    }
}
