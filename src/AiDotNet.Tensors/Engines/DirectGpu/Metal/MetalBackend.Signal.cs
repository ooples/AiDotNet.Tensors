// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - FFT, Signal Processing, and RNG operations.

using AiDotNet.Tensors.Helpers;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    private void DispatchSplitFftMetal(IGpuBuffer inputReal, IGpuBuffer inputImag,
        IGpuBuffer outputReal, IGpuBuffer outputImag, int sequences, int length,
        int baseStride, int elementStride, bool inverse)
    {
        if (length <= 0 || (length & (length - 1)) != 0)
            throw new ArgumentException($"FFT size must be a positive power of two, got {length}", nameof(length));
        if (ReferenceEquals(outputReal, outputImag))
            throw new ArgumentException("Real and imaginary FFT outputs must use distinct buffers.");
        DispatchResidentMetal("split_fft_strided_serial", sequences,
            new[] { inputReal, inputImag, outputReal, outputImag },
            (uint)sequences, (uint)length, (uint)baseStride, (uint)elementStride, inverse ? 1u : 0u);
    }

    #region FFT and Signal Processing

    /// <summary>
    /// Complex-to-complex 1D FFT.
    /// </summary>
    public void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
    {
        ThrowIfDisposed();
        DispatchSplitFftMetal(inputReal, inputImag, outputReal, outputImag, 1, n, n, 1, inverse);
    }

    private static void CooleyTukeyFFT(float[] inReal, float[] inImag, float[] outReal, float[] outImag, int n, bool inverse)
    {
        // Validate n is a power of two
        if (n <= 0 || (n & (n - 1)) != 0)
        {
            throw new ArgumentException($"FFT size must be a positive power of two, got {n}", nameof(n));
        }

        // Copy input to output for in-place computation
        Array.Copy(inReal, outReal, n);
        Array.Copy(inImag, outImag, n);

        // Compute bits using integer arithmetic to avoid float rounding errors
        int bits = 0;
        int temp = n;
        while (temp > 1) { temp >>= 1; bits++; }
        for (int i = 0; i < n; i++)
        {
            int j = BitReverse(i, bits);
            if (j > i)
            {
                (outReal[i], outReal[j]) = (outReal[j], outReal[i]);
                (outImag[i], outImag[j]) = (outImag[j], outImag[i]);
            }
        }

        // FFT
        float sign = inverse ? 1.0f : -1.0f;
        for (int len = 2; len <= n; len *= 2)
        {
            float angle = sign * 2 * MathF.PI / len;
            float wReal = MathF.Cos(angle);
            float wImag = MathF.Sin(angle);

            for (int i = 0; i < n; i += len)
            {
                float curReal = 1.0f;
                float curImag = 0.0f;

                for (int j = 0; j < len / 2; j++)
                {
                    int u = i + j;
                    int v = i + j + len / 2;

                    float tReal = curReal * outReal[v] - curImag * outImag[v];
                    float tImag = curReal * outImag[v] + curImag * outReal[v];

                    outReal[v] = outReal[u] - tReal;
                    outImag[v] = outImag[u] - tImag;
                    outReal[u] = outReal[u] + tReal;
                    outImag[u] = outImag[u] + tImag;

                    float nextReal = curReal * wReal - curImag * wImag;
                    curImag = curReal * wImag + curImag * wReal;
                    curReal = nextReal;
                }
            }
        }

        // Normalize for inverse FFT
        if (inverse)
        {
            for (int i = 0; i < n; i++)
            {
                outReal[i] /= n;
                outImag[i] /= n;
            }
        }
    }

    private static int BitReverse(int n, int bits)
    {
        int result = 0;
        for (int i = 0; i < bits; i++)
        {
            result = (result << 1) | (n & 1);
            n >>= 1;
        }
        return result;
    }

    /// <summary>
    /// Real-to-complex 1D FFT.
    /// </summary>
    public void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n)
    {
        ThrowIfDisposed();
        if (n <= 0 || (n & (n - 1)) != 0)
            throw new ArgumentException($"FFT size must be a positive power of two, got {n}", nameof(n));
        int halfLength = n / 2 + 1;
        using var zeroImaginary = AllocateBuffer(n);
        using var fullReal = AllocateBuffer(n);
        using var fullImaginary = AllocateBuffer(n);
        Fill(zeroImaginary, 0f, n);
        DispatchSplitFftMetal(input, zeroImaginary, fullReal, fullImaginary, 1, n, n, 1, false);
        Copy(fullReal, 0, outputReal, 0, halfLength);
        Copy(fullImaginary, 0, outputImag, 0, halfLength);
    }

    /// <summary>
    /// Complex-to-real inverse 1D FFT.
    /// </summary>
    public void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n)
    {
        ThrowIfDisposed();
        if (n <= 0 || (n & (n - 1)) != 0)
            throw new ArgumentException($"FFT size must be a positive power of two, got {n}", nameof(n));
        using var fullReal = AllocateBuffer(n);
        using var fullImaginary = AllocateBuffer(n);
        using var transformedReal = AllocateBuffer(n);
        using var transformedImaginary = AllocateBuffer(n);
        DispatchResidentMetal("irfft_reconstruct", n,
            new[] { inputReal, inputImag, fullReal, fullImaginary }, (uint)n);
        DispatchSplitFftMetal(fullReal, fullImaginary, transformedReal, transformedImaginary, 1, n, n, 1, true);
        Copy(transformedReal, output, n);
    }

    /// <summary>
    /// Batched 1D FFT.
    /// </summary>
    public void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int batch, int n, bool inverse)
    {
        ThrowIfDisposed();
        DispatchSplitFftMetal(inputReal, inputImag, outputReal, outputImag,
            batch, n, n, 1, inverse);
    }

    /// <summary>
    /// 2D FFT.
    /// </summary>
    public void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int height, int width, bool inverse)
    {
        ThrowIfDisposed();
        if (height <= 0 || width <= 0) return;
        if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0)
            throw new ArgumentException("FFT dimensions must be powers of two.");
        int count = checked(height * width);
        using var rowReal = AllocateBuffer(count);
        using var rowImaginary = AllocateBuffer(count);
        DispatchSplitFftMetal(inputReal, inputImag, rowReal, rowImaginary,
            height, width, width, 1, inverse);
        DispatchSplitFftMetal(rowReal, rowImaginary, outputReal, outputImag,
            width, height, 1, width, inverse);
    }

    /// <summary>
    /// Apply window function.
    /// </summary>
    /// <inheritdoc/>
    public void BatchedFFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag,
        IGpuBuffer outputReal, IGpuBuffer outputImag,
        int batch, int height, int width, bool inverse)
    {
        if (batch <= 0 || height <= 0 || width <= 0) return;
        int sliceSize = height * width;

        if (batch == 1)
        {
            // Single image — direct FFT2D, no temp buffers needed
            FFT2D(inputReal, inputImag, outputReal, outputImag, height, width, inverse);
            return;
        }

        // Allocate temp slice-sized buffers ONCE (reused across all slices)
        var tempInR = AllocateBuffer(sliceSize);
        var tempInI = AllocateBuffer(sliceSize);
        var tempOutR = AllocateBuffer(sliceSize);
        var tempOutI = AllocateBuffer(sliceSize);
        try
        {
            for (int b = 0; b < batch; b++)
            {
                int off = b * sliceSize;
                // GPU-to-GPU copy: extract slice from batched buffer
                Copy(inputReal, off, tempInR, 0, sliceSize);
                Copy(inputImag, off, tempInI, 0, sliceSize);
                // FFT2D on GPU (fully GPU-resident, no CPU round-trip)
                FFT2D(tempInR, tempInI, tempOutR, tempOutI, height, width, inverse);
                // GPU-to-GPU copy: write result back to batched output
                Copy(tempOutR, 0, outputReal, off, sliceSize);
                Copy(tempOutI, 0, outputImag, off, sliceSize);
            }
        }
        finally
        {
            tempInR.Dispose(); tempInI.Dispose();
            tempOutR.Dispose(); tempOutI.Dispose();
        }
    }

    public void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n)
    {
        ThrowIfDisposed();
        Multiply(input, window, output, n);
    }

    /// <summary>
    /// Complex magnitude: sqrt(real² + imag²).
    /// </summary>
    public void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
    {
        ThrowIfDisposed();
        SplitComplexMagnitude(real, imag, magnitude, n);
    }

    /// <summary>
    /// Complex phase: atan2(imag, real).
    /// </summary>
    public void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
    {
        ThrowIfDisposed();
        SplitComplexPhase(real, imag, phase, n);
    }

    /// <summary>
    /// Convert polar to complex.
    /// </summary>
    public void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        ThrowIfDisposed();
        SplitComplexFromPolar(magnitude, phase, real, imag, n);
    }

    /// <summary>
    /// Apply Mel filterbank.
    /// </summary>
    public void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec,
        int numFrames, int numFreqs, int nMels)
    {
        ThrowIfDisposed();
        MelFilterbankApply(powerSpec, filterbank, melSpec, numFrames, numFreqs, nMels);
    }

    /// <summary>
    /// Convert power to decibels.
    /// </summary>
    public void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("power_to_db", n, new[] { power, db },
            (uint)n, unchecked((uint)SingleToInt32BitsCompat(refValue)), unchecked((uint)SingleToInt32BitsCompat(minDb)));
    }

    /// <summary>
    /// Convert decibels to power.
    /// </summary>
    public void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("db_to_power", n, new[] { db, power },
            (uint)n, unchecked((uint)SingleToInt32BitsCompat(refValue)));
    }

    #endregion

    #region Random Number Generation

    /// <summary>
    /// Generate uniform random numbers.
    /// </summary>
    public void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        DispatchResidentMetal("random_uniform_resident", size, [output], (uint)size,
            unchecked((uint)SingleToInt32BitsCompat(min)), unchecked((uint)SingleToInt32BitsCompat(max)), (uint)seed, (uint)(seed >> 32));
    }

    public void GenerateStatelessDropoutMask(
        IGpuBuffer output, int size, uint threshold, float scale, uint seed)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        DispatchResidentMetal("stateless_dropout_mask", size, [output], (uint)size, threshold,
            unchecked((uint)SingleToInt32BitsCompat(scale)), seed);
    }

    /// <summary>
    /// Generate normal (Gaussian) random numbers using Box-Muller.
    /// </summary>
    public void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        DispatchResidentMetal("random_normal_resident", (size + 1) / 2, [output], (uint)size,
            unchecked((uint)SingleToInt32BitsCompat(mean)), unchecked((uint)SingleToInt32BitsCompat(stdDev)), (uint)seed, (uint)(seed >> 32));
    }

    public void GenerateSecureRandomUniform(IGpuBuffer output, int size, float min, float max)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        GenerateRandomUniform(output, size, min, max, GpuRandomSeed.Create());
    }

    #endregion

    #region Specialized Layer Operations

    /// <summary>
    /// RBF kernel forward pass.
    /// </summary>
    public void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output,
        int batchSize, int numCenters, int inputDim)
    {
        ThrowIfDisposed();
        int count = checked(batchSize * numCenters);
        DispatchResidentMetal("rbf_forward", count, new[] { input, centers, epsilons, output },
            (uint)batchSize, (uint)numCenters, (uint)inputDim);
    }

    /// <summary>
    /// STDP weight update.
    /// </summary>
    public void StdpUpdate(IGpuBuffer weights, IGpuBuffer preTrace, IGpuBuffer postTrace,
        IGpuBuffer preSpike, IGpuBuffer postSpike,
        float ltpRate, float ltdRate, float homeostasisRate,
        float minWeight, float maxWeight, int numPre, int numPost)
    {
        ThrowIfDisposed();
        int count = checked(numPre * numPost);
        DispatchResidentMetal("stdp_update", count,
            new[] { weights, preTrace, postTrace, preSpike, postSpike },
            (uint)numPre, (uint)numPost, unchecked((uint)SingleToInt32BitsCompat(ltpRate)), unchecked((uint)SingleToInt32BitsCompat(ltdRate)),
            unchecked((uint)SingleToInt32BitsCompat(homeostasisRate)), unchecked((uint)SingleToInt32BitsCompat(minWeight)), unchecked((uint)SingleToInt32BitsCompat(maxWeight)));
    }

    /// <summary>
    /// Update traces and detect spikes.
    /// </summary>
    public void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input,
        float decay, float threshold, int size)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("update_traces", size, new[] { traces, spikes, input },
            (uint)size, unchecked((uint)SingleToInt32BitsCompat(decay)), unchecked((uint)SingleToInt32BitsCompat(threshold)));
    }

    #endregion

    /// <inheritdoc/>
    public void SpectralFilter(IGpuBuffer inputReal, IGpuBuffer filterReal, IGpuBuffer filterImag,
        IGpuBuffer outputReal, int batch, int height, int width, int filterSliceCount)
    {
        ThrowIfDisposed();
        if (filterSliceCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(filterSliceCount), "Must be >= 1.");
        if (height <= 0 || width <= 0 || batch <= 0)
            throw new ArgumentOutOfRangeException("Dimensions must be positive.");
        if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0)
            throw new ArgumentException("height and width must be powers of 2 for FFT.");

        int sliceSize = height * width;
        int totalSize = batch * sliceSize;

        IGpuBuffer? fftR = null, fftI = null, mulR = null, mulI = null, ifftI = null, zeroI = null;
        try
        {
            fftR = AllocateBuffer(totalSize);
            fftI = AllocateBuffer(totalSize);
            mulR = AllocateBuffer(totalSize);
            mulI = AllocateBuffer(totalSize);
            ifftI = AllocateBuffer(totalSize);
            zeroI = AllocateBuffer(totalSize);
        Fill(zeroI, 0f, totalSize);

            BatchedFFT2D(inputReal, zeroI, fftR, fftI, batch, height, width, inverse: false);

            if (filterSliceCount == batch)
            {
                SplitComplexMultiply(fftR, fftI, filterReal, filterImag, mulR, mulI, totalSize);
            }
            else
            {
                var bcastFR = AllocateBuffer(totalSize);
                var bcastFI = AllocateBuffer(totalSize);
                try
                {
                    for (int b = 0; b < batch; b++)
                    {
                        Copy(filterReal, 0, bcastFR, b * sliceSize, sliceSize);
                        Copy(filterImag, 0, bcastFI, b * sliceSize, sliceSize);
                    }
                    SplitComplexMultiply(fftR, fftI, bcastFR, bcastFI, mulR, mulI, totalSize);
                }
                finally { bcastFR.Dispose(); bcastFI.Dispose(); }
            }

            BatchedFFT2D(mulR, mulI, outputReal, ifftI, batch, height, width, inverse: true);
        }
        finally
        {
            fftR?.Dispose(); fftI?.Dispose();
            mulR?.Dispose(); mulI?.Dispose();
            ifftI?.Dispose(); zeroI?.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Atan2Elementwise(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer output, int n)
    {
        ThrowIfDisposed();
        if (n <= 0) return;
        var pipeline = GetPipeline("SpectralPerf", _spectralPerfLibrary, "atan2_elementwise");
        var (groups, threads) = pipeline.Calculate1DDispatch(n);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        // Kernel signature is atan2_elementwise(imag, real, ...), keep binding order unchanged.
        encoder.SetBuffer((MetalGpuBuffer)imag, 0);
        encoder.SetBuffer((MetalGpuBuffer)real, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes((uint)n, 3);
        encoder.DispatchThreadgroups(groups, threads);
    }

    /// <inheritdoc/>
    public void NormalizeRowsFused(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        ThrowIfDisposed();
        if (rows <= 0 || cols <= 0) return;
        var pipeline = GetPipeline("SpectralPerf", _spectralPerfLibrary, "normalize_rows_fused");
        // The normalize_rows_fused kernel uses a tree reduction that requires a power-of-two threadgroup size.
        // Pick the largest power-of-two <= min(256, cols), then clamp to a minimum of 32.
        uint block = 32;
        uint cap = (uint)Math.Min(256, cols);
        while (block * 2 <= cap) block *= 2;
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes((uint)rows, 2);
        encoder.SetBytes((uint)cols, 3);
        encoder.SetThreadgroupMemoryLength(block * sizeof(float), 0);
        encoder.DispatchThreadgroups(new MTLSize((uint)rows, 1, 1), new MTLSize(block, 1, 1));
    }

    /// <inheritdoc/>
    public void AnalyticSignalMask(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batch, int fftSize, int binLow, int binHigh)
    {
        ThrowIfDisposed();
        if (batch <= 0 || fftSize <= 0) return;
        // Validate signed bin indices before casting to uint (negative values would wrap to
        // very large unsigned values and corrupt kernel indexing).
        if (binLow < 0 || binHigh < binLow || binHigh > fftSize)
            throw new ArgumentOutOfRangeException(nameof(binHigh),
                $"Require 0 <= binLow ({binLow}) <= binHigh ({binHigh}) <= fftSize ({fftSize}).");
        // Guard batch*fftSize against int overflow.
        long totalL = (long)batch * fftSize;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        var pipeline = GetPipeline("SpectralPerf", _spectralPerfLibrary, "analytic_signal_mask");
        var (groups, threads) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)specReal, 0);
        encoder.SetBuffer((MetalGpuBuffer)specImag, 1);
        encoder.SetBuffer((MetalGpuBuffer)outReal, 2);
        encoder.SetBuffer((MetalGpuBuffer)outImag, 3);
        encoder.SetBytes((uint)batch, 4);
        encoder.SetBytes((uint)fftSize, 5);
        encoder.SetBytes((uint)binLow, 6);
        encoder.SetBytes((uint)binHigh, 7);
        encoder.DispatchThreadgroups(groups, threads);
    }

    /// <inheritdoc/>
    public void BispectrumGather(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int maxF1, int maxF2)
    {
        ThrowIfDisposed();
        if (maxF1 <= 0 || maxF2 <= 0) return;
        long totalL = (long)maxF1 * maxF2;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        var pipeline = GetPipeline("SpectralPerf", _spectralPerfLibrary, "bispectrum_gather");
        var (groups, threads) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)specReal, 0);
        encoder.SetBuffer((MetalGpuBuffer)specImag, 1);
        encoder.SetBuffer((MetalGpuBuffer)outReal, 2);
        encoder.SetBuffer((MetalGpuBuffer)outImag, 3);
        encoder.SetBytes((uint)maxF1, 4);
        encoder.SetBytes((uint)maxF2, 5);
        encoder.DispatchThreadgroups(groups, threads);
    }

    /// <inheritdoc/>
    public void TrispectrumGather(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int maxF1, int maxF2, int maxF3)
    {
        ThrowIfDisposed();
        if (maxF1 <= 0 || maxF2 <= 0 || maxF3 <= 0) return;
        long totalL = (long)maxF1 * maxF2 * maxF3;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        var pipeline = GetPipeline("SpectralPerf", _spectralPerfLibrary, "trispectrum_gather");
        var (groups, threads) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)specReal, 0);
        encoder.SetBuffer((MetalGpuBuffer)specImag, 1);
        encoder.SetBuffer((MetalGpuBuffer)outReal, 2);
        encoder.SetBuffer((MetalGpuBuffer)outImag, 3);
        encoder.SetBytes((uint)maxF1, 4);
        encoder.SetBytes((uint)maxF2, 5);
        encoder.SetBytes((uint)maxF3, 6);
        encoder.DispatchThreadgroups(groups, threads);
    }

    /// <inheritdoc/>
    public void CavityBounceInplace(IGpuBuffer workReal, IGpuBuffer workImag, int total, float invN)
    {
        ThrowIfDisposed();
        if (total <= 0) return;
        var pipeline = GetPipeline("SpectralPerf", _spectralPerfLibrary, "cavity_bounce_inplace");
        var (groups, threads) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)workReal, 0);
        encoder.SetBuffer((MetalGpuBuffer)workImag, 1);
        encoder.SetBytes((uint)total, 2);
        encoder.SetBytes(invN, 3);
        encoder.DispatchThreadgroups(groups, threads);
    }

    /// <inheritdoc/>
    public void WidebandLogBinPool(IGpuBuffer magBuf, IGpuBuffer output,
        int totalSegBatch, int fftSize, int numBins, int usable)
    {
        ThrowIfDisposed();
        if (totalSegBatch <= 0 || fftSize <= 0 || numBins <= 0 || usable <= 0) return;
        long totalL = (long)totalSegBatch * numBins;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        var pipeline = GetPipeline("SpectralPerf", _spectralPerfLibrary, "wideband_log_bin_pool");
        var (groups, threads) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)magBuf, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes((uint)totalSegBatch, 2);
        encoder.SetBytes((uint)fftSize, 3);
        encoder.SetBytes((uint)numBins, 4);
        encoder.SetBytes((uint)usable, 5);
        encoder.DispatchThreadgroups(groups, threads);
    }

    /// <inheritdoc/>
    public void MelFilterbankApply(IGpuBuffer powerSpec, IGpuBuffer melFilters, IGpuBuffer melEnergy,
        int totalSegBatch, int specBins, int melBins)
    {
        ThrowIfDisposed();
        if (totalSegBatch <= 0 || specBins <= 0 || melBins <= 0) return;
        long totalL = (long)totalSegBatch * melBins;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        var pipeline = GetPipeline("SpectralPerf", _spectralPerfLibrary, "mel_filterbank_apply");
        var (groups, threads) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)powerSpec, 0);
        encoder.SetBuffer((MetalGpuBuffer)melFilters, 1);
        encoder.SetBuffer((MetalGpuBuffer)melEnergy, 2);
        encoder.SetBytes((uint)totalSegBatch, 3);
        encoder.SetBytes((uint)specBins, 4);
        encoder.SetBytes((uint)melBins, 5);
        encoder.DispatchThreadgroups(groups, threads);
    }

    /// <inheritdoc/>
    public void MfccLog1p(IGpuBuffer input, IGpuBuffer output, int n)
    {
        ThrowIfDisposed();
        if (n <= 0) return;
        var pipeline = GetPipeline("SpectralPerf", _spectralPerfLibrary, "mfcc_log1p");
        var (groups, threads) = pipeline.Calculate1DDispatch(n);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes((uint)n, 2);
        encoder.DispatchThreadgroups(groups, threads);
    }

    /// <inheritdoc/>
    public void PacPhaseBinMi(IGpuBuffer thetaPhase, IGpuBuffer gammaAmp, IGpuBuffer output,
        int batch, int numSamples, int numGammaBands, int gammaIdx)
    {
        ThrowIfDisposed();
        if (batch <= 0) return;
        if (numSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(numSamples), "numSamples must be positive.");
        if (numGammaBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGammaBands), "numGammaBands must be positive.");
        if (gammaIdx < 0 || gammaIdx >= numGammaBands)
            throw new ArgumentOutOfRangeException(nameof(gammaIdx), $"gammaIdx must be in [0, {numGammaBands}).");
        var pipeline = GetPipeline("SpectralPerf", _spectralPerfLibrary, "pac_phase_bin_mi");
        var (groups, threads) = pipeline.Calculate1DDispatch(batch);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)thetaPhase, 0);
        encoder.SetBuffer((MetalGpuBuffer)gammaAmp, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes((uint)batch, 3);
        encoder.SetBytes((uint)numSamples, 4);
        encoder.SetBytes((uint)numGammaBands, 5);
        encoder.SetBytes((uint)gammaIdx, 6);
        encoder.DispatchThreadgroups(groups, threads);
    }
}
