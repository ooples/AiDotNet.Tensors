// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - FFT, Signal Processing, and RNG operations.

using AiDotNet.Tensors.Helpers;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region FFT and Signal Processing

    /// <summary>
    /// Complex-to-complex 1D FFT.
    /// </summary>
    public void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
    {
        ThrowIfDisposed();

        var inReal = DownloadBuffer(inputReal);
        var inImag = DownloadBuffer(inputImag);
        var outReal = new float[n];
        var outImag = new float[n];

        // Cooley-Tukey FFT implementation
        CooleyTukeyFFT(inReal, inImag, outReal, outImag, n, inverse);

        UploadToBuffer(outputReal, outReal);
        UploadToBuffer(outputImag, outImag);
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

        var inData = DownloadBuffer(input);
        var inImag = new float[n];
        var outReal = new float[n];
        var outImag = new float[n];

        CooleyTukeyFFT(inData, inImag, outReal, outImag, n, false);

        // Copy only first n/2 + 1 elements
        var resultReal = new float[n / 2 + 1];
        var resultImag = new float[n / 2 + 1];
        Array.Copy(outReal, resultReal, n / 2 + 1);
        Array.Copy(outImag, resultImag, n / 2 + 1);

        UploadToBuffer(outputReal, resultReal);
        UploadToBuffer(outputImag, resultImag);
    }

    /// <summary>
    /// Complex-to-real inverse 1D FFT.
    /// </summary>
    public void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n)
    {
        ThrowIfDisposed();

        var inReal = DownloadBuffer(inputReal);
        var inImag = DownloadBuffer(inputImag);

        // Reconstruct full complex spectrum
        var fullReal = new float[n];
        var fullImag = new float[n];

        int halfN = n / 2 + 1;
        for (int i = 0; i < halfN; i++)
        {
            fullReal[i] = inReal[i];
            fullImag[i] = inImag[i];
        }
        // Conjugate symmetry
        for (int i = 1; i < n / 2; i++)
        {
            fullReal[n - i] = fullReal[i];
            fullImag[n - i] = -fullImag[i];
        }

        var outReal = new float[n];
        var outImag = new float[n];
        CooleyTukeyFFT(fullReal, fullImag, outReal, outImag, n, true);

        UploadToBuffer(output, outReal);
    }

    /// <summary>
    /// Batched 1D FFT.
    /// </summary>
    public void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int batch, int n, bool inverse)
    {
        ThrowIfDisposed();

        var inReal = DownloadBuffer(inputReal);
        var inImag = DownloadBuffer(inputImag);
        var outReal = new float[batch * n];
        var outImag = new float[batch * n];

        for (int b = 0; b < batch; b++)
        {
            var batchInReal = new float[n];
            var batchInImag = new float[n];
            var batchOutReal = new float[n];
            var batchOutImag = new float[n];

            Array.Copy(inReal, b * n, batchInReal, 0, n);
            Array.Copy(inImag, b * n, batchInImag, 0, n);

            CooleyTukeyFFT(batchInReal, batchInImag, batchOutReal, batchOutImag, n, inverse);

            Array.Copy(batchOutReal, 0, outReal, b * n, n);
            Array.Copy(batchOutImag, 0, outImag, b * n, n);
        }

        UploadToBuffer(outputReal, outReal);
        UploadToBuffer(outputImag, outImag);
    }

    /// <summary>
    /// 2D FFT.
    /// </summary>
    public void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int height, int width, bool inverse)
    {
        ThrowIfDisposed();

        var inReal = DownloadBuffer(inputReal);
        var inImag = DownloadBuffer(inputImag);
        var outReal = new float[height * width];
        var outImag = new float[height * width];

        // First pass: FFT on rows
        for (int h = 0; h < height; h++)
        {
            var rowReal = new float[width];
            var rowImag = new float[width];
            var rowOutReal = new float[width];
            var rowOutImag = new float[width];

            for (int w = 0; w < width; w++)
            {
                rowReal[w] = inReal[h * width + w];
                rowImag[w] = inImag[h * width + w];
            }

            CooleyTukeyFFT(rowReal, rowImag, rowOutReal, rowOutImag, width, inverse);

            for (int w = 0; w < width; w++)
            {
                outReal[h * width + w] = rowOutReal[w];
                outImag[h * width + w] = rowOutImag[w];
            }
        }

        // Second pass: FFT on columns
        var tempReal = new float[height * width];
        var tempImag = new float[height * width];
        Array.Copy(outReal, tempReal, height * width);
        Array.Copy(outImag, tempImag, height * width);

        for (int w = 0; w < width; w++)
        {
            var colReal = new float[height];
            var colImag = new float[height];
            var colOutReal = new float[height];
            var colOutImag = new float[height];

            for (int h = 0; h < height; h++)
            {
                colReal[h] = tempReal[h * width + w];
                colImag[h] = tempImag[h * width + w];
            }

            CooleyTukeyFFT(colReal, colImag, colOutReal, colOutImag, height, inverse);

            for (int h = 0; h < height; h++)
            {
                outReal[h * width + w] = colOutReal[h];
                outImag[h * width + w] = colOutImag[h];
            }
        }

        UploadToBuffer(outputReal, outReal);
        UploadToBuffer(outputImag, outImag);
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

        var inData = DownloadBuffer(input);
        var windowData = DownloadBuffer(window);
        var outData = new float[n];

        for (int i = 0; i < n; i++)
        {
            outData[i] = inData[i] * windowData[i];
        }

        UploadToBuffer(output, outData);
    }

    /// <summary>
    /// Complex magnitude: sqrt(real² + imag²).
    /// </summary>
    public void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
    {
        ThrowIfDisposed();

        var realData = DownloadBuffer(real);
        var imagData = DownloadBuffer(imag);
        var magData = new float[n];

        for (int i = 0; i < n; i++)
        {
            magData[i] = MathF.Sqrt(realData[i] * realData[i] + imagData[i] * imagData[i]);
        }

        UploadToBuffer(magnitude, magData);
    }

    /// <summary>
    /// Complex phase: atan2(imag, real).
    /// </summary>
    public void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
    {
        ThrowIfDisposed();

        var realData = DownloadBuffer(real);
        var imagData = DownloadBuffer(imag);
        var phaseData = new float[n];

        for (int i = 0; i < n; i++)
        {
            phaseData[i] = MathF.Atan2(imagData[i], realData[i]);
        }

        UploadToBuffer(phase, phaseData);
    }

    /// <summary>
    /// Convert polar to complex.
    /// </summary>
    public void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        ThrowIfDisposed();

        var magData = DownloadBuffer(magnitude);
        var phaseData = DownloadBuffer(phase);
        var realData = new float[n];
        var imagData = new float[n];

        for (int i = 0; i < n; i++)
        {
            realData[i] = magData[i] * MathF.Cos(phaseData[i]);
            imagData[i] = magData[i] * MathF.Sin(phaseData[i]);
        }

        UploadToBuffer(real, realData);
        UploadToBuffer(imag, imagData);
    }

    /// <summary>
    /// Apply Mel filterbank.
    /// </summary>
    public void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec,
        int numFrames, int numFreqs, int nMels)
    {
        ThrowIfDisposed();

        var powerData = DownloadBuffer(powerSpec);
        var filterData = DownloadBuffer(filterbank);
        var melData = new float[numFrames * nMels];

        for (int f = 0; f < numFrames; f++)
        {
            for (int m = 0; m < nMels; m++)
            {
                float sum = 0;
                for (int k = 0; k < numFreqs; k++)
                {
                    sum += filterData[m * numFreqs + k] * powerData[f * numFreqs + k];
                }
                melData[f * nMels + m] = sum;
            }
        }

        UploadToBuffer(melSpec, melData);
    }

    /// <summary>
    /// Convert power to decibels.
    /// </summary>
    public void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb)
    {
        ThrowIfDisposed();

        var powerData = DownloadBuffer(power);
        var dbData = new float[n];

        for (int i = 0; i < n; i++)
        {
            float val = 10.0f * MathF.Log10(MathF.Max(powerData[i], 1e-10f) / refValue);
            dbData[i] = MathF.Max(val, minDb);
        }

        UploadToBuffer(db, dbData);
    }

    /// <summary>
    /// Convert decibels to power.
    /// </summary>
    public void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
    {
        ThrowIfDisposed();

        var dbData = DownloadBuffer(db);
        var powerData = new float[n];

        for (int i = 0; i < n; i++)
        {
            powerData[i] = refValue * MathF.Pow(10.0f, dbData[i] / 10.0f);
        }

        UploadToBuffer(power, powerData);
    }

    #endregion

    #region Random Number Generation

    /// <summary>
    /// Generate uniform random numbers.
    /// </summary>
    public void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed)
    {
        ThrowIfDisposed();

        var rng = new Random((int)(seed & 0x7FFFFFFF));
        var data = new float[size];

        for (int i = 0; i < size; i++)
        {
            data[i] = min + (float)rng.NextDouble() * (max - min);
        }

        UploadToBuffer(output, data);
    }

    /// <summary>
    /// Generate normal (Gaussian) random numbers using Box-Muller.
    /// </summary>
    public void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
    {
        ThrowIfDisposed();

        var rng = new Random((int)(seed & 0x7FFFFFFF));
        var data = new float[size];

        for (int i = 0; i < size; i += 2)
        {
            // Box-Muller transform
            float u1 = (float)rng.NextDouble();
            float u2 = (float)rng.NextDouble();

            u1 = MathF.Max(u1, 1e-10f);

            float r = MathF.Sqrt(-2.0f * MathF.Log(u1));
            float theta = 2.0f * MathF.PI * u2;

            data[i] = mean + stdDev * r * MathF.Cos(theta);
            if (i + 1 < size)
            {
                data[i + 1] = mean + stdDev * r * MathF.Sin(theta);
            }
        }

        UploadToBuffer(output, data);
    }

    public void GenerateSecureRandomUniform(IGpuBuffer output, int size, float min, float max)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        var data = new float[size];
        try
        {
            Helpers.SimdRandom.SecureFillFloats(data.AsSpan());
            float range = max - min;
            for (int i = 0; i < size; i++) data[i] = data[i] * range + min;
            UploadToBuffer(output, data);
        }
        finally { Array.Clear(data, 0, size); }
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

        var inputData = DownloadBuffer(input);
        var centersData = DownloadBuffer(centers);
        var epsilonsData = DownloadBuffer(epsilons);
        var outputData = new float[batchSize * numCenters];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < numCenters; c++)
            {
                float distSq = 0;
                for (int d = 0; d < inputDim; d++)
                {
                    float diff = inputData[b * inputDim + d] - centersData[c * inputDim + d];
                    distSq += diff * diff;
                }
                outputData[b * numCenters + c] = MathF.Exp(-epsilonsData[c] * distSq);
            }
        }

        UploadToBuffer(output, outputData);
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

        var weightsData = DownloadBuffer(weights);
        var preTraceData = DownloadBuffer(preTrace);
        var postTraceData = DownloadBuffer(postTrace);
        var preSpikeData = DownloadBuffer(preSpike);
        var postSpikeData = DownloadBuffer(postSpike);

        for (int i = 0; i < numPre; i++)
        {
            for (int j = 0; j < numPost; j++)
            {
                int idx = i * numPost + j;

                // LTP: pre spike when post has trace
                if (preSpikeData[i] > 0)
                {
                    weightsData[idx] += ltpRate * postTraceData[j];
                }

                // LTD: post spike when pre has trace
                if (postSpikeData[j] > 0)
                {
                    weightsData[idx] -= ltdRate * preTraceData[i];
                }

                // Homeostasis
                weightsData[idx] -= homeostasisRate * weightsData[idx];

                // Clamp weights
                weightsData[idx] = MathF.Max(minWeight, MathF.Min(maxWeight, weightsData[idx]));
            }
        }

        UploadToBuffer(weights, weightsData);
    }

    /// <summary>
    /// Update traces and detect spikes.
    /// </summary>
    public void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input,
        float decay, float threshold, int size)
    {
        ThrowIfDisposed();

        var tracesData = DownloadBuffer(traces);
        var inputData = DownloadBuffer(input);
        var spikesData = new float[size];

        for (int i = 0; i < size; i++)
        {
            // Decay trace
            tracesData[i] *= decay;

            // Add input
            tracesData[i] += inputData[i];

            // Check for spike
            if (tracesData[i] >= threshold)
            {
                spikesData[i] = 1.0f;
                tracesData[i] -= threshold;
            }
            else
            {
                spikesData[i] = 0.0f;
            }
        }

        UploadToBuffer(traces, tracesData);
        UploadToBuffer(spikes, spikesData);
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
        int total = batch * fftSize;
        if (total <= 0) return;
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
