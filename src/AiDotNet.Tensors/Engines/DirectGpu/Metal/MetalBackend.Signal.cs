// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - FFT, Signal Processing, and RNG operations.

using AiDotNet.Tensors.Helpers;

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
        // Copy input to output for in-place computation
        Array.Copy(inReal, outReal, n);
        Array.Copy(inImag, outImag, n);

        // Bit-reversal permutation
        int bits = (int)MathHelper.Log2((float)n);
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
}
