// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Normalization, Dropout, and Embedding operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Normalization Operations

    /// <summary>
    /// Batch normalization forward pass.
    /// </summary>
    public void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
    {
        ThrowIfDisposed();

        // CPU fallback implementation
        var inputData = DownloadBuffer(input);
        var gammaData = DownloadBuffer(gamma);
        var betaData = DownloadBuffer(beta);
        var runningMeanData = DownloadBuffer(runningMean);
        var runningVarData = DownloadBuffer(runningVar);

        var outputData = new float[batch * channels * spatialSize];
        var saveMeanData = new float[channels];
        var saveInvVarData = new float[channels];

        for (int c = 0; c < channels; c++)
        {
            float mean, invVar;

            if (training)
            {
                // Compute mean
                float sum = 0;
                for (int b = 0; b < batch; b++)
                {
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = b * channels * spatialSize + c * spatialSize + s;
                        sum += inputData[idx];
                    }
                }
                mean = sum / (batch * spatialSize);

                // Compute variance
                float varSum = 0;
                for (int b = 0; b < batch; b++)
                {
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = b * channels * spatialSize + c * spatialSize + s;
                        float diff = inputData[idx] - mean;
                        varSum += diff * diff;
                    }
                }
                float variance = varSum / (batch * spatialSize);
                invVar = 1.0f / MathF.Sqrt(variance + epsilon);

                // Update running statistics
                runningMeanData[c] = (1 - momentum) * runningMeanData[c] + momentum * mean;
                runningVarData[c] = (1 - momentum) * runningVarData[c] + momentum * variance;

                saveMeanData[c] = mean;
                saveInvVarData[c] = invVar;
            }
            else
            {
                mean = runningMeanData[c];
                invVar = 1.0f / MathF.Sqrt(runningVarData[c] + epsilon);
            }

            // Normalize and scale
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = b * channels * spatialSize + c * spatialSize + s;
                    float normalized = (inputData[idx] - mean) * invVar;
                    outputData[idx] = gammaData[c] * normalized + betaData[c];
                }
            }
        }

        // Upload results
        UploadToBuffer(output, outputData);
        if (training)
        {
            UploadToBuffer(runningMean, runningMeanData);
            UploadToBuffer(runningVar, runningVarData);
            UploadToBuffer(saveMean, saveMeanData);
            UploadToBuffer(saveInvVar, saveInvVarData);
        }
    }

    /// <summary>
    /// Batch normalization backward pass.
    /// </summary>
    public void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var gammaData = DownloadBuffer(gamma);
        var saveMeanData = DownloadBuffer(saveMean);
        var saveInvVarData = DownloadBuffer(saveInvVar);

        var gradInputData = new float[batch * channels * spatialSize];
        var gradGammaData = new float[channels];
        var gradBetaData = new float[channels];

        int N = batch * spatialSize;

        for (int c = 0; c < channels; c++)
        {
            float mean = saveMeanData[c];
            float invVar = saveInvVarData[c];
            float g = gammaData[c];

            // Compute gradGamma and gradBeta
            float dgamma = 0, dbeta = 0;
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = b * channels * spatialSize + c * spatialSize + s;
                    float xhat = (inputData[idx] - mean) * invVar;
                    dgamma += gradOutputData[idx] * xhat;
                    dbeta += gradOutputData[idx];
                }
            }
            gradGammaData[c] = dgamma;
            gradBetaData[c] = dbeta;

            // Compute gradInput
            float sum1 = 0, sum2 = 0;
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = b * channels * spatialSize + c * spatialSize + s;
                    float xhat = (inputData[idx] - mean) * invVar;
                    sum1 += gradOutputData[idx];
                    sum2 += gradOutputData[idx] * xhat;
                }
            }

            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = b * channels * spatialSize + c * spatialSize + s;
                    float xhat = (inputData[idx] - mean) * invVar;
                    gradInputData[idx] = g * invVar * (gradOutputData[idx] - sum1 / N - xhat * sum2 / N);
                }
            }
        }

        UploadToBuffer(gradInput, gradInputData);
        UploadToBuffer(gradGamma, gradGammaData);
        UploadToBuffer(gradBeta, gradBetaData);
    }

    /// <summary>
    /// Layer normalization forward pass.
    /// </summary>
    public void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var gammaData = DownloadBuffer(gamma);
        var betaData = DownloadBuffer(beta);

        var outputData = new float[batchSize * normalizedSize];
        var saveMeanData = new float[batchSize];
        var saveInvVarData = new float[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            // Compute mean
            float sum = 0;
            for (int i = 0; i < normalizedSize; i++)
            {
                sum += inputData[b * normalizedSize + i];
            }
            float mean = sum / normalizedSize;

            // Compute variance
            float varSum = 0;
            for (int i = 0; i < normalizedSize; i++)
            {
                float diff = inputData[b * normalizedSize + i] - mean;
                varSum += diff * diff;
            }
            float variance = varSum / normalizedSize;
            float invVar = 1.0f / MathF.Sqrt(variance + epsilon);

            saveMeanData[b] = mean;
            saveInvVarData[b] = invVar;

            // Normalize and scale
            for (int i = 0; i < normalizedSize; i++)
            {
                int idx = b * normalizedSize + i;
                float normalized = (inputData[idx] - mean) * invVar;
                outputData[idx] = gammaData[i] * normalized + betaData[i];
            }
        }

        UploadToBuffer(output, outputData);
        UploadToBuffer(saveMean, saveMeanData);
        UploadToBuffer(saveInvVar, saveInvVarData);
    }

    /// <summary>
    /// Layer normalization backward pass.
    /// </summary>
    public void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize, float epsilon)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var gammaData = DownloadBuffer(gamma);
        var saveMeanData = DownloadBuffer(saveMean);
        var saveInvVarData = DownloadBuffer(saveInvVar);

        var gradInputData = new float[batchSize * normalizedSize];
        var gradGammaData = new float[normalizedSize];
        var gradBetaData = new float[normalizedSize];

        // Initialize gradGamma and gradBeta
        for (int i = 0; i < normalizedSize; i++)
        {
            gradGammaData[i] = 0;
            gradBetaData[i] = 0;
        }

        for (int b = 0; b < batchSize; b++)
        {
            float mean = saveMeanData[b];
            float invVar = saveInvVarData[b];

            // Accumulate gradGamma and gradBeta
            for (int i = 0; i < normalizedSize; i++)
            {
                int idx = b * normalizedSize + i;
                float xhat = (inputData[idx] - mean) * invVar;
                gradGammaData[i] += gradOutputData[idx] * xhat;
                gradBetaData[i] += gradOutputData[idx];
            }

            // Compute intermediate sums for gradInput
            float sum1 = 0, sum2 = 0;
            for (int i = 0; i < normalizedSize; i++)
            {
                int idx = b * normalizedSize + i;
                float xhat = (inputData[idx] - mean) * invVar;
                sum1 += gradOutputData[idx] * gammaData[i];
                sum2 += gradOutputData[idx] * gammaData[i] * xhat;
            }

            // Compute gradInput
            for (int i = 0; i < normalizedSize; i++)
            {
                int idx = b * normalizedSize + i;
                float xhat = (inputData[idx] - mean) * invVar;
                gradInputData[idx] = invVar * (gradOutputData[idx] * gammaData[i] - sum1 / normalizedSize - xhat * sum2 / normalizedSize);
            }
        }

        UploadToBuffer(gradInput, gradInputData);
        UploadToBuffer(gradGamma, gradGammaData);
        UploadToBuffer(gradBeta, gradBetaData);
    }

    /// <summary>
    /// Group normalization forward pass.
    /// </summary>
    public void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var gammaData = DownloadBuffer(gamma);
        var betaData = DownloadBuffer(beta);

        int channelsPerGroup = channels / numGroups;
        int groupSize = channelsPerGroup * spatialSize;

        var outputData = new float[batch * channels * spatialSize];
        var saveMeanData = new float[batch * numGroups];
        var saveInvVarData = new float[batch * numGroups];

        for (int b = 0; b < batch; b++)
        {
            for (int g = 0; g < numGroups; g++)
            {
                // Compute mean for this group
                float sum = 0;
                for (int c = 0; c < channelsPerGroup; c++)
                {
                    int channelIdx = g * channelsPerGroup + c;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = b * channels * spatialSize + channelIdx * spatialSize + s;
                        sum += inputData[idx];
                    }
                }
                float mean = sum / groupSize;

                // Compute variance
                float varSum = 0;
                for (int c = 0; c < channelsPerGroup; c++)
                {
                    int channelIdx = g * channelsPerGroup + c;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = b * channels * spatialSize + channelIdx * spatialSize + s;
                        float diff = inputData[idx] - mean;
                        varSum += diff * diff;
                    }
                }
                float variance = varSum / groupSize;
                float invVar = 1.0f / MathF.Sqrt(variance + epsilon);

                saveMeanData[b * numGroups + g] = mean;
                saveInvVarData[b * numGroups + g] = invVar;

                // Normalize and scale
                for (int c = 0; c < channelsPerGroup; c++)
                {
                    int channelIdx = g * channelsPerGroup + c;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = b * channels * spatialSize + channelIdx * spatialSize + s;
                        float normalized = (inputData[idx] - mean) * invVar;
                        outputData[idx] = gammaData[channelIdx] * normalized + betaData[channelIdx];
                    }
                }
            }
        }

        UploadToBuffer(output, outputData);
        UploadToBuffer(saveMean, saveMeanData);
        UploadToBuffer(saveInvVar, saveInvVarData);
    }

    /// <summary>
    /// Instance normalization forward pass.
    /// </summary>
    public void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var gammaData = DownloadBuffer(gamma);
        var betaData = DownloadBuffer(beta);

        var outputData = new float[batch * channels * spatialSize];
        var saveMeanData = new float[batch * channels];
        var saveInvVarData = new float[batch * channels];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                // Compute mean for this instance
                float sum = 0;
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = b * channels * spatialSize + c * spatialSize + s;
                    sum += inputData[idx];
                }
                float mean = sum / spatialSize;

                // Compute variance
                float varSum = 0;
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = b * channels * spatialSize + c * spatialSize + s;
                    float diff = inputData[idx] - mean;
                    varSum += diff * diff;
                }
                float variance = varSum / spatialSize;
                float invVar = 1.0f / MathF.Sqrt(variance + epsilon);

                saveMeanData[b * channels + c] = mean;
                saveInvVarData[b * channels + c] = invVar;

                // Normalize and scale
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = b * channels * spatialSize + c * spatialSize + s;
                    float normalized = (inputData[idx] - mean) * invVar;
                    outputData[idx] = gammaData[c] * normalized + betaData[c];
                }
            }
        }

        UploadToBuffer(output, outputData);
        UploadToBuffer(saveMean, saveMeanData);
        UploadToBuffer(saveInvVar, saveInvVarData);
    }

    /// <summary>
    /// Instance normalization backward pass.
    /// </summary>
    public void InstanceNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var gammaData = DownloadBuffer(gamma);
        var saveMeanData = DownloadBuffer(saveMean);
        var saveInvVarData = DownloadBuffer(saveInvVar);

        var gradInputData = new float[batch * channels * spatialSize];
        var gradGammaData = new float[channels];
        var gradBetaData = new float[channels];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float mean = saveMeanData[b * channels + c];
                float invVar = saveInvVarData[b * channels + c];
                float g = gammaData[c];

                // Compute gradGamma and gradBeta contributions
                float dgamma = 0, dbeta = 0;
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = b * channels * spatialSize + c * spatialSize + s;
                    float xhat = (inputData[idx] - mean) * invVar;
                    dgamma += gradOutputData[idx] * xhat;
                    dbeta += gradOutputData[idx];
                }
                gradGammaData[c] += dgamma;
                gradBetaData[c] += dbeta;

                // Compute gradInput
                float sum1 = 0, sum2 = 0;
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = b * channels * spatialSize + c * spatialSize + s;
                    float xhat = (inputData[idx] - mean) * invVar;
                    sum1 += gradOutputData[idx];
                    sum2 += gradOutputData[idx] * xhat;
                }

                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = b * channels * spatialSize + c * spatialSize + s;
                    float xhat = (inputData[idx] - mean) * invVar;
                    gradInputData[idx] = g * invVar * (gradOutputData[idx] - sum1 / spatialSize - xhat * sum2 / spatialSize);
                }
            }
        }

        UploadToBuffer(gradInput, gradInputData);
        UploadToBuffer(gradGamma, gradGammaData);
        UploadToBuffer(gradBeta, gradBetaData);
    }

    /// <summary>
    /// RMS normalization forward pass.
    /// </summary>
    public void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var gammaData = DownloadBuffer(gamma);

        var outputData = new float[batchSize * normalizedSize];
        var saveRmsData = new float[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            // Compute RMS
            float sumSq = 0;
            for (int i = 0; i < normalizedSize; i++)
            {
                float val = inputData[b * normalizedSize + i];
                sumSq += val * val;
            }
            float rms = MathF.Sqrt(sumSq / normalizedSize + epsilon);
            saveRmsData[b] = rms;

            // Normalize and scale
            for (int i = 0; i < normalizedSize; i++)
            {
                int idx = b * normalizedSize + i;
                outputData[idx] = inputData[idx] / rms * gammaData[i];
            }
        }

        UploadToBuffer(output, outputData);
        UploadToBuffer(saveRms, saveRmsData);
    }

    /// <summary>
    /// RMS normalization backward pass.
    /// </summary>
    public void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var gammaData = DownloadBuffer(gamma);
        var saveRmsData = DownloadBuffer(saveRms);

        var gradInputData = new float[batchSize * normalizedSize];
        var gradGammaData = new float[normalizedSize];

        for (int b = 0; b < batchSize; b++)
        {
            float rms = saveRmsData[b];
            float invRms = 1.0f / rms;

            // Accumulate gradGamma
            for (int i = 0; i < normalizedSize; i++)
            {
                int idx = b * normalizedSize + i;
                gradGammaData[i] += gradOutputData[idx] * inputData[idx] * invRms;
            }

            // Compute sum for gradInput
            float sum = 0;
            for (int i = 0; i < normalizedSize; i++)
            {
                int idx = b * normalizedSize + i;
                sum += gradOutputData[idx] * gammaData[i] * inputData[idx];
            }

            // Compute gradInput
            for (int i = 0; i < normalizedSize; i++)
            {
                int idx = b * normalizedSize + i;
                float gradNorm = gradOutputData[idx] * gammaData[i];
                gradInputData[idx] = invRms * (gradNorm - inputData[idx] * sum / (rms * rms * normalizedSize));
            }
        }

        UploadToBuffer(gradInput, gradInputData);
        UploadToBuffer(gradGamma, gradGammaData);
    }

    #endregion

    #region Dropout and Regularization

    /// <summary>
    /// Dropout forward pass.
    /// </summary>
    public void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var outputData = new float[size];
        var maskData = new float[size];

        if (training && dropoutRate > 0)
        {
            var rng = new Random((int)(seed & 0x7FFFFFFF));
            float scale = 1.0f / (1.0f - dropoutRate);

            for (int i = 0; i < size; i++)
            {
                if (rng.NextDouble() >= dropoutRate)
                {
                    maskData[i] = scale;
                    outputData[i] = inputData[i] * scale;
                }
                else
                {
                    maskData[i] = 0;
                    outputData[i] = 0;
                }
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                maskData[i] = 1.0f;
                outputData[i] = inputData[i];
            }
        }

        UploadToBuffer(output, outputData);
        UploadToBuffer(mask, maskData);
    }

    /// <summary>
    /// Dropout backward pass.
    /// </summary>
    public void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        var maskData = DownloadBuffer(mask);
        var gradInputData = new float[size];

        for (int i = 0; i < size; i++)
        {
            gradInputData[i] = gradOutputData[i] * maskData[i];
        }

        UploadToBuffer(gradInput, gradInputData);
    }

    #endregion

    #region Embedding Operations

    /// <summary>
    /// Embedding lookup forward pass.
    /// </summary>
    public void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
    {
        ThrowIfDisposed();

        // Download indices as int buffer
        var indicesData = DownloadIntBuffer(indices, numIndices);
        var tableData = DownloadBuffer(embeddingTable);
        var outputData = new float[numIndices * embeddingDim];

        for (int i = 0; i < numIndices; i++)
        {
            int idx = indicesData[i];
            for (int d = 0; d < embeddingDim; d++)
            {
                outputData[i * embeddingDim + d] = tableData[idx * embeddingDim + d];
            }
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// Embedding backward pass.
    /// </summary>
    public void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        var indicesData = DownloadIntBuffer(indices, numIndices);
        var gradEmbeddingData = new float[vocabSize * embeddingDim];

        // Accumulate gradients
        for (int i = 0; i < numIndices; i++)
        {
            int idx = indicesData[i];
            for (int d = 0; d < embeddingDim; d++)
            {
                gradEmbeddingData[idx * embeddingDim + d] += gradOutputData[i * embeddingDim + d];
            }
        }

        UploadToBuffer(gradEmbedding, gradEmbeddingData);
    }

    /// <summary>
    /// Downloads an integer buffer.
    /// </summary>
    private int[] DownloadIntBuffer(IGpuBuffer buffer, int size)
    {
        var floatData = DownloadBuffer(buffer);
        var intData = new int[size];
        for (int i = 0; i < size; i++)
        {
            intData[i] = SingleToInt32BitsCompat(floatData[i]);
        }
        return intData;
    }

    #endregion
}
