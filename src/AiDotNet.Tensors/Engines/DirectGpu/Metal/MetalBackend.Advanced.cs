// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Hyperbolic Geometry, Octonion Algebra, and Quantum Computing operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Hyperbolic Geometry Operations (Poincare Ball Model)

    /// <summary>
    /// Project points to Poincare ball.
    /// </summary>
    public void PoincareProject(IGpuBuffer input, IGpuBuffer output, int batchSize, int dim, float curvature, float epsilon = 1e-5f)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var outputData = new float[batchSize * dim];

        float maxNorm = 1.0f / MathF.Sqrt(curvature) - epsilon;

        for (int b = 0; b < batchSize; b++)
        {
            float norm = 0;
            for (int d = 0; d < dim; d++)
            {
                norm += inputData[b * dim + d] * inputData[b * dim + d];
            }
            norm = MathF.Sqrt(norm);

            float scale = norm > maxNorm ? maxNorm / norm : 1.0f;
            for (int d = 0; d < dim; d++)
            {
                outputData[b * dim + d] = inputData[b * dim + d] * scale;
            }
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// Mobius addition in Poincare ball.
    /// </summary>
    public void MobiusAdd(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        ThrowIfDisposed();

        var xData = DownloadBuffer(x);
        var yData = DownloadBuffer(y);
        var outputData = new float[batchSize * dim];

        for (int b = 0; b < batchSize; b++)
        {
            float xNormSq = 0, yNormSq = 0, xyDot = 0;
            for (int d = 0; d < dim; d++)
            {
                float xv = xData[b * dim + d];
                float yv = yData[b * dim + d];
                xNormSq += xv * xv;
                yNormSq += yv * yv;
                xyDot += xv * yv;
            }

            float c = curvature;
            float numerator1 = (1 + 2 * c * xyDot + c * yNormSq);
            float numerator2 = (1 - c * xNormSq);
            float denominator = 1 + 2 * c * xyDot + c * c * xNormSq * yNormSq;

            for (int d = 0; d < dim; d++)
            {
                outputData[b * dim + d] = (numerator1 * xData[b * dim + d] + numerator2 * yData[b * dim + d]) / denominator;
            }
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// Poincare exponential map.
    /// </summary>
    public void PoincareExpMap(IGpuBuffer basePoint, IGpuBuffer tangentVec, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        ThrowIfDisposed();

        var baseData = DownloadBuffer(basePoint);
        var tangentData = DownloadBuffer(tangentVec);
        var outputData = new float[batchSize * dim];

        for (int b = 0; b < batchSize; b++)
        {
            float baseNormSq = 0, tangentNorm = 0;
            for (int d = 0; d < dim; d++)
            {
                baseNormSq += baseData[b * dim + d] * baseData[b * dim + d];
                tangentNorm += tangentData[b * dim + d] * tangentData[b * dim + d];
            }
            tangentNorm = MathF.Sqrt(tangentNorm);

            float c = curvature;
            float lambda = 2.0f / (1 - c * baseNormSq);
            float sqrtC = MathF.Sqrt(c);

            if (tangentNorm < 1e-10f)
            {
                Array.Copy(baseData, b * dim, outputData, b * dim, dim);
            }
            else
            {
                float coeff = MathF.Tanh(sqrtC * lambda * tangentNorm / 2) / (sqrtC * tangentNorm);
                var direction = new float[dim];
                for (int d = 0; d < dim; d++)
                {
                    direction[d] = tangentData[b * dim + d] * coeff;
                }

                // Mobius addition of base and direction
                float dirNormSq = 0, baseDirDot = 0;
                for (int d = 0; d < dim; d++)
                {
                    dirNormSq += direction[d] * direction[d];
                    baseDirDot += baseData[b * dim + d] * direction[d];
                }

                float num1 = 1 + 2 * c * baseDirDot + c * dirNormSq;
                float num2 = 1 - c * baseNormSq;
                float denom = 1 + 2 * c * baseDirDot + c * c * baseNormSq * dirNormSq;

                for (int d = 0; d < dim; d++)
                {
                    outputData[b * dim + d] = (num1 * baseData[b * dim + d] + num2 * direction[d]) / denom;
                }
            }
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// Poincare distance.
    /// </summary>
    public void PoincareDistance(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        ThrowIfDisposed();

        var xData = DownloadBuffer(x);
        var yData = DownloadBuffer(y);
        var outputData = new float[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            float xNormSq = 0, yNormSq = 0, diffNormSq = 0;
            for (int d = 0; d < dim; d++)
            {
                float xv = xData[b * dim + d];
                float yv = yData[b * dim + d];
                xNormSq += xv * xv;
                yNormSq += yv * yv;
                float diff = xv - yv;
                diffNormSq += diff * diff;
            }

            float c = curvature;
            float sqrtC = MathF.Sqrt(c);
            float num = 2 * c * diffNormSq;
            float denom = (1 - c * xNormSq) * (1 - c * yNormSq);
            float delta = num / denom;

            outputData[b] = 2 / sqrtC * Artanh(sqrtC * MathF.Sqrt(delta / (1 + delta)));
        }

        UploadToBuffer(output, outputData);
    }

    private static float Artanh(float x)
    {
        return 0.5f * MathF.Log((1 + x) / (1 - x));
    }

    /// <summary>
    /// Hyperbolic linear forward pass.
    /// </summary>
    public void HyperbolicLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var weightsData = DownloadBuffer(weights);
        var biasesData = DownloadBuffer(biases);
        var outputData = new float[batchSize * outputFeatures];

        for (int b = 0; b < batchSize; b++)
        {
            // First apply linear transformation in tangent space
            var linearOut = new float[outputFeatures];
            for (int o = 0; o < outputFeatures; o++)
            {
                float sum = 0;
                for (int i = 0; i < inputFeatures; i++)
                {
                    sum += weightsData[o * inputFeatures + i] * inputData[b * inputFeatures + i];
                }
                linearOut[o] = sum;
            }

            // Add bias in hyperbolic space using Mobius addition
            // For simplicity, using Euclidean bias addition
            for (int o = 0; o < outputFeatures; o++)
            {
                outputData[b * outputFeatures + o] = linearOut[o] + biasesData[o];
            }
        }

        // Project to Poincare ball
        var tempOutput = AllocateBuffer(outputData);
        PoincareProject(tempOutput, output, batchSize, outputFeatures, curvature, epsilon);
        ((MetalGpuBuffer)tempOutput).Dispose();
    }

    /// <summary>
    /// Hyperbolic linear backward for input.
    /// </summary>
    public void HyperbolicLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        ThrowIfDisposed();

        var gradOutData = DownloadBuffer(gradOutput);
        var weightsData = DownloadBuffer(weights);
        var gradInputData = new float[batchSize * inputFeatures];

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputFeatures; i++)
            {
                float sum = 0;
                for (int o = 0; o < outputFeatures; o++)
                {
                    sum += gradOutData[b * outputFeatures + o] * weightsData[o * inputFeatures + i];
                }
                gradInputData[b * inputFeatures + i] = sum;
            }
        }

        UploadToBuffer(gradInput, gradInputData);
    }

    /// <summary>
    /// Hyperbolic linear backward for weights.
    /// </summary>
    public void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        ThrowIfDisposed();

        var gradOutData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var gradWeightsData = new float[outputFeatures * inputFeatures];

        for (int o = 0; o < outputFeatures; o++)
        {
            for (int i = 0; i < inputFeatures; i++)
            {
                float sum = 0;
                for (int b = 0; b < batchSize; b++)
                {
                    sum += gradOutData[b * outputFeatures + o] * inputData[b * inputFeatures + i];
                }
                gradWeightsData[o * inputFeatures + i] = sum;
            }
        }

        UploadToBuffer(gradWeights, gradWeightsData);
    }

    /// <summary>
    /// Hyperbolic linear backward for biases.
    /// </summary>
    public void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        ThrowIfDisposed();

        var gradOutData = DownloadBuffer(gradOutput);
        var gradBiasesData = new float[outputFeatures];

        for (int o = 0; o < outputFeatures; o++)
        {
            float sum = 0;
            for (int b = 0; b < batchSize; b++)
            {
                sum += gradOutData[b * outputFeatures + o];
            }
            // Bias gradient is the sum over batch for each output feature
            gradBiasesData[o] = sum;
        }

        UploadToBuffer(gradBiases, gradBiasesData);
    }

    #endregion

    #region Octonion Algebra Operations

    /// <summary>
    /// Octonion multiplication using Cayley-Dickson rules.
    /// </summary>
    public void OctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(a);
        var bData = DownloadBuffer(b);
        var outputData = new float[count * 8];

        // Octonion multiplication table
        for (int n = 0; n < count; n++)
        {
            int offset = n * 8;
            float a0 = aData[offset], a1 = aData[offset + 1], a2 = aData[offset + 2], a3 = aData[offset + 3];
            float a4 = aData[offset + 4], a5 = aData[offset + 5], a6 = aData[offset + 6], a7 = aData[offset + 7];
            float b0 = bData[offset], b1 = bData[offset + 1], b2 = bData[offset + 2], b3 = bData[offset + 3];
            float b4 = bData[offset + 4], b5 = bData[offset + 5], b6 = bData[offset + 6], b7 = bData[offset + 7];

            // Cayley-Dickson construction multiplication
            outputData[offset] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4 - a5 * b5 - a6 * b6 - a7 * b7;
            outputData[offset + 1] = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2 + a4 * b5 - a5 * b4 - a6 * b7 + a7 * b6;
            outputData[offset + 2] = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1 + a4 * b6 + a5 * b7 - a6 * b4 - a7 * b5;
            outputData[offset + 3] = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0 + a4 * b7 - a5 * b6 + a6 * b5 - a7 * b4;
            outputData[offset + 4] = a0 * b4 - a1 * b5 - a2 * b6 - a3 * b7 + a4 * b0 + a5 * b1 + a6 * b2 + a7 * b3;
            outputData[offset + 5] = a0 * b5 + a1 * b4 - a2 * b7 + a3 * b6 - a4 * b1 + a5 * b0 - a6 * b3 + a7 * b2;
            outputData[offset + 6] = a0 * b6 + a1 * b7 + a2 * b4 - a3 * b5 - a4 * b2 + a5 * b3 + a6 * b0 - a7 * b1;
            outputData[offset + 7] = a0 * b7 - a1 * b6 + a2 * b5 + a3 * b4 - a4 * b3 - a5 * b2 + a6 * b1 + a7 * b0;
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// Octonion addition.
    /// </summary>
    public void OctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(a);
        var bData = DownloadBuffer(b);
        var outputData = new float[count * 8];

        for (int i = 0; i < count * 8; i++)
        {
            outputData[i] = aData[i] + bData[i];
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// Octonion linear forward pass.
    /// </summary>
    public void OctonionLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var weightsData = DownloadBuffer(weights);
        var biasesData = DownloadBuffer(biases);
        var outputData = new float[batchSize * outputFeatures * 8];

        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < outputFeatures; o++)
            {
                // Initialize with bias
                for (int c = 0; c < 8; c++)
                {
                    outputData[(b * outputFeatures + o) * 8 + c] = biasesData[o * 8 + c];
                }

                // Accumulate octonion products
                for (int i = 0; i < inputFeatures; i++)
                {
                    var prod = new float[8];
                    int inputOffset = (b * inputFeatures + i) * 8;
                    int weightOffset = (o * inputFeatures + i) * 8;

                    // Extract octonions
                    float a0 = inputData[inputOffset], a1 = inputData[inputOffset + 1];
                    float a2 = inputData[inputOffset + 2], a3 = inputData[inputOffset + 3];
                    float a4 = inputData[inputOffset + 4], a5 = inputData[inputOffset + 5];
                    float a6 = inputData[inputOffset + 6], a7 = inputData[inputOffset + 7];

                    float w0 = weightsData[weightOffset], w1 = weightsData[weightOffset + 1];
                    float w2 = weightsData[weightOffset + 2], w3 = weightsData[weightOffset + 3];
                    float w4 = weightsData[weightOffset + 4], w5 = weightsData[weightOffset + 5];
                    float w6 = weightsData[weightOffset + 6], w7 = weightsData[weightOffset + 7];

                    // Octonion product
                    prod[0] = a0 * w0 - a1 * w1 - a2 * w2 - a3 * w3 - a4 * w4 - a5 * w5 - a6 * w6 - a7 * w7;
                    prod[1] = a0 * w1 + a1 * w0 + a2 * w3 - a3 * w2 + a4 * w5 - a5 * w4 - a6 * w7 + a7 * w6;
                    prod[2] = a0 * w2 - a1 * w3 + a2 * w0 + a3 * w1 + a4 * w6 + a5 * w7 - a6 * w4 - a7 * w5;
                    prod[3] = a0 * w3 + a1 * w2 - a2 * w1 + a3 * w0 + a4 * w7 - a5 * w6 + a6 * w5 - a7 * w4;
                    prod[4] = a0 * w4 - a1 * w5 - a2 * w6 - a3 * w7 + a4 * w0 + a5 * w1 + a6 * w2 + a7 * w3;
                    prod[5] = a0 * w5 + a1 * w4 - a2 * w7 + a3 * w6 - a4 * w1 + a5 * w0 - a6 * w3 + a7 * w2;
                    prod[6] = a0 * w6 + a1 * w7 + a2 * w4 - a3 * w5 - a4 * w2 + a5 * w3 + a6 * w0 - a7 * w1;
                    prod[7] = a0 * w7 - a1 * w6 + a2 * w5 + a3 * w4 - a4 * w3 - a5 * w2 + a6 * w1 + a7 * w0;

                    // Accumulate
                    for (int c = 0; c < 8; c++)
                    {
                        outputData[(b * outputFeatures + o) * 8 + c] += prod[c];
                    }
                }
            }
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// Octonion linear backward for input.
    /// </summary>
    public void OctonionLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        ThrowIfDisposed();

        var gradOutData = DownloadBuffer(gradOutput);
        var weightsData = DownloadBuffer(weights);
        var gradInputData = new float[batchSize * inputFeatures * 8];

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputFeatures; i++)
            {
                for (int c = 0; c < 8; c++)
                {
                    float sum = 0;
                    for (int o = 0; o < outputFeatures; o++)
                    {
                        // Simplified gradient computation
                        sum += gradOutData[(b * outputFeatures + o) * 8 + c] * weightsData[(o * inputFeatures + i) * 8 + c];
                    }
                    gradInputData[(b * inputFeatures + i) * 8 + c] = sum;
                }
            }
        }

        UploadToBuffer(gradInput, gradInputData);
    }

    /// <summary>
    /// Octonion linear backward for weights.
    /// </summary>
    public void OctonionLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        ThrowIfDisposed();

        var gradOutData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var gradWeightsData = new float[outputFeatures * inputFeatures * 8];

        for (int o = 0; o < outputFeatures; o++)
        {
            for (int i = 0; i < inputFeatures; i++)
            {
                for (int c = 0; c < 8; c++)
                {
                    float sum = 0;
                    for (int b = 0; b < batchSize; b++)
                    {
                        sum += gradOutData[(b * outputFeatures + o) * 8 + c] * inputData[(b * inputFeatures + i) * 8 + c];
                    }
                    gradWeightsData[(o * inputFeatures + i) * 8 + c] = sum;
                }
            }
        }

        UploadToBuffer(gradWeights, gradWeightsData);
    }

    /// <summary>
    /// Octonion linear backward for biases.
    /// </summary>
    public void OctonionLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer gradBiases, int batchSize, int outputFeatures)
    {
        ThrowIfDisposed();

        var gradOutData = DownloadBuffer(gradOutput);
        var gradBiasesData = new float[outputFeatures * 8];

        for (int o = 0; o < outputFeatures; o++)
        {
            for (int c = 0; c < 8; c++)
            {
                float sum = 0;
                for (int b = 0; b < batchSize; b++)
                {
                    sum += gradOutData[(b * outputFeatures + o) * 8 + c];
                }
                gradBiasesData[o * 8 + c] = sum;
            }
        }

        UploadToBuffer(gradBiases, gradBiasesData);
    }

    #endregion

    #region Quantum Computing Operations

    /// <summary>
    /// Quantum measurement: probabilities = |amplitude|^2.
    /// </summary>
    public void QuantumMeasurement(IGpuBuffer realPart, IGpuBuffer imagPart, IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        ThrowIfDisposed();

        var realData = DownloadBuffer(realPart);
        var imagData = DownloadBuffer(imagPart);
        var probData = new float[batchSize * stateSize];

        for (int i = 0; i < batchSize * stateSize; i++)
        {
            probData[i] = realData[i] * realData[i] + imagData[i] * imagData[i];
        }

        UploadToBuffer(probabilities, probData);
    }

    /// <summary>
    /// Normalize probabilities to sum to 1.
    /// </summary>
    public void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        ThrowIfDisposed();

        var probData = DownloadBuffer(probabilities);

        for (int b = 0; b < batchSize; b++)
        {
            float sum = 0;
            for (int s = 0; s < stateSize; s++)
            {
                sum += probData[b * stateSize + s];
            }

            if (sum > 0)
            {
                for (int s = 0; s < stateSize; s++)
                {
                    probData[b * stateSize + s] /= sum;
                }
            }
        }

        UploadToBuffer(probabilities, probData);
    }

    /// <summary>
    /// Complex matrix-vector multiplication.
    /// </summary>
    public void ComplexMatVec(IGpuBuffer matReal, IGpuBuffer matImag, IGpuBuffer vecReal, IGpuBuffer vecImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batchSize, int dim)
    {
        ThrowIfDisposed();

        var mReal = DownloadBuffer(matReal);
        var mImag = DownloadBuffer(matImag);
        var vReal = DownloadBuffer(vecReal);
        var vImag = DownloadBuffer(vecImag);
        var oReal = new float[batchSize * dim];
        var oImag = new float[batchSize * dim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < dim; i++)
            {
                float sumReal = 0, sumImag = 0;
                for (int j = 0; j < dim; j++)
                {
                    int mIdx = i * dim + j;
                    int vIdx = b * dim + j;
                    // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                    sumReal += mReal[mIdx] * vReal[vIdx] - mImag[mIdx] * vImag[vIdx];
                    sumImag += mReal[mIdx] * vImag[vIdx] + mImag[mIdx] * vReal[vIdx];
                }
                oReal[b * dim + i] = sumReal;
                oImag[b * dim + i] = sumImag;
            }
        }

        UploadToBuffer(outReal, oReal);
        UploadToBuffer(outImag, oImag);
    }

    /// <summary>
    /// Apply quantum rotation gates (Ry rotation).
    /// </summary>
    public void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
        IGpuBuffer angles, int numQubits, int batchSize)
    {
        ThrowIfDisposed();

        int dim = 1 << numQubits;
        var sReal = DownloadBuffer(stateReal);
        var sImag = DownloadBuffer(stateImag);
        var anglesData = DownloadBuffer(angles);
        var oReal = new float[batchSize * dim];
        var oImag = new float[batchSize * dim];

        Array.Copy(sReal, oReal, batchSize * dim);
        Array.Copy(sImag, oImag, batchSize * dim);

        // Apply Ry rotation to each qubit
        for (int q = 0; q < numQubits; q++)
        {
            float theta = anglesData[q];
            float cosT = MathF.Cos(theta / 2);
            float sinT = MathF.Sin(theta / 2);

            int stride = 1 << q;
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < dim; i += 2 * stride)
                {
                    for (int j = 0; j < stride; j++)
                    {
                        int idx0 = b * dim + i + j;
                        int idx1 = b * dim + i + j + stride;

                        float r0 = oReal[idx0], i0 = oImag[idx0];
                        float r1 = oReal[idx1], i1 = oImag[idx1];

                        oReal[idx0] = cosT * r0 - sinT * r1;
                        oImag[idx0] = cosT * i0 - sinT * i1;
                        oReal[idx1] = sinT * r0 + cosT * r1;
                        oImag[idx1] = sinT * i0 + cosT * i1;
                    }
                }
            }
        }

        UploadToBuffer(outReal, oReal);
        UploadToBuffer(outImag, oImag);
    }

    /// <summary>
    /// Measurement layer forward with interleaved input.
    /// </summary>
    public void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var outputData = new float[batchSize * stateSize];

        for (int b = 0; b < batchSize; b++)
        {
            float sum = 0;
            for (int s = 0; s < stateSize; s++)
            {
                int idx = (b * stateSize + s) * 2;
                float real = inputData[idx];
                float imag = inputData[idx + 1];
                float prob = real * real + imag * imag;
                outputData[b * stateSize + s] = prob;
                sum += prob;
            }

            // Normalize
            if (sum > 0)
            {
                for (int s = 0; s < stateSize; s++)
                {
                    outputData[b * stateSize + s] /= sum;
                }
            }
        }

        UploadToBuffer(output, outputData);
    }

    #endregion
}
