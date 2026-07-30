#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class OpenClComplexTopKTests : IDisposable
{
    private readonly OpenClBackend? _backend;
    private readonly Exception? _initException;

    public OpenClComplexTopKTests()
    {
        try
        {
            _backend = new OpenClBackend();
        }
        catch (Exception ex)
        {
            _initException = ex;
        }
    }

    public void Dispose() => _backend?.Dispose();

    private OpenClBackend RequireBackend()
    {
        bool ready = _backend?.IsAvailable == true;
        if (!ready && string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException("OpenCL was required but unavailable.", _initException);
        Skip.If(!ready, "OpenCL backend unavailable.");
        return _backend!;
    }

    [SkippableFact]
    public void SplitComplexTopK_IsExactAndStableForTies()
    {
        var backend = RequireBackend();
        float[] real = [3, -3, 0, 4, float.NaN, 0, 2];
        float[] imag = [4, 4, 5, 0, 0, -5, 0];
        float[] expectedReal = [3, -3, 0, 0, 0, 0, 0];
        float[] expectedImag = [4, 4, 5, 0, 0, 0, 0];

        using var inReal = backend.AllocateBuffer(real);
        using var inImag = backend.AllocateBuffer(imag);
        using var outReal = backend.AllocateBuffer(real.Length);
        using var outImag = backend.AllocateBuffer(imag.Length);

        backend.SplitComplexTopK(inReal, inImag, outReal, outImag, real.Length, 3);

        Assert.Equal(expectedReal, backend.DownloadBuffer(outReal));
        Assert.Equal(expectedImag, backend.DownloadBuffer(outImag));
    }

    [SkippableFact]
    public void SplitComplexTopK_SortsNaNMagnitudesAfterNumericValues()
    {
        var backend = RequireBackend();
        float[] real = [3, -3, 0, 4, float.NaN, 0, 2];
        float[] imag = [4, 4, 5, 0, 0, -5, 0];
        float[] expectedReal = [3, -3, 0, 4, 0, 0, 2];
        float[] expectedImag = [4, 4, 5, 0, 0, -5, 0];

        using var inReal = backend.AllocateBuffer(real);
        using var inImag = backend.AllocateBuffer(imag);
        using var outReal = backend.AllocateBuffer(real.Length);
        using var outImag = backend.AllocateBuffer(imag.Length);

        backend.SplitComplexTopK(inReal, inImag, outReal, outImag, real.Length, 6);

        Assert.Equal(expectedReal, backend.DownloadBuffer(outReal));
        Assert.Equal(expectedImag, backend.DownloadBuffer(outImag));
    }

    [SkippableTheory]
    [InlineData(false, 0)]
    [InlineData(true, 8)]
    public void UnitPhaseCodebook_IsBitIdenticalToCpu(bool kPsk, int k)
    {
        var backend = RequireBackend();
        const int seed = -1729, vocabulary = 5, dimension = 17;
        int length = vocabulary * dimension;
        var expectedReal = new float[length];
        var expectedImag = new float[length];
        SimdHrrKernels.UnitPhaseCodebookFloat(
            expectedReal, expectedImag, seed, vocabulary, dimension, kPsk, k);

        using var realBuffer = backend.AllocateBuffer(length);
        using var imagBuffer = backend.AllocateBuffer(length);
        backend.SplitComplexUnitPhaseCodebook(
            realBuffer, imagBuffer, seed, vocabulary, dimension, kPsk, k);

        Assert.Equal(expectedReal, backend.DownloadBuffer(realBuffer));
        Assert.Equal(expectedImag, backend.DownloadBuffer(imagBuffer));
    }

    [SkippableFact]
    public void HyperbolicLinear_NonsquareForwardAndBackwardMatchCommonContract()
    {
        var backend = RequireBackend();
        const int batch = 2, inputFeatures = 3, outputFeatures = 2;
        const float curvature = 0.5f, epsilon = 1e-5f;
        float[] input = [2.2f, -1.7f, 1.3f, -0.4f, 0.8f, 1.1f];
        float[] weights = [0.2f, -0.5f, 0.7f, -0.3f, 0.4f, 0.6f];
        float[] biases = [0.15f, -0.25f];
        float[] gradOutput = [0.4f, -0.6f, 0.8f, 0.3f];

        var expectedOutput = new float[batch * outputFeatures];
        for (int b = 0; b < batch; b++)
        {
            for (int o = 0; o < outputFeatures; o++)
            {
                float sum = biases[o];
                for (int i = 0; i < inputFeatures; i++)
                    sum += input[b * inputFeatures + i] * weights[o * inputFeatures + i];
                expectedOutput[b * outputFeatures + o] = sum;
            }

            float normSquared = 0f;
            for (int o = 0; o < outputFeatures; o++)
                normSquared += expectedOutput[b * outputFeatures + o] * expectedOutput[b * outputFeatures + o];
            float maximumNorm = 1f / MathF.Sqrt(curvature) - epsilon;
            if (normSquared >= maximumNorm * maximumNorm)
            {
                float scale = maximumNorm / MathF.Sqrt(normSquared);
                for (int o = 0; o < outputFeatures; o++) expectedOutput[b * outputFeatures + o] *= scale;
            }
        }

        var expectedGradInput = new float[batch * inputFeatures];
        var expectedGradWeights = new float[outputFeatures * inputFeatures];
        var expectedGradBiases = new float[outputFeatures];
        for (int b = 0; b < batch; b++)
        {
            for (int o = 0; o < outputFeatures; o++)
            {
                float gradient = gradOutput[b * outputFeatures + o];
                expectedGradBiases[o] += gradient;
                for (int i = 0; i < inputFeatures; i++)
                {
                    expectedGradInput[b * inputFeatures + i] += gradient * weights[o * inputFeatures + i];
                    expectedGradWeights[o * inputFeatures + i] += gradient * input[b * inputFeatures + i];
                }
            }
        }

        using var inputBuffer = backend.AllocateBuffer(input);
        using var weightsBuffer = backend.AllocateBuffer(weights);
        using var biasesBuffer = backend.AllocateBuffer(biases);
        using var outputBuffer = backend.AllocateBuffer(expectedOutput.Length);
        using var gradOutputBuffer = backend.AllocateBuffer(gradOutput);
        using var gradInputBuffer = backend.AllocateBuffer(expectedGradInput.Length);
        using var gradWeightsBuffer = backend.AllocateBuffer(expectedGradWeights.Length);
        using var gradBiasesBuffer = backend.AllocateBuffer(expectedGradBiases.Length);

        backend.HyperbolicLinearForward(inputBuffer, weightsBuffer, biasesBuffer, outputBuffer,
            batch, inputFeatures, outputFeatures, curvature, epsilon);
        backend.HyperbolicLinearBackwardInput(gradOutputBuffer, inputBuffer, weightsBuffer, gradInputBuffer,
            batch, inputFeatures, outputFeatures, curvature);
        backend.HyperbolicLinearBackwardWeights(gradOutputBuffer, inputBuffer, gradWeightsBuffer,
            batch, inputFeatures, outputFeatures, curvature);
        backend.HyperbolicLinearBackwardBiases(gradOutputBuffer, inputBuffer, gradBiasesBuffer,
            batch, inputFeatures, outputFeatures, curvature);

        AssertClose(expectedOutput, backend.DownloadBuffer(outputBuffer));
        AssertClose(expectedGradInput, backend.DownloadBuffer(gradInputBuffer));
        AssertClose(expectedGradWeights, backend.DownloadBuffer(gradWeightsBuffer));
        AssertClose(expectedGradBiases, backend.DownloadBuffer(gradBiasesBuffer));
    }

    [SkippableFact]
    public void ScaledDotProductAttention_MultiBatchMaskedForwardAndBackwardMatchReference()
    {
        var backend = RequireBackend();
        const int batch = 2, heads = 2, sequence = 3, dimension = 2;
        const float scale = 0.5f;
        int tensorSize = batch * heads * sequence * dimension;
        int weightsSize = batch * heads * sequence * sequence;
        float[] query = CreateSequence(tensorSize, 0.07f, -0.35f);
        float[] key = CreateSequence(tensorSize, -0.05f, 0.4f);
        float[] value = CreateSequence(tensorSize, 0.09f, -0.2f);
        float[] gradOutput = CreateSequence(tensorSize, -0.04f, 0.3f);
        float[] mask =
        [
            1f, 1f, 0f,
            1f, 1f, 0f,
            1f, 1f, 1f
        ];

        ReferenceAttention(query, key, value, gradOutput, mask, batch, heads, sequence, dimension, scale,
            out float[] expectedOutput, out float[] expectedWeights,
            out float[] expectedGradQuery, out float[] expectedGradKey, out float[] expectedGradValue);

        using var queryBuffer = backend.AllocateBuffer(query);
        using var keyBuffer = backend.AllocateBuffer(key);
        using var valueBuffer = backend.AllocateBuffer(value);
        using var maskBuffer = backend.AllocateBuffer(mask);
        using var outputBuffer = backend.AllocateBuffer(tensorSize);
        using var weightsBuffer = backend.AllocateBuffer(weightsSize);
        using var gradOutputBuffer = backend.AllocateBuffer(gradOutput);
        using var gradQueryBuffer = backend.AllocateBuffer(tensorSize);
        using var gradKeyBuffer = backend.AllocateBuffer(tensorSize);
        using var gradValueBuffer = backend.AllocateBuffer(tensorSize);

        backend.ScaledDotProductAttention(queryBuffer, keyBuffer, valueBuffer, outputBuffer, weightsBuffer, maskBuffer,
            batch, heads, sequence, sequence, dimension, scale, isCausal: true);
        backend.ScaledDotProductAttentionBackward(gradOutputBuffer, queryBuffer, keyBuffer, valueBuffer, weightsBuffer,
            gradQueryBuffer, gradKeyBuffer, gradValueBuffer, batch, heads, sequence, sequence, dimension, scale, isCausal: true);

        AssertClose(expectedOutput, backend.DownloadBuffer(outputBuffer));
        AssertClose(expectedWeights, backend.DownloadBuffer(weightsBuffer));
        AssertClose(expectedGradQuery, backend.DownloadBuffer(gradQueryBuffer));
        AssertClose(expectedGradKey, backend.DownloadBuffer(gradKeyBuffer));
        AssertClose(expectedGradValue, backend.DownloadBuffer(gradValueBuffer));
    }

    [SkippableFact]
    public void ScaledDotProductAttention_CrossAttentionUsesDistinctQueryAndKeyLengths()
    {
        var backend = RequireBackend();
        const int batch = 2, heads = 2, seqQ = 2, seqK = 3, dimension = 3;
        const float scale = 0.75f;
        float[] query = CreateSequence(batch * heads * seqQ * dimension, 0.03f, -0.2f);
        float[] key = CreateSequence(batch * heads * seqK * dimension, -0.025f, 0.5f);
        float[] value = CreateSequence(batch * heads * seqK * dimension, 0.04f, -0.1f);
        float[] gradOutput = CreateSequence(batch * heads * seqQ * dimension, -0.02f, 0.3f);
        ReferenceCrossAttentionForward(query, key, value, batch, heads, seqQ, seqK, dimension, scale,
            out float[] expectedOutput, out float[] expectedWeights);
        ReferenceCrossAttentionBackward(query, key, value, gradOutput, expectedWeights,
            batch, heads, seqQ, seqK, dimension, scale,
            out float[] expectedGradQuery, out float[] expectedGradKey, out float[] expectedGradValue);

        using var queryBuffer = backend.AllocateBuffer(query);
        using var keyBuffer = backend.AllocateBuffer(key);
        using var valueBuffer = backend.AllocateBuffer(value);
        using var gradOutputBuffer = backend.AllocateBuffer(gradOutput);
        using var outputBuffer = backend.AllocateBuffer(expectedOutput.Length);
        using var weightsBuffer = backend.AllocateBuffer(expectedWeights.Length);
        using var gradQueryBuffer = backend.AllocateBuffer(expectedGradQuery.Length);
        using var gradKeyBuffer = backend.AllocateBuffer(expectedGradKey.Length);
        using var gradValueBuffer = backend.AllocateBuffer(expectedGradValue.Length);

        backend.ScaledDotProductAttention(queryBuffer, keyBuffer, valueBuffer, outputBuffer, weightsBuffer, null,
            batch, heads, seqQ, seqK, dimension, scale, isCausal: false);
        backend.ScaledDotProductAttentionBackward(gradOutputBuffer, queryBuffer, keyBuffer, valueBuffer, weightsBuffer,
            gradQueryBuffer, gradKeyBuffer, gradValueBuffer,
            batch, heads, seqQ, seqK, dimension, scale, isCausal: false);

        AssertClose(expectedOutput, backend.DownloadBuffer(outputBuffer));
        AssertClose(expectedWeights, backend.DownloadBuffer(weightsBuffer));
        AssertClose(expectedGradQuery, backend.DownloadBuffer(gradQueryBuffer));
        AssertClose(expectedGradKey, backend.DownloadBuffer(gradKeyBuffer));
        AssertClose(expectedGradValue, backend.DownloadBuffer(gradValueBuffer));
    }

    private static float[] CreateSequence(int length, float step, float offset)
    {
        var result = new float[length];
        for (int i = 0; i < result.Length; i++) result[i] = offset + step * i;
        return result;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 2e-5f,
                $"Mismatch at {i}: expected {expected[i]}, actual {actual[i]}");
    }

    private static void ReferenceAttention(float[] query, float[] key, float[] value, float[] gradOutput,
        float[] mask, int batch, int heads, int sequence, int dimension, float scale,
        out float[] output, out float[] weights, out float[] gradQuery, out float[] gradKey, out float[] gradValue)
    {
        output = new float[query.Length];
        weights = new float[batch * heads * sequence * sequence];
        gradQuery = new float[query.Length];
        gradKey = new float[key.Length];
        gradValue = new float[value.Length];

        for (int bh = 0; bh < batch * heads; bh++)
        {
            int tensorBase = bh * sequence * dimension;
            int weightsBase = bh * sequence * sequence;
            for (int qi = 0; qi < sequence; qi++)
            {
                float maximum = float.NegativeInfinity;
                for (int ki = 0; ki < sequence; ki++)
                {
                    if (ki > qi || mask[qi * sequence + ki] == 0f) continue;
                    float score = 0f;
                    for (int d = 0; d < dimension; d++)
                        score += query[tensorBase + qi * dimension + d] * key[tensorBase + ki * dimension + d];
                    maximum = MathF.Max(maximum, score * scale);
                }

                float denominator = 0f;
                for (int ki = 0; ki < sequence; ki++)
                {
                    if (ki > qi || mask[qi * sequence + ki] == 0f) continue;
                    float score = 0f;
                    for (int d = 0; d < dimension; d++)
                        score += query[tensorBase + qi * dimension + d] * key[tensorBase + ki * dimension + d];
                    float weight = MathF.Exp(score * scale - maximum);
                    weights[weightsBase + qi * sequence + ki] = weight;
                    denominator += weight;
                }

                for (int ki = 0; ki < sequence; ki++)
                    weights[weightsBase + qi * sequence + ki] /= denominator;
                for (int d = 0; d < dimension; d++)
                    for (int ki = 0; ki < sequence; ki++)
                        output[tensorBase + qi * dimension + d] +=
                            weights[weightsBase + qi * sequence + ki] * value[tensorBase + ki * dimension + d];
            }

            for (int qi = 0; qi < sequence; qi++)
            {
                float softmaxDot = 0f;
                for (int ki = 0; ki < sequence; ki++)
                {
                    float gradWeight = 0f;
                    for (int d = 0; d < dimension; d++)
                        gradWeight += gradOutput[tensorBase + qi * dimension + d] * value[tensorBase + ki * dimension + d];
                    softmaxDot += weights[weightsBase + qi * sequence + ki] * gradWeight;
                }

                for (int ki = 0; ki < sequence; ki++)
                {
                    float gradWeight = 0f;
                    for (int d = 0; d < dimension; d++)
                        gradWeight += gradOutput[tensorBase + qi * dimension + d] * value[tensorBase + ki * dimension + d];
                    float weight = weights[weightsBase + qi * sequence + ki];
                    float gradScore = weight * (gradWeight - softmaxDot) * scale;
                    for (int d = 0; d < dimension; d++)
                    {
                        gradQuery[tensorBase + qi * dimension + d] += gradScore * key[tensorBase + ki * dimension + d];
                        gradKey[tensorBase + ki * dimension + d] += gradScore * query[tensorBase + qi * dimension + d];
                        gradValue[tensorBase + ki * dimension + d] += weight * gradOutput[tensorBase + qi * dimension + d];
                    }
                }
            }
        }
    }

    private static void ReferenceCrossAttentionForward(float[] query, float[] key, float[] value,
        int batch, int heads, int seqQ, int seqK, int dimension, float scale,
        out float[] output, out float[] weights)
    {
        output = new float[batch * heads * seqQ * dimension];
        weights = new float[batch * heads * seqQ * seqK];
        for (int bh = 0; bh < batch * heads; bh++)
        {
            int queryBase = bh * seqQ * dimension;
            int keyBase = bh * seqK * dimension;
            int weightsBase = bh * seqQ * seqK;
            for (int qi = 0; qi < seqQ; qi++)
            {
                float maximum = float.NegativeInfinity;
                for (int ki = 0; ki < seqK; ki++)
                {
                    float score = 0f;
                    for (int d = 0; d < dimension; d++)
                        score += query[queryBase + qi * dimension + d] * key[keyBase + ki * dimension + d];
                    maximum = MathF.Max(maximum, score * scale);
                }

                float denominator = 0f;
                for (int ki = 0; ki < seqK; ki++)
                {
                    float score = 0f;
                    for (int d = 0; d < dimension; d++)
                        score += query[queryBase + qi * dimension + d] * key[keyBase + ki * dimension + d];
                    float weight = MathF.Exp(score * scale - maximum);
                    weights[weightsBase + qi * seqK + ki] = weight;
                    denominator += weight;
                }

                for (int ki = 0; ki < seqK; ki++)
                {
                    float weight = weights[weightsBase + qi * seqK + ki] / denominator;
                    weights[weightsBase + qi * seqK + ki] = weight;
                    for (int d = 0; d < dimension; d++)
                        output[queryBase + qi * dimension + d] += weight * value[keyBase + ki * dimension + d];
                }
            }
        }
    }

    private static void ReferenceCrossAttentionBackward(float[] query, float[] key, float[] value,
        float[] gradOutput, float[] weights, int batch, int heads, int seqQ, int seqK,
        int dimension, float scale, out float[] gradQuery, out float[] gradKey, out float[] gradValue)
    {
        gradQuery = new float[query.Length];
        gradKey = new float[key.Length];
        gradValue = new float[value.Length];
        for (int bh = 0; bh < batch * heads; bh++)
        {
            int queryBase = bh * seqQ * dimension;
            int keyBase = bh * seqK * dimension;
            int weightsBase = bh * seqQ * seqK;
            for (int qi = 0; qi < seqQ; qi++)
            {
                float softmaxDot = 0f;
                for (int ki = 0; ki < seqK; ki++)
                {
                    float gradWeight = 0f;
                    for (int d = 0; d < dimension; d++)
                        gradWeight += gradOutput[queryBase + qi * dimension + d] * value[keyBase + ki * dimension + d];
                    softmaxDot += weights[weightsBase + qi * seqK + ki] * gradWeight;
                }

                for (int ki = 0; ki < seqK; ki++)
                {
                    float gradWeight = 0f;
                    for (int d = 0; d < dimension; d++)
                        gradWeight += gradOutput[queryBase + qi * dimension + d] * value[keyBase + ki * dimension + d];
                    float weight = weights[weightsBase + qi * seqK + ki];
                    float gradScore = weight * (gradWeight - softmaxDot) * scale;
                    for (int d = 0; d < dimension; d++)
                    {
                        gradQuery[queryBase + qi * dimension + d] += gradScore * key[keyBase + ki * dimension + d];
                        gradKey[keyBase + ki * dimension + d] += gradScore * query[queryBase + qi * dimension + d];
                        gradValue[keyBase + ki * dimension + d] += weight * gradOutput[queryBase + qi * dimension + d];
                    }
                }
            }
        }
    }
}

#endif
