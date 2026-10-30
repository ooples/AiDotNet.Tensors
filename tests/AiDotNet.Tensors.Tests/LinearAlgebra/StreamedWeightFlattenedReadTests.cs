// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// AiDotNet#1767: a full-scale (661M-parameter) eager training step threw "Source array was not long
/// enough" from <c>TensorBase.GetFlattenedData</c>, called by <c>CpuEngine.LayerNormBackward</c> on the
/// LayerNorm gamma. The model had auto-enabled weight streaming, so gamma was a registered streaming
/// weight whose resident copy the pool had dropped: empty storage, logical Length unchanged.
/// GetFlattenedData was the one engine read accessor that did not page a dropped weight back in.
/// </summary>
[Collection("WeightRegistry")]
public class StreamedWeightFlattenedReadTests
{
    private const int Features = 64;
    private const int Rows = 3;

    [Fact]
    public void LayerNormBackward_ReadsADroppedStreamingGamma_AsTheResidentOneWould()
    {
        var engine = new CpuEngine();
        var input = Fill(new Tensor<float>(new[] { Rows, Features }), i => (float)Math.Sin(i * 0.37) * 2f);
        var gradOutput = Fill(new Tensor<float>(new[] { Rows, Features }), i => (float)Math.Cos(i * 0.11));
        var beta = Fill(new Tensor<float>(new[] { Features }), i => 0.01f * i);

        // Reference: the same gamma, never streamed.
        var residentGamma = Fill(new Tensor<float>(new[] { Features }), Gamma);
        engine.LayerNorm(input, residentGamma, beta, 1e-5, out var mean, out var variance);
        var expected = engine.LayerNormBackward(gradOutput, input, residentGamma, mean, variance, 1e-5,
            out var expectedGradGamma, out var expectedGradBeta);

        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-1767-" + Guid.NewGuid().ToString("N"));
        try
        {
            WeightRegistry.Configure(new GpuOffloadOptions { StreamingBackingStorePath = dir });
            var streamedGamma = Fill(new Tensor<float>(new[] { Features }), Gamma);
            streamedGamma.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(streamedGamma);

            // The state the failing step saw: registered, resident copy dropped, logical length intact.
            Assert.True(streamedGamma.StreamingPoolHandle >= 0);
            Assert.Equal(Features, streamedGamma.Length);
            Assert.NotEqual(Features, streamedGamma.DataVector.Length);

            var actual = engine.LayerNormBackward(gradOutput, input, streamedGamma, mean, variance, 1e-5,
                out var actualGradGamma, out var actualGradBeta);

            Assert.Equal(expected.ToArray(), actual.ToArray());
            Assert.Equal(expectedGradGamma.ToArray(), actualGradGamma.ToArray());
            Assert.Equal(expectedGradBeta.ToArray(), actualGradBeta.ToArray());
        }
        finally
        {
            WeightRegistry.Reset();
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    private static float Gamma(int i) => 0.5f + 0.01f * i;

    private static Tensor<float> Fill(Tensor<float> tensor, Func<int, float> value)
    {
        for (int i = 0; i < tensor.Length; i++) tensor[i] = value(i);
        return tensor;
    }
}