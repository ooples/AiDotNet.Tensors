using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Verifies the composed RMSNorm graph used by higher-level layers. The input
/// fans out through both x-squared and x-times-inverse-RMS paths, making this a
/// high-value gradient-ownership contract rather than an isolated op check.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public sealed class RmsNormCompositeGradientTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    private readonly CpuEngine _engine;

    public RmsNormCompositeGradientTests()
    {
        AiDotNetEngine.Current = new CpuEngine();
        _engine = (CpuEngine)AiDotNetEngine.Current;
    }

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    [Fact]
    public async Task InputAndScaleGradients_MatchFiniteDifferences()
    {
        await Task.Yield();

        const int featureSize = 32;
        var input = CreateTensor([8, featureSize], 0.02f);
        var scale = CreateTensor([featureSize], 0.04f, offset: 1f);
        var projection = CreateTensor([8, featureSize], 0.03f);

        Tensor<float> Forward()
        {
            var squared = _engine.TensorSquare(input);
            var sumSquared = _engine.ReduceSum(squared, [1], keepDims: true);
            var meanSquared = _engine.TensorDivideScalar(sumSquared, featureSize);
            var stabilized = _engine.TensorAddScalar(meanSquared, 1e-6f);
            var inverseRms = _engine.TensorReciprocal(_engine.TensorSqrt(stabilized));
            var normalized = _engine.TensorBroadcastMultiply(input, inverseRms);
            return _engine.TensorBroadcastMultiply(normalized, scale);
        }

        Tensor<float> inputGradient;
        Tensor<float> scaleGradient;
        using (var tape = new GradientTape<float>())
        {
            var objective = _engine.ReduceSum(
                _engine.TensorMultiply(Forward(), projection),
                axes: null);
            var gradients = tape.ComputeGradients(objective, [input, scale]);
            inputGradient = gradients[input];
            scaleGradient = gradients[scale];
        }

        AssertSampledGradient(input, inputGradient, Forward, projection, sampleCount: 16);
        AssertSampledGradient(scale, scaleGradient, Forward, projection, sampleCount: 16);
    }

    private static void AssertSampledGradient(
        Tensor<float> source,
        Tensor<float> analytical,
        Func<Tensor<float>> forward,
        Tensor<float> projection,
        int sampleCount)
    {
        const float step = 1e-3f;
        for (int sample = 0; sample < sampleCount; sample++)
        {
            int index = sample * (source.Length - 1) / Math.Max(1, sampleCount - 1);
            float original = source[index];
            source[index] = original + step;
            float plus = Project(forward(), projection);
            source[index] = original - step;
            float minus = Project(forward(), projection);
            source[index] = original;

            float numerical = (plus - minus) / (2f * step);
            float difference = Math.Abs(analytical[index] - numerical);
            float scale = Math.Max(Math.Abs(analytical[index]), Math.Abs(numerical));
            float tolerance = Math.Max(2e-3f, scale * 3e-2f);
            Assert.True(
                difference <= tolerance,
                $"index {index}: analytical={analytical[index]:G9}, numerical={numerical:G9}, " +
                $"difference={difference:G9}, tolerance={tolerance:G9}");
        }
    }

    private static float Project(Tensor<float> output, Tensor<float> projection)
    {
        float sum = 0f;
        for (int i = 0; i < output.Length; i++) sum += output[i] * projection[i];
        return sum;
    }

    private static Tensor<float> CreateTensor(int[] shape, float scale, float offset = 0f)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = offset + (scale * (((i * 17) % 31) - 15) / 15f);
        return tensor;
    }
}
