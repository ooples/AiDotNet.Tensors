using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

[Collection("EngineCurrentGlobalState")]
public sealed class FusedLinearLargeInputGradientTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    private readonly CpuEngine _engine = new();

    public FusedLinearLargeInputGradientTests() => AiDotNetEngine.Current = _engine;

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    [Fact]
    public async Task FusedLinear_LargeParallelBackward_InputGradientMatchesReference()
    {
        await Task.Yield();

        const int rows = 64;
        const int inputFeatures = 32;
        const int outputFeatures = 128;
        var input = CreatePattern([rows, inputFeatures], 0.013f);
        var weights = CreatePattern([inputFeatures, outputFeatures], -0.017f);
        var bias = CreatePattern([outputFeatures], 0.007f);
        var projection = CreatePattern([rows, outputFeatures], 0.011f);

        Tensor<float> inputGradient;
        using (var tape = new GradientTape<float>())
        {
            var output = _engine.FusedLinear(
                input, weights, bias, FusedActivationType.None);
            var projected = _engine.TensorMultiply(output, projection);
            var objective = _engine.ReduceSum(projected, [0, 1], keepDims: false);
            inputGradient = tape.ComputeGradients(objective, [input])[input];
        }

        for (int row = 0; row < rows; row++)
        {
            for (int feature = 0; feature < inputFeatures; feature++)
            {
                double expected = 0.0;
                for (int output = 0; output < outputFeatures; output++)
                    expected += projection[row, output] * weights[feature, output];

                double actual = inputGradient[row, feature];
                double scale = Math.Max(Math.Abs(expected), Math.Abs(actual));
                double tolerance = Math.Max(2e-5, scale * 1e-4);
                Assert.True(Math.Abs(expected - actual) <= tolerance,
                    $"Input gradient [{row},{feature}] mismatch: expected={expected:R}, actual={actual:R}, tolerance={tolerance:R}.");
            }
        }
    }

    [Fact]
    public async Task SwishAfterLinear_LargeBackward_InputGradientMatchesReference()
    {
        await Task.Yield();

        const int rows = 64;
        const int inputFeatures = 32;
        const int outputFeatures = 128;
        var input = CreatePatternDouble([rows, inputFeatures], 0.013);
        var weights = CreatePatternDouble([inputFeatures, outputFeatures], -0.017);
        var bias = CreatePatternDouble([outputFeatures], 0.007);
        var projection = CreatePatternDouble([rows, outputFeatures], 0.011);

        Tensor<double> inputGradient;
        using (var tape = new GradientTape<double>())
        {
            var linear = _engine.FusedLinear(
                input, weights, bias, FusedActivationType.None);
            var output = _engine.Swish(linear);
            var projected = _engine.TensorMultiply(output, projection);
            var objective = _engine.ReduceSum(projected, [0, 1], keepDims: false);
            inputGradient = tape.ComputeGradients(objective, [input])[input];
        }

        for (int row = 0; row < rows; row++)
        {
            for (int feature = 0; feature < inputFeatures; feature++)
            {
                double expected = 0.0;
                for (int output = 0; output < outputFeatures; output++)
                {
                    double preActivation = bias[output];
                    for (int inner = 0; inner < inputFeatures; inner++)
                        preActivation += input[row, inner] * weights[inner, output];
                    double sigmoid = 1.0 / (1.0 + Math.Exp(-preActivation));
                    double derivative = sigmoid + preActivation * sigmoid * (1.0 - sigmoid);
                    expected += projection[row, output] * derivative * weights[feature, output];
                }

                double actual = inputGradient[row, feature];
                Assert.True(Math.Abs(expected - actual) <= 1e-9,
                    $"Swish input gradient [{row},{feature}] mismatch: expected={expected:R}, actual={actual:R}.");
            }
        }
    }

    private static Tensor<float> CreatePattern(int[] shape, float scale)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = scale * (((i * 17) % 29) - 14);
        return tensor;
    }

    private static Tensor<double> CreatePatternDouble(int[] shape, double scale)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = scale * (((i * 17) % 29) - 14);
        return tensor;
    }
}
