using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Guards convolution VJPs when padding makes a kernel larger than the input's
/// spatial extent. This geometry occurs in encoder/decoder models whose final
/// feature map is 1x1 but whose decoder still uses padded 3x3 convolutions.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public sealed class Conv2DDegenerateSpatialGradientTests : IDisposable
{
    private const float Epsilon = 1e-3f;
    private const float RelativeTolerance = 2e-2f;

    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    private readonly IEngine _engine;

    public Conv2DDegenerateSpatialGradientTests()
    {
        AiDotNetEngine.Current = new CpuEngine();
        _engine = AiDotNetEngine.Current;
    }

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    [Fact]
    public async Task Conv2D_PaddedThreeByThreeOverOneByOne_MatchesFiniteDifferences()
    {
        await Task.Yield();

        var input = CreateTensor([1, 2, 1, 1], 0.35f, -0.20f);
        var kernel = CreateSequence([2, 2, 3, 3], -0.18f, 0.011f);

        AssertGradientsMatch(
            () => _engine.Conv2D(input, kernel, [1, 1], [1, 1], [1, 1]),
            input,
            kernel);
    }

    [Fact]
    public async Task FusedConv2D_PaddedThreeByThreeOverOneByOne_MatchesFiniteDifferences()
    {
        await Task.Yield();

        var input = CreateTensor([1, 2, 1, 1], 0.35f, -0.20f);
        var kernel = CreateSequence([2, 2, 3, 3], -0.18f, 0.011f);
        // Keep both outputs away from ReLU's non-differentiable boundary.
        var bias = CreateTensor([2], 0.40f, 0.55f);

        AssertGradientsMatch(
            () => _engine.FusedConv2D(
                input,
                kernel,
                bias,
                strideH: 1,
                strideW: 1,
                padH: 1,
                padW: 1,
                dilationH: 1,
                dilationW: 1,
                FusedActivationType.ReLU),
            input,
            kernel,
            bias);
    }

    [Fact]
    public async Task Conv2DBackwardInput_PaddedThreeByThreeOverOneByOne_MatchesReference()
    {
        await Task.Yield();

        var cpu = Assert.IsType<CpuEngine>(_engine);
        var gradOutput = CreateTensor([1, 2, 1, 1], 1f, 1f);
        var kernel = CreateSequence([2, 2, 3, 3], -0.18f, 0.011f);
        var into = new Tensor<float>([1, 2, 1, 1]);
        var flippedKernel = FlipAndTranspose(kernel, inputChannels: 2, outputChannels: 2);
        var forwardIdentity = cpu.Conv2D(gradOutput, flippedKernel, [1, 1], [1, 1], [1, 1]);
        var scalarForwardIdentity = cpu.Conv2D(gradOutput, flippedKernel, stride: 1, padding: 1, dilation: 1);

        cpu.Conv2DBackwardInputInto(
            into,
            gradOutput,
            kernel,
            [1, 2, 1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            accumulate: false);
        var allocating = cpu.Conv2DBackwardInput(
            gradOutput,
            kernel,
            [1, 2, 1, 1],
            [1, 1],
            [1, 1],
            [1, 1]);

        for (int inputChannel = 0; inputChannel < 2; inputChannel++)
        {
            float expected = 0f;
            for (int outputChannel = 0; outputChannel < 2; outputChannel++)
            {
                int centerIndex = (((outputChannel * 2) + inputChannel) * 3 * 3) + 4;
                expected += kernel[centerIndex];
            }

            AssertClose(allocating[inputChannel], expected, $"allocating input channel {inputChannel}");
            AssertClose(forwardIdentity[inputChannel], expected, $"forward identity input channel {inputChannel}");
            AssertClose(scalarForwardIdentity[inputChannel], expected, $"scalar forward identity input channel {inputChannel}");
            AssertClose(into[inputChannel], expected, $"into input channel {inputChannel}");
        }
    }

    private void AssertGradientsMatch(Func<Tensor<float>> forward, params Tensor<float>[] sources)
    {
        Dictionary<Tensor<float>, Tensor<float>> gradients;
        using (var tape = new GradientTape<float>())
        {
            var loss = _engine.ReduceSum(forward(), axes: null);
            gradients = tape.ComputeGradients(loss, sources);
        }

        foreach (var source in sources)
        {
            Assert.True(gradients.TryGetValue(source, out var analytical), "Autodiff did not produce a source gradient.");
            Assert.NotNull(analytical);

            for (int i = 0; i < source.Length; i++)
            {
                float original = source[i];
                source[i] = original + Epsilon;
                float plus = Sum(forward());
                source[i] = original - Epsilon;
                float minus = Sum(forward());
                source[i] = original;

                float numerical = (plus - minus) / (2f * Epsilon);
                AssertClose(analytical![i], numerical, $"source shape [{string.Join(",", source._shape)}], index {i}");
            }
        }
    }

    private static void AssertClose(float analytical, float numerical, string coordinate)
    {
        float difference = Math.Abs(analytical - numerical);
        float scale = Math.Max(Math.Abs(analytical), Math.Abs(numerical));
        float tolerance = Math.Max(1e-4f, scale * RelativeTolerance);
        Assert.True(
            difference <= tolerance,
            $"{coordinate}: analytical={analytical:G9}, numerical={numerical:G9}, difference={difference:G9}, tolerance={tolerance:G9}");
    }

    private static float Sum(Tensor<float> tensor)
    {
        float sum = 0f;
        for (int i = 0; i < tensor.Length; i++)
        {
            sum += tensor[i];
        }

        return sum;
    }

    private static Tensor<float> CreateSequence(int[] shape, float start, float step)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = start + (i * step);
        }

        return tensor;
    }

    private static Tensor<float> FlipAndTranspose(Tensor<float> kernel, int inputChannels, int outputChannels)
    {
        var flipped = new Tensor<float>([inputChannels, outputChannels, 3, 3]);
        for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++)
        {
            for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++)
            {
                for (int kernelRow = 0; kernelRow < 3; kernelRow++)
                {
                    for (int kernelColumn = 0; kernelColumn < 3; kernelColumn++)
                    {
                        int source = (((outputChannel * inputChannels) + inputChannel) * 3 * 3)
                            + ((2 - kernelRow) * 3)
                            + (2 - kernelColumn);
                        int destination = (((inputChannel * outputChannels) + outputChannel) * 3 * 3)
                            + (kernelRow * 3)
                            + kernelColumn;
                        flipped[destination] = kernel[source];
                    }
                }
            }
        }

        return flipped;
    }

    private static Tensor<float> CreateTensor(int[] shape, params float[] values)
        => new(values, shape);
}
