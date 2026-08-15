using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CpuParallelSettings")]
public sealed class VideoWarpOperationsTests
{
    private readonly CpuEngine _engine = new();

    [Fact]
    public void PartialCorrelationVolume_RadiusOne_MatchesNeighborhoodOrdering()
    {
        var first = new Tensor<float>(Enumerable.Repeat(1f, 9).ToArray(), [1, 1, 3, 3]);
        var second = new Tensor<float>(Enumerable.Range(1, 9).Select(x => (float)x).ToArray(), [1, 1, 3, 3]);

        var result = _engine.PartialCorrelationVolume(first, second, radius: 1);

        Assert.Equal([1, 9, 3, 3], result.Shape.ToArray());
        for (int offset = 0; offset < 9; offset++)
            Assert.Equal(offset + 1f, result[0, offset, 1, 1], 5);
    }

    [Fact]
    public void PartialCorrelationVolume_BackwardMatchesFiniteDifferencesForBothInputs()
    {
        var firstData = Enumerable.Range(1, 18).Select(i => i * 0.03f - 0.2f).ToArray();
        var secondData = Enumerable.Range(1, 18).Select(i => 0.4f - i * 0.02f).ToArray();
        var outputGradient = new Tensor<float>(
            Enumerable.Range(1, 81).Select(i => i * 0.005f).ToArray(),
            [1, 9, 3, 3]);
        var first = new Tensor<float>((float[])firstData.Clone(), [1, 2, 3, 3]);
        var second = new Tensor<float>((float[])secondData.Clone(), [1, 2, 3, 3]);

        Dictionary<Tensor<float>, Tensor<float>> gradients;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
        {
            var correlation = _engine.PartialCorrelationVolume(first, second, radius: 1);
            var loss = _engine.ReduceSum(
                _engine.TensorMultiply(correlation, outputGradient), null);
            gradients = tape.ComputeGradients(loss, [first, second]);
        }

        var firstGradient = gradients[first].GetDataArray();
        var secondGradient = gradients[second].GetDataArray();
        const float step = 1e-3f;
        for (int index = 0; index < firstData.Length; index++)
        {
            float numerical = CentralDifference(firstData, index, step, values =>
                CorrelationLoss(
                    new Tensor<float>(values, [1, 2, 3, 3]), second, outputGradient));
            Assert.True(Math.Abs(numerical - firstGradient[index]) < 2e-3f,
                $"first[{index}] analytic={firstGradient[index]}, numerical={numerical}");
        }

        for (int index = 0; index < secondData.Length; index++)
        {
            float numerical = CentralDifference(secondData, index, step, values =>
                CorrelationLoss(
                    first, new Tensor<float>(values, [1, 2, 3, 3]), outputGradient));
            Assert.True(Math.Abs(numerical - secondGradient[index]) < 2e-3f,
                $"second[{index}] analytic={secondGradient[index]}, numerical={numerical}");
        }
    }

    [Fact]
    public void ForwardSplat_ZeroFlow_IsIdentity()
    {
        var input = new Tensor<float>([1f, 2f, 3f, 4f], [1, 1, 2, 2]);
        var flow = new Tensor<float>(new float[8], [1, 2, 2, 2]);

        var result = _engine.ForwardSplat(input, flow);

        AssertClose(input.GetDataArray(), result.GetDataArray(), 1e-5f);
    }

    [Fact]
    public void ForwardSplat_OnePixelRight_UsesForwardFlowConvention()
    {
        var input = new Tensor<float>([1f, 2f, 3f], [1, 1, 1, 3]);
        var flow = new Tensor<float>([1f, 1f, 1f, 0f, 0f, 0f], [1, 2, 1, 3]);

        var result = _engine.ForwardSplat(input, flow, normalize: false);

        AssertClose([0f, 1f, 2f], result.GetDataArray(), 1e-6f);
    }

    [Fact]
    public void ForwardSplat_AverageMatchesReleasedZeroWeightFallbackExactly()
    {
        var input = new Tensor<float>([2f, 6f], [1, 1, 1, 2]);
        var flow = new Tensor<float>([1f, 0f, 0f, 0f], [1, 2, 1, 2]);

        var result = _engine.ForwardSplat(input, flow);

        Assert.Equal(0f, result[0, 0, 0, 0]);
        Assert.Equal(4f, result[0, 0, 0, 1]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ForwardSplat_BatchParallelOutputIsBitwiseDeterministic(bool normalize)
    {
        const int batch = 4;
        const int channels = 3;
        const int height = 64;
        const int width = 64;
        var random = new Random(1789);
        var input = new Tensor<float>(
            Enumerable.Range(0, batch * channels * height * width)
                .Select(_ => random.NextSingle() * 2f - 1f).ToArray(),
            [batch, channels, height, width]);
        var flow = new Tensor<float>(
            Enumerable.Range(0, batch * 2 * height * width)
                .Select(_ => random.NextSingle() * 1.5f - 0.75f).ToArray(),
            [batch, 2, height, width]);
        int savedParallelism = CpuParallelSettings.MaxDegreeOfParallelism;

        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            var serial = _engine.ForwardSplat(input, flow, normalize).GetDataArray();
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(4, Environment.ProcessorCount);
            var parallel = _engine.ForwardSplat(input, flow, normalize).GetDataArray();

            Assert.Equal(serial, parallel);
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = savedParallelism;
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ForwardSplat_BackwardInputAndFlow_MatchFiniteDifferences(bool normalize)
    {
        var inputData = new[] { 0.2f, -0.4f, 0.7f, 1.1f, -0.3f, 0.5f, 0.9f, -0.8f, 0.6f };
        var flowData = Enumerable.Range(0, 18).Select(i => i < 9 ? 0.21f : 0.34f).ToArray();
        var gradData = Enumerable.Range(1, 9).Select(i => i * 0.07f).ToArray();
        var input = new Tensor<float>((float[])inputData.Clone(), [1, 1, 3, 3]);
        var flow = new Tensor<float>((float[])flowData.Clone(), [1, 2, 3, 3]);
        var gradOutput = new Tensor<float>(gradData, [1, 1, 3, 3]);
        var output = _engine.ForwardSplat(input, flow, normalize);

        var gradInput = _engine.ForwardSplatBackwardInput(
            gradOutput, input, flow, normalize).GetDataArray();
        var gradFlow = _engine.ForwardSplatBackwardFlow(
            gradOutput, input, flow, output, normalize).GetDataArray();

        const float step = 1e-3f;
        for (int i = 0; i < inputData.Length; i++)
        {
            float numerical = CentralDifference(inputData, i, step,
                values => Loss(
                    new Tensor<float>(values, [1, 1, 3, 3]), flow, gradOutput, normalize));
            Assert.True(Math.Abs(numerical - gradInput[i]) < 2e-3f,
                $"input[{i}] analytic={gradInput[i]}, numerical={numerical}");
        }

        for (int i = 0; i < flowData.Length; i++)
        {
            float numerical = CentralDifference(flowData, i, step,
                values => Loss(
                    input, new Tensor<float>(values, [1, 2, 3, 3]), gradOutput, normalize));
            Assert.True(Math.Abs(numerical - gradFlow[i]) < 3e-3f,
                $"flow[{i}] analytic={gradFlow[i]}, numerical={numerical}");
        }
    }

    [Fact]
    public void ForwardSplatBackwardFlow_ValidatesOnlyValuesUsedByTheSelectedMode()
    {
        var input = new Tensor<float>([1f, 2f, 3f, 4f], [1, 1, 2, 2]);
        var flow = new Tensor<float>(new float[8], [1, 2, 2, 2]);
        var gradOutput = new Tensor<float>(Enumerable.Repeat(1f, 4).ToArray(), [1, 1, 2, 2]);
        var wrongShape = new Tensor<float>([1f, 2f], [1, 1, 1, 2]);

        var gradError = Assert.Throws<ArgumentException>(() =>
            _engine.ForwardSplatBackwardFlow(wrongShape, input, flow, input));
        Assert.Equal("gradOutput", gradError.ParamName);

        var outputError = Assert.Throws<ArgumentException>(() =>
            _engine.ForwardSplatBackwardFlow(gradOutput, input, flow, wrongShape));
        Assert.Equal("output", outputError.ParamName);

        var nullOutputError = Assert.Throws<ArgumentNullException>(() =>
            _engine.ForwardSplatBackwardFlow(gradOutput, input, flow, null!));
        Assert.Equal("output", nullOutputError.ParamName);

        var unnormalized = _engine.ForwardSplatBackwardFlow(
            gradOutput, input, flow, null!, normalize: false);
        Assert.Equal(flow.Shape.ToArray(), unnormalized.Shape.ToArray());
    }

    [Fact]
    public void ForwardSplat_CompiledReplay_MatchesEager()
    {
        var input = new Tensor<float>(Enumerable.Range(1, 9).Select(i => (float)i).ToArray(), [1, 1, 3, 3]);
        var flow = new Tensor<float>(Enumerable.Repeat(0.25f, 18).ToArray(), [1, 2, 3, 3]);
        Func<Tensor<float>> forward = () => _engine.ForwardSplat(input, flow);
        var eager = forward().GetDataArray();
        using var cache = new CompiledModelCache<float>();
        var plan = cache.GetOrCompileInference([input, flow], forward);
        plan.SetInputs([input, flow]);

        var compiled = plan.Execute().GetDataArray();

        AssertClose(eager, compiled, 1e-5f);
    }

    [Fact]
    public void ScaledDotProductAttention_CompiledReplay_PreservesBooleanMask()
    {
        var q = new Tensor<float>([1f, 0f, 0f, 1f, 1f, 1f], [1, 1, 3, 2]);
        var mask = new Tensor<bool>(
            [true, false, false, true, true, false, true, true, true],
            [1, 1, 3, 3]);
        Func<Tensor<float>> forward = () =>
            _engine.ScaledDotProductAttention(q, q, q, mask, scale: 1.0, out _);
        var eager = forward().GetDataArray();
        using var cache = new CompiledModelCache<float>();
        var plan = cache.GetOrCompileInference(q, forward);
        plan.SetInputs([q]);

        var compiled = plan.Execute().GetDataArray();

        AssertClose(eager, compiled, 1e-5f);
    }

    private float Loss(
        Tensor<float> input, Tensor<float> flow, Tensor<float> gradOutput, bool normalize)
    {
        var output = _engine.ForwardSplat(input, flow, normalize).GetDataArray();
        var grad = gradOutput.GetDataArray();
        float result = 0;
        for (int i = 0; i < output.Length; i++) result += output[i] * grad[i];
        return result;
    }

    private float CorrelationLoss(
        Tensor<float> first, Tensor<float> second, Tensor<float> outputGradient)
    {
        var correlation = _engine.PartialCorrelationVolume(first, second, radius: 1);
        return _engine.TensorSum(_engine.TensorMultiply(correlation, outputGradient));
    }

    private static float CentralDifference(
        float[] source, int index, float step, Func<float[], float> function)
    {
        var plus = (float[])source.Clone();
        var minus = (float[])source.Clone();
        plus[index] += step;
        minus[index] -= step;
        return (function(plus) - function(minus)) / (2 * step);
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"[{i}] expected {expected[i]}, actual {actual[i]}");
    }
}
