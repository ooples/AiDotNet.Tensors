using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

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
    public void ForwardSplat_BackwardInputAndFlow_MatchFiniteDifferences()
    {
        var inputData = new[] { 0.2f, -0.4f, 0.7f, 1.1f, -0.3f, 0.5f, 0.9f, -0.8f, 0.6f };
        var flowData = Enumerable.Range(0, 18).Select(i => i < 9 ? 0.21f : 0.34f).ToArray();
        var gradData = Enumerable.Range(1, 9).Select(i => i * 0.07f).ToArray();
        var input = new Tensor<float>((float[])inputData.Clone(), [1, 1, 3, 3]);
        var flow = new Tensor<float>((float[])flowData.Clone(), [1, 2, 3, 3]);
        var gradOutput = new Tensor<float>(gradData, [1, 1, 3, 3]);
        var output = _engine.ForwardSplat(input, flow);

        var gradInput = _engine.ForwardSplatBackwardInput(gradOutput, input, flow).GetDataArray();
        var gradFlow = _engine.ForwardSplatBackwardFlow(gradOutput, input, flow, output).GetDataArray();

        const float step = 1e-3f;
        for (int i = 0; i < inputData.Length; i++)
        {
            float numerical = CentralDifference(inputData, i, step,
                values => Loss(new Tensor<float>(values, [1, 1, 3, 3]), flow, gradOutput));
            Assert.True(Math.Abs(numerical - gradInput[i]) < 2e-3f,
                $"input[{i}] analytic={gradInput[i]}, numerical={numerical}");
        }

        for (int i = 0; i < flowData.Length; i++)
        {
            float numerical = CentralDifference(flowData, i, step,
                values => Loss(input, new Tensor<float>(values, [1, 2, 3, 3]), gradOutput));
            Assert.True(Math.Abs(numerical - gradFlow[i]) < 3e-3f,
                $"flow[{i}] analytic={gradFlow[i]}, numerical={numerical}");
        }
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

    private float Loss(Tensor<float> input, Tensor<float> flow, Tensor<float> gradOutput)
    {
        var output = _engine.ForwardSplat(input, flow).GetDataArray();
        var grad = gradOutput.GetDataArray();
        float result = 0;
        for (int i = 0; i < output.Length; i++) result += output[i] * grad[i];
        return result;
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
