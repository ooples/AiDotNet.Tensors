using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

[Collection("EngineCurrentGlobalState")]
public sealed class ScaledDotProductAttentionGradientTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    private readonly CpuEngine _engine = new();

    public ScaledDotProductAttentionGradientTests() => AiDotNetEngine.Current = _engine;

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    [Fact]
    public async Task Double_QkvGradients_MatchFiniteDifferencesAtTransformerShape()
    {
        await Task.Yield();

        const int headDimension = 8;
        double scale = 1.0 / Math.Sqrt(headDimension);
        var query = CreatePattern([4, 4, 16, headDimension], 0.013);
        var key = CreatePattern([4, 4, 16, headDimension], -0.017);
        var value = CreatePattern([4, 4, 16, headDimension], 0.019);
        var projection = CreatePattern([4, 4, 16, headDimension], 0.011);

        Tensor<double> queryGradient;
        Tensor<double> keyGradient;
        Tensor<double> valueGradient;
        using (var tape = new GradientTape<double>())
        {
            var output = _engine.ScaledDotProductAttention(
                query, key, value, mask: null, scale, out _);
            var projected = _engine.TensorMultiply(output, projection);
            var objective = _engine.ReduceSum(projected, [0, 1, 2, 3], keepDims: false);
            var gradients = tape.ComputeGradients(objective, [query, key, value]);
            queryGradient = gradients[query];
            keyGradient = gradients[key];
            valueGradient = gradients[value];
        }

        AssertFiniteDifference(query, queryGradient, key, value, projection, scale, "query");
        AssertFiniteDifference(key, keyGradient, query, value, projection, scale, "key", sourceIsKey: true);
        AssertFiniteDifference(value, valueGradient, query, key, projection, scale, "value", sourceIsValue: true);
    }

    [Fact]
    public async Task Double_PermutedQkvGradients_MatchFiniteDifferencesAtTransformerShape()
    {
        await Task.Yield();

        const int batch = 4;
        const int sequence = 16;
        const int heads = 4;
        const int headDimension = 8;
        double scale = 1.0 / Math.Sqrt(headDimension);
        var queryBase = CreatePattern([batch, sequence, heads, headDimension], 0.013);
        var keyBase = CreatePattern([batch, sequence, heads, headDimension], -0.017);
        var valueBase = CreatePattern([batch, sequence, heads, headDimension], 0.019);
        var projection = CreatePattern([batch, heads, sequence, headDimension], 0.011);

        Tensor<double> queryGradient;
        Tensor<double> keyGradient;
        Tensor<double> valueGradient;
        using (var tape = new GradientTape<double>())
        {
            var query = _engine.TensorPermute(queryBase, [0, 2, 1, 3]);
            var key = _engine.TensorPermute(keyBase, [0, 2, 1, 3]);
            var value = _engine.TensorPermute(valueBase, [0, 2, 1, 3]);
            var output = _engine.ScaledDotProductAttention(
                query, key, value, mask: null, scale, out _);
            var projected = _engine.TensorMultiply(output, projection);
            var objective = _engine.ReduceSum(projected, [0, 1, 2, 3], keepDims: false);
            var gradients = tape.ComputeGradients(objective, [queryBase, keyBase, valueBase]);
            queryGradient = gradients[queryBase];
            keyGradient = gradients[keyBase];
            valueGradient = gradients[valueBase];
        }

        AssertPermutedFiniteDifference(
            queryBase, queryGradient,
            () => ProjectPermuted(queryBase, keyBase, valueBase, projection, scale),
            "query");
        AssertPermutedFiniteDifference(
            keyBase, keyGradient,
            () => ProjectPermuted(queryBase, keyBase, valueBase, projection, scale),
            "key");
        AssertPermutedFiniteDifference(
            valueBase, valueGradient,
            () => ProjectPermuted(queryBase, keyBase, valueBase, projection, scale),
            "value");
    }

    [Fact]
    public async Task Double_SelfAttentionInputGradient_MatchesFiniteDifferencesThroughOutputProjection()
    {
        await Task.Yield();

        const int batch = 4;
        const int sequence = 16;
        const int heads = 4;
        const int headDimension = 8;
        const int embedding = heads * headDimension;
        var input = CreatePattern([batch * sequence, embedding], 0.007);
        var queryWeights = CreatePattern([embedding, embedding], 0.003);
        var keyWeights = CreatePattern([embedding, embedding], -0.004);
        var valueWeights = CreatePattern([embedding, embedding], 0.005);
        var outputWeights = CreatePattern([embedding, embedding], -0.006);
        var outputBias = CreatePattern([embedding], 0.002);
        var projection = CreatePattern([batch * sequence, embedding], 0.009);

        Tensor<double> inputGradient;
        using (var tape = new GradientTape<double>())
        {
            var output = ProjectSelfAttention(
                input, queryWeights, keyWeights, valueWeights, outputWeights, outputBias,
                batch, sequence, heads, headDimension);
            var projected = _engine.TensorMultiply(output, projection);
            var objective = _engine.ReduceSum(projected, [0, 1], keepDims: false);
            inputGradient = tape.ComputeGradients(objective, [input])[input];
        }

        AssertPermutedFiniteDifference(
            input, inputGradient,
            () => EvaluateSelfAttention(
                input, queryWeights, keyWeights, valueWeights, outputWeights, outputBias,
                projection, batch, sequence, heads, headDimension),
            "self-attention input");
    }

    private void AssertFiniteDifference(
        Tensor<double> source,
        Tensor<double> analytical,
        Tensor<double> firstOther,
        Tensor<double> secondOther,
        Tensor<double> projection,
        double scale,
        string name,
        bool sourceIsKey = false,
        bool sourceIsValue = false)
    {
        const double step = 1e-6;
        int[] samples = [0, source.Length / 3, source.Length - 1];
        foreach (int index in samples)
        {
            double original = source[index];
            source[index] = original + step;
            double plus = sourceIsValue
                ? Project(firstOther, secondOther, source, projection, scale)
                : sourceIsKey
                    ? Project(firstOther, source, secondOther, projection, scale)
                    : Project(source, firstOther, secondOther, projection, scale);
            source[index] = original - step;
            double minus = sourceIsValue
                ? Project(firstOther, secondOther, source, projection, scale)
                : sourceIsKey
                    ? Project(firstOther, source, secondOther, projection, scale)
                    : Project(source, firstOther, secondOther, projection, scale);
            source[index] = original;

            double numerical = (plus - minus) / (2.0 * step);
            double actual = analytical[index];
            double tolerance = Math.Max(1e-9, Math.Abs(numerical) * 1e-6);
            Assert.True(Math.Abs(numerical - actual) <= tolerance,
                $"{name}[{index}] mismatch: numerical={numerical:R}, analytical={actual:R}, " +
                $"difference={Math.Abs(numerical - actual):R}, tolerance={tolerance:R}.");
        }
    }

    private double Project(
        Tensor<double> query,
        Tensor<double> key,
        Tensor<double> value,
        Tensor<double> projection,
        double scale)
    {
        using var noGrad = new NoGradScope<double>();
        var output = _engine.ScaledDotProductAttention(
            query, key, value, mask: null, scale, out _);
        double sum = 0.0;
        for (int i = 0; i < output.Length; i++) sum += output[i] * projection[i];
        return sum;
    }

    private void AssertPermutedFiniteDifference(
        Tensor<double> source,
        Tensor<double> analytical,
        Func<double> evaluate,
        string name)
    {
        const double step = 1e-6;
        int[] samples = [0, source.Length / 3, source.Length - 1];
        foreach (int index in samples)
        {
            double original = source[index];
            source[index] = original + step;
            double plus = evaluate();
            source[index] = original - step;
            double minus = evaluate();
            source[index] = original;

            double numerical = (plus - minus) / (2.0 * step);
            double actual = analytical[index];
            double tolerance = Math.Max(1e-9, Math.Abs(numerical) * 1e-6);
            Assert.True(Math.Abs(numerical - actual) <= tolerance,
                $"{name}[{index}] mismatch after QKV permutation: numerical={numerical:R}, " +
                $"analytical={actual:R}, difference={Math.Abs(numerical - actual):R}, " +
                $"tolerance={tolerance:R}.");
        }
    }

    private double ProjectPermuted(
        Tensor<double> queryBase,
        Tensor<double> keyBase,
        Tensor<double> valueBase,
        Tensor<double> projection,
        double scale)
    {
        using var noGrad = new NoGradScope<double>();
        var query = _engine.TensorPermute(queryBase, [0, 2, 1, 3]);
        var key = _engine.TensorPermute(keyBase, [0, 2, 1, 3]);
        var value = _engine.TensorPermute(valueBase, [0, 2, 1, 3]);
        var output = _engine.ScaledDotProductAttention(
            query, key, value, mask: null, scale, out _);
        double sum = 0.0;
        for (int i = 0; i < output.Length; i++) sum += output[i] * projection[i];
        return sum;
    }

    private Tensor<double> ProjectSelfAttention(
        Tensor<double> input,
        Tensor<double> queryWeights,
        Tensor<double> keyWeights,
        Tensor<double> valueWeights,
        Tensor<double> outputWeights,
        Tensor<double> outputBias,
        int batch,
        int sequence,
        int heads,
        int headDimension)
    {
        int embedding = heads * headDimension;
        var queryFlat = _engine.TensorMatMul(input, queryWeights);
        var keyFlat = _engine.TensorMatMul(input, keyWeights);
        var valueFlat = _engine.TensorMatMul(input, valueWeights);
        var query = _engine.TensorPermute(
            _engine.Reshape(queryFlat, [batch, sequence, heads, headDimension]),
            [0, 2, 1, 3]);
        var key = _engine.TensorPermute(
            _engine.Reshape(keyFlat, [batch, sequence, heads, headDimension]),
            [0, 2, 1, 3]);
        var value = _engine.TensorPermute(
            _engine.Reshape(valueFlat, [batch, sequence, heads, headDimension]),
            [0, 2, 1, 3]);
        var context = _engine.ScaledDotProductAttention(
            query, key, value, mask: null, 1.0 / Math.Sqrt(headDimension), out _);
        var contextTransposed = _engine.TensorPermute(context, [0, 2, 1, 3]);
        var contextFlat = _engine.Reshape(contextTransposed, [batch * sequence, embedding]);
        return _engine.FusedLinear(
            contextFlat, outputWeights, outputBias, FusedActivationType.None);
    }

    private double EvaluateSelfAttention(
        Tensor<double> input,
        Tensor<double> queryWeights,
        Tensor<double> keyWeights,
        Tensor<double> valueWeights,
        Tensor<double> outputWeights,
        Tensor<double> outputBias,
        Tensor<double> projection,
        int batch,
        int sequence,
        int heads,
        int headDimension)
    {
        using var noGrad = new NoGradScope<double>();
        var output = ProjectSelfAttention(
            input, queryWeights, keyWeights, valueWeights, outputWeights, outputBias,
            batch, sequence, heads, headDimension);
        double sum = 0.0;
        for (int i = 0; i < output.Length; i++) sum += output[i] * projection[i];
        return sum;
    }

    private static Tensor<double> CreatePattern(int[] shape, double scale)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = scale * (((i * 17) % 29) - 14);
        return tensor;
    }
}
