using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public sealed class FusedAttentionRectangularTests
{
    public enum ForwardRoute { Engine, Tiled, ExplicitWeights, Generic }

    [Theory]
    [InlineData(3, false, ForwardRoute.Engine)]
    [InlineData(3, true, ForwardRoute.Engine)]
    [InlineData(77, true, ForwardRoute.Engine)]
    [InlineData(129, true, ForwardRoute.Engine)]
    [InlineData(3, false, ForwardRoute.Tiled)]
    [InlineData(77, true, ForwardRoute.Tiled)]
    [InlineData(77, true, ForwardRoute.ExplicitWeights)]
    [InlineData(3, false, ForwardRoute.Generic)]
    [InlineData(77, true, ForwardRoute.Generic)]
    public void MoreQueriesThanMemory_UsesActualKernelAndMatchesManualSoftmax(int queryCount, bool rank4, ForwardRoute route)
    {
        const int batch = 2, dimension = 4, memoryCount = 2;
        int heads = rank4 ? 2 : 1;
        var query = Data(rank4 ? new[] { batch, heads, queryCount, dimension } : new[] { batch, queryCount, dimension }, 0.7);
        var key = Data(rank4 ? new[] { batch, heads, memoryCount, dimension } : new[] { batch, memoryCount, dimension }, 1.1);
        var value = Data(key.Shape.ToArray(), 1.9);
        var beforeQuery = query.ToArray();
        var beforeKey = key.ToArray();
        var beforeValue = value.ToArray();
        var config = new FlashAttentionConfig
        {
            IsCausal = false,
            UseFlashAttention2 = route == ForwardRoute.Tiled,
            ReturnAttentionWeights = route == ForwardRoute.ExplicitWeights
        };
        Tensor<float> actual;
        Tensor<float>? weights;
        if (route == ForwardRoute.Generic)
        {
            actual = FlashAttention<float>.Forward(query, key, value).Output;
            weights = null;
        }
        else (actual, weights) = FusedAttention<float>.Forward(query, key, value, config, engine: new CpuEngine());
        var expected = Reference(beforeQuery, beforeKey, beforeValue, batch * heads, queryCount, memoryCount, dimension);
        Assert.Equal(query.Shape.ToArray(), actual.Shape.ToArray());
        Close(expected, actual.ToArray());
        Assert.Equal(beforeQuery, query.ToArray());
        Assert.Equal(beforeKey, key.ToArray());
        Assert.Equal(beforeValue, value.ToArray());
        if (route == ForwardRoute.ExplicitWeights)
        {
            Assert.NotNull(weights);
            Assert.Equal(new[] { batch, heads, queryCount, memoryCount }, weights.Shape.ToArray());
        }
        else Assert.Null(weights);
    }

    [Theory]
    [InlineData(3, false, false)]
    [InlineData(77, true, false)]
    [InlineData(129, true, false)]
    [InlineData(3, false, true)]
    [InlineData(77, true, true)]
    public void RectangularGradients_MatchIndependentAnalyticReference(int queryCount, bool rank4, bool generic)
    {
        const int dimension = 4, memoryCount = 2;
        var query = Data(rank4 ? new[] { 1, 1, queryCount, dimension } : new[] { 1, queryCount, dimension }, 0.7);
        var key = Data(rank4 ? new[] { 1, 1, memoryCount, dimension } : new[] { 1, memoryCount, dimension }, 1.1);
        var value = Data(key.Shape.ToArray(), 1.9);
        var upstream = Data(query.Shape.ToArray(), 2.3);
        var expectedQuery = new double[query.Length];
        var expectedKey = new double[key.Length];
        var expectedValue = new double[value.Length];
        double scale = 1.0 / Math.Sqrt(dimension);
        for (int row = 0; row < queryCount; row++)
        {
            var probabilities = Probabilities(query.ToArray(), key.ToArray(), row * dimension, 0, memoryCount, dimension);
            var probabilityGradient = new double[memoryCount];
            double average = 0;
            for (int memory = 0; memory < memoryCount; memory++)
            {
                for (int column = 0; column < dimension; column++)
                {
                    probabilityGradient[memory] += upstream[row * dimension + column] * value[memory * dimension + column];
                    expectedValue[memory * dimension + column] += probabilities[memory] * upstream[row * dimension + column];
                }
                average += probabilities[memory] * probabilityGradient[memory];
            }
            for (int memory = 0; memory < memoryCount; memory++)
            {
                double scoreGradient = probabilities[memory] * (probabilityGradient[memory] - average) * scale;
                for (int column = 0; column < dimension; column++)
                {
                    expectedQuery[row * dimension + column] += scoreGradient * key[memory * dimension + column];
                    expectedKey[memory * dimension + column] += scoreGradient * query[row * dimension + column];
                }
            }
        }
        (Tensor<float> actualQuery, Tensor<float> actualKey, Tensor<float> actualValue) =
            generic ? GenericBackward(upstream, query, key, value)
                : FusedAttention<float>.Backward(upstream, query, key, value, engine: new CpuEngine());
        Close(expectedQuery, actualQuery.ToArray());
        Close(expectedKey, actualKey.ToArray());
        Close(expectedValue, actualValue.ToArray());
        Assert.Contains(expectedQuery, gradient => Math.Abs(gradient) > 1e-5);
        Assert.Contains(expectedKey, gradient => Math.Abs(gradient) > 1e-5);
        Assert.Contains(expectedValue, gradient => Math.Abs(gradient) > 1e-5);
    }

    [Theory]
    [InlineData(-1, false)]
    [InlineData(-1, true)]
    [InlineData(2, false)]
    [InlineData(2, true)]
    [InlineData(int.MaxValue, false)]
    [InlineData(int.MaxValue, true)]
    public void InvalidQueryOffsets_StillRejectIncludingOverflow(int offset, bool causal)
    {
        var query = new Tensor<float>(new[] { 1, 2, 2 });
        var key = new Tensor<float>(new[] { 1, 3, 2 });
        Assert.Throws<ArgumentException>(() => FusedAttention<float>.Forward(query, key, key,
            new FlashAttentionConfig { IsCausal = causal, QueryOffset = offset }, engine: new CpuEngine()));
    }

    [Fact]
    public void CausalWindow_AtExactEndPreservesMaskedValues()
    {
        var query = new Tensor<float>(new[] { 1, 2, 2 });
        var key = new Tensor<float>(new[] { 1, 3, 2 });
        var value = new Tensor<float>(new[] { 1, 3, 2 }, new Vector<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }));
        var actual = FusedAttention<float>.Forward(query, key, value,
            new FlashAttentionConfig { IsCausal = true, QueryOffset = 1 }, engine: new CpuEngine()).Output;
        Close(new[] { 2.0, 3.0, 3.0, 4.0 }, actual.ToArray());
    }

    [Theory]
    [InlineData(-1, false)]
    [InlineData(-1, true)]
    [InlineData(2, false)]
    [InlineData(2, true)]
    [InlineData(int.MaxValue, false)]
    [InlineData(int.MaxValue, true)]
    public void DirectTiledEntry_RejectsInvalidExplicitWindows(int offset, bool causal)
    {
        var query = new Tensor<float>(new[] { 1, 1, 2, 2 });
        var key = new Tensor<float>(new[] { 1, 1, 3, 2 });
        Assert.Throws<ArgumentException>(() => FlashAttention2.Forward(query, key, key, isCausal: causal, queryOffset: offset));
    }

    [Fact]
    public void CausalQueries_CannotExtendBeyondMemory()
    {
        var query = new Tensor<float>(new[] { 1, 1, 3, 2 });
        var key = new Tensor<float>(new[] { 1, 1, 2, 2 });
        Assert.Throws<ArgumentException>(() => FusedAttention<float>.Forward(query, key, key,
            new FlashAttentionConfig { IsCausal = true }, engine: new CpuEngine()));
        Assert.Throws<ArgumentException>(() => FlashAttention2.Forward(query, key, key, isCausal: true));
        Assert.Throws<ArgumentException>(() => FlashAttention<float>.Forward(query, key, key, isCausal: true));
    }

    [Theory]
    [InlineData(-1, false)]
    [InlineData(-1, true)]
    [InlineData(2, false)]
    [InlineData(2, true)]
    [InlineData(int.MaxValue, false)]
    [InlineData(int.MaxValue, true)]
    public void GenericEntry_RejectsInvalidExplicitWindows(int offset, bool causal)
    {
        var query = new Tensor<float>(new[] { 1, 1, 2, 2 });
        var key = new Tensor<float>(new[] { 1, 1, 3, 2 });
        Assert.Throws<ArgumentException>(() => FlashAttention<float>.Forward(query, key, key, isCausal: causal, queryOffset: offset));
    }

    private static (Tensor<float>, Tensor<float>, Tensor<float>) GenericBackward(
        Tensor<float> upstream, Tensor<float> query, Tensor<float> key, Tensor<float> value)
    {
        var (output, logSumExp) = FlashAttention<float>.Forward(query, key, value);
        return FlashAttention<float>.Backward(upstream, query, key, value, output, logSumExp);
    }

    private static Tensor<float> Data(int[] shape, double phase)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = (float)Math.Sin(i * 0.37 + phase);
        return tensor;
    }

    private static double[] Reference(float[] query, float[] key, float[] value, int groups, int queries, int memory, int dimension)
    {
        var output = new double[query.Length];
        for (int group = 0; group < groups; group++)
        for (int row = 0; row < queries; row++)
        {
            int queryOffset = (group * queries + row) * dimension;
            int memoryOffset = group * memory * dimension;
            var probabilities = Probabilities(query, key, queryOffset, memoryOffset, memory, dimension);
            for (int column = 0; column < dimension; column++)
            for (int keyRow = 0; keyRow < memory; keyRow++)
                output[queryOffset + column] += probabilities[keyRow] * value[memoryOffset + keyRow * dimension + column];
        }
        return output;
    }

    private static double[] Probabilities(float[] query, float[] key, int queryOffset, int memoryOffset, int memory, int dimension)
    {
        var probabilities = new double[memory];
        double maximum = double.NegativeInfinity;
        for (int row = 0; row < memory; row++)
        {
            for (int column = 0; column < dimension; column++)
                probabilities[row] += (double)query[queryOffset + column] * key[memoryOffset + row * dimension + column];
            probabilities[row] /= Math.Sqrt(dimension);
            maximum = Math.Max(maximum, probabilities[row]);
        }
        double total = 0;
        for (int row = 0; row < memory; row++) total += probabilities[row] = Math.Exp(probabilities[row] - maximum);
        for (int row = 0; row < memory; row++) probabilities[row] /= total;
        return probabilities;
    }

    private static void Close(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]) || float.IsInfinity(actual[i]));
            Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, 2e-5 * (1 + Math.Abs(expected[i])));
        }
    }
}
