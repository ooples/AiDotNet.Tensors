using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class HashGridReviewRegressionTests
{
    [Theory]
    [InlineData(1.999999999, 31f)]
    [InlineData(1.9999999999999998, 32f)]
    public void Paper_rounding_only_covers_evaluation_error(double growth, float expected)
        => Assert.Equal(expected, InstantNGPOperations.ComputeHashGridLevel(1, 16, growth, HashGridAlignment.Paper).Scale);

    [Theory]
    [InlineData(8, 2)]
    [InlineData(8, 1, 1)]
    [InlineData(8)]
    public void Rejects_incompatible_table_geometry(params int[] shape)
    {
        var positions = new Tensor<float>(new[] { 1, 3 });
        var table = new Tensor<float>(shape);
        Assert.Throws<ArgumentException>(() => InstantNGPOperations.ValidateHashEncodingArguments(positions, new[] { table }, 1));
    }
}
