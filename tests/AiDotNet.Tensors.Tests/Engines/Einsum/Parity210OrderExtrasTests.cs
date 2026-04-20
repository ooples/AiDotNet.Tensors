using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210OrderExtrasTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void NanMedian_IgnoresNaN()
    {
        var x = T(new[] { 1f, 2f, float.NaN, 3f, 4f }, 5);
        var m = E.TensorNanMedian(x);
        // Non-NaN values sorted: 1, 2, 3, 4 → lower median = 2.
        Assert.Equal(2f, m);
    }

    [Fact]
    public void NanMedian_AllNaN_ReturnsNaN()
    {
        var x = T(new[] { float.NaN, float.NaN }, 2);
        var m = E.TensorNanMedian(x);
        Assert.True(float.IsNaN(m));
    }

    [Fact]
    public void Mode_PicksMostFrequent()
    {
        var x = T(new[] { 1f, 2f, 2f, 3f, 2f }, 5);
        var (v, c) = E.TensorMode(x);
        Assert.Equal(2f, v);
        Assert.Equal(3, c);
    }

    [Fact]
    public void Mode_Tie_PrefersSmallerValue()
    {
        var x = T(new[] { 3f, 1f, 3f, 1f }, 4);
        var (v, c) = E.TensorMode(x);
        // Both 1 and 3 appear twice — smaller (1) wins.
        Assert.Equal(1f, v);
        Assert.Equal(2, c);
    }

    [Fact]
    public void Bucketize_MatchesSearchSorted()
    {
        var boundaries = T(new[] { 1f, 3f, 5f, 7f }, 4);
        var values = T(new[] { 2f, 4f, 6f, 8f }, 4);
        var r = E.TensorBucketize(values, boundaries);
        Assert.Equal(new[] { 1, 2, 3, 4 }, r.AsSpan().ToArray());
    }
}
