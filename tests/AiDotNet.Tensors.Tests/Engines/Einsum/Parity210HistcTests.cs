using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210HistcTests
{
    private static CpuEngine E => new CpuEngine();

    [Fact]
    public void Histc_BasicCounting()
    {
        // torch.histc(torch.tensor([1., 2., 1.]), bins=4, min=0., max=3.)
        // => tensor([0., 2., 1., 0.])
        var x = new Tensor<float>(new[] { 1f, 2f, 1f }, new[] { 3 });
        var r = E.TensorHistc(x, bins: 4, min: 0f, max: 3f);
        Assert.Equal(new[] { 4 }, r.Shape.ToArray());
        Assert.Equal(new[] { 0f, 2f, 1f, 0f }, r.GetDataArray());
    }

    [Fact]
    public void Histc_AutoDetectBounds_WhenMinEqualsMax()
    {
        // PyTorch: histc uses the tensor's own min/max when min == max.
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var r = E.TensorHistc(x, bins: 4, min: 0f, max: 0f);
        // Auto-range [1, 4], bin width 0.75. Values 1→bin0, 2→bin1, 3→bin2, 4→bin3.
        Assert.Equal(new[] { 1f, 1f, 1f, 1f }, r.GetDataArray());
    }

    [Fact]
    public void Histc_OutOfRangeDropped()
    {
        // 5 is outside [0, 3] and must be excluded.
        var x = new Tensor<float>(new[] { 5f, 1f, 2f }, new[] { 3 });
        var r = E.TensorHistc(x, bins: 3, min: 0f, max: 3f);
        Assert.Equal(2f, r[0] + r[1] + r[2]);
    }

    [Fact]
    public void Histc_MinGreaterThanMax_Throws()
    {
        var x = new Tensor<float>(new[] { 1f }, new[] { 1 });
        Assert.Throws<ArgumentException>(() => E.TensorHistc(x, bins: 3, min: 5f, max: 1f));
    }

    [Fact]
    public void Histc_MaxGoesIntoLastBin_NotOneBeyond()
    {
        // Boundary test: upper-boundary value should land in bin[bins-1], not overflow.
        var x = new Tensor<float>(new[] { 3f }, new[] { 1 });
        var r = E.TensorHistc(x, bins: 3, min: 0f, max: 3f);
        Assert.Equal(new[] { 0f, 0f, 1f }, r.GetDataArray());
    }
}
