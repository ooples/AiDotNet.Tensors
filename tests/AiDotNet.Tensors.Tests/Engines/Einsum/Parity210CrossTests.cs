using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210CrossTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void Cross_XYZ_BasisVectors()
    {
        // x̂ × ŷ = ẑ = (0, 0, 1)
        var x = T(new[] { 1f, 0f, 0f }, 3);
        var y = T(new[] { 0f, 1f, 0f }, 3);
        var r = E.TensorCross(x, y, dim: 0);
        Assert.Equal(new[] { 0f, 0f, 1f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Cross_GeneralVectors()
    {
        // (1,2,3) × (4,5,6) = (2·6-3·5, 3·4-1·6, 1·5-2·4) = (-3, 6, -3)
        var a = T(new[] { 1f, 2f, 3f }, 3);
        var b = T(new[] { 4f, 5f, 6f }, 3);
        var r = E.TensorCross(a, b, dim: 0);
        Assert.Equal(new[] { -3f, 6f, -3f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Cross_BatchedAlongLastDim()
    {
        // Two pairs, each of three-vectors along last dim.
        var a = T(new[] { 1f, 0f, 0f, 0f, 1f, 0f }, 2, 3);
        var b = T(new[] { 0f, 1f, 0f, 0f, 0f, 1f }, 2, 3);
        var r = E.TensorCross(a, b, dim: -1);
        // x̂×ŷ=ẑ; ŷ×ẑ=x̂.
        Assert.Equal(new[] { 0f, 0f, 1f, 1f, 0f, 0f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Cross_WrongSize_Throws()
    {
        var a = T(new[] { 1f, 2f }, 2);
        var b = T(new[] { 3f, 4f }, 2);
        Assert.Throws<System.ArgumentException>(() => E.TensorCross(a, b, dim: 0));
    }
}
