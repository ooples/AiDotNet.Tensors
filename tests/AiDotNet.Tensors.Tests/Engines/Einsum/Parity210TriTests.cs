using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210TriTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void Triu_KeepsUpperTriangle()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f }, 3, 3);
        var r = E.TensorTriu(x);
        Assert.Equal(new[] { 1f, 2f, 3f, 0f, 5f, 6f, 0f, 0f, 9f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Tril_KeepsLowerTriangle()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f }, 3, 3);
        var r = E.TensorTril(x);
        Assert.Equal(new[] { 1f, 0f, 0f, 4f, 5f, 0f, 7f, 8f, 9f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Triu_PositiveDiagonal_ShiftsBoundaryUp()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f }, 3, 3);
        var r = E.TensorTriu(x, diagonal: 1);
        // diagonal=1 → keep col-row >= 1 → only super-diagonal and above.
        Assert.Equal(new[] { 0f, 2f, 3f, 0f, 0f, 6f, 0f, 0f, 0f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Nonzero_ReturnsCoordsOf1DTensor()
    {
        var x = T(new[] { 0f, 3f, 0f, 5f }, 4);
        var r = E.TensorNonzero(x);
        // Shape [2, 1]: two nonzero indices (1, 3).
        Assert.Equal(new[] { 2, 1 }, r.Shape.ToArray());
        Assert.Equal(1, r[0, 0]);
        Assert.Equal(3, r[1, 0]);
    }

    [Fact]
    public void Nonzero_ReturnsCoordsOf2DTensor()
    {
        var x = T(new[] { 0f, 1f, 0f, 2f, 0f, 0f }, 2, 3);
        var r = E.TensorNonzero(x);
        Assert.Equal(new[] { 2, 2 }, r.Shape.ToArray());
        Assert.Equal(0, r[0, 0]); Assert.Equal(1, r[0, 1]);
        Assert.Equal(1, r[1, 0]); Assert.Equal(0, r[1, 1]);
    }

    [Fact]
    public void CountNonzero_Counts()
    {
        var x = T(new[] { 0f, 1f, 0f, 2f, 3f, 0f }, 6);
        Assert.Equal(3, E.TensorCountNonzero(x));
    }

    [Fact]
    public void Triu_RequiresRank2_Throws()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        Assert.Throws<System.ArgumentException>(() => E.TensorTriu(x));
    }
}
