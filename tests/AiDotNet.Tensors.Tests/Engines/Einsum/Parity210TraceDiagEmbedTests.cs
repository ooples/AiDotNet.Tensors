using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210TraceDiagEmbedTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void Trace_2D_SumsDiagonal()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f }, 3, 3);
        Assert.Equal(1 + 5 + 9, E.TensorTrace(x));
    }

    [Fact]
    public void Trace_Rectangular_StopsAtMinDim()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        Assert.Equal(1 + 5, E.TensorTrace(x));
    }

    [Fact]
    public void Trace_NonMatrix_Throws()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        Assert.Throws<System.ArgumentException>(() => E.TensorTrace(x));
    }

    [Fact]
    public void DiagEmbed_1D_MainDiagonal()
    {
        var x = T(new[] { 7f, 8f, 9f }, 3);
        var r = E.TensorDiagEmbed(x);
        Assert.Equal(new[] { 3, 3 }, r.Shape.ToArray());
        var expected = new[] {
            7f, 0f, 0f,
            0f, 8f, 0f,
            0f, 0f, 9f
        };
        Assert.Equal(expected, r.AsSpan().ToArray());
    }

    [Fact]
    public void DiagEmbed_SuperDiagonal()
    {
        var x = T(new[] { 1f, 2f }, 2);
        var r = E.TensorDiagEmbed(x, offset: 1);
        // 3x3 zero matrix with 1 on (0,1) and 2 on (1,2).
        Assert.Equal(new[] { 3, 3 }, r.Shape.ToArray());
        Assert.Equal(1f, r[0, 1]);
        Assert.Equal(2f, r[1, 2]);
    }

    [Fact]
    public void DiagEmbed_SubDiagonal()
    {
        var x = T(new[] { 1f, 2f }, 2);
        var r = E.TensorDiagEmbed(x, offset: -1);
        Assert.Equal(1f, r[1, 0]);
        Assert.Equal(2f, r[2, 1]);
    }

    [Fact]
    public void DiagEmbed_Batched()
    {
        // 2 batches of rank-1 vectors of length 3.
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var r = E.TensorDiagEmbed(x);
        Assert.Equal(new[] { 2, 3, 3 }, r.Shape.ToArray());
        Assert.Equal(1f, r[0, 0, 0]);
        Assert.Equal(2f, r[0, 1, 1]);
        Assert.Equal(3f, r[0, 2, 2]);
        Assert.Equal(4f, r[1, 0, 0]);
        Assert.Equal(6f, r[1, 2, 2]);
    }
}
