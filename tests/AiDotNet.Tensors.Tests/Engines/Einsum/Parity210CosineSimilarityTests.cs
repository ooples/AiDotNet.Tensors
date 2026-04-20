using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210CosineSimilarityTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b, float tol = 1e-4f) => MathF.Abs(a - b) <= tol * (1 + MathF.Abs(a) + MathF.Abs(b));

    [Fact]
    public void Cosine_IdenticalVectors_IsOne()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var r = E.TensorCosineSimilarity(x, x);
        Assert.True(Close(1f, r[Array.Empty<int>()]));
    }

    [Fact]
    public void Cosine_Orthogonal_IsZero()
    {
        var a = T(new[] { 1f, 0f }, 2);
        var b = T(new[] { 0f, 1f }, 2);
        var r = E.TensorCosineSimilarity(a, b);
        Assert.True(Close(0f, r[Array.Empty<int>()]));
    }

    [Fact]
    public void Cosine_Opposite_IsNegativeOne()
    {
        var a = T(new[] { 1f, 2f }, 2);
        var b = T(new[] { -1f, -2f }, 2);
        var r = E.TensorCosineSimilarity(a, b);
        Assert.True(Close(-1f, r[Array.Empty<int>()]));
    }

    [Fact]
    public void Cosine_2D_Along_LastDim_ReturnsPerRowSimilarity()
    {
        var a = T(new[] { 1f, 0f, 0f, 1f }, 2, 2);
        var b = T(new[] { 1f, 0f, 1f, 0f }, 2, 2);
        var r = E.TensorCosineSimilarity(a, b, dim: 1);
        Assert.Equal(new[] { 2 }, r.Shape.ToArray());
        Assert.True(Close(1f, r[0]));  // (1,0) · (1,0) = 1
        Assert.True(Close(0f, r[1]));  // (0,1) · (1,0) = 0
    }

    [Fact]
    public void Cosine_ShapeMismatch_Throws()
    {
        var a = T(new[] { 1f, 2f }, 2);
        var b = T(new[] { 1f, 2f, 3f }, 3);
        Assert.Throws<ArgumentException>(() => E.TensorCosineSimilarity(a, b));
    }
}
