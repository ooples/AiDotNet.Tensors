using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210BinCountMultiDotTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static Tensor<int> I(int[] data, params int[] shape) => new Tensor<int>(data, shape);
    private static bool Close(float a, float b) => MathF.Abs(a - b) < 1e-3f;

    // --- BinCount -----------------------------------------------------

    [Fact]
    public void BinCount_Basic()
    {
        var x = I(new[] { 0, 1, 1, 2, 2, 2 }, 6);
        var r = E.TensorBinCount(x);
        Assert.Equal(new[] { 1, 2, 3 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void BinCount_WithMinLength_PadsTrailingZeros()
    {
        var x = I(new[] { 0, 2 }, 2);
        var r = E.TensorBinCount(x, minLength: 5);
        Assert.Equal(new[] { 1, 0, 1, 0, 0 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void BinCount_Empty_WithNoMinLength_IsEmpty()
    {
        var x = I(Array.Empty<int>(), 0);
        var r = E.TensorBinCount(x);
        Assert.Equal(new[] { 0 }, r.Shape.ToArray());
    }

    [Fact]
    public void BinCount_Negative_Throws()
    {
        var x = I(new[] { -1, 0 }, 2);
        Assert.Throws<ArgumentException>(() => E.TensorBinCount(x));
    }

    // --- MultiDot -----------------------------------------------------

    [Fact]
    public void MultiDot_SingleMatrix_IsIdentity()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var r = E.TensorMultiDot(new[] { a });
        Assert.Equal(a.AsSpan().ToArray(), r.AsSpan().ToArray());
    }

    [Fact]
    public void MultiDot_TwoMatrices_EqualsMatMul()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var b = T(new[] { 7f, 8f, 9f, 10f, 11f, 12f }, 3, 2);
        var r = E.TensorMultiDot(new[] { a, b });
        var matmul = E.TensorMatMul(a, b);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.True(Close(matmul[i, j], r[i, j]));
    }

    [Fact]
    public void MultiDot_ThreeMatrices_MatchesLeftToRight()
    {
        // A(2,3) · B(3,4) · C(4,2) → expect a 2×2.
        var a = T(new float[6], 2, 3);
        var b = T(new float[12], 3, 4);
        var c = T(new float[8], 4, 2);
        // Fill deterministically.
        for (int i = 0; i < 6; i++) a[i / 3, i % 3] = i + 1;
        for (int i = 0; i < 12; i++) b[i / 4, i % 4] = i + 1;
        for (int i = 0; i < 8; i++) c[i / 2, i % 2] = i + 1;

        var r = E.TensorMultiDot(new[] { a, b, c });
        // Reference via two matmuls.
        var ab = E.TensorMatMul(a, b);
        var abc = E.TensorMatMul(ab, c);
        Assert.Equal(abc.Shape.ToArray(), r.Shape.ToArray());
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.True(Close(abc[i, j], r[i, j]));
    }
}
