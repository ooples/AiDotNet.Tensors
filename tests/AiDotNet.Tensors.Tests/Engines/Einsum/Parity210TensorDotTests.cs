using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210TensorDotTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b) => MathF.Abs(a - b) < 1e-4f;

    [Fact]
    public void TensorDot_MatMul_ViaAxis1AndAxis0()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var b = T(new[] { 7f, 8f, 9f, 10f, 11f, 12f }, 3, 2);
        // Contract a's axis 1 with b's axis 0 → matmul.
        var r = E.TensorDot(a, b, new[] { 1 }, new[] { 0 });
        Assert.Equal(new[] { 2, 2 }, r.Shape.ToArray());
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                float expected = 0;
                for (int k = 0; k < 3; k++) expected += a[i, k] * b[k, j];
                Assert.True(Close(expected, r[i, j]));
            }
    }

    [Fact]
    public void TensorDot_FullReduction_ReturnsScalar()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = T(new[] { 5f, 6f, 7f, 8f }, 2, 2);
        // Contract both axes → scalar sum of element-wise products.
        var r = E.TensorDot(a, b, new[] { 0, 1 }, new[] { 0, 1 });
        Assert.Empty(r.Shape.ToArray());
        Assert.True(Close(1 * 5 + 2 * 6 + 3 * 7 + 4 * 8, r[System.Array.Empty<int>()]));
    }

    [Fact]
    public void TensorDot_NoContraction_IsOuterProduct()
    {
        var a = T(new[] { 1f, 2f }, 2);
        var b = T(new[] { 3f, 4f, 5f }, 3);
        var r = E.TensorDot(a, b, System.Array.Empty<int>(), System.Array.Empty<int>());
        Assert.Equal(new[] { 2, 3 }, r.Shape.ToArray());
        Assert.True(Close(1 * 3, r[0, 0]));
        Assert.True(Close(2 * 5, r[1, 2]));
    }

    [Fact]
    public void VecDot_ComputesInnerProduct()
    {
        var a = T(new[] { 1f, 2f, 3f }, 3);
        var b = T(new[] { 4f, 5f, 6f }, 3);
        Assert.True(Close(32f, E.TensorVecDot(a, b)));  // 1*4 + 2*5 + 3*6
    }

    [Fact]
    public void VecDot_LengthMismatch_Throws()
    {
        var a = T(new[] { 1f, 2f }, 2);
        var b = T(new[] { 3f, 4f, 5f }, 3);
        Assert.Throws<System.ArgumentException>(() => E.TensorVecDot(a, b));
    }
}
