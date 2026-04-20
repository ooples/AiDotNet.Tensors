using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210KronInnerTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b) => System.MathF.Abs(a - b) < 1e-4f;

    [Fact]
    public void Kron_2x2_2x2_ProducesBlock4x4()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = T(new[] { 0f, 5f, 6f, 7f }, 2, 2);
        var r = E.TensorKron(a, b);
        Assert.Equal(new[] { 4, 4 }, r.Shape.ToArray());
        // A[0,0]=1 → block B = [0 5; 6 7]
        Assert.Equal(0f, r[0, 0]); Assert.Equal(5f, r[0, 1]);
        // A[0,1]=2 → block 2·B = [0 10; 12 14]
        Assert.Equal(10f, r[0, 3]);
        // A[1,1]=4 → block 4·B
        Assert.Equal(28f, r[3, 3]);
    }

    [Fact]
    public void Kron_Rank1_ProducesFlatOuterProduct()
    {
        var a = T(new[] { 1f, 2f }, 2);
        var b = T(new[] { 3f, 4f }, 2);
        var r = E.TensorKron(a, b);
        // General form (torch.kron): 1-D inputs stay 1-D; shape [a.len * b.len].
        Assert.Equal(new[] { 4 }, r.Shape.ToArray());
        Assert.Equal(new[] { 3f, 4f, 6f, 8f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Inner_Rank1_EqualsVecDot()
    {
        var a = T(new[] { 1f, 2f, 3f }, 3);
        var b = T(new[] { 4f, 5f, 6f }, 3);
        var r = E.TensorInner(a, b);
        Assert.Empty(r.Shape.ToArray());
        Assert.True(Close(32f, r[System.Array.Empty<int>()]));
    }

    [Fact]
    public void Inner_2D_ContractsLastAxis()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = T(new[] { 5f, 6f, 7f, 8f }, 2, 2);
        var r = E.TensorInner(a, b);
        // Shape: a.shape[:-1] + b.shape[:-1] = (2,) + (2,) = (2, 2).
        Assert.Equal(new[] { 2, 2 }, r.Shape.ToArray());
        // r[i,j] = Σ_k a[i,k] · b[j,k]
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                float expected = 0;
                for (int k = 0; k < 2; k++) expected += a[i, k] * b[j, k];
                Assert.True(Close(expected, r[i, j]));
            }
    }

    [Fact]
    public void Inner_LastAxisMismatch_Throws()
    {
        var a = T(new[] { 1f, 2f, 3f }, 3);
        var b = T(new[] { 4f, 5f }, 2);
        Assert.Throws<System.ArgumentException>(() => E.TensorInner(a, b));
    }
}
