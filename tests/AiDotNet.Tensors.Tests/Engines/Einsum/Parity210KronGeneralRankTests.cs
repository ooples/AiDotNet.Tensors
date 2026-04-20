using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

/// <summary>
/// Kron rank > 2 coverage — the previous implementation only handled rank 1
/// and 2; the general implementation in CpuEngine.Parity210.cs handles
/// arbitrary rank via per-axis index decomposition.
/// </summary>
public class Parity210KronGeneralRankTests
{
    private static CpuEngine E => new CpuEngine();

    [Fact]
    public void Kron_1D_1D_ProducesOuterProductVector()
    {
        // [2] ⊗ [3] should give a 6-element vector (or [1,6] with promotion).
        var a = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var b = new Tensor<float>(new[] { 3f, 4f, 5f }, new[] { 3 });
        var r = E.TensorKron(a, b);
        // General form: outShape[0] = 2 * 3 = 6
        Assert.Equal(new[] { 6 }, r.Shape.ToArray());
        // Values: 1*[3,4,5], 2*[3,4,5] = [3,4,5, 6,8,10]
        Assert.Equal(new[] { 3f, 4f, 5f, 6f, 8f, 10f }, r.GetDataArray());
    }

    [Fact]
    public void Kron_3D_3D_Generalises()
    {
        // Shape [2, 1, 2] ⊗ [1, 2, 1] → [2, 2, 2]
        var a = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 1, 2 });
        var b = new Tensor<float>(new[] { 5f, 6f }, new[] { 1, 2, 1 });
        var r = E.TensorKron(a, b);
        Assert.Equal(new[] { 2, 2, 2 }, r.Shape.ToArray());

        // Verify a couple of positions:
        // y[0, 0, 0] = a[0,0,0] * b[0,0,0] = 1 * 5 = 5
        // y[0, 1, 1] = a[0,0,1] * b[0,1,0] = 2 * 6 = 12
        Assert.Equal(5f, r[0, 0, 0]);
        Assert.Equal(12f, r[0, 1, 1]);
    }

    [Fact]
    public void Kron_RankMismatch_RightAlignsWithOnes()
    {
        // [2, 3] ⊗ [2] should right-align b to shape [1, 2].
        // Output shape = [2*1, 3*2] = [2, 6].
        var a = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });
        var b = new Tensor<float>(new[] { 10f, 20f }, new[] { 2 });
        var r = E.TensorKron(a, b);
        Assert.Equal(new[] { 2, 6 }, r.Shape.ToArray());
        // Row 0 of a = [1, 2, 3] → [1*10, 1*20, 2*10, 2*20, 3*10, 3*20] = [10, 20, 20, 40, 30, 60]
        Assert.Equal(10f, r[0, 0]);
        Assert.Equal(20f, r[0, 1]);
        Assert.Equal(60f, r[0, 5]);
    }

    [Fact]
    public void Kron_2D_AgreesWith_OldPath()
    {
        // The pre-generalisation 2-D implementation is the reference.
        // Manually compute A⊗B = [[a00*B, a01*B], [a10*B, a11*B]]
        var a = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var b = new Tensor<float>(new[] { 5f, 6f, 7f, 8f }, new[] { 2, 2 });
        var r = E.TensorKron(a, b);
        Assert.Equal(new[] { 4, 4 }, r.Shape.ToArray());
        // Top-left 2x2 block = 1 * B
        Assert.Equal(5f, r[0, 0]);
        Assert.Equal(8f, r[1, 1]);
        // Top-right 2x2 block = 2 * B
        Assert.Equal(10f, r[0, 2]);
        Assert.Equal(16f, r[1, 3]);
        // Bottom-right = 4 * B
        Assert.Equal(32f, r[3, 3]);
    }
}
