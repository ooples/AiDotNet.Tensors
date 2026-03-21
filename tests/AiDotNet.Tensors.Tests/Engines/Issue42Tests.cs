using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Regression tests for Issue #42:
/// CpuEngine.TensorMultiply doesn't support broadcasting — breaks backward pass in 3+ layers
/// </summary>
public class Issue42Tests
{
    private readonly CpuEngine _engine = new();
    private const double Tolerance = 1e-10;
    private const float FloatTolerance = 1e-5f;

    #region Broadcasting multiply

    [Fact]
    public void TensorMultiply_BroadcastChannelScale_Works()
    {
        // [1,4,4,4] * [1,1,1,4] → broadcast b across spatial dims
        var a = new Tensor<double>(new[] { 1, 4, 4, 4 });
        var b = new Tensor<double>(new[] { 1, 1, 1, 4 });

        // Fill a with 1s, b with scale factors
        for (int i = 0; i < a.Length; i++) a.SetFlat(i, 1.0);
        b[0, 0, 0, 0] = 2.0;
        b[0, 0, 0, 1] = 3.0;
        b[0, 0, 0, 2] = 4.0;
        b[0, 0, 0, 3] = 5.0;

        var result = _engine.TensorMultiply(a, b);

        Assert.Equal(new[] { 1, 4, 4, 4 }, result.Shape);
        // Channel 0 should be scaled by 2, channel 3 by 5
        Assert.Equal(2.0, result[0, 0, 0, 0], Tolerance);
        Assert.Equal(5.0, result[0, 0, 0, 3], Tolerance);
    }

    [Fact]
    public void TensorMultiply_Broadcast2D_Works()
    {
        // [4,2] * [1,1] → broadcast scalar across all elements
        var a = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 4, 2 });
        var b = new Tensor<double>(new double[] { 10 }, new[] { 1, 1 });

        var result = _engine.TensorMultiply(a, b);

        Assert.Equal(new[] { 4, 2 }, result.Shape);
        Assert.Equal(10.0, result[0, 0], Tolerance);
        Assert.Equal(80.0, result[3, 1], Tolerance);
    }

    [Fact]
    public void TensorMultiply_ExactShapeMatch_StillWorks()
    {
        // Same-shape multiply should still work (no regression)
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });

        var result = _engine.TensorMultiply(a, b);

        Assert.Equal(new[] { 2, 2 }, result.Shape);
        Assert.Equal(5f, result[0, 0], FloatTolerance);
        Assert.Equal(32f, result[1, 1], FloatTolerance);
    }

    [Fact]
    public void TensorMultiply_BroadcastColumnVector_Works()
    {
        // [3,4] * [3,1] → broadcast column across columns
        var a = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, new[] { 3, 4 });
        var b = new Tensor<double>(new double[] { 10, 20, 30 }, new[] { 3, 1 });

        var result = _engine.TensorMultiply(a, b);

        Assert.Equal(new[] { 3, 4 }, result.Shape);
        Assert.Equal(10.0, result[0, 0], Tolerance);   // 1 * 10
        Assert.Equal(40.0, result[0, 3], Tolerance);   // 4 * 10
        Assert.Equal(100.0, result[1, 0], Tolerance);  // 5 * 20
        Assert.Equal(360.0, result[2, 3], Tolerance);  // 12 * 30
    }

    #endregion

    #region Reshape error message

    [Fact]
    public void Reshape_MismatchedElements_ShowsElementCounts()
    {
        var tensor = new Tensor<float>(new float[32], new[] { 4, 8 });

        var ex = Assert.Throws<ArgumentException>(() => tensor.Reshape(8));

        Assert.Contains("32 elements", ex.Message);
        Assert.Contains("8 elements", ex.Message);
    }

    #endregion
}
