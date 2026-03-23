using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Regression tests for Issue #41:
/// Bug 1: Engine.TensorMatMul returns zeros for 2D matrix multiplication with column vectors
/// Bug 2: Tensor indexer writes don't persist for Engine-created tensors
/// </summary>
public class Issue41Tests
{
    private readonly CpuEngine _engine = new();
    private const float Tolerance = 1e-5f;

    #region Bug 1: TensorMatMul column vector

    [Fact]
    public void TensorMatMul_MatrixTimesColumnVector_ReturnsNonZero()
    {
        // [3,3] @ [3,1] should produce non-zero result
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new[] { 3, 3 });
        var b = new Tensor<float>(new float[] { 1, 1, 1 }, new[] { 3, 1 });

        var result = _engine.TensorMatMul(a, b);

        Assert.Equal(new[] { 3, 1 }, result.Shape.ToArray());
        Assert.Equal(6f, result[0, 0], Tolerance);   // 1+2+3
        Assert.Equal(15f, result[1, 0], Tolerance);   // 4+5+6
        Assert.Equal(24f, result[2, 0], Tolerance);   // 7+8+9
    }

    [Fact]
    public void TensorMatMul_100x100_Times_100x1_ReturnsNonZero()
    {
        // The exact shape from the bug report
        var aData = new float[100 * 100];
        var bData = new float[100];
        for (int i = 0; i < aData.Length; i++) aData[i] = 0.01f * (i % 100);
        for (int i = 0; i < bData.Length; i++) bData[i] = 1f;

        var a = new Tensor<float>(aData, new[] { 100, 100 });
        var b = new Tensor<float>(bData, new[] { 100, 1 });

        var result = _engine.TensorMatMul(a, b);

        Assert.Equal(new[] { 100, 1 }, result.Shape.ToArray());
        // Result should NOT be all zeros
        float sum = 0;
        for (int i = 0; i < 100; i++) sum += result[i, 0];
        Assert.True(sum > 0, $"TensorMatMul [100,100]@[100,1] returned all zeros. Sum={sum}");
    }

    [Fact]
    public void TensorMatMul_RowTimesMatrix_ReturnsNonZero()
    {
        // [1,3] @ [3,2] should produce [1,2]
        var a = new Tensor<float>(new float[] { 1, 2, 3 }, new[] { 1, 3 });
        var b = new Tensor<float>(new float[] { 1, 0, 0, 1, 1, 1 }, new[] { 3, 2 });

        var result = _engine.TensorMatMul(a, b);

        Assert.Equal(new[] { 1, 2 }, result.Shape.ToArray());
        Assert.Equal(4f, result[0, 0], Tolerance);   // 1*1+2*0+3*1
        Assert.Equal(5f, result[0, 1], Tolerance);   // 1*0+2*1+3*1
    }

    [Fact]
    public void TensorMatMul_OuterProduct_ReturnsNonZero()
    {
        // [3,1] @ [1,3] = outer product [3,3]
        var a = new Tensor<float>(new float[] { 1, 2, 3 }, new[] { 3, 1 });
        var b = new Tensor<float>(new float[] { 4, 5, 6 }, new[] { 1, 3 });

        var result = _engine.TensorMatMul(a, b);

        Assert.Equal(new[] { 3, 3 }, result.Shape.ToArray());
        Assert.Equal(4f, result[0, 0], Tolerance);
        Assert.Equal(12f, result[2, 0], Tolerance);
        Assert.Equal(18f, result[2, 2], Tolerance);
    }

    #endregion

    #region Bug 2: Indexer writes persist

    [Fact]
    public void EngineCreatedTensor_IndexerWrite_Persists()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var b = new Tensor<float>(new float[] { 1, 1, 1, 1 }, new[] { 2, 2 });

        var result = _engine.TensorSubtract(a, b);

        // Write via indexer
        result[0, 0] = 99f;

        // Read should return the written value
        Assert.Equal(99f, result[0, 0], Tolerance);
    }

    [Fact]
    public void EngineCreatedTensor_SetFlat_Persists()
    {
        var a = new Tensor<float>(new float[] { 10, 20, 30, 40 }, new[] { 4 });
        var scalar = _engine.TensorMultiplyScalar(a, 2f);

        // Write via SetFlat
        scalar.SetFlat(0, 999f);

        // Read should return written value
        Assert.Equal(999f, scalar.GetFlat(0), Tolerance);
    }

    [Fact]
    public void EngineCreatedTensor_SpanWrite_Persists()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var b = new Tensor<float>(new float[] { 1, 1, 1, 1 }, new[] { 2, 2 });

        var result = _engine.TensorSubtract(a, b);

        // Write via Memory.Span
        result.Memory.Span[0] = 42f;

        // Read should return written value
        Assert.Equal(42f, result.Memory.Span[0], Tolerance);
        Assert.Equal(42f, result[0, 0], Tolerance);
    }

    #endregion
}
