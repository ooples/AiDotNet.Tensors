using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tests proving that CpuEngine operations work correctly on non-contiguous
/// stride-based views (transposed, sliced) without requiring Contiguous() copies.
/// These validate the stride-aware implementation from issue #59.
/// </summary>
public class StrideAwareOpsTests
{
    private readonly CpuEngine E = new();
    private const float Tol = 1e-4f;

    // ================================================================
    // MatMul with transposed views
    // ================================================================

    [Fact]
    public void MatMul2D_TransposedA_MatchesContiguous()
    {
        // A is [4,3] transposed to [3,4], B is [4,5]
        // Result should be [3,5]
        var aOrig = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, [4, 3]);
        var aT = aOrig.Transpose(new[] { 1, 0 }); // [3,4] view, non-contiguous
        Assert.False(aT.IsContiguous);

        var b = new Tensor<float>(new float[] { 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0 }, [4, 5]);

        // Compute via strided view
        var result = E.BatchMatMul(aT, b);
        Assert.Equal(new[] { 3, 5 }, result.Shape.ToArray());

        // Compute via explicit contiguous copy (reference)
        var aTContiguous = aT.Contiguous();
        Assert.True(aTContiguous.IsContiguous);
        var expected = E.BatchMatMul(aTContiguous, b);

        // Results must match
        var rArr = result.GetDataArray();
        var eArr = expected.GetDataArray();
        for (int i = 0; i < result.Length; i++)
            Assert.True(Math.Abs(rArr[i] - eArr[i]) < Tol, $"Mismatch at [{i}]: {rArr[i]} vs {eArr[i]}");
    }

    [Fact]
    public void MatMul2D_TransposedB_MatchesContiguous()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);
        var bOrig = new Tensor<float>(new float[] { 1, 4, 2, 5, 3, 6 }, [2, 3]);
        var bT = bOrig.Transpose(new[] { 1, 0 }); // [3,2] view
        Assert.False(bT.IsContiguous);

        var result = E.BatchMatMul(a, bT);
        Assert.Equal(new[] { 2, 2 }, result.Shape.ToArray());

        var expected = E.BatchMatMul(a, bT.Contiguous());
        var rArr = result.GetDataArray();
        var eArr = expected.GetDataArray();
        for (int i = 0; i < result.Length; i++)
            Assert.True(Math.Abs(rArr[i] - eArr[i]) < Tol, $"Mismatch at [{i}]: {rArr[i]} vs {eArr[i]}");
    }

    // ================================================================
    // Element-wise ops with transposed views (real strided iteration)
    // ================================================================

    [Fact]
    public void Add_TransposedView_MatchesContiguous()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);
        var aT = a.Transpose(new[] { 1, 0 }); // [3,2] non-contiguous

        var b = new Tensor<float>(new float[] { 10, 20, 30, 40, 50, 60 }, [3, 2]);

        var result = E.TensorAdd(aT, b);
        var expected = E.TensorAdd(aT.Contiguous(), b);

        AssertClose(result, expected);
    }

    [Fact]
    public void Subtract_TransposedView_MatchesContiguous()
    {
        var a = new Tensor<float>(Enumerable.Range(0, 12).Select(i => (float)i).ToArray(), [3, 4]);
        var aT = a.Transpose(new[] { 1, 0 }); // [4,3]
        var b = new Tensor<float>(Enumerable.Range(0, 12).Select(i => (float)(i * 2)).ToArray(), [4, 3]);

        var result = E.TensorSubtract(aT, b);
        var expected = E.TensorSubtract(aT.Contiguous(), b);

        AssertClose(result, expected);
    }

    [Fact]
    public void Multiply_TransposedView_MatchesContiguous()
    {
        var a = new Tensor<float>(Enumerable.Range(1, 6).Select(i => (float)i).ToArray(), [2, 3]);
        var aT = a.Transpose(new[] { 1, 0 }); // [3,2]
        var b = new Tensor<float>(new float[] { 2, 3, 4, 5, 6, 7 }, [3, 2]);

        var result = E.TensorMultiply(aT, b);
        var expected = E.TensorMultiply(aT.Contiguous(), b);

        AssertClose(result, expected);
    }

    [Fact]
    public void Divide_TransposedView_MatchesContiguous()
    {
        var a = new Tensor<float>(Enumerable.Range(1, 6).Select(i => (float)(i * 10)).ToArray(), [2, 3]);
        var aT = a.Transpose(new[] { 1, 0 }); // [3,2]
        var b = new Tensor<float>(new float[] { 2, 3, 4, 5, 6, 7 }, [3, 2]);

        var result = E.TensorDivide(aT, b);
        var expected = E.TensorDivide(aT.Contiguous(), b);

        AssertClose(result, expected);
    }

    [Fact]
    public void Add_BothTransposed_MatchesContiguous()
    {
        var a = new Tensor<float>(Enumerable.Range(0, 6).Select(i => (float)i).ToArray(), [2, 3]);
        var b = new Tensor<float>(Enumerable.Range(6, 6).Select(i => (float)i).ToArray(), [2, 3]);
        var aT = a.Transpose(new[] { 1, 0 });
        var bT = b.Transpose(new[] { 1, 0 });

        var result = E.TensorAdd(aT, bT);
        var expected = E.TensorAdd(aT.Contiguous(), bT.Contiguous());

        AssertClose(result, expected);
    }

    // ================================================================
    // Scalar ops with transposed views
    // ================================================================

    [Fact]
    public void MultiplyScalar_TransposedView_MatchesContiguous()
    {
        var a = new Tensor<float>(Enumerable.Range(1, 6).Select(i => (float)i).ToArray(), [2, 3]);
        var aT = a.Transpose(new[] { 1, 0 });

        var result = E.TensorMultiplyScalar(aT, 3.0f);
        var expected = E.TensorMultiplyScalar(aT.Contiguous(), 3.0f);

        AssertClose(result, expected);
    }

    [Fact]
    public void AddScalar_TransposedView_MatchesContiguous()
    {
        var a = new Tensor<float>(Enumerable.Range(1, 6).Select(i => (float)i).ToArray(), [2, 3]);
        var aT = a.Transpose(new[] { 1, 0 });

        var result = E.TensorAddScalar(aT, 100f);
        var expected = E.TensorAddScalar(aT.Contiguous(), 100f);

        AssertClose(result, expected);
    }

    // ================================================================
    // Unary ops with transposed views
    // ================================================================

    [Fact]
    public void Exp_TransposedView_MatchesContiguous()
    {
        var a = new Tensor<float>(new float[] { 0, 0.5f, 1, 1.5f, 2, 2.5f }, [2, 3]);
        var aT = a.Transpose(new[] { 1, 0 });

        var result = E.TensorExp(aT);
        var expected = E.TensorExp(aT.Contiguous());

        AssertClose(result, expected, 1e-3f);
    }

    [Fact]
    public void Sigmoid_TransposedView_MatchesContiguous()
    {
        var a = new Tensor<float>(new float[] { -2, -1, 0, 1, 2, 3 }, [2, 3]);
        var aT = a.Transpose(new[] { 1, 0 });

        var result = E.Sigmoid(aT);
        var expected = E.Sigmoid(aT.Contiguous());

        AssertClose(result, expected, 1e-3f);
    }

    [Fact]
    public void ReLU_TransposedView_MatchesContiguous()
    {
        var a = new Tensor<float>(new float[] { -3, -1, 0, 1, 2, 5 }, [2, 3]);
        var aT = a.Transpose(new[] { 1, 0 });

        var result = E.ReLU(aT);
        var expected = E.ReLU(aT.Contiguous());

        AssertClose(result, expected);
    }

    // ================================================================
    // Comparison ops with transposed views
    // ================================================================

    [Fact]
    public void GreaterThan_TransposedView_NoThrow()
    {
        var a = new Tensor<float>(new float[] { 1, 5, 3, 4, 2, 6 }, [2, 3]);
        var aT = a.Transpose(new[] { 1, 0 });
        var b = new Tensor<float>(new float[] { 3, 3, 3, 3, 3, 3 }, [3, 2]);

        var result = E.TensorGreaterThan(aT, b);
        Assert.Equal(new[] { 3, 2 }, result.Shape.ToArray());
    }

    // ================================================================
    // 3D BatchMatMul with transposed views
    // ================================================================

    [Fact]
    public void BatchMatMul3D_TransposedInput_MatchesContiguous()
    {
        var rng = new Random(42);
        var aOrig = new Tensor<float>(Enumerable.Range(0, 2 * 4 * 3).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray(), [2, 4, 3]);
        var aT = aOrig.Transpose(new[] { 0, 2, 1 }); // [2, 3, 4]
        var b = new Tensor<float>(Enumerable.Range(0, 2 * 4 * 5).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray(), [2, 4, 5]);

        var result = E.TensorBatchMatMul(aT, b);
        Assert.Equal(new[] { 2, 3, 5 }, result.Shape.ToArray());

        var expected = E.TensorBatchMatMul(aT.Contiguous(), b);
        AssertClose(result, expected, 1e-3f);
    }

    // ================================================================
    // Normalization with transposed views
    // ================================================================

    [Fact]
    public void LayerNorm_TransposedInput_NoThrow()
    {
        var x = new Tensor<float>(Enumerable.Range(0, 12).Select(i => (float)i).ToArray(), [3, 4]);
        var xT = x.Transpose(new[] { 1, 0 }); // [4,3]
        var gamma = new Tensor<float>(new float[] { 1, 1, 1 }, [3]);
        var beta = new Tensor<float>(new float[] { 0, 0, 0 }, [3]);

        // Should not throw
        var result = E.TensorLayerNorm(xT, gamma, beta, 1e-5);
        Assert.Equal(new[] { 4, 3 }, result.Shape.ToArray());
    }

    // ================================================================
    // Helpers
    // ================================================================

    private void AssertClose(Tensor<float> a, Tensor<float> b, float tol = Tol)
    {
        Assert.Equal(a.Shape.ToArray(), b.Shape.ToArray());
        var aArr = a.GetDataArray();
        var bArr = b.GetDataArray();
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(aArr[i] - bArr[i]) < tol, $"[{i}]: {aArr[i]} vs {bArr[i]}");
    }
}
