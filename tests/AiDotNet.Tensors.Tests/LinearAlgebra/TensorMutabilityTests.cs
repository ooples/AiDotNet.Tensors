using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Tests that tensors returned by Engine operations are mutable (issue #47).
/// </summary>
public class TensorMutabilityTests
{
    [Fact]
    public void TensorSubtract_ResultIsMutable()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [4]);
        var b = new Tensor<float>(new float[] { 0.5f, 0.5f, 0.5f, 0.5f }, [4]);

        var result = AiDotNetEngine.Current.TensorSubtract(a, b);

        Assert.Equal(0.5f, result[0]);
        result[0] = 99.0f;
        Assert.Equal(99.0f, result[0]);
    }

    [Fact]
    public void TensorAdd_ResultIsMutable()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [4]);
        var b = new Tensor<float>(new float[] { 10, 20, 30, 40 }, [4]);

        var result = AiDotNetEngine.Current.TensorAdd(a, b);

        Assert.Equal(11f, result[0]);
        result[0] = 99.0f;
        Assert.Equal(99.0f, result[0]);
    }

    [Fact]
    public void TensorMultiplyScalar_ResultIsMutable()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [4]);

        var result = AiDotNetEngine.Current.TensorMultiplyScalar(a, 2.0f);

        Assert.Equal(2.0f, result[0]);
        result[0] = 99.0f;
        Assert.Equal(99.0f, result[0]);
    }

    [Fact]
    public void TensorMatMul_ResultIsMutable()
    {
        var a = new Tensor<float>(new float[] { 1, 0, 0, 1 }, [2, 2]);
        var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, [2, 2]);

        var result = AiDotNetEngine.Current.TensorMatMul(a, b);

        Assert.Equal(5.0f, result[0, 0]);
        result[0, 0] = 99.0f;
        Assert.Equal(99.0f, result[0, 0]);
    }

    [Fact]
    public void TensorSubtract_DataSpanIsMutable()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [4]);
        var b = new Tensor<float>(new float[] { 0.5f, 0.5f, 0.5f, 0.5f }, [4]);

        var result = AiDotNetEngine.Current.TensorSubtract(a, b);

        result.Data.Span[0] = 99.0f;
        Assert.Equal(99.0f, result[0]);
    }

    [Fact]
    public void TensorSubtract_LargePooled_ResultIsMutable()
    {
        // Large tensor to trigger ArrayPool/ThreadLocalCache allocation path
        int size = 100_000;
        var a = new Tensor<float>(new int[] { size });
        var b = new Tensor<float>(new int[] { size });
        for (int i = 0; i < size; i++) { a[i] = 1.0f; b[i] = 0.5f; }

        var result = AiDotNetEngine.Current.TensorSubtract(a, b);

        Assert.Equal(0.5f, result[0]);
        result[0] = 99.0f;
        Assert.Equal(99.0f, result[0]);

        // Also check via Data.Span
        result.Data.Span[1] = 77.0f;
        Assert.Equal(77.0f, result[1]);
    }

    [Fact]
    public void TensorMultiplyScalar_LargePooled_ResultIsMutable()
    {
        int size = 100_000;
        var a = new Tensor<float>(new int[] { size });
        for (int i = 0; i < size; i++) a[i] = 2.0f;

        var result = AiDotNetEngine.Current.TensorMultiplyScalar(a, 3.0f);

        Assert.Equal(6.0f, result[0]);
        result[0] = 99.0f;
        Assert.Equal(99.0f, result[0]);
    }

    [Fact]
    public void EngineResult_SetFlat_Persists()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [4]);
        var b = new Tensor<float>(new float[] { 0.5f, 0.5f, 0.5f, 0.5f }, [4]);

        var result = AiDotNetEngine.Current.TensorSubtract(a, b);

        result.SetFlat(0, 99.0f);
        Assert.Equal(99.0f, result[0]);
    }
}
