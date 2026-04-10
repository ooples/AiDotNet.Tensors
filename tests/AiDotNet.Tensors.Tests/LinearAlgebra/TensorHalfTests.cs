#if NET7_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Integration tests for Tensor&lt;Half&gt; support (issue #118).
/// Verifies that Half-precision tensors work throughout the system.
/// </summary>
public class TensorHalfTests
{
    [Fact]
    public void CreateHalfTensor_Works()
    {
        var data = new Half[] { (Half)1.0f, (Half)2.0f, (Half)3.0f, (Half)4.0f };
        var tensor = new Tensor<Half>(data, new[] { 2, 2 });

        Assert.Equal(4, tensor.Length);
        Assert.Equal(2, tensor.Rank);
        Assert.Equal((Half)1.0f, tensor[0, 0]);
        Assert.Equal((Half)4.0f, tensor[1, 1]);
    }

    [Fact]
    public void HalfTensor_Add()
    {
        var engine = new CpuEngine();
        var a = new Tensor<Half>(new Half[] { (Half)1.0f, (Half)2.0f, (Half)3.0f, (Half)4.0f }, new[] { 4 });
        var b = new Tensor<Half>(new Half[] { (Half)10.0f, (Half)20.0f, (Half)30.0f, (Half)40.0f }, new[] { 4 });

        var result = engine.TensorAdd(a, b);

        Assert.Equal((Half)11.0f, result[0]);
        Assert.Equal((Half)22.0f, result[1]);
        Assert.Equal((Half)33.0f, result[2]);
        Assert.Equal((Half)44.0f, result[3]);
    }

    [Fact]
    public void HalfTensor_Multiply()
    {
        var engine = new CpuEngine();
        var a = new Tensor<Half>(new Half[] { (Half)2.0f, (Half)3.0f }, new[] { 2 });
        var b = new Tensor<Half>(new Half[] { (Half)4.0f, (Half)5.0f }, new[] { 2 });

        var result = engine.TensorMultiply(a, b);

        Assert.Equal((Half)8.0f, result[0]);
        Assert.Equal((Half)15.0f, result[1]);
    }

    [Fact]
    public void HalfTensor_Sigmoid()
    {
        var engine = new CpuEngine();
        var input = new Tensor<Half>(new Half[] { (Half)0.0f, (Half)1.0f, (Half)(-1.0f) }, new[] { 3 });

        var result = engine.Sigmoid(input);

        // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269
        float s0 = (float)result[0];
        float s1 = (float)result[1];
        float s2 = (float)result[2];
        Assert.True(MathF.Abs(s0 - 0.5f) < 0.01f, $"sigmoid(0) = {s0}, expected ~0.5");
        Assert.True(s1 > 0.7f && s1 < 0.75f, $"sigmoid(1) = {s1}, expected ~0.731");
        Assert.True(s2 > 0.25f && s2 < 0.3f, $"sigmoid(-1) = {s2}, expected ~0.269");
    }

    [Fact]
    public void HalfTensor_Reshape()
    {
        var data = new Half[] { (Half)1.0f, (Half)2.0f, (Half)3.0f, (Half)4.0f, (Half)5.0f, (Half)6.0f };
        var tensor = new Tensor<Half>(data, new[] { 2, 3 });
        var reshaped = tensor.Reshape(3, 2);

        Assert.Equal(new[] { 3, 2 }, reshaped.Shape.ToArray());
        Assert.Equal((Half)1.0f, reshaped[0, 0]);
        Assert.Equal((Half)6.0f, reshaped[2, 1]);
    }

    [Fact]
    public void HalfTensor_Exp()
    {
        var engine = new CpuEngine();
        var input = new Tensor<Half>(new Half[] { (Half)0.0f, (Half)1.0f }, new[] { 2 });

        var result = engine.TensorExp(input);

        float e0 = (float)result[0];
        float e1 = (float)result[1];
        Assert.True(MathF.Abs(e0 - 1.0f) < 0.01f, $"exp(0) = {e0}, expected 1.0");
        Assert.True(MathF.Abs(e1 - 2.718f) < 0.05f, $"exp(1) = {e1}, expected ~2.718");
    }

    [Fact]
    public void HalfTensor_MatMul()
    {
        var engine = new CpuEngine();
        // Identity matrix @ vector
        var identity = new Tensor<Half>(new Half[] { (Half)1.0f, (Half)0.0f, (Half)0.0f, (Half)1.0f }, new[] { 2, 2 });
        var vec = new Tensor<Half>(new Half[] { (Half)3.0f, (Half)7.0f }, new[] { 2, 1 });

        var result = engine.TensorMatMul(identity, vec);

        Assert.Equal((Half)3.0f, result[0, 0]);
        Assert.Equal((Half)7.0f, result[1, 0]);
    }

    [Fact]
    public void HalfTensor_Sum()
    {
        var engine = new CpuEngine();
        var input = new Tensor<Half>(new Half[] { (Half)1.0f, (Half)2.0f, (Half)3.0f, (Half)4.0f }, new[] { 4 });

        var sum = engine.TensorSum(input);

        Assert.Equal((Half)10.0f, sum);
    }
}
#endif
