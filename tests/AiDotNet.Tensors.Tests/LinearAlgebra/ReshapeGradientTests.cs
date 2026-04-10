using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Integration tests proving Tensor.Reshape() propagates GradFn for gradient tape.
/// Verifies fix for issue #123.
/// </summary>
public class ReshapeGradientTests
{
    [Fact]
    public void Reshape_SetsGradFn_WhenTapeActive()
    {
        using var tape = new GradientTape<float>();
        var original = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var reshaped = original.Reshape(3, 2);

        Assert.NotNull(reshaped.GradFn);
    }

    [Fact]
    public void Reshape_NoGradFn_WhenNoTape()
    {
        var original = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var reshaped = original.Reshape(4);

        Assert.Null(reshaped.GradFn);
    }

    [Fact]
    public void Reshape_GradientsFlowBackward()
    {
        var engine = new CpuEngine();

        // weight is the parameter we want gradients for
        var weight = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var input = new Tensor<float>(new float[] { 1, 0, 0, 1 }, new[] { 2, 2 });

        using var tape = new GradientTape<float>();

        // Forward: matmul -> reshape -> sum
        var matmulResult = engine.TensorMatMul(input, weight); // [2,2]
        var reshaped = matmulResult.Reshape(4);                // [4] — this was breaking gradients
        var loss = engine.ReduceSum(reshaped, new[] { 0 }, false); // scalar

        var grads = tape.ComputeGradients(loss);

        // Gradient must flow through reshape back to weight
        Assert.True(grads.ContainsKey(weight),
            "Gradient for weight must exist — Reshape must not break the gradient chain");
        Assert.True(grads[weight].Length == 4, "Gradient shape must match weight shape");

        // Verify gradient is non-zero (sum of all outputs = gradient of 1 everywhere)
        var gradData = grads[weight].GetDataArray();
        bool anyNonZero = false;
        for (int i = 0; i < gradData.Length; i++)
            if (gradData[i] != 0f) anyNonZero = true;
        Assert.True(anyNonZero, "Gradient values must be non-zero");
    }

    [Fact]
    public void Reshape_MultipleReshapes_GradientsFlowThrough()
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });

        using var tape = new GradientTape<float>();

        // Multiple reshapes in sequence
        var r1 = weight.Reshape(3, 2);    // [3,2]
        var r2 = r1.Reshape(6);           // [6]
        var r3 = r2.Reshape(2, 3);        // [2,3]
        var loss = engine.ReduceSum(r3, new[] { 0, 1 }, false);

        var grads = tape.ComputeGradients(loss);

        Assert.True(grads.ContainsKey(weight),
            "Gradient must flow through multiple reshapes");
    }

    [Fact]
    public void Reshape_3DTo2D_GradientsFlowForTraining()
    {
        // Simulates the common pattern: batch reshape in Forward method
        var engine = new CpuEngine();
        var input = new Tensor<float>(
            new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
            new[] { 2, 2, 3 }); // [batch=2, seq=2, features=3]
        var weight = new Tensor<float>(
            new float[] { 1, 0, 0, 1, 0, 0 },
            new[] { 3, 2 }); // [features=3, output=2]

        using var tape = new GradientTape<float>();

        // Reshape [2,2,3] -> [4,3] for matmul (common in transformer layers)
        var flat = input.Reshape(4, 3);
        var output = engine.TensorMatMul(flat, weight); // [4, 2]
        var loss = engine.ReduceSum(output, new[] { 0, 1 }, false);

        var grads = tape.ComputeGradients(loss);

        Assert.True(grads.ContainsKey(weight),
            "Weight gradient must exist after reshape+matmul");
        Assert.True(grads.ContainsKey(input),
            "Input gradient must flow through reshape");
    }
}
