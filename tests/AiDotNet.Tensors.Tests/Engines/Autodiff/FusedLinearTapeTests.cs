using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Integration tests proving FusedLinear is recorded by GradientTape
/// and produces non-zero gradients for all paths (fixes issue #102).
/// </summary>
public class FusedLinearTapeTests
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public FusedLinearTapeTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.ReLU)]
    [InlineData(FusedActivationType.Sigmoid)]
    [InlineData(FusedActivationType.Tanh)]
    [InlineData(FusedActivationType.GELU)]
    [InlineData(FusedActivationType.Swish)]
    [InlineData(FusedActivationType.LeakyReLU)]
    public void FusedLinear_Float_AllActivations_ProducesGradients(FusedActivationType activation)
    {
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var weights = new Tensor<float>(new float[] { 0.5f, -0.3f, 0.1f, 0.8f }, [2, 2]);
        var bias = new Tensor<float>(new float[] { 0.1f, -0.1f }, [1, 2]);

        using var tape = new GradientTape<float>();
        var output = _engine.FusedLinear(input, weights, bias, activation);
        var loss = _engine.TensorMeanDiff(output);

        Assert.True(tape.EntryCount > 0, $"Tape should have entries for activation={activation}");

        var grads = tape.ComputeGradients(loss, sources: new[] { weights, bias });

        Assert.True(grads.ContainsKey(weights), $"Should have weight gradient for {activation}");
        Assert.True(grads.ContainsKey(bias), $"Should have bias gradient for {activation}");

        // Gradients must be non-zero (input data is non-zero, weights are non-zero)
        var wGrad = grads[weights];
        bool anyNonZeroW = false;
        for (int i = 0; i < wGrad.Length; i++)
            if (wGrad.GetFlat(i) != 0f) anyNonZeroW = true;
        Assert.True(anyNonZeroW, $"Weight gradient should be non-zero for {activation}");

        var bGrad = grads[bias];
        bool anyNonZeroB = false;
        for (int i = 0; i < bGrad.Length; i++)
            if (bGrad.GetFlat(i) != 0f) anyNonZeroB = true;
        Assert.True(anyNonZeroB, $"Bias gradient should be non-zero for {activation}");

        _output.WriteLine($"{activation}: wGrad[0]={wGrad.GetFlat(0):F6}, bGrad[0]={bGrad.GetFlat(0):F6}");
    }

    [Fact]
    public void FusedLinear_Double_ProducesGradients()
    {
        var input = new Tensor<double>(new double[] { 1, 2, 3, 4 }, [2, 2]);
        var weights = new Tensor<double>(new double[] { 0.5, -0.3, 0.1, 0.8 }, [2, 2]);
        var bias = new Tensor<double>(new double[] { 0.1, -0.1 }, [1, 2]);

        using var tape = new GradientTape<double>();
        var output = _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        var loss = _engine.TensorMeanDiff(output);
        var grads = tape.ComputeGradients(loss, sources: new[] { weights, bias });

        Assert.True(grads.ContainsKey(weights), "Should have weight gradient for double");
        Assert.True(grads.ContainsKey(bias), "Should have bias gradient for double");

        var wGrad = grads[weights];
        bool anyNonZero = false;
        for (int i = 0; i < wGrad.Length; i++)
            if (wGrad.GetFlat(i) != 0.0) anyNonZero = true;
        Assert.True(anyNonZero, "Double weight gradient should be non-zero");
    }

    [Fact]
    public void FusedLinear_NoBias_ProducesGradients()
    {
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var weights = new Tensor<float>(new float[] { 0.5f, -0.3f, 0.1f, 0.8f }, [2, 2]);

        using var tape = new GradientTape<float>();
        var output = _engine.FusedLinear(input, weights, null, FusedActivationType.None);
        var loss = _engine.TensorMeanDiff(output);
        var grads = tape.ComputeGradients(loss, sources: new[] { weights });

        Assert.True(grads.ContainsKey(weights), "Should have weight gradient without bias");
    }

    [Fact]
    public void FusedLinear_LargerMatrix_ProducesGradients()
    {
        // Test with sizes typical of neural networks
        var input = Tensor<float>.CreateRandom([8, 64]);
        var weights = Tensor<float>.CreateRandom([64, 32]);
        var bias = Tensor<float>.CreateRandom([1, 32]);

        using var tape = new GradientTape<float>();
        var output = _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        var loss = _engine.TensorMeanDiff(output);
        var grads = tape.ComputeGradients(loss, sources: new[] { weights, bias });

        Assert.True(grads.ContainsKey(weights));
        Assert.Equal(64 * 32, grads[weights].Length);
        Assert.True(grads.ContainsKey(bias));
        Assert.Equal(32, grads[bias].Length);
    }

    [Fact]
    public void FusedLinear_WithoutTape_StillWorks()
    {
        // Ensure inference (no tape) still uses the fast BLAS path
        var input = Tensor<float>.CreateRandom([4, 16]);
        var weights = Tensor<float>.CreateRandom([16, 8]);
        var bias = Tensor<float>.CreateRandom([1, 8]);

        // No tape active — should use BLAS fast path
        var result = _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        Assert.Equal(4, result._shape[0]);
        Assert.Equal(8, result._shape[1]);

        // Verify output is reasonable (not all zeros, no NaN)
        bool anyNonZero = false;
        for (int i = 0; i < result.Length; i++)
        {
            float v = result.GetFlat(i);
            Assert.False(float.IsNaN(v), "Output should not contain NaN");
            if (v != 0f) anyNonZero = true;
        }
        Assert.True(anyNonZero, "Output should have non-zero values with random inputs");
    }

    [Fact]
    public void FusedLinear_TrainingStep_ParametersActuallyChange()
    {
        // Simulate a full training step to prove parameters update
        var input = new Tensor<float>(new float[] { 1, 0.5f, -1, 2, 0.3f, 1.5f }, [2, 3]);
        var weights = new Tensor<float>(new float[] { 0.1f, 0.2f, -0.1f, 0.3f, 0.4f, -0.2f }, [3, 2]);
        var bias = new Tensor<float>(new float[] { 0.01f, -0.01f }, [1, 2]);
        var target = new Tensor<float>(new float[] { 1, 0, 0, 1 }, [2, 2]);

        // Save initial parameter values
        float w0Before = weights.GetFlat(0);
        float b0Before = bias.GetFlat(0);

        // Forward + backward
        using var tape = new GradientTape<float>();
        var output = _engine.FusedLinear(input, weights, bias, FusedActivationType.None);
        var diff = _engine.TensorSubtract(output, target);
        var loss = _engine.TensorMeanDiff(_engine.TensorMultiply(diff, diff)); // MSE

        var grads = tape.ComputeGradients(loss, sources: new[] { weights, bias });

        // Apply SGD update: param -= lr * grad
        float lr = 0.01f;
        var wGrad = grads[weights];
        var bGrad = grads[bias];

        for (int i = 0; i < weights.Length; i++)
            weights.SetFlat(i, weights.GetFlat(i) - lr * wGrad.GetFlat(i));
        for (int i = 0; i < bias.Length; i++)
            bias.SetFlat(i, bias.GetFlat(i) - lr * bGrad.GetFlat(i));

        // Parameters should have changed
        Assert.NotEqual(w0Before, weights.GetFlat(0));
        Assert.NotEqual(b0Before, bias.GetFlat(0));

        _output.WriteLine($"Weight[0]: {w0Before:F6} -> {weights.GetFlat(0):F6}");
        _output.WriteLine($"Bias[0]: {b0Before:F6} -> {bias.GetFlat(0):F6}");
    }
}
