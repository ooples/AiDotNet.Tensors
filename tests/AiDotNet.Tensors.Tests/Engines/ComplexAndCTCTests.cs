using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class ComplexAndCTCTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    [Fact]
    public void ComplexMultiply_BasicTest()
    {
        // (3 + 4i) * (1 + 2i) = (3*1 - 4*2) + (3*2 + 4*1)i = -5 + 10i
        var a = new Tensor<float>(new float[] { 3, 4 }, new[] { 2 });
        var b = new Tensor<float>(new float[] { 1, 2 }, new[] { 2 });
        var result = _engine.TensorComplexMultiply(a, b);

        Assert.Equal(2, result.Length);
        Assert.True(Math.Abs(result[0] - (-5f)) < 1e-5f, $"Real part: expected -5, got {result[0]}");
        Assert.True(Math.Abs(result[1] - 10f) < 1e-5f, $"Imag part: expected 10, got {result[1]}");
    }

    [Fact]
    public void ComplexMultiply_BatchTest()
    {
        // Batch of 2 complex numbers
        var a = new Tensor<float>(new float[] { 1, 0, 0, 1 }, new[] { 4 }); // [1+0i, 0+1i]
        var b = new Tensor<float>(new float[] { 0, 1, 0, 1 }, new[] { 4 }); // [0+1i, 0+1i]
        var result = _engine.TensorComplexMultiply(a, b);

        // (1+0i)*(0+1i) = 0+1i
        Assert.True(Math.Abs(result[0] - 0f) < 1e-5f);
        Assert.True(Math.Abs(result[1] - 1f) < 1e-5f);
        // (0+1i)*(0+1i) = -1+0i
        Assert.True(Math.Abs(result[2] - (-1f)) < 1e-5f);
        Assert.True(Math.Abs(result[3] - 0f) < 1e-5f);
    }

    [Fact]
    public void ComplexConjugate_NegatesImaginary()
    {
        var a = new Tensor<float>(new float[] { 3, 4, -1, 2 }, new[] { 4 });
        var result = _engine.TensorComplexConjugate(a);

        Assert.Equal(3f, result[0]);   // re unchanged
        Assert.Equal(-4f, result[1]);  // im negated
        Assert.Equal(-1f, result[2]);  // re unchanged
        Assert.Equal(-2f, result[3]);  // im negated
    }

    [Fact]
    public void ComplexMagnitude_ComputesCorrectly()
    {
        // |3+4i| = 5, |0+1i| = 1
        var a = new Tensor<float>(new float[] { 3, 4, 0, 1 }, new[] { 4 });
        var result = _engine.TensorComplexMagnitude(a);

        Assert.Equal(2, result.Length); // half the input length
        Assert.True(Math.Abs(result[0] - 5f) < 1e-5f, $"Expected 5, got {result[0]}");
        Assert.True(Math.Abs(result[1] - 1f) < 1e-5f, $"Expected 1, got {result[1]}");
    }

    [Fact]
    public void ComplexMultiply_GradientFlows()
    {
        var a = new Tensor<float>(new float[] { 2, 3 }, new[] { 2 });
        var b = new Tensor<float>(new float[] { 1, -1 }, new[] { 2 });

        using var tape = new GradientTape<float>();
        var result = _engine.TensorComplexMultiply(a, b);
        var sum = _engine.ReduceSum(result, new[] { 0 }, keepDims: false);

        var grads = tape.ComputeGradients(sum, [a, b]);
        Assert.True(grads.ContainsKey(a), "Should have gradient for a");
        Assert.True(grads.ContainsKey(b), "Should have gradient for b");
    }

    [Fact]
    public void ComplexMagnitude_GradientFlows()
    {
        var a = new Tensor<float>(new float[] { 3, 4 }, new[] { 2 });

        using var tape = new GradientTape<float>();
        var mag = _engine.TensorComplexMagnitude(a);
        var sum = _engine.ReduceSum(mag, new[] { 0 }, keepDims: false);

        var grads = tape.ComputeGradients(sum, [a]);
        Assert.True(grads.ContainsKey(a), "Should have gradient for a");

        // d|z|/d(re) = re/|z| = 3/5 = 0.6
        // d|z|/d(im) = im/|z| = 4/5 = 0.8
        var grad = grads[a];
        Assert.True(Math.Abs(grad[0] - 0.6f) < 0.01f, $"d/d(re): expected 0.6, got {grad[0]}");
        Assert.True(Math.Abs(grad[1] - 0.8f) < 0.01f, $"d/d(im): expected 0.8, got {grad[1]}");
    }

    [Fact]
    public void CTCLoss_BasicTest()
    {
        // Simple case: T=3, N=1, C=3 (blank=0, classes 1,2)
        // Target: [1] (single label)
        int T = 3, N = 1, C = 3;
        var logProbs = new Tensor<float>(new[] { T, N, C });

        // Set uniform log probabilities (log(1/3))
        float logUniform = MathF.Log(1f / 3);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
                logProbs[t, 0, c] = logUniform;

        var targets = new Tensor<int>(new int[] { 1 }, new[] { 1 });
        int[] inputLengths = { T };
        int[] targetLengths = { 1 };

        var loss = _engine.TensorCTCLoss(logProbs, targets, inputLengths, targetLengths);

        // Loss should be finite and positive
        Assert.True(loss[0] > 0, $"CTC loss should be positive, got {loss[0]}");
        Assert.True(!float.IsNaN(loss[0]) && !float.IsInfinity(loss[0]),
            $"CTC loss should be finite, got {loss[0]}");
    }

    [Fact]
    public void CTCLoss_PerfectPrediction_LowLoss()
    {
        // T=3, target=[1], class 1 has very high probability
        int T = 3, N = 1, C = 3;
        var logProbs = new Tensor<float>(new[] { T, N, C });

        // High prob for class 1 at each timestep
        for (int t = 0; t < T; t++)
        {
            logProbs[t, 0, 0] = MathF.Log(0.05f); // blank
            logProbs[t, 0, 1] = MathF.Log(0.9f);  // target class
            logProbs[t, 0, 2] = MathF.Log(0.05f); // other
        }

        var targets = new Tensor<int>(new int[] { 1 }, new[] { 1 });
        var loss = _engine.TensorCTCLoss(logProbs, targets, new[] { T }, new[] { 1 });

        // Loss should be low (high probability path)
        Assert.True(loss[0] < 2f, $"CTC loss for near-perfect prediction should be low, got {loss[0]}");
    }
}
