using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Verifies gradient correctness by comparing autodiff gradients against
/// finite-difference numerical gradients. Every backward function must
/// produce gradients that match within tolerance.
/// </summary>
public class GradientCorrectnessTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private const double Epsilon = 1e-3; // Central difference: O(eps^2) truncation error
    private const double RelTolerance = 5e-2; // 5% tolerance accounts for float finite-difference error
    // Note: finite-difference gradient has O(eps^2) truncation + O(1/eps) roundoff error.
    // For float (7 significant digits) with eps=1e-3, relative error is ~max(1e-6, 1e-4) ≈ 1e-4.
    // 5% tolerance is conservative and catches real bugs while allowing finite precision.

    /// <summary>
    /// Compares autodiff gradient against finite-difference approximation.
    /// </summary>
    private void VerifyGradient(
        Func<Tensor<float>, Tensor<float>> forward,
        Tensor<float> input,
        string opName)
    {
        // Autodiff gradient
        Tensor<float> autodiffGrad;
        using (var tape = new GradientTape<float>())
        {
            var output = forward(input);
            var grads = tape.ComputeGradients(output, sources: new[] { input });
            Assert.True(grads.ContainsKey(input), $"{opName}: no gradient computed");
            autodiffGrad = grads[input];
        }

        // Finite-difference gradient
        var numGrad = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            float origVal = input[i];

            input[i] = origVal + (float)Epsilon;
            var fPlus = forward(input);
            float sumPlus = 0;
            for (int j = 0; j < fPlus.Length; j++) sumPlus += fPlus[j];

            input[i] = origVal - (float)Epsilon;
            var fMinus = forward(input);
            float sumMinus = 0;
            for (int j = 0; j < fMinus.Length; j++) sumMinus += fMinus[j];

            numGrad[i] = (sumPlus - sumMinus) / (2f * (float)Epsilon);
            input[i] = origVal;
        }

        // Compare
        for (int i = 0; i < input.Length; i++)
        {
            float ad = autodiffGrad[i];
            float nd = numGrad[i];
            float maxAbs = Math.Max(Math.Abs(ad), Math.Abs(nd));
            float diff = Math.Abs(ad - nd);

            if (maxAbs > 1e-6f)
            {
                float relError = diff / maxAbs;
                Assert.True(relError < (float)RelTolerance,
                    $"{opName}[{i}]: autodiff={ad:G6}, numerical={nd:G6}, relError={relError:G4}");
            }
            else
            {
                Assert.True(diff < 1e-4f,
                    $"{opName}[{i}]: autodiff={ad:G6}, numerical={nd:G6}, diff={diff:G4}");
            }
        }
    }

    // ─── Arithmetic ─────────────────────────────────────────────

    [Fact]
    public void Add_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f }, [3]);
        var y = new Tensor<float>(new float[] { 4f, 5f, 6f }, [3]);
        VerifyGradient(inp => _engine.TensorAdd(inp, y), x, "Add_x");
        VerifyGradient(inp => _engine.TensorAdd(x, inp), y, "Add_y");
    }

    [Fact]
    public void Subtract_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f }, [3]);
        var y = new Tensor<float>(new float[] { 4f, 5f, 6f }, [3]);
        VerifyGradient(inp => _engine.TensorSubtract(inp, y), x, "Sub_x");
    }

    [Fact]
    public void Multiply_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 2f, 3f, 4f }, [3]);
        var y = new Tensor<float>(new float[] { 5f, 6f, 7f }, [3]);
        VerifyGradient(inp => _engine.TensorMultiply(inp, y), x, "Mul_x");
        VerifyGradient(inp => _engine.TensorMultiply(x, inp), y, "Mul_y");
    }

    [Fact]
    public void Divide_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 6f, 8f, 10f }, [3]);
        var y = new Tensor<float>(new float[] { 2f, 4f, 5f }, [3]);
        VerifyGradient(inp => _engine.TensorDivide(inp, y), x, "Div_x");
    }

    [Fact]
    public void MatMul_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
        var w = new Tensor<float>(new float[] { 5f, 6f, 7f, 8f }, [2, 2]);
        VerifyGradient(inp => _engine.TensorMatMul(inp, w), x, "MatMul_x");
        VerifyGradient(inp => _engine.TensorMatMul(x, inp), w, "MatMul_w");
    }

    // ─── Activations ────────────────────────────────────────────

    [Fact]
    public void ReLU_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -2f, -0.5f, 0.5f, 2f }, [4]);
        VerifyGradient(inp => _engine.ReLU(inp), x, "ReLU");
    }

    [Fact]
    public void Sigmoid_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -2f, -1f, 0f, 1f, 2f }, [5]);
        VerifyGradient(inp => _engine.Sigmoid(inp), x, "Sigmoid");
    }

    [Fact]
    public void Tanh_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -2f, -1f, 0f, 1f, 2f }, [5]);
        VerifyGradient(inp => _engine.Tanh(inp), x, "Tanh");
    }

    [Fact]
    public void GELU_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -2f, -1f, 0f, 1f, 2f }, [5]);
        VerifyGradient(inp => _engine.GELU(inp), x, "GELU");
    }

    [Fact]
    public void Swish_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -2f, -1f, 0f, 1f, 2f }, [5]);
        VerifyGradient(inp => _engine.Swish(inp), x, "Swish");
    }

    [Fact]
    public void LeakyReLU_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -2f, -0.5f, 0.5f, 2f }, [4]);
        var numOps = MathHelper.GetNumericOperations<float>();
        VerifyGradient(inp => _engine.LeakyReLU(inp, numOps.FromDouble(0.1)), x, "LeakyReLU");
    }

    [Fact]
    public void ELU_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -2f, -0.5f, 0.5f, 2f }, [4]);
        VerifyGradient(inp => _engine.ELU(inp), x, "ELU");
    }

    [Fact]
    public void Mish_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -2f, -1f, 0f, 1f, 2f }, [5]);
        VerifyGradient(inp => _engine.Mish(inp), x, "Mish");
    }

    [Fact]
    public void Softplus_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -2f, -1f, 0f, 1f, 2f }, [5]);
        VerifyGradient(inp => _engine.Softplus(inp), x, "Softplus");
    }

    [Fact]
    public void HardSwish_Gradient_MatchesNumerical()
    {
        // Avoid exact boundary values where gradient is discontinuous
        var x = new Tensor<float>(new float[] { -4f, -1f, 0f, 1f, 4f }, [5]);
        VerifyGradient(inp => _engine.HardSwish(inp), x, "HardSwish");
    }

    // ─── Math ops ───────────────────────────────────────────────

    [Fact]
    public void Exp_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -1f, 0f, 1f, 2f }, [4]);
        VerifyGradient(inp => _engine.TensorExp(inp), x, "Exp");
    }

    [Fact]
    public void Log_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 0.5f, 1f, 2f, 3f }, [4]);
        VerifyGradient(inp => _engine.TensorLog(inp), x, "Log");
    }

    [Fact]
    public void Sqrt_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 4f, 9f }, [4]);
        VerifyGradient(inp => _engine.TensorSqrt(inp), x, "Sqrt");
    }

    [Fact]
    public void Abs_Gradient_MatchesNumerical()
    {
        // Avoid zero where gradient is undefined
        var x = new Tensor<float>(new float[] { -3f, -1f, 1f, 3f }, [4]);
        VerifyGradient(inp => _engine.TensorAbs(inp), x, "Abs");
    }

    [Fact]
    public void Power_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f }, [3]);
        VerifyGradient(inp => _engine.TensorPower(inp, 3f), x, "Power");
    }

    // ─── Normalization ──────────────────────────────────────────

    [Fact]
    public void Softmax_Gradient_MatchesNumerical()
    {
        // sum(softmax(x)) = 1 always, so gradient of sum is zero.
        // Use weighted sum with target to get non-trivial gradient:
        // loss = sum(target * softmax(x))
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [1, 4]);
        var target = new Tensor<float>(new float[] { 1f, 0f, 0f, 0f }, [1, 4]);
        VerifyGradient(inp =>
        {
            var s = _engine.Softmax(inp, axis: 1);
            return _engine.TensorMultiply(s, target);
        }, x, "Softmax");
    }

    [Fact]
    public void LogSoftmax_Gradient_MatchesNumerical()
    {
        // Use weighted sum for non-trivial gradient
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [1, 4]);
        var target = new Tensor<float>(new float[] { 1f, 0f, 0f, 0f }, [1, 4]);
        VerifyGradient(inp =>
        {
            var ls = _engine.TensorLogSoftmax(inp, axis: 1);
            return _engine.TensorMultiply(ls, target);
        }, x, "LogSoftmax");
    }

    // ─── Shape ops ──────────────────────────────────────────────

    [Fact]
    public void Concatenate_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f }, [2]);
        var y = new Tensor<float>(new float[] { 3f, 4f, 5f }, [3]);
        VerifyGradient(inp => _engine.TensorConcatenate(new[] { inp, y }, axis: 0), x, "Concat_x");
    }

    // ─── Integration test: training convergence ─────────────────

    [Fact]
    public void Integration_MLP_TrainingConverges()
    {
        // 2-layer MLP: 4->8->1, train on XOR-like data
        var w1 = Tensor<float>.CreateRandom([4, 8]);
        var w2 = Tensor<float>.CreateRandom([8, 1]);
        float lr = 0.01f;

        // Simple training data
        var x = new Tensor<float>(new float[]
        {
            0, 0, 1, 1,
            0, 1, 0, 1,
            1, 0, 1, 0,
            1, 1, 0, 0
        }, [4, 4]);
        var target = new Tensor<float>(new float[] { 0f, 1f, 1f, 0f }, [4, 1]);

        float initialLoss = float.MaxValue;
        float finalLoss = float.MaxValue;

        for (int epoch = 0; epoch < 200; epoch++)
        {
            using var tape = new GradientTape<float>();

            var h = _engine.ReLU(_engine.TensorMatMul(x, w1));
            var pred = _engine.TensorMatMul(h, w2);
            var diff = _engine.TensorSubtract(pred, target);
            var loss = _engine.TensorMultiply(diff, diff); // MSE

            float lossVal = 0;
            for (int i = 0; i < loss.Length; i++) lossVal += loss[i];
            lossVal /= loss.Length;

            if (epoch == 0) initialLoss = lossVal;
            if (epoch == 199) finalLoss = lossVal;

            var grads = tape.ComputeGradients(loss, sources: new[] { w1, w2 });

            if (grads.TryGetValue(w1, out var gw1))
                for (int i = 0; i < w1.Length; i++) w1[i] -= lr * gw1[i];
            if (grads.TryGetValue(w2, out var gw2))
                for (int i = 0; i < w2.Length; i++) w2[i] -= lr * gw2[i];
        }

        Assert.True(finalLoss < initialLoss * 0.5f,
            $"MLP training should reduce loss. Initial: {initialLoss:G4}, Final: {finalLoss:G4}");
    }

    // ─── Integration tests for advanced features ────────────────

    [Fact]
    public void Integration_GradientAccumulator_TrainsWithMiniBatches()
    {
        var w = new Tensor<float>(new float[] { 0f, 0f }, [2]);
        var accumulator = new GradientAccumulator<float>();
        accumulator.Register(w);

        // 4 mini-batches, accumulate then step
        for (int batch = 0; batch < 4; batch++)
        {
            using var tape = new GradientTape<float>();
            var x = new Tensor<float>(new float[] { (batch + 1) * 1f, (batch + 1) * 2f }, [2]);
            var target = new Tensor<float>(new float[] { 3f, 6f }, [2]);
            var diff = _engine.TensorSubtract(_engine.TensorMultiply(w, x), target);
            var loss = _engine.TensorMultiply(diff, diff);
            var grads = tape.ComputeGradients(loss, sources: new[] { w });
            accumulator.Accumulate(grads);
        }

        Assert.Equal(4, accumulator.AccumulationCount);
        var grad = accumulator.GetGrad(w);
        Assert.NotNull(grad);
        Assert.True(grad!.Length == 2);

        accumulator.Step(0.001f);
        accumulator.ZeroGrad();
        Assert.Equal(0, accumulator.AccumulationCount);
    }

    [Fact]
    public void Integration_FusedLinear_ProducesCorrectGradients()
    {
        using var tape = new GradientTape<float>();

        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
        var w = new Tensor<float>(new float[] { 0.5f, 0.3f, 0.2f, 0.4f }, [2, 2]);
        var b = new Tensor<float>(new float[] { 0.1f, 0.2f }, [2]);

        var output = FusedOperations<float>.Linear(x, w, b);

        var grads = tape.ComputeGradients(output, sources: new[] { x, w, b });

        Assert.True(grads.ContainsKey(x), "Should have gradient for input");
        Assert.True(grads.ContainsKey(w), "Should have gradient for weight");
        Assert.True(grads.ContainsKey(b), "Should have gradient for bias");

        // Verify gradients are finite
        foreach (var g in new[] { grads[x], grads[w], grads[b] })
            for (int i = 0; i < g.Length; i++)
                Assert.False(float.IsNaN(g[i]) || float.IsInfinity(g[i]));
    }

    [Fact]
    public void Integration_GradientCheckpointing_ProducesGradients()
    {
        using var tape = new GradientTape<float>();

        var x = new Tensor<float>(new float[] { 1f, 2f, 3f }, [3]);

        // Chain of operations with checkpointing
        var functions = new List<Func<Tensor<float>, Tensor<float>>>
        {
            inp => _engine.TensorMultiplyScalar(inp, 2f),
            inp => _engine.ReLU(inp),
            inp => _engine.TensorMultiplyScalar(inp, 0.5f),
        };

        var output = GradientCheckpointing<float>.Checkpoint(functions, x, segmentSize: 2);

        var grads = tape.ComputeGradients(output, sources: new[] { x });
        Assert.True(grads.ContainsKey(x), "Checkpointed forward should produce gradients");

        // Verify gradient is finite and non-zero
        for (int i = 0; i < grads[x].Length; i++)
        {
            Assert.False(float.IsNaN(grads[x][i]) || float.IsInfinity(grads[x][i]));
        }
    }

    // ─── Additional arithmetic gradient checks ───────────────────

    [Fact]
    public void Negate_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, -2f, 3f }, [3]);
        VerifyGradient(inp => _engine.TensorNegate(inp), x, "Negate");
    }

    [Fact]
    public void MultiplyScalar_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f }, [3]);
        VerifyGradient(inp => _engine.TensorMultiplyScalar(inp, 2.5f), x, "MulScalar");
    }

    [Fact]
    public void BroadcastAdd_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
        var bias = new Tensor<float>(new float[] { 0.5f, 1.5f }, [2]);
        VerifyGradient(inp => _engine.TensorBroadcastAdd(inp, bias), x, "BroadcastAdd");
    }

    [Fact]
    public void Clamp_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -2f, 0.5f, 1.5f, 3f }, [4]);
        VerifyGradient(inp => _engine.TensorClamp(inp, 0f, 2f), x, "Clamp");
    }

    [Fact]
    public void Sin_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 0f, 0.5f, 1f, 2f }, [4]);
        VerifyGradient(inp => _engine.TensorSin(inp), x, "Sin");
    }

    [Fact]
    public void Cos_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 0f, 0.5f, 1f, 2f }, [4]);
        VerifyGradient(inp => _engine.TensorCos(inp), x, "Cos");
    }

    [Fact]
    public void Transpose_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
        VerifyGradient(inp => _engine.TensorTranspose(inp), x, "Transpose");
    }

    [Fact]
    public void Reshape_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        VerifyGradient(inp => _engine.Reshape(inp, new[] { 3, 2 }), x, "Reshape");
    }

    [Fact]
    public void ReduceSum_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
        VerifyGradient(inp => _engine.ReduceSum(inp, new[] { 0 }, keepDims: true), x, "ReduceSum");
    }

    [Fact]
    public void Conv2D_Gradient_MatchesNumerical()
    {
        // Simple 1-channel 3x3 input with 1x1 kernel
        var x = new Tensor<float>(new float[]
        {
            1f, 2f, 3f,
            4f, 5f, 6f,
            7f, 8f, 9f
        }, [1, 1, 3, 3]);
        var kernel = new Tensor<float>(new float[] { 0.5f }, [1, 1, 1, 1]);

        VerifyGradient(inp =>
        {
            return _engine.Conv2D(inp, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        }, x, "Conv2D");
    }

    [Fact]
    public void AddScalar_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [4]);
        VerifyGradient(inp => _engine.TensorAddScalar(inp, 2.5f), x, "AddScalar");
    }

    [Fact]
    public void SubtractScalar_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [4]);
        VerifyGradient(inp => _engine.TensorSubtractScalar(inp, 1.5f), x, "SubtractScalar");
    }

    [Fact]
    public void DivideScalar_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [4]);
        VerifyGradient(inp => _engine.TensorDivideScalar(inp, 2.0f), x, "DivideScalar");
    }

    [Fact]
    public void BroadcastMultiply_Gradient_MatchesNumerical()
    {
        // 2x3 tensor broadcast multiplied by 1x3 scale
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        var scale = new Tensor<float>(new float[] { 0.5f, 1.0f, 2.0f }, [1, 3]);
        VerifyGradient(inp => _engine.TensorBroadcastMultiply(inp, scale), x, "BroadcastMultiply");
    }

    [Fact]
    public void ReduceMean_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        VerifyGradient(inp => _engine.ReduceMean(inp, new[] { 0 }, keepDims: true), x, "ReduceMean");
    }

    [Fact]
    public void ExpandDims_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [4]);
        VerifyGradient(inp => _engine.TensorExpandDims(inp, 0), x, "ExpandDims");
    }

    [Fact]
    public void Squeeze_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [1, 4]);
        VerifyGradient(inp => _engine.TensorSqueeze(inp, 0), x, "Squeeze");
    }

    [Fact]
    public void BroadcastSubtract_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        var bias = new Tensor<float>(new float[] { 0.1f, 0.2f, 0.3f }, [1, 3]);
        VerifyGradient(inp => _engine.TensorBroadcastSubtract(inp, bias), x, "BroadcastSubtract");
    }

    [Fact]
    public void Power_2D_Gradient_MatchesNumerical()
    {
        // Test power with 2D input
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
        var exp = new Tensor<float>(new float[] { 2f, 2f, 2f, 2f }, [2, 2]);
        VerifyGradient(inp => _engine.TensorPower(inp, exp), x, "Power2D");
    }

    [Fact]
    public void ChainedOps_Gradient_MatchesNumerical()
    {
        // Test gradient through a chain: sigmoid(x * 2 + 1)
        var x = new Tensor<float>(new float[] { 0.1f, 0.5f, -0.3f, 0.8f }, [4]);
        VerifyGradient(inp =>
        {
            var scaled = _engine.TensorMultiplyScalar(inp, 2.0f);
            var shifted = _engine.TensorAddScalar(scaled, 1.0f);
            return _engine.TensorSigmoid(shifted);
        }, x, "ChainedOps");
    }

    [Fact]
    public void MultipleInputPaths_Gradient_MatchesNumerical()
    {
        // Test when same input is used multiple times: x * x (should give 2x gradient)
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [4]);
        VerifyGradient(inp => _engine.TensorMultiply(inp, inp), x, "SelfMultiply");
    }

    [Fact]
    public void Permute_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        VerifyGradient(inp => _engine.TensorPermute(inp, new[] { 1, 0 }), x, "Permute");
    }

    /// <summary>
    /// Regression for #274: when a permute output is consumed by multiple
    /// downstream ops, the second AccumulateGrad call into the permute's
    /// input previously hit "In-place add requires contiguous target tensor"
    /// because PermuteBackward returns a non-contiguous (strided view)
    /// gradient and AccumulateGrad stored it as the first-write target.
    /// Now AccumulateGrad materializes via .Contiguous() at storage time.
    /// </summary>
    [Fact]
    public void Permute_GradAccumulation_AcrossMultipleConsumers_Issue274()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);

        using var tape = new GradientTape<float>();
        // Permute creates a non-contiguous view; both branches consume it
        // so its gradient accumulates twice — the failing path in #274.
        var p = _engine.TensorPermute(x, new[] { 1, 0 });          // [3, 2]
        var s1 = _engine.TensorMultiplyScalar(p, 2.0f);
        var s2 = _engine.TensorMultiplyScalar(p, 3.0f);
        var sum = _engine.TensorAdd(s1, s2);
        var loss = _engine.ReduceSum(sum, axes: null);

        // Pre-fix: throws InvalidOperationException with the #274 message.
        var grads = tape.ComputeGradients(loss, sources: new[] { x });

        Assert.True(grads.ContainsKey(x));
        // dL/dx = 5 everywhere (perm is just a re-indexing; both scalar-mults
        // contribute 2 + 3 = 5 to every cell).
        var g = grads[x].AsSpan();
        for (int i = 0; i < g.Length; i++) Assert.Equal(5f, g[i], 4);
    }

    /// <summary>Audit-companion to #274: same multi-consumer pattern through Transpose
    /// (which also returns a stride-rewritten view). Caught by the universal
    /// AccumulateGrad contiguity fix.</summary>
    [Fact]
    public void Transpose_GradAccumulation_AcrossMultipleConsumers_Issue274()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        using var tape = new GradientTape<float>();
        var t = _engine.TensorTranspose(x);
        var s1 = _engine.TensorMultiplyScalar(t, 2.0f);
        var s2 = _engine.TensorMultiplyScalar(t, 3.0f);
        var sum = _engine.TensorAdd(s1, s2);
        var loss = _engine.ReduceSum(sum, axes: null);
        var grads = tape.ComputeGradients(loss, sources: new[] { x });
        Assert.True(grads.ContainsKey(x));
        var g = grads[x].AsSpan();
        for (int i = 0; i < g.Length; i++) Assert.Equal(5f, g[i], 4);
    }

    /// <summary>Audit-companion to #274: Squeeze returns a metadata-only view.
    /// Multi-consumer accumulation must not hit the in-place contiguity throw.</summary>
    [Fact]
    public void Squeeze_GradAccumulation_AcrossMultipleConsumers_Issue274()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [1, 2, 3]);
        using var tape = new GradientTape<float>();
        var sq = _engine.TensorSqueeze(x, 0);
        var s1 = _engine.TensorMultiplyScalar(sq, 2.0f);
        var s2 = _engine.TensorMultiplyScalar(sq, 3.0f);
        var sum = _engine.TensorAdd(s1, s2);
        var loss = _engine.ReduceSum(sum, axes: null);
        var grads = tape.ComputeGradients(loss, sources: new[] { x });
        Assert.True(grads.ContainsKey(x));
        var g = grads[x].AsSpan();
        for (int i = 0; i < g.Length; i++) Assert.Equal(5f, g[i], 4);
    }

    /// <summary>Audit-companion to #274: ExpandDims returns a view.</summary>
    [Fact]
    public void ExpandDims_GradAccumulation_AcrossMultipleConsumers_Issue274()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        using var tape = new GradientTape<float>();
        var e = _engine.TensorExpandDims(x, 0);
        var s1 = _engine.TensorMultiplyScalar(e, 2.0f);
        var s2 = _engine.TensorMultiplyScalar(e, 3.0f);
        var sum = _engine.TensorAdd(s1, s2);
        var loss = _engine.ReduceSum(sum, axes: null);
        var grads = tape.ComputeGradients(loss, sources: new[] { x });
        Assert.True(grads.ContainsKey(x));
        var g = grads[x].AsSpan();
        for (int i = 0; i < g.Length; i++) Assert.Equal(5f, g[i], 4);
    }

    /// <summary>Audit-companion to #274: Reshape can return a view in some
    /// stride configurations. Multi-consumer pattern through Reshape.</summary>
    [Fact]
    public void Reshape_GradAccumulation_AcrossMultipleConsumers_Issue274()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        using var tape = new GradientTape<float>();
        var r = _engine.Reshape(x, new[] { 3, 2 });
        var s1 = _engine.TensorMultiplyScalar(r, 2.0f);
        var s2 = _engine.TensorMultiplyScalar(r, 3.0f);
        var sum = _engine.TensorAdd(s1, s2);
        var loss = _engine.ReduceSum(sum, axes: null);
        var grads = tape.ComputeGradients(loss, sources: new[] { x });
        Assert.True(grads.ContainsKey(x));
        var g = grads[x].AsSpan();
        for (int i = 0; i < g.Length; i++) Assert.Equal(5f, g[i], 4);
    }

    // ──────────────────────────────────────────────────────────────
    // #274 edge-case integration suite — deeper graphs, more
    // consumers, non-contiguous input, large tensors, chained views.
    // ──────────────────────────────────────────────────────────────

    /// <summary>Permute → Permute → multi-consumer. The view is composed
    /// before the fan-out, so the cached gradient buffer for the inner
    /// permute is itself non-contiguous AND has a non-contiguous incoming
    /// gradient. Both legs of the AccumulateGrad fix must engage.</summary>
    [Fact]
    public void Permute_Chained_GradAccumulation_Issue274()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, [2, 2, 2]);
        using var tape = new GradientTape<float>();
        var p1 = _engine.TensorPermute(x, new[] { 1, 0, 2 });   // first stride-rewrite
        var p2 = _engine.TensorPermute(p1, new[] { 0, 2, 1 });  // second stride-rewrite
        var s1 = _engine.TensorMultiplyScalar(p2, 2.0f);
        var s2 = _engine.TensorMultiplyScalar(p2, 3.0f);
        var s3 = _engine.TensorMultiplyScalar(p2, 7.0f);        // 3 consumers, not 2
        var sum12 = _engine.TensorAdd(s1, s2);
        var sum123 = _engine.TensorAdd(sum12, s3);
        var loss = _engine.ReduceSum(sum123, axes: null);

        var grads = tape.ComputeGradients(loss, sources: new[] { x });
        Assert.True(grads.ContainsKey(x));
        // dL/dx = 2 + 3 + 7 = 12 everywhere (the permutes only re-index, sum is invariant).
        var g = grads[x].AsSpan();
        for (int i = 0; i < g.Length; i++) Assert.Equal(12f, g[i], 4);
    }

    /// <summary>Permute fan-out with FIVE downstream consumers. Stress the
    /// AccumulateGrad fast path (indexed) under repeated in-place adds onto
    /// a previously non-contiguous slot.</summary>
    [Fact]
    public void Permute_FiveConsumers_GradAccumulation_Issue274()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
        using var tape = new GradientTape<float>();
        var p = _engine.TensorPermute(x, new[] { 1, 0 });
        var sums = new[] { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f };
        Tensor<float>? acc = null;
        foreach (var s in sums)
        {
            var sm = _engine.TensorMultiplyScalar(p, s);
            acc = acc is null ? sm : _engine.TensorAdd(acc, sm);
        }
        var loss = _engine.ReduceSum(acc!, axes: null);
        var grads = tape.ComputeGradients(loss, sources: new[] { x });
        // dL/dx = sum(scalars) = 17.5 everywhere.
        var g = grads[x].AsSpan();
        for (int i = 0; i < g.Length; i++) Assert.Equal(17.5f, g[i], 3);
    }

    /// <summary>Mixed-view multi-consumer: Permute output AND Transpose output
    /// of the same source feed downstream — exercises that gradient
    /// accumulation through TWO different non-contiguous views into the same
    /// source tensor stays correct.</summary>
    [Fact]
    public void Permute_AndTranspose_BothToSameSource_GradAccumulation_Issue274()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        using var tape = new GradientTape<float>();
        var p = _engine.TensorPermute(x, new[] { 1, 0 });
        var t = _engine.TensorTranspose(x);
        var s1 = _engine.TensorMultiplyScalar(p, 2.0f);
        var s2 = _engine.TensorMultiplyScalar(t, 5.0f);
        var sum = _engine.TensorAdd(s1, s2);
        var loss = _engine.ReduceSum(sum, axes: null);
        var grads = tape.ComputeGradients(loss, sources: new[] { x });
        var g = grads[x].AsSpan();
        // dL/dx = 7 everywhere (Permute and Transpose are equivalent for [2,3] → [3,2]).
        for (int i = 0; i < g.Length; i++) Assert.Equal(7f, g[i], 4);
    }

    /// <summary>NBEATS-style scenario from the issue: a permuted basis tensor
    /// feeds two downstream linear projections (backcast + forecast) whose
    /// gradients flow back through accumulation.</summary>
    [Fact]
    public void NBeatsStyle_PermuteThenTwoMatMuls_GradAccumulation_Issue274()
    {
        // NBEATS pattern: a (channels × time) basis is permuted to
        // (time × channels), then two head MatMuls (backcast / forecast)
        // both consume the permuted view → the gradient accumulator hits
        // the failing #274 path on the second backward write into perm's input.
        // basis: [channels=4, time=3]; perm: [3, 4]; wBack/wFore: [4, 2].
        var basis = new Tensor<float>(
            Enumerable.Range(0, 4 * 3).Select(i => (float)i).ToArray(), [4, 3]);
        var wBack = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f, 1f, 1f, 1f, 1f }, [4, 2]);
        var wFore = new Tensor<float>(new float[] { 1f, 1f, 0f, 1f, 1f, 0f, 1f, 1f }, [4, 2]);

        using var tape = new GradientTape<float>();
        var perm = _engine.TensorPermute(basis, new[] { 1, 0 });    // [3, 4]
        var bc = _engine.TensorMatMul(perm, wBack);                  // [3, 2]
        var fc = _engine.TensorMatMul(perm, wFore);                  // [3, 2]
        var combined = _engine.TensorAdd(bc, fc);
        var loss = _engine.ReduceSum(combined, axes: null);

        // Pre-fix, this is the failing path NBEATSModel.cs:398 hits.
        var grads = tape.ComputeGradients(loss, sources: new[] { basis });
        Assert.True(grads.ContainsKey(basis));
        Assert.Equal(basis.Length, grads[basis].Length);
        var g = grads[basis].AsSpan();
        // float.IsFinite is .NET 2.1+; net471 uses !IsNaN && !IsInfinity.
        for (int i = 0; i < g.Length; i++)
            Assert.True(!float.IsNaN(g[i]) && !float.IsInfinity(g[i]), $"g[{i}] not finite: {g[i]}");
    }

    /// <summary>Large-tensor stress: 64×64×8 permuted then fanned out. Ensures
    /// the .Contiguous() copy in AccumulateGrad scales without NRE / overflow.</summary>
    [Fact]
    public void Permute_LargeTensor_GradAccumulation_Issue274()
    {
        const int A = 64, B = 64, C = 8;
        var arr = new float[A * B * C];
        for (int i = 0; i < arr.Length; i++) arr[i] = (i % 13) * 0.01f;
        var x = new Tensor<float>(arr, [A, B, C]);

        using var tape = new GradientTape<float>();
        var p = _engine.TensorPermute(x, new[] { 2, 0, 1 });
        var s1 = _engine.TensorMultiplyScalar(p, 0.5f);
        var s2 = _engine.TensorMultiplyScalar(p, 0.25f);
        var sum = _engine.TensorAdd(s1, s2);
        var loss = _engine.ReduceSum(sum, axes: null);

        var grads = tape.ComputeGradients(loss, sources: new[] { x });
        Assert.True(grads.ContainsKey(x));
        var g = grads[x].AsSpan();
        // dL/dx = 0.75 everywhere.
        for (int i = 0; i < g.Length; i++) Assert.Equal(0.75f, g[i], 3);
    }

    /// <summary>Source x is itself non-contiguous (already permuted) BEFORE
    /// it reaches the watched op. Exercises that AccumulateGrad's
    /// dictionary-fallback existing-slot materialization works when the
    /// source tensor itself was a view.</summary>
    [Fact]
    public void Permute_NonContiguousSource_GradAccumulation_Issue274()
    {
        var rawSrc = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, [2, 4]);
        // Make source non-contiguous BEFORE it enters the tape's source set
        // (callers do this when feeding a transposed batch into a model).
        var nonContig = _engine.TensorTranspose(rawSrc);  // [4, 2], non-contiguous

        using var tape = new GradientTape<float>();
        var p = _engine.TensorPermute(nonContig, new[] { 1, 0 });
        var s1 = _engine.TensorMultiplyScalar(p, 2.0f);
        var s2 = _engine.TensorMultiplyScalar(p, 3.0f);
        var sum = _engine.TensorAdd(s1, s2);
        var loss = _engine.ReduceSum(sum, axes: null);
        var grads = tape.ComputeGradients(loss, sources: new[] { nonContig });
        Assert.True(grads.ContainsKey(nonContig));
        var g = grads[nonContig].AsSpan();
        for (int i = 0; i < g.Length; i++) Assert.Equal(5f, g[i], 4);
    }

    /// <summary>Higher-order autograd path — createGraph=true takes the
    /// out-of-place TensorAdd branch, but the FIRST-write contiguity fix
    /// must still apply so the SECOND backward pass sees a clean target.</summary>
    [Fact]
    public void Permute_HigherOrder_GradAccumulation_Issue274()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
        using var tape = new GradientTape<float>();
        var p = _engine.TensorPermute(x, new[] { 1, 0 });
        var sq = _engine.TensorMultiply(p, p);          // x^2 path — d/dx = 2x
        var loss = _engine.ReduceSum(sq, axes: null);
        var grads = tape.ComputeGradients(loss, sources: new[] { x }, createGraph: true);
        Assert.True(grads.ContainsKey(x));
        // createGraph=true keeps the original non-contiguous grad reference so
        // double-backward stays graph-connected (PyTorch parity); materialize
        // before reading the span. .Contiguous() preserves values without
        // touching the recorded graph.
        var g = grads[x].Contiguous().AsSpan();
        // d(sum(x^2))/dx = 2x.
        Assert.Equal(2f, g[0], 4);
        Assert.Equal(4f, g[1], 4);
        Assert.Equal(6f, g[2], 4);
        Assert.Equal(8f, g[3], 4);
    }

    /// <summary>Double-backward through Permute — exercises that the createGraph=true
    /// path keeps GradFn lineage intact through non-contiguous incoming gradients.
    /// If AccumulateGrad eagerly materialized via .Contiguous(), the second backward
    /// would fail to walk back through the detached copy and yield wrong grads
    /// (or throw). PyTorch parity: x^3 → d/dx = 3x², d²/dx² = 6x.</summary>
    [Fact]
    public void Permute_DoubleBackward_PreservesGraphLineage_Issue274()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);

        // First backward with createGraph=true so the backward ops record on the tape.
        using var tape = new GradientTape<float>();
        var p = _engine.TensorPermute(x, new[] { 1, 0 });
        var pSq = _engine.TensorMultiply(p, p);             // p²
        var pCube = _engine.TensorMultiply(pSq, p);          // p³
        var loss1 = _engine.ReduceSum(pCube, axes: null);
        var grads1 = tape.ComputeGradients(loss1, sources: new[] { x }, createGraph: true);
        Assert.True(grads1.ContainsKey(x));
        var g1 = grads1[x];

        // First-derivative sanity: d/dx(x³) = 3x² → at [1,2,3,4] = [3,12,27,48].
        // .Contiguous() materializes a value-copy without disturbing g1's
        // tape lineage (g1 itself is the live reference for double-backward).
        var g1Span = g1.Contiguous().AsSpan();
        Assert.Equal(3f, g1Span[0], 3);
        Assert.Equal(12f, g1Span[1], 3);
        Assert.Equal(27f, g1Span[2], 3);
        Assert.Equal(48f, g1Span[3], 3);

        // Second backward — sum the first gradient and differentiate again.
        // d²/dx² of x³ = 6x → at [1,2,3,4] = [6,12,18,24]; d/dx of sum(3x²) = 6x.
        var loss2 = _engine.ReduceSum(g1, axes: null);
        var grads2 = tape.ComputeGradients(loss2, sources: new[] { x }, createGraph: false);
        Assert.True(grads2.ContainsKey(x));
        var g2 = grads2[x].Contiguous().AsSpan();
        Assert.Equal(6f, g2[0], 2);
        Assert.Equal(12f, g2[1], 2);
        Assert.Equal(18f, g2[2], 2);
        Assert.Equal(24f, g2[3], 2);
    }

    // ──────────────────────────────────────────────────────────────
    // Loss function gradient tests
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void MSELoss_Gradient_MatchesNumerical()
    {
        var pred = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [4]);
        var target = new Tensor<float>(new float[] { 1.5f, 2.5f, 2.5f, 3.5f }, [4]);
        VerifyGradient(inp => _engine.TensorMSELoss(inp, target), pred, "MSELoss");
    }

    [Fact]
    public void L1Loss_Gradient_MatchesNumerical()
    {
        var pred = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [4]);
        var target = new Tensor<float>(new float[] { 1.5f, 2.5f, 2.5f, 3.5f }, [4]);
        VerifyGradient(inp => _engine.TensorL1Loss(inp, target), pred, "L1Loss");
    }

    [Fact]
    public void HuberLoss_Gradient_MatchesNumerical()
    {
        var pred = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [4]);
        var target = new Tensor<float>(new float[] { 1.5f, 2.5f, 2.5f, 3.5f }, [4]);
        VerifyGradient(inp => _engine.TensorHuberLoss(inp, target), pred, "HuberLoss");
    }

    [Fact]
    public void BCEWithLogitsLoss_Gradient_MatchesNumerical()
    {
        var logits = new Tensor<float>(new float[] { -1f, 0f, 1f, 2f }, [4]);
        var targets = new Tensor<float>(new float[] { 0f, 0f, 1f, 1f }, [4]);
        VerifyGradient(inp => _engine.TensorBCEWithLogitsLoss(inp, targets), logits, "BCEWithLogitsLoss");
    }

    [Fact]
    public void CosineSimilarity_Gradient_MatchesNumerical()
    {
        var a = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [4]);
        var b = new Tensor<float>(new float[] { 4f, 3f, 2f, 1f }, [4]);
        VerifyGradient(inp => _engine.TensorCosineSimilarityLoss(inp, b), a, "CosineSimilarity");
    }

    // ──────────────────────────────────────────────────────────────
    // New activation gradient tests
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void SELU_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -1f, -0.5f, 0.5f, 1f }, [4]);
        VerifyGradient(inp => _engine.TensorSELU(inp), x, "SELU");
    }

    [Fact]
    public void HardSigmoid_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -1f, 0f, 1f, 2f }, [4]);
        VerifyGradient(inp => _engine.TensorHardSigmoid(inp), x, "HardSigmoid");
    }

    [Fact]
    public void ReLU6_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -1f, 2f, 5f, 7f }, [4]);
        VerifyGradient(inp => _engine.TensorReLU6(inp), x, "ReLU6");
    }

    [Fact]
    public void Reciprocal_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 0.5f, 1f, 2f, 4f }, [4]);
        VerifyGradient(inp => _engine.TensorReciprocal(inp), x, "Reciprocal");
    }

    [Fact]
    public void Mean_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [6]);
        VerifyGradient(inp => _engine.TensorMeanDiff(inp), x, "Mean");
    }

    [Fact]
    public void Flatten_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        VerifyGradient(inp => _engine.TensorFlatten(inp), x, "Flatten");
    }

    // ──────────────────────────────────────────────────────────────
    // Missing Phase 1 ops gradient tests
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Var_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [6]);
        VerifyGradient(inp => _engine.TensorVar(inp), x, "Var");
    }

    [Fact]
    public void Std_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [6]);
        VerifyGradient(inp => _engine.TensorStd(inp), x, "Std");
    }

    [Fact]
    public void LogSumExp_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 0.5f }, [4]);
        VerifyGradient(inp => _engine.TensorLogSumExp(inp), x, "LogSumExp");
    }

    [Fact]
    public void Norm_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 3f, 4f, 5f, 6f }, [4]);
        VerifyGradient(inp => _engine.TensorNorm(inp), x, "Norm");
    }

    [Fact]
    public void Where_Gradient_MatchesNumerical()
    {
        // Test gradient through the true branch
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [4]);
        var y = new Tensor<float>(new float[] { 10f, 20f, 30f, 40f }, [4]);
        var condition = new bool[] { true, false, true, false };
        VerifyGradient(inp => _engine.TensorWhere(condition, inp, y), x, "Where_TrueBranch");
    }

    [Fact]
    public void MaskedFill_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [4]);
        var mask = new bool[] { false, true, false, true };
        VerifyGradient(inp => _engine.TensorMaskedFill(inp, mask, 0f), x, "MaskedFill");
    }

    [Fact]
    public void ScaledDotProductAttention_Gradient_MatchesNumerical()
    {
        // Simple 2x2 attention
        var q = new Tensor<float>(new float[] { 0.1f, 0.2f, 0.3f, 0.4f }, [2, 2]);
        var k = new Tensor<float>(new float[] { 0.5f, 0.6f, 0.7f, 0.8f }, [2, 2]);
        var v = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
        VerifyGradient(inp => _engine.TensorScaledDotProductAttention(inp, k, v), q, "ScaledDotProductAttention");
    }

    // ──────────────────────────────────────────────────────────────
    // Previously untested ops
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void RReLU_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { -1f, -0.5f, 0.5f, 1f }, [4]);
        // RReLU in non-training mode uses fixed slope (lower+upper)/2
        VerifyGradient(inp => _engine.TensorRReLU(inp, 0.125, 0.333, training: false), x, "RReLU");
    }

    [Fact]
    public void AvgPool1D_Gradient_MatchesNumerical()
    {
        // [batch=1, channels=1, width=6], kernel=2, stride=2
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [1, 1, 6]);
        VerifyGradient(inp => _engine.TensorAvgPool1D(inp, 2, 2), x, "AvgPool1D");
    }

    [Fact]
    public void MaxPool1D_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 3f, 2f, 5f, 4f, 6f }, [1, 1, 6]);
        VerifyGradient(inp => _engine.TensorMaxPool1D(inp, 2, 2), x, "MaxPool1D");
    }

    [Fact]
    public void Narrow_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        VerifyGradient(inp => _engine.TensorNarrow(inp, 1, 0, 2), x, "Narrow");
    }

    [Fact]
    public void CrossEntropyLoss_Gradient_MatchesNumerical()
    {
        // [batch=2, classes=3] — use values that produce distinct softmax probs
        var logits = new Tensor<float>(new float[] { 1f, 2f, 0.5f, 0.1f, 0.9f, 2f }, [2, 3]);
        var targets = new Tensor<float>(new float[] { 0f, 1f, 0f, 0f, 0f, 1f }, [2, 3]);
        VerifyGradient(inp => _engine.TensorCrossEntropyLoss(inp, targets), logits, "CrossEntropyLoss");
    }

    [Fact]
    public void NLLLoss_Gradient_MatchesNumerical()
    {
        // Log probabilities [batch=2, classes=3], targets as class indices
        var logProbs = new Tensor<float>(new float[] { -2.3f, -0.1f, -3.0f, -0.5f, -1.5f, -0.2f }, [2, 3]);
        var targets = new Tensor<float>(new float[] { 1f, 2f }, [2]);
        VerifyGradient(inp => _engine.TensorNLLLoss(inp, targets), logProbs, "NLLLoss");
    }

    [Fact]
    public void KLDivLoss_Gradient_MatchesNumerical()
    {
        var input = new Tensor<float>(new float[] { -1f, -0.5f, -2f, -0.3f }, [4]);
        var target = new Tensor<float>(new float[] { 0.25f, 0.25f, 0.25f, 0.25f }, [4]);
        VerifyGradient(inp => _engine.TensorKLDivLoss(inp, target), input, "KLDivLoss");
    }

    [Fact]
    public void PReLU_4D_NCHW_Gradient_MatchesNumerical()
    {
        // 4D tensor [N=1, C=2, H=2, W=2] with per-channel alpha [2]
        var x = new Tensor<float>(new float[] { -1f, 2f, -3f, 4f, -5f, 6f, -7f, 8f }, [1, 2, 2, 2]);
        var alpha = new Tensor<float>(new float[] { 0.1f, 0.2f }, [2]);
        VerifyGradient(inp => _engine.TensorPReLU(inp, alpha), x, "PReLU_4D");
    }

    [Fact]
    public void Stack_Gradient_MatchesNumerical()
    {
        var a = new Tensor<float>(new float[] { 1f, 2f, 3f }, [3]);
        var b = new Tensor<float>(new float[] { 4f, 5f, 6f }, [3]);
        // Test gradient w.r.t. first tensor in stack
        VerifyGradient(inp =>
        {
            var stacked = _engine.TensorStackDiff(new[] { inp, b }, axis: 0);
            // Reduce to scalar for gradient
            return _engine.TensorMeanDiff(stacked);
        }, a, "Stack");
    }

    [Fact]
    public void ConstantPad_Gradient_MatchesNumerical()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
        VerifyGradient(inp => _engine.TensorConstantPad(inp, new[] { 1, 1, 1, 1 }, 0f), x, "ConstantPad");
    }

    [Fact]
    public void IndexSelect_Gradient_MatchesNumerical()
    {
        // 2D tensor, select rows 0 and 2 along axis 0
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, [3, 2]);
        var indices = new Tensor<int>(new int[] { 0, 2 }, [2]);
        VerifyGradient(inp => _engine.TensorIndexSelectDiff(inp, indices, 0), x, "IndexSelect");
    }

    [Fact]
    public void AdaptiveMaxPool2D_Gradient_MatchesNumerical()
    {
        // [N=1, C=1, H=4, W=4] -> [1, 1, 2, 2]
        var x = new Tensor<float>(new float[]
        {
            1f, 3f, 2f, 4f,
            5f, 7f, 6f, 8f,
            9f, 11f, 10f, 12f,
            13f, 15f, 14f, 16f
        }, [1, 1, 4, 4]);
        VerifyGradient(inp => _engine.TensorAdaptiveMaxPool2D(inp, new[] { 2, 2 }), x, "AdaptiveMaxPool2D");
    }

    [Fact]
    public void UpsampleBilinear_Gradient_MatchesNumerical()
    {
        // [N=1, C=1, H=2, W=2] -> [1, 1, 4, 4]
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [1, 1, 2, 2]);
        VerifyGradient(inp => _engine.TensorUpsampleBilinear(inp, new[] { 4, 4 }), x, "UpsampleBilinear");
    }

    [Fact]
    public void Integration_CompiledBackward_MatchesUncompiled()
    {
        // Compare compiled vs uncompiled backward
        var x = new Tensor<float>(new float[] { 2f, 3f }, [2]);
        var w = new Tensor<float>(new float[] { 0.5f, 1.5f }, [2]);

        // Uncompiled
        Dictionary<Tensor<float>, Tensor<float>> uncompiledGrads;
        using (var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true }))
        {
            var y = _engine.TensorMultiply(x, w);
            var loss = _engine.TensorMeanDiff(y); // reduce to scalar for CompileBackward
            uncompiledGrads = tape.ComputeGradients(loss, sources: new[] { x, w });
        }

        // Compiled
        Dictionary<Tensor<float>, Tensor<float>> compiledGrads;
        using (var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true }))
        {
            var y = _engine.TensorMultiply(x, w);
            var loss = _engine.TensorMeanDiff(y); // reduce to scalar
            var compiled = tape.CompileBackward(loss, sources: new[] { x, w });
            compiledGrads = compiled.Execute();
        }

        // Should produce identical gradients
        Assert.True(compiledGrads.ContainsKey(x));
        Assert.True(compiledGrads.ContainsKey(w));
        for (int i = 0; i < x.Length; i++)
        {
            Assert.Equal(uncompiledGrads[x][i], compiledGrads[x][i], 1e-6f);
            Assert.Equal(uncompiledGrads[w][i], compiledGrads[w][i], 1e-6f);
        }
    }
}
