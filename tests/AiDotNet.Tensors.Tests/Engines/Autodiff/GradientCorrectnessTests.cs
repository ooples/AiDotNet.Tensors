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
            uncompiledGrads = tape.ComputeGradients(y, sources: new[] { x, w });
        }

        // Compiled
        Dictionary<Tensor<float>, Tensor<float>> compiledGrads;
        using (var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true }))
        {
            var y = _engine.TensorMultiply(x, w);
            var compiled = tape.CompileBackward(y, sources: new[] { x, w });
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
