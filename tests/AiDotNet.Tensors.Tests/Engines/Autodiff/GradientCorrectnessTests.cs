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
}
