using System;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Gradient-correctness guards for the abs/softplus op chain and for the
/// AutoTrainingCompiler REPLAY path on realistic (parameter → matmul → unary →
/// loss) computations.
///
/// Origin: AiDotNet's BCE-with-logits loss (the numerically-stable softplus
/// max(x,0)+log(1+exp(-|x|))) was suspected of producing a wrong-direction tape
/// gradient. Direct testing here showed that suspicion was WRONG — both the
/// interpreted autodiff (every op) and the compiled-replay autodiff (for any
/// computation where the parameter reaches the loss through a binary op, i.e.
/// all real training) compute the correct gradient. These tests pin that down so
/// the conclusion is durable and the replay path's gradient correctness — which
/// was previously untested — is now guarded.
/// </summary>
public class AbsSoftplusGradientRepro
{
    private readonly ITestOutputHelper _out;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public AbsSoftplusGradientRepro(ITestOutputHelper output) => _out = output;

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    // ── Interpreted-path correctness (the path small models use) ──────────────

    [Fact]
    public void AbsForm_Softplus_Gradient_MatchesSigmoid()
    {
        var data = new double[] { -2.0, -0.5, 0.5, 2.0 };
        var x = new Tensor<double>(data, new[] { 4 });

        using var tape = new GradientTape<double>();
        // softplus(x) = max(x,0) + log(1 + exp(-|x|))  →  d/dx = sigmoid(x)
        var logT = _engine.TensorLog(_engine.TensorAddScalar(
            _engine.TensorExp(_engine.TensorNegate(_engine.TensorAbs(x))), 1.0));
        var sp = _engine.TensorAdd(_engine.ReLU(x), logT);
        var loss = _engine.ReduceSum(sp, null);
        var gx = tape.ComputeGradients(loss, new[] { x })[x];

        for (int i = 0; i < data.Length; i++)
            Assert.True(Math.Abs(gx[i] - Sigmoid(data[i])) < 1e-6,
                $"abs-form softplus grad at x={data[i]}: got {gx[i]}, expected {Sigmoid(data[i])}");
    }

    [Fact]
    public void AbsFreeForm_Softplus_Gradient_MatchesSigmoid()
    {
        var data = new double[] { -2.0, -0.5, 0.5, 2.0 };
        var x = new Tensor<double>(data, new[] { 4 });

        using var tape = new GradientTape<double>();
        var logT = _engine.TensorLog(_engine.TensorAddScalar(_engine.TensorExp(x), 1.0));
        var loss = _engine.ReduceSum(logT, null);
        var gx = tape.ComputeGradients(loss, new[] { x })[x];

        for (int i = 0; i < data.Length; i++)
            Assert.True(Math.Abs(gx[i] - Sigmoid(data[i])) < 1e-6,
                $"abs-free softplus grad at x={data[i]}: got {gx[i]}, expected {Sigmoid(data[i])}");
    }

    [Fact]
    public void Abs_Gradient_IsSign()
    {
        var data = new double[] { -2.0, -0.5, 0.5, 2.0 };
        var x = new Tensor<double>(data, new[] { 4 });

        using var tape = new GradientTape<double>();
        var loss = _engine.ReduceSum(_engine.TensorAbs(x), null);
        var gx = tape.ComputeGradients(loss, new[] { x })[x];

        for (int i = 0; i < data.Length; i++)
            Assert.True(Math.Abs(gx[i] - Math.Sign(data[i])) < 1e-6,
                $"|x| grad at x={data[i]}: got {gx[i]}, expected {Math.Sign(data[i])}");
    }

    // ── Compiled-replay correctness on realistic patterns (was untested) ──────

    [Fact]
    public void CompiledReplay_MatMulChain_WeightGradientPresent()
    {
        var input = new Tensor<double>(new double[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var wData = new double[] { 0.5, -0.5, 0.25, 0.75 };

        bool prev = AutoTrainingCompiler.Enabled;
        AutoTrainingCompiler.Enabled = true;
        AutoTrainingCompiler.TestMinForwardElementsOverride = 1;
        try
        {
            var weight = new Tensor<double>((double[])wData.Clone(), new[] { 2, 2 });
            using var tape = new GradientTape<double>(new GradientTapeOptions { Persistent = true });
            bool present = false;
            for (int step = 0; step < 5; step++)
            {
                tape.Reset();
                var h = _engine.TensorMatMul(input, weight);
                var loss = _engine.ReduceMean(h, new[] { 0, 1 }, keepDims: false);
                present = tape.ComputeGradients(loss, new[] { weight }).ContainsKey(weight);
            }
            Assert.True(present, "compiled replay dropped the matmul weight gradient");
        }
        finally
        {
            AutoTrainingCompiler.TestMinForwardElementsOverride = null;
            AutoTrainingCompiler.Enabled = prev;
        }
    }

    [Fact]
    public void CompiledReplay_MatMulThenUnary_GradientMatchesReference()
    {
        // BCE-with-logits-shaped pattern: weight → matmul → unary(exp) → loss.
        var input = new Tensor<double>(new double[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var wData = new double[] { 0.1, -0.2, 0.3, 0.4 };

        double[] refGrad;
        {
            var weight = new Tensor<double>((double[])wData.Clone(), new[] { 2, 2 });
            using var tape = new GradientTape<double>();
            var loss = _engine.ReduceMean(
                _engine.TensorExp(_engine.TensorMatMul(input, weight)), new[] { 0, 1 }, keepDims: false);
            var g = tape.ComputeGradients(loss, new[] { weight })[weight];
            refGrad = new double[4];
            for (int i = 0; i < 4; i++) refGrad[i] = g[i];
        }

        bool prev = AutoTrainingCompiler.Enabled;
        AutoTrainingCompiler.Enabled = true;
        AutoTrainingCompiler.TestMinForwardElementsOverride = 1;
        try
        {
            var weight = new Tensor<double>((double[])wData.Clone(), new[] { 2, 2 });
            using var tape = new GradientTape<double>(new GradientTapeOptions { Persistent = true });
            double[]? cg = null;
            for (int step = 0; step < 5; step++)
            {
                tape.Reset();
                var loss = _engine.ReduceMean(
                    _engine.TensorExp(_engine.TensorMatMul(input, weight)), new[] { 0, 1 }, keepDims: false);
                var grads = tape.ComputeGradients(loss, new[] { weight });
                cg = grads.TryGetValue(weight, out var g) ? new[] { g[0], g[1], g[2], g[3] } : null;
            }
            Assert.True(cg is not null, "compiled replay dropped the weight gradient through a unary intermediate");
            for (int i = 0; i < 4; i++)
                Assert.True(Math.Abs(cg![i] - refGrad[i]) < 1e-9,
                    $"compiled replay grad[{i}]={cg[i]} != interpreted reference {refGrad[i]}");
        }
        finally
        {
            AutoTrainingCompiler.TestMinForwardElementsOverride = null;
            AutoTrainingCompiler.Enabled = prev;
        }
    }
}
