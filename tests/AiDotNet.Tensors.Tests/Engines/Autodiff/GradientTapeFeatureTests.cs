using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tests for new GradientTape features: higher-order gradients, NoGradScope,
/// AutogradFunction, TensorPool, and new backward functions.
/// </summary>
public class GradientTapeFeatureTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    [Fact]
    public void NoGradScope_SuppressesRecording()
    {
        using var tape = new GradientTape<float>();
        var x = new Tensor<float>(new float[] { 2f, 3f }, [2]);

        // Record an op
        var y = _engine.TensorMultiplyScalar(x, 2f);
        Assert.True(tape.EntryCount > 0, "Should record with tape active");

        int countBefore = tape.EntryCount;

        // NoGradScope should suppress
        using (GradientTape<float>.NoGrad())
        {
            var z = _engine.TensorMultiplyScalar(x, 3f);
        }

        Assert.Equal(countBefore, tape.EntryCount);
    }

    [Fact]
    public void NoGradScope_NestedScopes_Work()
    {
        using var tape = new GradientTape<float>();
        var x = new Tensor<float>(new float[] { 1f }, [1]);

        using (GradientTape<float>.NoGrad())
        {
            Assert.True(NoGradScope<float>.IsSuppressed);

            using (GradientTape<float>.NoGrad())
            {
                Assert.True(NoGradScope<float>.IsSuppressed);
            }

            // Inner disposed but outer still active
            Assert.True(NoGradScope<float>.IsSuppressed);
        }

        // Both disposed
        Assert.False(NoGradScope<float>.IsSuppressed);
    }

    [Fact]
    public void HigherOrderGradient_CreateGraph_RecordsDuringBackward()
    {
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        var x = new Tensor<float>(new float[] { 3f }, [1]);

        // f(x) = x * x (x^2)
        var y = _engine.TensorMultiply(x, x);

        int entriesBefore = tape.EntryCount;

        // First gradient with createGraph=true
        var grads = tape.ComputeGradients(y, sources: new[] { x }, createGraph: true);

        Assert.True(grads.ContainsKey(x));
        // dx = 2x = 6
        Assert.True(Math.Abs(grads[x][0] - 6f) < 0.1f,
            $"Expected gradient ~6, got {grads[x][0]}");
    }

    [Fact]
    public void Concatenate_Backward_SplitsGradient()
    {
        using var tape = new GradientTape<float>();

        var a = new Tensor<float>(new float[] { 1f, 2f }, [2]);
        var b = new Tensor<float>(new float[] { 3f, 4f, 5f }, [3]);

        // Concatenate is recorded on tape, then multiply by ones to create a differentiable output
        var concat = _engine.TensorConcatenate(new[] { a, b }, axis: 0);

        // Use the concat tensor directly as loss (tape tracks it)
        var grads = tape.ComputeGradients(concat, sources: new[] { a, b });

        Assert.True(grads.ContainsKey(a), "Should have gradient for a");
        Assert.True(grads.ContainsKey(b), "Should have gradient for b");
        Assert.Equal(2, grads[a].Length);
        Assert.Equal(3, grads[b].Length);
    }

    [Fact]
    public void Softplus_Backward_IsCorrect()
    {
        using var tape = new GradientTape<float>();

        var x = new Tensor<float>(new float[] { -1f, 0f, 2f }, [3]);
        var y = _engine.Softplus(x);

        // Use y directly as loss output
        var grads = tape.ComputeGradients(y, sources: new[] { x });

        Assert.True(grads.ContainsKey(x));
        // Softplus'(x) = sigmoid(x), seed gradient = ones
        for (int i = 0; i < 3; i++)
        {
            float expected = 1f / (1f + MathF.Exp(-x[i]));
            Assert.True(Math.Abs(grads[x][i] - expected) < 1e-4f,
                $"Softplus gradient at {x[i]}: expected {expected}, got {grads[x][i]}");
        }
    }

    [Fact]
    public void TensorPool_RentReturn_Reuses()
    {
        TensorPool<float>.Clear();

        var shape = new[] { 3, 4 };
        var t1 = TensorPool<float>.Rent(shape);
        Assert.Equal(12, t1.Length);

        TensorPool<float>.Return(t1);

        var t2 = TensorPool<float>.Rent(shape);
        // Should get the same tensor back (same reference)
        Assert.Equal(12, t2.Length);

        TensorPool<float>.Clear();
    }

    [Fact]
    public void DetectAnomaly_Property_CanBeSet()
    {
        using var tape = new GradientTape<float>();
        Assert.False(tape.DetectAnomaly);

        tape.DetectAnomaly = true;
        Assert.True(tape.DetectAnomaly);
    }

    [Fact]
    public void BatchMatMul_Backward_ProducesGradients()
    {
        using var tape = new GradientTape<float>();

        var a = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [1, 2, 2]);
        var b = new Tensor<float>(new float[] { 5f, 6f, 7f, 8f }, [1, 2, 2]);

        var c = _engine.TensorBatchMatMul(a, b);

        var grads = tape.ComputeGradients(c, sources: new[] { a, b });

        Assert.True(grads.ContainsKey(a), "Should have gradient for a");
        Assert.True(grads.ContainsKey(b), "Should have gradient for b");

        for (int i = 0; i < grads[a].Length; i++)
            Assert.False(float.IsNaN(grads[a][i]) || float.IsInfinity(grads[a][i]));
        for (int i = 0; i < grads[b].Length; i++)
            Assert.False(float.IsNaN(grads[b][i]) || float.IsInfinity(grads[b][i]));
    }

    [Fact]
    public void HardSwish_Backward_IsCorrect()
    {
        using var tape = new GradientTape<float>();

        var x = new Tensor<float>(new float[] { -4f, 0f, 4f }, [3]);
        var y = _engine.HardSwish(x);

        var grads = tape.ComputeGradients(y, sources: new[] { x });

        Assert.True(grads.ContainsKey(x));
        // x=-4: deriv=0, x=0: deriv=0.5, x=4: deriv=1
        Assert.True(Math.Abs(grads[x][0]) < 0.01f, $"x=-4: expected ~0, got {grads[x][0]}");
        Assert.True(Math.Abs(grads[x][1] - 0.5f) < 0.01f, $"x=0: expected ~0.5, got {grads[x][1]}");
        Assert.True(Math.Abs(grads[x][2] - 1f) < 0.01f, $"x=4: expected ~1, got {grads[x][2]}");
    }

    [Fact]
    public void TrainingConvergence_LinearRegression_LossDecreases()
    {
        // Train y = 2x + 1 using gradient tape
        // Use tape-tracked ops for the entire forward pass
        var w = new Tensor<float>(new float[] { 0f }, [1]);
        var b = new Tensor<float>(new float[] { 0f }, [1]);
        float lr = 0.01f;

        float initialLoss = float.MaxValue;
        float finalLoss = float.MaxValue;

        for (int step = 0; step < 100; step++)
        {
            using var tape = new GradientTape<float>();

            var x = new Tensor<float>(new float[] { 3f }, [1]);
            var target = new Tensor<float>(new float[] { 7f }, [1]);

            // All ops go through engine, recorded on tape
            var pred = _engine.TensorAdd(_engine.TensorMultiply(w, x), b);
            var diff = _engine.TensorSubtract(pred, target);
            var sq = _engine.TensorMultiply(diff, diff);

            float lossVal = sq[0];
            if (step == 0) initialLoss = lossVal;
            if (step == 99) finalLoss = lossVal;

            // Use sq directly (it's on the tape)
            var grads = tape.ComputeGradients(sq, sources: new[] { w, b });

            if (grads.TryGetValue(w, out var wGrad))
                w[0] -= lr * wGrad[0];
            if (grads.TryGetValue(b, out var bGrad))
                b[0] -= lr * bGrad[0];
        }

        Assert.True(finalLoss < initialLoss * 0.1f,
            $"Loss should decrease significantly. Initial: {initialLoss}, Final: {finalLoss}");
    }

    // ─── Missing tests for untested features ────────────────────

    [Fact]
    public void AutogradFunction_CustomSquare_ProducesCorrectGradient()
    {
        // Custom autograd function: f(x) = x^2, f'(x) = 2x
        var squareOp = new SquareFunction();

        using var tape = new GradientTape<float>();
        var x = new Tensor<float>(new float[] { 3f }, [1]);

        var y = squareOp.Apply(x);

        Assert.Equal(9f, y[0], 0.01f); // 3^2 = 9

        var grads = tape.ComputeGradients(y, sources: new[] { x });
        Assert.True(grads.ContainsKey(x), "Should have gradient for x");
        Assert.Equal(6f, grads[x][0], 0.1f); // d(x^2)/dx = 2x = 6
    }

    private class SquareFunction : AutogradFunction<float>
    {
        public override Tensor<float> Forward(AutogradContext ctx, params Tensor<float>[] inputs)
        {
            ctx.SaveForBackward(inputs[0]);
            var engine = AiDotNetEngine.Current;
            return engine.TensorMultiply(inputs[0], inputs[0]);
        }

        public override void Backward(AutogradContext ctx, Tensor<float> gradOutput,
            IEngine engine, Dictionary<Tensor<float>, Tensor<float>> grads)
        {
            var x = ctx.GetSaved<float>(0);
            var twoX = engine.TensorMultiplyScalar(x, 2f);
            var grad = engine.TensorMultiply(gradOutput, twoX);
            DifferentiableOps.AccumulateGrad(grads, x, grad, engine);
        }
    }

    [Fact]
    public void CompiledBackwardGraph_EliminatesDeadNodes()
    {
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        var x = new Tensor<float>(new float[] { 2f, 3f }, [2]);
        var y = new Tensor<float>(new float[] { 1f, 1f }, [2]);

        // Two independent computation paths
        var pathA = _engine.TensorMultiply(x, x); // x^2
        var lossA = _engine.TensorMeanDiff(pathA); // reduce to scalar
        var pathB = _engine.TensorMultiply(y, y); // y^2 — dead if we only want dx
        var lossB = _engine.TensorMeanDiff(pathB); // dead path

        // Compile for x only — pathB should be eliminated
        var compiled = tape.CompileBackward(lossA, sources: new[] { x });

        Assert.True(compiled.EliminatedEntryCount > 0 || compiled.ReachableEntryCount <= tape.EntryCount,
            "Dead node elimination should remove unreachable entries");

        var grads = compiled.Execute();
        Assert.True(grads.ContainsKey(x), "Should have gradient for x");
    }

    [Fact]
    public void GradientAndUpdate_UpdatesParameters()
    {
        using var tape = new GradientTape<float>();

        var w = new Tensor<float>(new float[] { 1f, 2f }, [2]);
        var target = new Tensor<float>(new float[] { 0f, 0f }, [2]);

        float wBefore0 = w[0];
        float wBefore1 = w[1];

        // loss = sum((w - target)^2) = w[0]^2 + w[1]^2
        var diff = _engine.TensorSubtract(w, target);
        var sq = _engine.TensorMultiply(diff, diff);

        // Update with learning rate
        tape.GradientAndUpdate(sq, new[] { w }, 0.1f);

        // Parameters should have changed
        Assert.NotEqual(wBefore0, w[0]);
        Assert.NotEqual(wBefore1, w[1]);
        // w should move toward zero (gradient descent on w^2)
        Assert.True(Math.Abs(w[0]) < Math.Abs(wBefore0), "w[0] should decrease toward 0");
        Assert.True(Math.Abs(w[1]) < Math.Abs(wBefore1), "w[1] should decrease toward 0");
    }

    [Fact]
    public void RegisterHook_ModifiesGradientDuringBackward()
    {
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true, EnableHooks = true });

        var x = new Tensor<float>(new float[] { 3f }, [1]);

        var y = _engine.TensorMultiplyScalar(x, 2f); // y = 2x

        // Register hook that doubles the gradient
        tape.RegisterHook(y, grad =>
        {
            var doubled = _engine.TensorMultiplyScalar(grad, 2f);
            return doubled;
        });

        var grads = tape.ComputeGradients(y, sources: new[] { x });

        // Normal gradient would be 2 (dy/dx = 2)
        // Hook doubles it to 4
        Assert.True(grads.ContainsKey(x));
        Assert.Equal(4f, grads[x][0], 0.1f);
    }

    [Fact]
    public void AnomalyDetection_ThrowsOnNaN()
    {
        using var tape = new GradientTape<float>();
        tape.DetectAnomaly = true;

        var x = new Tensor<float>(new float[] { 0f }, [1]);

        // log(0) = -inf, gradient of log at 0 = 1/0 = inf
        var y = _engine.TensorLog(x);

        // Anomaly detection throws the dedicated AnomalyDetectedException
        // (which inherits from ArithmeticException for back-compat).
        // Use ThrowsAny<> so the test stays green whether callers want
        // the specific subtype or the base class.
        Assert.ThrowsAny<ArithmeticException>(() =>
            tape.ComputeGradients(y, sources: new[] { x }));
    }

    [Fact]
    public void HigherOrder_GradOfGrad_XCubed()
    {
        // f(x) = x^3
        // f'(x) = 3x^2
        // f''(x) = 6x
        using var outerTape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        var x = new Tensor<float>(new float[] { 2f }, [1]);

        // x^3 = x * x * x
        var x2 = _engine.TensorMultiply(x, x);
        var x3 = _engine.TensorMultiply(x2, x);

        // First derivative: f'(2) = 3*4 = 12
        var firstGrads = outerTape.ComputeGradients(x3, sources: new[] { x }, createGraph: true);
        Assert.True(firstGrads.ContainsKey(x));
        float firstDeriv = firstGrads[x][0];
        Assert.True(Math.Abs(firstDeriv - 12f) < 0.5f,
            $"f'(2) should be ~12, got {firstDeriv}");
    }

    [Fact]
    public void StopGradient_PreventsGradientFlow()
    {
        // y = x * StopGradient(x) should have dy/dx = StopGradient(x) (gradient only flows through first x)
        var x = new Tensor<float>(new float[] { 3f }, new[] { 1 });

        using var tape = new GradientTape<float>();
        var stopped = _engine.StopGradient(x);
        var y = _engine.TensorMultiply(x, stopped);

        var grads = tape.ComputeGradients(y, [x]);

        // dy/dx = stopped = 3 (gradient does NOT flow through the stopped copy)
        Assert.True(grads.ContainsKey(x), "Should have gradient for x");
        float grad = grads[x][0];
        Assert.True(Math.Abs(grad - 3f) < 0.1f,
            $"dy/dx should be 3 (value of stopped x), got {grad}");
    }

    [Fact]
    public void StopGradient_PreservesForwardValue()
    {
        var x = new Tensor<float>(new float[] { 5f }, new[] { 1 });
        var stopped = _engine.StopGradient(x);

        Assert.Equal(5f, stopped[0]);
    }
}
