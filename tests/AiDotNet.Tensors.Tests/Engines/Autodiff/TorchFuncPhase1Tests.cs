// Copyright (c) AiDotNet. All rights reserved.
// Phase 1 of issue #214: InferenceMode scope, higher-order gradients,
// TensorFunc.Grad / GradAndValue.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Autodiff.Transforms;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Issue #214 Phase 1 coverage: <see cref="InferenceModeScope{T}"/>,
/// higher-order grad via <c>createGraph: true</c>, and
/// <see cref="TensorFunc{T}"/> entry points.
/// </summary>
public class TorchFuncPhase1Tests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    // ─── InferenceMode ────────────────────────────────────────────────

    [Fact]
    public void InferenceMode_SuppressesRecordingLikeNoGrad()
    {
        using var tape = new GradientTape<float>();
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f }, new[] { 3 });

        _ = _engine.TensorMultiplyScalar(x, 2f);
        int countBeforeInference = tape.EntryCount;
        Assert.True(countBeforeInference > 0);

        using (GradientTape<float>.InferenceMode())
        {
            _ = _engine.TensorMultiplyScalar(x, 3f);
        }

        Assert.Equal(countBeforeInference, tape.EntryCount);
    }

    [Fact]
    public void InferenceMode_IsActiveFlagReflectsScope()
    {
        Assert.False(InferenceModeScope<float>.IsActive);
        using (GradientTape<float>.InferenceMode())
        {
            Assert.True(InferenceModeScope<float>.IsActive);
            Assert.True(NoGradScope<float>.IsSuppressed,
                "InferenceMode is strictly stronger than NoGrad — IsSuppressed must also be true.");
        }
        Assert.False(InferenceModeScope<float>.IsActive);
        Assert.False(NoGradScope<float>.IsSuppressed);
    }

    [Fact]
    public void InferenceMode_NestedScopesRestoreCleanly()
    {
        using (GradientTape<float>.InferenceMode())
        {
            Assert.True(InferenceModeScope<float>.IsActive);
            using (GradientTape<float>.InferenceMode())
            {
                Assert.True(InferenceModeScope<float>.IsActive);
            }
            Assert.True(InferenceModeScope<float>.IsActive);
        }
        Assert.False(InferenceModeScope<float>.IsActive);
    }

    [Fact]
    public void InferenceMode_DoesNotLeakNoGradAcrossScopes()
    {
        // NoGrad nested inside InferenceMode and vice-versa must not
        // leak state when unwinding in either order.
        using (GradientTape<float>.NoGrad())
        {
            using (GradientTape<float>.InferenceMode())
            {
                Assert.True(InferenceModeScope<float>.IsActive);
                Assert.True(NoGradScope<float>.IsSuppressed);
            }
            Assert.False(InferenceModeScope<float>.IsActive);
            Assert.True(NoGradScope<float>.IsSuppressed);
        }
        Assert.False(NoGradScope<float>.IsSuppressed);
    }

    // ─── Higher-order grad via createGraph ───────────────────────────

    [Fact]
    public void GradientTape_CreateGraph_EnablesSecondOrderDerivative()
    {
        // f(x) = x^2  →  f'(x) = 2x  →  f''(x) = 2.
        // At x=3: f' = 6, f'' = 2.
        //
        // Higher-order AD works by recording the backward pass onto
        // the SAME tape as the forward (createGraph keeps the tape
        // current during backward), then calling ComputeGradients a
        // second time on the gradient tensor. Requires Persistent=true
        // so the tape is not reset between the two calls.
        var x = new Tensor<float>(new float[] { 3f }, new[] { 1 });

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        var y = _engine.TensorMultiply(x, x); // y = x^2

        // createGraph=true → backward ops recorded onto `tape` for
        // a second derivative call.
        var firstGrads = tape.ComputeGradients(y, new[] { x }, createGraph: true);
        var dy_dx = firstGrads[x];

        // dy/dx at x=3 should be 2·3 = 6.
        Assert.Equal(6f, dy_dx[0], precision: 3);

        // Second derivative: d²y/dx² = d(2x)/dx = 2.
        var secondGrads = tape.ComputeGradients(dy_dx, new[] { x });
        Assert.True(secondGrads.ContainsKey(x),
            "Tape must have captured the backward op graph so d²y/dx² is computable.");
        var d2y_dx2 = secondGrads[x];
        Assert.Equal(2f, d2y_dx2[0], precision: 3);
    }

    [Fact]
    public void GradientTape_CreateGraph_CubicFunction()
    {
        // f(x) = x^3  →  f'(x) = 3x^2  →  f''(x) = 6x.
        // At x=2: f' = 12, f'' = 12.
        var x = new Tensor<float>(new float[] { 2f }, new[] { 1 });

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var xx = _engine.TensorMultiply(x, x);
        var y = _engine.TensorMultiply(xx, x);

        var firstGrads = tape.ComputeGradients(y, new[] { x }, createGraph: true);
        Assert.Equal(12f, firstGrads[x][0], precision: 3);

        var secondGrads = tape.ComputeGradients(firstGrads[x], new[] { x });
        Assert.Equal(12f, secondGrads[x][0], precision: 3);
    }

    // ─── TensorFunc.Grad / GradAndValue ───────────────────────────────

    [Fact]
    public void TensorFunc_Grad_UnaryMatchesAnalyticDerivative()
    {
        // f(x) = x^2  →  f'(x) = 2x.
        var gradFn = TensorFunc<float>.Grad(
            x => _engine.TensorMultiply(x, x));
        var x = new Tensor<float>(new float[] { 3f, -4f, 0.5f }, new[] { 3 });
        var g = gradFn(x);

        Assert.Equal(6f, g[0], precision: 4);
        Assert.Equal(-8f, g[1], precision: 4);
        Assert.Equal(1f, g[2], precision: 4);
    }

    [Fact]
    public void TensorFunc_Grad_MultiArg_RespectsArgIndex()
    {
        // f(a, b) = a * b  →  df/da = b, df/db = a.
        Func<Tensor<float>[], Tensor<float>> fn =
            args => _engine.TensorMultiply(args[0], args[1]);

        var gradA = TensorFunc<float>.Grad(fn, argIndex: 0);
        var gradB = TensorFunc<float>.Grad(fn, argIndex: 1);

        var a = new Tensor<float>(new float[] { 2f, 3f }, new[] { 2 });
        var b = new Tensor<float>(new float[] { 5f, 7f }, new[] { 2 });

        var gA = gradA(new[] { a, b });
        var gB = gradB(new[] { a, b });

        Assert.Equal(5f, gA[0], precision: 4);
        Assert.Equal(7f, gA[1], precision: 4);
        Assert.Equal(2f, gB[0], precision: 4);
        Assert.Equal(3f, gB[1], precision: 4);
    }

    [Fact]
    public void TensorFunc_GradAndValue_ReturnsBothInSinglePass()
    {
        // f(x) = x^2 at x=4  →  value = 16, grad = 8.
        var gradValueFn = TensorFunc<float>.GradAndValue(
            x => _engine.TensorMultiply(x, x));
        var x = new Tensor<float>(new float[] { 4f }, new[] { 1 });
        var (g, v) = gradValueFn(x);

        Assert.Equal(16f, v[0], precision: 4);
        Assert.Equal(8f, g[0], precision: 4);
    }

    [Fact]
    public void TensorFunc_Grad_InputNotUsed_ReturnsZeroGrad()
    {
        // f(a, b) = a * a (b unused) → df/db = 0.
        Func<Tensor<float>[], Tensor<float>> fn =
            args => _engine.TensorMultiply(args[0], args[0]);

        var gradB = TensorFunc<float>.Grad(fn, argIndex: 1);
        var a = new Tensor<float>(new float[] { 5f }, new[] { 1 });
        var b = new Tensor<float>(new float[] { 99f }, new[] { 1 });

        var gB = gradB(new[] { a, b });
        Assert.Equal(0f, gB[0], precision: 6);
    }

    [Fact]
    public void TensorFunc_Grad_ArgumentValidation()
    {
        Assert.Throws<ArgumentNullException>(
            () => TensorFunc<float>.Grad((Func<Tensor<float>[], Tensor<float>>)null!));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => TensorFunc<float>.Grad(args => args[0], argIndex: -1));

        var good = TensorFunc<float>.Grad(args => args[0], argIndex: 5);
        Assert.Throws<ArgumentOutOfRangeException>(
            () => good(new[] { new Tensor<float>(new float[] { 1f }, new[] { 1 }) }));
    }
}
