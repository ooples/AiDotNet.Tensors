// Copyright (c) AiDotNet. All rights reserved.
// Phase 3 of issue #214: Vjp, JacRev, JacFwd, Hessian, Vmap, FunctionalCall.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Autodiff.ForwardAD;
using AiDotNet.Tensors.Engines.Autodiff.Transforms;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Phase 3 torch.func-equivalent transforms — Jacobians, Hessian,
/// Vmap, FunctionalCall.
/// </summary>
public class TorchFuncPhase3Tests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    // ─── JacRev ───────────────────────────────────────────────────────

    [Fact]
    public void JacRev_ElementwiseSquare_IsDiagonal()
    {
        // f(x) = x·x (elementwise) with x ∈ ℝ³. Jacobian is diag(2x).
        // TensorSum returns a scalar T and breaks the graph — we use
        // element-wise ops only so the tape stays intact.
        Func<Tensor<float>, Tensor<float>> fn = x => _engine.TensorMultiply(x, x);

        var J = TensorFunc<float>.JacRev(fn)(
            new Tensor<float>(new[] { 2f, 3f, 5f }, new[] { 3 }));

        Assert.Equal(new[] { 3, 3 }, J._shape);
        Assert.Equal(4f, J[0, 0], precision: 3);
        Assert.Equal(6f, J[1, 1], precision: 3);
        Assert.Equal(10f, J[2, 2], precision: 3);
        Assert.Equal(0f, J[0, 1], precision: 3);
        Assert.Equal(0f, J[1, 0], precision: 3);
    }

    [Fact]
    public void JacRev_IdentityFunction_IsIdentityMatrix()
    {
        // f(x) = x → Jacobian is the 3×3 identity.
        Func<Tensor<float>, Tensor<float>> fn = x => x;
        var x = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var J = TensorFunc<float>.JacRev(fn)(x);

        Assert.Equal(new[] { 3, 3 }, J._shape);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(i == j ? 1f : 0f, J[i, j], precision: 3);
    }

    // ─── JacFwd ───────────────────────────────────────────────────────

    [Fact]
    public void JacFwd_MatchesJacRev_OnAnalyticalFunction()
    {
        // f(x) = x² (element-wise). Jacobian is diag(2x).
        Func<Dual<float>, Dual<float>> dualFn = x => DualOps<float>.Square(_engine, x);
        Func<Tensor<float>, Tensor<float>> reverseFn = x =>
            _engine.TensorMultiply(x, x);

        var input = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var jFwd = TensorFunc<float>.JacFwd(_engine, dualFn)(input);
        var jRev = TensorFunc<float>.JacRev(reverseFn)(input);

        Assert.Equal(jRev._shape, jFwd._shape);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(jRev[i, j], jFwd[i, j], precision: 3);

        // Also verify the analytic value: J[i,i] = 2·x[i], J[i,j] = 0 otherwise.
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(2f * (i + 1), jFwd[i, i], precision: 3);
            for (int j = 0; j < 3; j++)
                if (i != j) Assert.Equal(0f, jFwd[i, j], precision: 3);
        }
    }

    // ─── Hessian ──────────────────────────────────────────────────────

    [Fact]
    public void Hessian_QuadraticForm_IsConstantTwiceIdentity()
    {
        // f(x) = ½·(x₁² + x₂²). Gradient: [x₁, x₂]. Hessian: diag([1, 1]).
        // SumToScalarTensor is used so the scalar output stays connected
        // to the tape for higher-order differentiation.
        Func<Tensor<float>, Tensor<float>> fn = x =>
        {
            var squared = _engine.TensorMultiply(x, x);
            var summed = TensorFunc<float>.SumToScalarTensor(squared);
            return _engine.TensorMultiplyScalar(summed, 0.5f);
        };

        var H = TensorFunc<float>.Hessian(fn)(
            new Tensor<float>(new[] { 3f, 4f }, new[] { 2 }));

        Assert.Equal(new[] { 2, 2 }, H._shape);
        Assert.Equal(1f, H[0, 0], precision: 3);
        Assert.Equal(0f, H[0, 1], precision: 3);
        Assert.Equal(0f, H[1, 0], precision: 3);
        Assert.Equal(1f, H[1, 1], precision: 3);
    }

    // ─── Vmap ─────────────────────────────────────────────────────────

    [Fact]
    public void Vmap_AppliesFunctionPerBatchRow()
    {
        // fn doubles its input. Batched over rows of a [3, 2] tensor:
        // input  = [[1, 2], [3, 4], [5, 6]]
        // output = [[2, 4], [6, 8], [10, 12]]
        Func<Tensor<float>, Tensor<float>> fn = x =>
            _engine.TensorMultiplyScalar(x, 2f);

        var input = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 3, 2 });
        var result = TensorFunc<float>.Vmap(fn, inDim: 0, outDim: 0)(input);

        Assert.Equal(new[] { 3, 2 }, result._shape);
        Assert.Equal(2f, result[0, 0]);
        Assert.Equal(4f, result[0, 1]);
        Assert.Equal(6f, result[1, 0]);
        Assert.Equal(8f, result[1, 1]);
        Assert.Equal(10f, result[2, 0]);
        Assert.Equal(12f, result[2, 1]);
    }

    [Fact]
    public void Vmap_PerSampleGradient_ComposesWithGrad()
    {
        // Per-sample gradient of f(x) = x² (single-element tensor) is 2x.
        // Batch of 4 inputs — expect 2x per sample.
        // Using TensorMultiply keeps everything on the tape; TensorSum
        // returns a scalar T (not Tensor<T>) and would break the graph.
        Func<Tensor<float>, Tensor<float>> scalarFn = x =>
            _engine.TensorMultiply(x, x); // [1]-shape output for [1]-shape x

        var gradFn = TensorFunc<float>.Grad(scalarFn);
        var perSample = TensorFunc<float>.Vmap(gradFn);

        var batch = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4, 1 });
        var result = perSample(batch);

        Assert.Equal(new[] { 4, 1 }, result._shape);
        Assert.Equal(2f, result[0, 0], precision: 3);
        Assert.Equal(4f, result[1, 0], precision: 3);
        Assert.Equal(6f, result[2, 0], precision: 3);
        Assert.Equal(8f, result[3, 0], precision: 3);
    }

    [Fact]
    public void Vmap_ArgumentValidation()
    {
        Assert.Throws<ArgumentNullException>(
            () => TensorFunc<float>.Vmap((Func<Tensor<float>, Tensor<float>>)null!));
        var fn = TensorFunc<float>.Vmap((Tensor<float> x) => x, inDim: 5);
        Assert.Throws<ArgumentOutOfRangeException>(
            () => fn(new Tensor<float>(new[] { 1f }, new[] { 1 })));
    }

    // ─── FunctionalCall ───────────────────────────────────────────────

    [Fact]
    public void FunctionalCall_RunsWithSubstitutedParametersAndRestores()
    {
        // ParameterBuffer holding a single 3-element parameter.
        var pb = new ParameterBuffer<float>(new int[][] { new[] { 3 } });
        var originalView = pb.CreateAllViews()[0];
        originalView[0] = 10f;
        originalView[1] = 20f;
        originalView[2] = 30f;

        // fn reads the parameter.
        Func<Tensor<float>> fn = () =>
        {
            var view = pb.CreateAllViews()[0];
            return new Tensor<float>(new[] { view[0], view[1], view[2] }, new[] { 3 });
        };

        var substituted = new Vector<float>(new[] { 1f, 2f, 3f });
        var result = TensorFunc<float>.FunctionalCall(pb, substituted, fn);

        Assert.Equal(1f, result[0]);
        Assert.Equal(2f, result[1]);
        Assert.Equal(3f, result[2]);

        // Original values must be restored.
        var restored = pb.CreateAllViews()[0];
        Assert.Equal(10f, restored[0]);
        Assert.Equal(20f, restored[1]);
        Assert.Equal(30f, restored[2]);
    }

    [Fact]
    public void FunctionalCall_ArgumentValidation()
    {
        var pb = new ParameterBuffer<float>(new int[][] { new[] { 1 } });
        var v = new Vector<float>(1);
        Assert.Throws<ArgumentNullException>(
            () => TensorFunc<float>.FunctionalCall(null!, v, () => null!));
        Assert.Throws<ArgumentNullException>(
            () => TensorFunc<float>.FunctionalCall(pb, null!, () => null!));
        Assert.Throws<ArgumentNullException>(
            () => TensorFunc<float>.FunctionalCall(pb, v, null!));
    }
}
