// Copyright (c) AiDotNet. All rights reserved.
// Phase 2 of issue #214: forward-mode AD (Dual<T>, DualOps<T>, Jvp).
//
// Nullable disabled for the ArgumentNullException-testing scenarios
// — see CLAUDE.md ban on null-forgiving operators in production code.

#nullable disable

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Autodiff.ForwardAD;
using AiDotNet.Tensors.Engines.Autodiff.Transforms;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Forward-mode AD correctness: every op rule in
/// <see cref="DualOps{T}"/> must produce a tangent that agrees with
/// the reverse-mode gradient (as computed by
/// <see cref="TensorFunc{T}.Grad"/>) to within numerical tolerance.
/// </summary>
public class TorchFuncPhase2Tests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    [Fact]
    public void Dual_Constant_HasZeroTangent()
    {
        var primal = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var c = Dual<float>.Constant(primal);

        Assert.Equal(primal, c.Primal);
        for (int i = 0; i < 3; i++)
            Assert.Equal(0f, c.Tangent[i]);
    }

    [Fact]
    public void Dual_Seed_MatchesSuppliedTangent()
    {
        var primal = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var tangent = new Tensor<float>(new[] { 1f, 0f }, new[] { 2 });
        var d = Dual<float>.Seed(primal, tangent);

        Assert.Equal(1f, d.Tangent[0]);
        Assert.Equal(0f, d.Tangent[1]);
    }

    [Fact]
    public void Dual_ShapeMismatch_Throws()
    {
        var p = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var t = new Tensor<float>(new[] { 1f }, new[] { 1 });
        Assert.Throws<ArgumentException>(() => new Dual<float>(p, t));
    }

    // ─── Op rule verification ─────────────────────────────────────────

    [Fact]
    public void DualOps_Add_Tangent_IsDirectSum()
    {
        var a = new Dual<float>(
            new Tensor<float>(new[] { 3f }, new[] { 1 }),
            new Tensor<float>(new[] { 1f }, new[] { 1 })); // dy/da = 1
        var b = new Dual<float>(
            new Tensor<float>(new[] { 4f }, new[] { 1 }),
            new Tensor<float>(new[] { 1f }, new[] { 1 })); // dy/db = 1

        var result = DualOps<float>.Add(_engine, a, b);
        Assert.Equal(7f, result.Primal[0]);
        Assert.Equal(2f, result.Tangent[0]); // 1 + 1
    }

    [Fact]
    public void DualOps_Multiply_AgreesWithProductRule()
    {
        // f(x) = x * c where c=5. df/dx = 5.
        var x = new Dual<float>(
            new Tensor<float>(new[] { 3f }, new[] { 1 }),
            new Tensor<float>(new[] { 1f }, new[] { 1 }));
        var c = Dual<float>.Constant(new Tensor<float>(new[] { 5f }, new[] { 1 }));

        var result = DualOps<float>.Multiply(_engine, x, c);
        Assert.Equal(15f, result.Primal[0]);
        Assert.Equal(5f, result.Tangent[0]);
    }

    [Fact]
    public void DualOps_Square_TangentEqualsTwoX()
    {
        // f(x) = x^2 at x=3.5. df/dx = 7.
        var x = new Dual<float>(
            new Tensor<float>(new[] { 3.5f }, new[] { 1 }),
            new Tensor<float>(new[] { 1f }, new[] { 1 }));
        var result = DualOps<float>.Square(_engine, x);
        Assert.Equal(12.25f, result.Primal[0], precision: 4);
        Assert.Equal(7f, result.Tangent[0], precision: 4);
    }

    [Fact]
    public void DualOps_Sin_TangentEqualsCos()
    {
        // f(x) = sin(x) at x=0. df/dx = cos(0) = 1.
        var x = new Dual<float>(
            new Tensor<float>(new[] { 0f }, new[] { 1 }),
            new Tensor<float>(new[] { 1f }, new[] { 1 }));
        var result = DualOps<float>.Sin(_engine, x);
        Assert.Equal(0f, result.Primal[0], precision: 4);
        Assert.Equal(1f, result.Tangent[0], precision: 4);
    }

    [Fact]
    public void DualOps_Exp_TangentEqualsPrimal()
    {
        // d(exp(x))/dx = exp(x). At x=1: exp(1) ≈ 2.71828.
        var x = new Dual<float>(
            new Tensor<float>(new[] { 1f }, new[] { 1 }),
            new Tensor<float>(new[] { 1f }, new[] { 1 }));
        var result = DualOps<float>.Exp(_engine, x);
        Assert.Equal((float)Math.E, result.Primal[0], precision: 3);
        Assert.Equal((float)Math.E, result.Tangent[0], precision: 3);
    }

    // ─── Cross-validation with reverse-mode ───────────────────────────

    [Fact]
    public void Jvp_vs_Grad_Agreement_OnChainedPolynomial()
    {
        // f(x) = (x * x) * (x + 1)  →  df/dx = 2x(x+1) + x^2 = 3x^2 + 2x.
        // At x=2: f = 4*3 = 12, df/dx = 12 + 4 = 16.
        Func<Dual<float>, Dual<float>> dualFn = x =>
        {
            var xSq = DualOps<float>.Square(_engine, x);
            var one = Dual<float>.Constant(new Tensor<float>(new[] { 1f }, new[] { 1 }));
            var xPlusOne = DualOps<float>.Add(_engine, x, one);
            return DualOps<float>.Multiply(_engine, xSq, xPlusOne);
        };

        var xPrimal = new Tensor<float>(new[] { 2f }, new[] { 1 });
        var xTangent = new Tensor<float>(new[] { 1f }, new[] { 1 });

        var (primal, tangent) = TensorFunc<float>.Jvp(_engine, dualFn, xPrimal, xTangent);
        Assert.Equal(12f, primal[0], precision: 3);
        Assert.Equal(16f, tangent[0], precision: 3);

        // Reverse-mode reference — same function in plain Tensor<T> land.
        var grad = TensorFunc<float>.Grad(x =>
        {
            var xSq = _engine.TensorMultiply(x, x);
            var oneT = new Tensor<float>(new[] { 1f }, new[] { 1 });
            var xPlusOne = _engine.TensorAdd(x, oneT);
            return _engine.TensorMultiply(xSq, xPlusOne);
        })(xPrimal);

        // Reverse-mode gradient at x=2 == forward-mode tangent (both = 16).
        Assert.Equal(tangent[0], grad[0], precision: 3);
    }

    [Fact]
    public void Jvp_vs_Grad_Agreement_OnExpSinComposition()
    {
        // f(x) = exp(sin(x))  →  df/dx = cos(x) * exp(sin(x)).
        // At x=0.5: sin(0.5)≈0.479, exp(0.479)≈1.615, cos(0.5)≈0.877.
        // df/dx ≈ 0.877 * 1.615 ≈ 1.417.
        Func<Dual<float>, Dual<float>> dualFn = x =>
            DualOps<float>.Exp(_engine, DualOps<float>.Sin(_engine, x));

        var xPrimal = new Tensor<float>(new[] { 0.5f }, new[] { 1 });
        var xTangent = new Tensor<float>(new[] { 1f }, new[] { 1 });
        var (_, tangent) = TensorFunc<float>.Jvp(_engine, dualFn, xPrimal, xTangent);

        var grad = TensorFunc<float>.Grad(x =>
            _engine.TensorExp(_engine.TensorSin(x)))(xPrimal);

        Assert.Equal(grad[0], tangent[0], precision: 3);
    }

    [Fact]
    public void Jvp_MatMul_Respects_ProductRule()
    {
        // d(A·B) = dA·B + A·dB. Verify the product rule on 2×2 matrices.
        var A = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var B = new Tensor<float>(new[] { 5f, 6f, 7f, 8f }, new[] { 2, 2 });
        var dA = new Tensor<float>(new[] { 0f, 0f, 0f, 0f }, new[] { 2, 2 });
        var dB = new Tensor<float>(new[] { 1f, 0f, 0f, 0f }, new[] { 2, 2 });

        Func<Dual<float>[], Dual<float>> fn = duals =>
            DualOps<float>.MatMul(_engine, duals[0], duals[1]);

        var (primal, tangent) = TensorFunc<float>.Jvp(_engine, fn,
            new[] { A, B }, new[] { dA, dB });

        // Primal A·B at entries: (1,1)=19, (1,2)=22, (2,1)=43, (2,2)=50.
        Assert.Equal(19f, primal[0, 0], precision: 3);
        Assert.Equal(50f, primal[1, 1], precision: 3);

        // Tangent = 0·B + A·dB = A·dB.
        // A·dB where dB has only (0,0)=1: result has column 0 = column 0 of A.
        //   [1 2] · [1 0]   [1  0]
        //   [3 4]   [0 0] = [3  0]
        Assert.Equal(1f, tangent[0, 0], precision: 3);
        Assert.Equal(0f, tangent[0, 1], precision: 3);
        Assert.Equal(3f, tangent[1, 0], precision: 3);
        Assert.Equal(0f, tangent[1, 1], precision: 3);
    }

    [Fact]
    public void Jvp_ArgumentValidation()
    {
        Assert.Throws<ArgumentNullException>(
            () => TensorFunc<float>.Jvp(null, (Dual<float>[] _) => default, new Tensor<float>[0], new Tensor<float>[0]));
        Assert.Throws<ArgumentNullException>(
            () => TensorFunc<float>.Jvp(_engine, (Func<Dual<float>[], Dual<float>>)null, new Tensor<float>[0], new Tensor<float>[0]));

        var x = new Tensor<float>(new[] { 1f }, new[] { 1 });
        Assert.Throws<ArgumentException>(
            () => TensorFunc<float>.Jvp(_engine, duals => duals[0], new[] { x }, new Tensor<float>[0]));
    }
}
