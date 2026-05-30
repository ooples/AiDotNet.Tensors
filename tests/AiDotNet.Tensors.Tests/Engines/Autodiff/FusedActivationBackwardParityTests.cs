using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// #499: every <see cref="FusedActivationType"/> must have a registered backward
/// handler (so the fused-linear backward pass never throws "No handler registered"),
/// and each handler's analytic VJP must match a finite-difference vector-Jacobian
/// product of its own forward. This is the rigorous gradient check — it validates the
/// derivative against the forward directly, for both pointwise and row-wise/Jacobian
/// (softmax-family) activations.
/// </summary>
public class FusedActivationBackwardParityTests
{
    private const int Rows = 3, M = 5;

    // Knots where a piecewise activation is non-differentiable; finite differences are
    // invalid within ε of these, so the safe-input generator nudges away from them.
    private static readonly double[] Knots = { 0.0, 1.0, -1.0, 3.0, -3.0, 6.0 };

    public static IEnumerable<object[]> AllNonNoneActivations()
    {
        foreach (FusedActivationType a in Enum.GetValues(typeof(FusedActivationType)))
            if (a != FusedActivationType.None)
                yield return new object[] { a };
    }

    // Differentiable + deterministic: excludes the hard nonlinearities (Sign,
    // BinarySpiking — 0 derivative a.e.) and the stochastic GumbelSoftmax.
    public static IEnumerable<object[]> FiniteDiffActivations()
    {
        foreach (FusedActivationType a in Enum.GetValues(typeof(FusedActivationType)))
        {
            if (a == FusedActivationType.None || a == FusedActivationType.Sign
                || a == FusedActivationType.BinarySpiking || a == FusedActivationType.GumbelSoftmax)
                continue;
            yield return new object[] { a };
        }
    }

    [Theory]
    [MemberData(nameof(AllNonNoneActivations))]
    public void EveryActivation_HasRegisteredBackwardHandler(FusedActivationType act)
    {
        // The core #499 guarantee: ActivationRegistry.Get no longer throws for any of them.
        var handler = ActivationRegistry.Get(act);
        Assert.NotNull(handler);
    }

    [Theory]
    [MemberData(nameof(FiniteDiffActivations))]
    public void FusedActivationBackward_MatchesFiniteDifferenceVJP(FusedActivationType act)
    {
        var engine = new CpuEngine();
        var handler = ActivationRegistry.Get(act)!;
        var shape = new[] { Rows, M };
        int n = Rows * M;

        var x = MakeSafeInput(n, seed: 1234 + (int)act);
        var g = MakeGrad(n, seed: 5678 + (int)act);

        // Analytic VJP: gradInput = Jᵀ·g.
        var xTensor = new Tensor<double>((double[])x.Clone(), shape);
        var gTensor = new Tensor<double>((double[])g.Clone(), shape);
        var analytic = handler.ApplyBackward(engine, gTensor, xTensor).AsSpan().ToArray();

        // Numerical VJP via central differences of the handler's own forward:
        //   (Jᵀg)_i ≈ Σ_k g_k · (f(x+εeᵢ)_k − f(x−εeᵢ)_k) / (2ε)
        const double eps = 1e-5;
        for (int i = 0; i < n; i++)
        {
            var xp = (double[])x.Clone(); xp[i] += eps;
            var xm = (double[])x.Clone(); xm[i] -= eps;
            var fp = handler.Apply(engine, new Tensor<double>(xp, shape)).AsSpan();
            var fm = handler.Apply(engine, new Tensor<double>(xm, shape)).AsSpan();

            double numerical = 0.0;
            for (int k = 0; k < n; k++)
                numerical += g[k] * (fp[k] - fm[k]) / (2.0 * eps);

            double tol = 2e-3 + 5e-3 * Math.Abs(numerical);
            Assert.True(Math.Abs(analytic[i] - numerical) < tol,
                $"{act}: dInput[{i}] analytic {analytic[i]:E4} vs finite-diff {numerical:E4} (tol {tol:E4}).");
        }
    }

    [Theory]
    [InlineData(FusedActivationType.Sign)]
    [InlineData(FusedActivationType.BinarySpiking)]
    public void HardNonlinearity_Backward_IsZeroAlmostEverywhere(FusedActivationType act)
    {
        // Sign and BinarySpiking are piecewise-constant; their derivative is 0 a.e.
        // The handler returns exactly 0 (a surrogate-gradient layer is used for SNN training).
        var engine = new CpuEngine();
        var handler = ActivationRegistry.Get(act)!;
        var shape = new[] { Rows, M };
        int n = Rows * M;
        var x = new Tensor<double>(MakeSafeInput(n, seed: 99 + (int)act), shape);
        var g = new Tensor<double>(MakeGrad(n, seed: 7 + (int)act), shape);

        var grad = handler.ApplyBackward(engine, g, x).AsSpan();
        for (int i = 0; i < grad.Length; i++)
            Assert.True(Math.Abs(grad[i]) < 1e-12, $"{act}: derivative is 0 a.e.; got {grad[i]} at {i}.");
    }

    [Fact]
    public void GumbelSoftmax_Backward_RunsAndIsFinite()
    {
        // Stochastic forward → no finite-difference check; verify the deterministic
        // relaxation backward produces a finite, correctly-shaped gradient (no throw).
        var engine = new CpuEngine();
        var handler = ActivationRegistry.Get(FusedActivationType.GumbelSoftmax)!;
        var shape = new[] { Rows, M };
        int n = Rows * M;
        var x = new Tensor<double>(MakeSafeInput(n, seed: 321), shape);
        var g = new Tensor<double>(MakeGrad(n, seed: 654), shape);

        var grad = handler.ApplyBackward(engine, g, x).AsSpan();
        Assert.Equal(n, grad.Length);
        for (int i = 0; i < grad.Length; i++)
            Assert.True(!double.IsNaN(grad[i]) && !double.IsInfinity(grad[i]), $"non-finite gradient at {i}.");
    }

    [Fact]
    public void ApplyFusedActivationBackward_NoLongerThrows_ForNewlyWiredActivations()
    {
        // Regression for the original gap: these used to throw ArgumentException
        // ("No handler registered") through the fused-linear backward entry point.
        var engine = new CpuEngine();
        var shape = new[] { Rows, M };
        var preAct = new Tensor<float>(new float[Rows * M], shape);
        var grad = new Tensor<float>(new float[Rows * M], shape);
        for (int i = 0; i < Rows * M; i++) { preAct[i] = 0.3f * (i - 7); grad[i] = 1f; }

        foreach (var act in new[]
        {
            FusedActivationType.Mish, FusedActivationType.CELU, FusedActivationType.SELU,
            FusedActivationType.ISRU, FusedActivationType.Softmin, FusedActivationType.Sparsemax,
            FusedActivationType.PReLU, FusedActivationType.LogSoftmax,
        })
        {
            var ex = Record.Exception(() => engine.ApplyFusedActivationBackward(grad, preAct, act));
            Assert.Null(ex);
        }
    }

    /// <summary>
    /// #506 review: the fused-activation backward must honor a caller-supplied
    /// <see cref="FusedActivationParams"/> (slope/alpha), not silently use the default.
    /// For each parametric activation: the params-aware backward matches a finite-difference
    /// VJP of the params-aware forward, AND differs from the default-params backward (proving
    /// the parameter actually flows through — the bug CodeRabbit flagged).
    /// </summary>
    [Theory]
    [InlineData(FusedActivationType.PReLU)]   // PReluSlope[0]
    [InlineData(FusedActivationType.ELU)]     // Alpha
    [InlineData(FusedActivationType.CELU)]    // Alpha
    [InlineData(FusedActivationType.LeakyReLU)] // Alpha
    public void ParametricActivation_Backward_HonorsNonDefaultParams(FusedActivationType act)
    {
        var engine = new CpuEngine();
        var handler = ActivationRegistry.Get(act)!;
        int n = 12;
        var x = MakeSafeInput(n, 4242 + (int)act);
        var g = MakeGrad(n, 2424 + (int)act);
        // Non-default params: slope/alpha = 0.3 (defaults are 0.25 / 1.0 / 1.0 / 0.01).
        var p = act == FusedActivationType.PReLU
            ? new FusedActivationParams { PReluSlope = new[] { 0.3f } }
            : new FusedActivationParams { Alpha = 0.3f };

        var xT = new Tensor<double>((double[])x.Clone(), new[] { n });
        var gT = new Tensor<double>((double[])g.Clone(), new[] { n });
        var analytic = handler.ApplyBackward(engine, gT, xT, p).AsSpan().ToArray();

        // Finite-difference VJP of the params-aware forward (pointwise: per-element).
        const double eps = 1e-6;
        for (int i = 0; i < n; i++)
        {
            var xp = (double[])x.Clone(); xp[i] += eps;
            var xm = (double[])x.Clone(); xm[i] -= eps;
            var fp = handler.Apply(engine, new Tensor<double>(xp, new[] { n }), p).AsSpan();
            var fm = handler.Apply(engine, new Tensor<double>(xm, new[] { n }), p).AsSpan();
            double numerical = g[i] * (fp[i] - fm[i]) / (2.0 * eps);
            Assert.True(Math.Abs(analytic[i] - numerical) < 2e-3,
                $"{act} params-aware backward[{i}] {analytic[i]:E4} != finite-diff {numerical:E4}");
        }

        // And it must DIFFER from the default-params backward on the negative region
        // (where the slope/alpha applies), proving the parameter took effect.
        var defaultGrad = handler.ApplyBackward(engine, gT, xT).AsSpan().ToArray();
        double maxDiff = 0;
        for (int i = 0; i < n; i++) maxDiff = Math.Max(maxDiff, Math.Abs(analytic[i] - defaultGrad[i]));
        Assert.True(maxDiff > 1e-6,
            $"{act}: params-aware backward identical to default — FusedActivationParams not honored.");
    }

    private static double[] MakeSafeInput(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new double[n];
        for (int i = 0; i < n; i++)
        {
            double v;
            do { v = rng.NextDouble() * 3.6 - 1.8; } // (-1.8, 1.8)
            while (TooCloseToKnot(v));
            a[i] = v;
        }
        return a;
    }

    private static bool TooCloseToKnot(double v)
    {
        foreach (var k in Knots)
            if (Math.Abs(v - k) < 0.05) return true;
        return false;
    }

    private static double[] MakeGrad(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.NextDouble() * 2.0 - 1.0;
        return a;
    }
}
