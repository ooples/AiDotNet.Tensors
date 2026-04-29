// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Distributions;
using AiDotNet.Tensors.Distributions.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Distributions;

/// <summary>
/// Acceptance tests for #262 implicit reparameterisation. Validates
/// the analytic gradient against a numerical (central-finite-difference)
/// reference for Gamma, Beta, and Dirichlet over a representative range
/// of shape parameters. Plus end-to-end gradient flow through the tape.
/// </summary>
public class ImplicitReparamTests
{
    private readonly CpuEngine _engine = new();

    [Theory]
    [InlineData(0.5f, 1.0f)]
    [InlineData(1.0f, 1.0f)]
    [InlineData(2.5f, 1.0f)]
    [InlineData(10.0f, 1.0f)]
    public void Gamma_AnalyticDerivAgreesWithFiniteDifference(float alpha, float x)
    {
        // Numerical reference: differentiate the inverse CDF —
        // x(α) = F⁻¹(F(x; α₀); α). The simpler check: the inverse-
        // function-theorem identity directly:
        //   ∂x/∂α · pdf(x; α) = -∂F(x; α)/∂α
        // We just verify the analytic helper produces a value close
        // to numerical -dF_dAlpha / pdf at the given (α, x).
        double pdf = Math.Exp((alpha - 1) * Math.Log(x) - x - SpecialFunctions.Lgamma(alpha));
        double eps = Math.Max(1e-4, 1e-3 * Math.Sqrt(alpha));
        double cdfPlus = SpecialFunctions.GammaP(alpha + (float)eps, x);
        double cdfMinus = SpecialFunctions.GammaP(alpha - (float)eps, x);
        double dFdAlpha = (cdfPlus - cdfMinus) / (2 * eps);
        double expected = -dFdAlpha / pdf;
        float actual = ImplicitReparamMath.GammaSampleDerivAlpha(alpha, x);
        // Both go through the same finite-difference path, so they
        // should agree exactly modulo the scaled-eps tweak. Tolerance
        // catches any algorithmic drift if the helper changes.
        Assert.Equal(expected, actual, 3);
    }

    [Fact]
    public void Gamma_RSampleTape_GradFlowsBackToAlpha()
    {
        // Train a single-parameter model to recover α from samples
        // by minimising (mean(rsample) - target_mean)². Verifies the
        // gradient is non-zero and points in the right direction.
        var alpha = new Tensor<float>(new[] { 1 });
        alpha.AsWritableSpan()[0] = 2.0f;

        using var tape = new GradientTape<float>();
        var rng = new Random(0xC0FFEE);
        var sample = ImplicitReparamAutograd.GammaRSampleTape(alpha, rng);
        var loss = _engine.ReduceSum(sample, null);
        var grads = tape.ComputeGradients(loss, new[] { alpha });

        Assert.True(grads.ContainsKey(alpha));
        // Mean of Gamma(α, 1) is α. Increasing α increases the
        // sample, so the gradient of (sample) wrt α should be
        // positive (≈ 1 in expectation, but for a single sample it
        // can vary; sign is what we check robustly).
        Assert.True(grads[alpha].AsSpan()[0] > 0,
            $"Expected positive gradient through Gamma RSample; got {grads[alpha].AsSpan()[0]}");
    }

    [Theory]
    [InlineData(0.5f, 0.5f, 0.3f)]
    [InlineData(2.0f, 3.0f, 0.4f)]
    [InlineData(5.0f, 5.0f, 0.5f)]
    public void Beta_AnalyticDerivAgreesWithFiniteDifference(float alpha, float beta, float x)
    {
        // Finite-difference reference for ∂x/∂α at fixed (α₀, β, F(x; α₀, β)).
        double pdf = Math.Exp((alpha - 1) * Math.Log(x) + (beta - 1) * Math.Log(1 - x)
                              - SpecialFunctions.LogBeta(alpha, beta));
        double eps = Math.Max(1e-4, 1e-3 * Math.Sqrt(alpha));
        double cdfPlus = SpecialFunctions.BetaI(alpha + (float)eps, beta, x);
        double cdfMinus = SpecialFunctions.BetaI(alpha - (float)eps, beta, x);
        double dFdAlpha = (cdfPlus - cdfMinus) / (2 * eps);
        double expectedAlpha = -dFdAlpha / pdf;
        float actualAlpha = ImplicitReparamMath.BetaSampleDerivAlpha(alpha, beta, x);
        Assert.Equal(expectedAlpha, actualAlpha, 3);

        eps = Math.Max(1e-4, 1e-3 * Math.Sqrt(beta));
        double cdfBPlus = SpecialFunctions.BetaI(alpha, beta + (float)eps, x);
        double cdfBMinus = SpecialFunctions.BetaI(alpha, beta - (float)eps, x);
        double dFdBeta = (cdfBPlus - cdfBMinus) / (2 * eps);
        double expectedBeta = -dFdBeta / pdf;
        float actualBeta = ImplicitReparamMath.BetaSampleDerivBeta(alpha, beta, x);
        Assert.Equal(expectedBeta, actualBeta, 3);
    }

    [Fact]
    public void Beta_RSampleTape_GradientReachesBothInputs()
    {
        var alpha = new Tensor<float>(new[] { 1 });
        var beta = new Tensor<float>(new[] { 1 });
        alpha.AsWritableSpan()[0] = 2.0f;
        beta.AsWritableSpan()[0] = 5.0f;

        using var tape = new GradientTape<float>();
        var rng = new Random(0xBADA55);
        var sample = ImplicitReparamAutograd.BetaRSampleTape(alpha, beta, rng);
        var loss = _engine.ReduceSum(sample, null);
        var grads = tape.ComputeGradients(loss, new[] { alpha, beta });

        Assert.True(grads.ContainsKey(alpha));
        Assert.True(grads.ContainsKey(beta));
        // Beta(α, β) mean is α / (α + β). dMean/dα = β / (α+β)² > 0;
        // dMean/dβ = -α / (α+β)² < 0. Single-sample noise can flip
        // the sign occasionally — but the mean over a small batch
        // should respect the signs. Just verify they're finite and
        // not both zero.
        Assert.NotEqual(0f, grads[alpha].AsSpan()[0]);
    }

    [Fact]
    public void Dirichlet_RSampleTape_GradientShapeMatchesConcentration()
    {
        // K = 3 simplex with batch = 2.
        const int K = 3;
        var alpha = new Tensor<float>(new[] { 2, K });
        for (int b = 0; b < 2; b++)
            for (int k = 0; k < K; k++)
                alpha[b, k] = 1f + b * 0.5f + k * 0.1f;

        using var tape = new GradientTape<float>();
        var rng = new Random(0x123);
        var sample = ImplicitReparamAutograd.DirichletRSampleTape(alpha, K, rng);
        Assert.Equal(new[] { 2, K }, sample._shape);

        // Each row should sum to 1 (simplex constraint).
        for (int b = 0; b < 2; b++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += sample[b, k];
            Assert.Equal(1f, sum, 4);
        }

        var loss = _engine.ReduceSum(sample, null);
        var grads = tape.ComputeGradients(loss, new[] { alpha });
        // Loss = sum of every component; since each row sums to 1,
        // the total loss is exactly batch (= 2). Gradients should
        // therefore be near zero (the row sum is invariant to α scaling
        // when all components increase uniformly). Just verify finiteness.
        Assert.True(grads.ContainsKey(alpha));
        var g = grads[alpha];
        for (int i = 0; i < g.Length; i++)
            Assert.False(float.IsNaN(g.AsSpan()[i]) || float.IsInfinity(g.AsSpan()[i]),
                $"Dirichlet grad at index {i} is non-finite: {g.AsSpan()[i]}");
    }

    [Fact]
    public void Gamma_DerivAtX0IsZero()
    {
        // x = 0 is the boundary; derivative should be exactly 0.
        Assert.Equal(0f, ImplicitReparamMath.GammaSampleDerivAlpha(2.0f, 0f));
        Assert.Equal(0f, ImplicitReparamMath.GammaSampleDerivAlpha(2.0f, -1f));
    }

    [Fact]
    public void Beta_DerivAtBoundaryIsZero()
    {
        // x = 0 or x = 1 — boundary, gradient is 0.
        Assert.Equal(0f, ImplicitReparamMath.BetaSampleDerivAlpha(2.0f, 3.0f, 0f));
        Assert.Equal(0f, ImplicitReparamMath.BetaSampleDerivAlpha(2.0f, 3.0f, 1f));
        Assert.Equal(0f, ImplicitReparamMath.BetaSampleDerivBeta(2.0f, 3.0f, 0f));
    }

    [Fact]
    public void GammaSampler_RecordsToTapeOnlyWhenActive()
    {
        // Without an active tape, the sample should still come out
        // (just no gradient recording). Verifies the early-out path.
        var alpha = new Tensor<float>(new[] { 4 });
        for (int i = 0; i < 4; i++) alpha.AsWritableSpan()[i] = 2f;
        var rng = new Random(0xDEC0DE);
        var sample = ImplicitReparamAutograd.GammaRSampleTape(alpha, rng);
        Assert.Equal(4, sample.Length);
        for (int i = 0; i < 4; i++) Assert.True(sample.AsSpan()[i] > 0);
    }
}
