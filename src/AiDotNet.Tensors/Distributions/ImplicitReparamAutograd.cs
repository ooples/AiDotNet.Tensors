// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Distributions.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Tape-aware reparameterised samplers for Gamma, Beta, and Dirichlet —
/// the implicit-reparameterisation gradients (Figurnov, Mohamed &amp;
/// Mnih 2018) that PyTorch ships only for Normal / Student-T / Gumbel.
///
/// <para>Each <c>RSampleTape</c> entry takes the distribution's shape
/// parameters as <see cref="Tensor{T}"/>, draws a sample from the
/// underlying rejection sampler, and records a backward closure that
/// computes <c>∂X / ∂α</c> via the inverse function theorem applied
/// to the CDF (<see cref="ImplicitReparamMath"/>). Downstream tape ops
/// can then differentiate through the sample and the gradient flows
/// back to <c>α</c> as if the sampler were a smooth function.</para>
///
/// <para>Numerical contract: gradients agree with a central finite
/// difference on the sample function within ~1e-3 over α ∈ [0.1, 100],
/// validated by the acceptance tests on this PR.</para>
/// </summary>
public static class ImplicitReparamAutograd
{
    /// <summary>
    /// Tape-aware Gamma sample with rate = 1. Parameter
    /// <paramref name="alpha"/> is the concentration tensor; the output
    /// has the same shape and is autograd-recorded.
    /// </summary>
    public static Tensor<float> GammaRSampleTape(Tensor<float> alpha, Random rng)
    {
        if (alpha is null) throw new ArgumentNullException(nameof(alpha));
        if (rng is null) throw new ArgumentNullException(nameof(rng));

        var output = new Tensor<float>((int[])alpha._shape.Clone());
        var aSpan = alpha.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < aSpan.Length; i++)
            dst[i] = GammaFamilyDistributions_MarsagliaTsangProxy(rng, aSpan[i]);

        if (DifferentiableOps._anyTapeActive == 0) return output;

        // Snapshot α so backward sees the values that produced these
        // samples even if α gets mutated downstream.
        var alphaSnapshot = new Tensor<float>((int[])alpha._shape.Clone());
        aSpan.CopyTo(alphaSnapshot.AsWritableSpan());
        var savedState = new object[] { alphaSnapshot };
        DifferentiableOps.RecordUnary(
            "GammaRSample",
            output,
            alpha,
            GammaRSampleBackward,
            savedState);
        return output;
    }

    /// <summary>
    /// Tape-aware Beta sample. Both <paramref name="alpha"/> and
    /// <paramref name="beta"/> get autograd-recorded gradient flow.
    /// </summary>
    public static Tensor<float> BetaRSampleTape(Tensor<float> alpha, Tensor<float> beta, Random rng)
    {
        if (alpha is null) throw new ArgumentNullException(nameof(alpha));
        if (beta is null) throw new ArgumentNullException(nameof(beta));
        if (alpha.Length != beta.Length) throw new ArgumentException("alpha and beta length mismatch.");

        var output = new Tensor<float>((int[])alpha._shape.Clone());
        var aSpan = alpha.AsSpan();
        var bSpan = beta.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < aSpan.Length; i++)
        {
            float ga = GammaFamilyDistributions_MarsagliaTsangProxy(rng, aSpan[i]);
            float gb = GammaFamilyDistributions_MarsagliaTsangProxy(rng, bSpan[i]);
            dst[i] = ga / (ga + gb);
        }

        if (DifferentiableOps._anyTapeActive == 0) return output;

        var alphaSnapshot = new Tensor<float>((int[])alpha._shape.Clone());
        aSpan.CopyTo(alphaSnapshot.AsWritableSpan());
        var betaSnapshot = new Tensor<float>((int[])beta._shape.Clone());
        bSpan.CopyTo(betaSnapshot.AsWritableSpan());
        var savedState = new object[] { alphaSnapshot, betaSnapshot };
        DifferentiableOps.RecordBinary(
            "BetaRSample",
            output,
            alpha,
            beta,
            BetaRSampleBackward,
            savedState);
        return output;
    }

    /// <summary>
    /// Tape-aware Dirichlet sample. <paramref name="concentration"/>
    /// has shape <c>[..., K]</c>; output has the same shape with each
    /// length-K row a sample on the simplex.
    /// </summary>
    public static Tensor<float> DirichletRSampleTape(Tensor<float> concentration, int k, Random rng)
    {
        if (concentration is null) throw new ArgumentNullException(nameof(concentration));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (concentration.Length % k != 0)
            throw new ArgumentException("concentration length must be divisible by K.");

        int batch = concentration.Length / k;
        var output = new Tensor<float>((int[])concentration._shape.Clone());
        var aSpan = concentration.AsSpan();
        var dst = output.AsWritableSpan();

        // Capture the underlying Gamma samples so backward can chain-rule
        // through the normalisation.
        var gammaSamples = new float[batch * k];
        var gammaSums = new float[batch];

        for (int b = 0; b < batch; b++)
        {
            float sum = 0f;
            for (int i = 0; i < k; i++)
            {
                float gi = GammaFamilyDistributions_MarsagliaTsangProxy(rng, aSpan[b * k + i]);
                gammaSamples[b * k + i] = gi;
                sum += gi;
            }
            gammaSums[b] = sum;
            for (int i = 0; i < k; i++) dst[b * k + i] = gammaSamples[b * k + i] / sum;
        }

        if (DifferentiableOps._anyTapeActive == 0) return output;

        var alphaSnapshot = new Tensor<float>((int[])concentration._shape.Clone());
        aSpan.CopyTo(alphaSnapshot.AsWritableSpan());
        var savedState = new object[] { alphaSnapshot, gammaSamples, gammaSums, k };
        DifferentiableOps.RecordUnary(
            "DirichletRSample",
            output,
            concentration,
            DirichletRSampleBackward,
            savedState);
        return output;
    }

    private static void GammaRSampleBackward(
        Tensor<float> gradOutput,
        Tensor<float>[] inputs,
        Tensor<float> output,
        object[] savedState,
        Engines.IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> gradAccumulator)
    {
        var alpha = inputs[0];
        var alphaSnapshot = (Tensor<float>)savedState[0];
        var gradAlpha = new Tensor<float>((int[])alpha._shape.Clone());
        var aSpan = alphaSnapshot.AsSpan();
        var xSpan = output.AsSpan();
        var goSpan = gradOutput.AsSpan();
        var dst = gradAlpha.AsWritableSpan();
        for (int i = 0; i < aSpan.Length; i++)
        {
            // d output_i / d alpha_i — only diagonal contribution
            // since each sample is independent.
            float dXdA = ImplicitReparamMath.GammaSampleDerivAlpha(aSpan[i], xSpan[i]);
            dst[i] = goSpan[i] * dXdA;
        }
        AccumulateGrad(alpha, gradAlpha, gradAccumulator, engine);
    }

    private static void BetaRSampleBackward(
        Tensor<float> gradOutput,
        Tensor<float>[] inputs,
        Tensor<float> output,
        object[] savedState,
        Engines.IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> gradAccumulator)
    {
        var alpha = inputs[0];
        var beta = inputs[1];
        var alphaSnapshot = (Tensor<float>)savedState[0];
        var betaSnapshot = (Tensor<float>)savedState[1];
        var gradAlpha = new Tensor<float>((int[])alpha._shape.Clone());
        var gradBeta = new Tensor<float>((int[])beta._shape.Clone());
        var aSpan = alphaSnapshot.AsSpan();
        var bSpan = betaSnapshot.AsSpan();
        var xSpan = output.AsSpan();
        var goSpan = gradOutput.AsSpan();
        var dstA = gradAlpha.AsWritableSpan();
        var dstB = gradBeta.AsWritableSpan();
        for (int i = 0; i < aSpan.Length; i++)
        {
            float dXdA = ImplicitReparamMath.BetaSampleDerivAlpha(aSpan[i], bSpan[i], xSpan[i]);
            float dXdB = ImplicitReparamMath.BetaSampleDerivBeta(aSpan[i], bSpan[i], xSpan[i]);
            dstA[i] = goSpan[i] * dXdA;
            dstB[i] = goSpan[i] * dXdB;
        }
        AccumulateGrad(alpha, gradAlpha, gradAccumulator, engine);
        AccumulateGrad(beta, gradBeta, gradAccumulator, engine);
    }

    private static void DirichletRSampleBackward(
        Tensor<float> gradOutput,
        Tensor<float>[] inputs,
        Tensor<float> output,
        object[] savedState,
        Engines.IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> gradAccumulator)
    {
        var alpha = inputs[0];
        var alphaSnapshot = (Tensor<float>)savedState[0];
        var gammaSamples = (float[])savedState[1];
        var gammaSums = (float[])savedState[2];
        int k = (int)savedState[3];

        var gradAlpha = new Tensor<float>((int[])alpha._shape.Clone());
        var aSpan = alphaSnapshot.AsSpan();
        var goSpan = gradOutput.AsSpan();
        var dst = gradAlpha.AsWritableSpan();
        int batch = alpha.Length / k;
        for (int b = 0; b < batch; b++)
        {
            for (int kIdx = 0; kIdx < k; kIdx++)
            {
                // Accumulate dL/dα_k = Σ_i goSpan[b*K+i] · ∂X_i/∂α_k
                float gradAk = 0f;
                for (int i = 0; i < k; i++)
                {
                    float dXi_dAk = ImplicitReparamMath.DirichletSampleDerivAlpha(
                        aSpan.Slice(b * k, k), gammaSamples.AsSpan(b * k, k), gammaSums[b], i, kIdx);
                    gradAk += goSpan[b * k + i] * dXi_dAk;
                }
                dst[b * k + kIdx] = gradAk;
            }
        }
        AccumulateGrad(alpha, gradAlpha, gradAccumulator, engine);
    }

    private static void AccumulateGrad(
        Tensor<float> target, Tensor<float> grad,
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> gradAccumulator,
        Engines.IEngine engine)
    {
        if (gradAccumulator.TryGetValue(target, out var existing))
            gradAccumulator[target] = engine.TensorAdd(existing, grad);
        else
            gradAccumulator[target] = grad;
    }

    // The Marsaglia-Tsang sampler lives as `internal` on
    // GammaDistribution. We reach for it through this small proxy so the
    // public surface here doesn't need to expose it.
    private static float GammaFamilyDistributions_MarsagliaTsangProxy(Random rng, float alpha)
        => GammaDistribution.MarsagliaTsang(rng, alpha);
}
