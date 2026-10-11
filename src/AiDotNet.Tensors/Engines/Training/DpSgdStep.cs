using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Training;

/// <summary>
/// Fused per-example DP-SGD step (Abadi et al. 2016 §3, Algorithm 1). Runs a batch
/// of examples through per-example forward+backward, clips each per-example gradient
/// against the GLOBAL parameter-vector L2 norm, aggregates the clipped gradients,
/// adds a single Gaussian noise draw <c>N(0, σ² C² I)</c>, and averages by batch size.
///
/// <para>The clip-BEFORE-aggregate order is the L2-sensitivity bound the DP proof
/// requires — reversing it (aggregate-then-clip) breaks the privacy guarantee. This
/// helper enforces the correct order so callers cannot regress it.</para>
///
/// <para>Enables the DP-SGD paths in AiDotNet's WGAN-GP variants (DPCTGAN critic,
/// MedSynth DP disc) to route through a shared, tested primitive instead of each
/// re-implementing per-example clip+aggregate.</para>
/// </summary>
/// <typeparam name="T">Numeric type of the parameter tensors.</typeparam>
public static class DpSgdStep<T>
{
    /// <summary>
    /// Runs a DP-SGD training step over <paramref name="batchSize"/> examples.
    /// Each example gets its own <see cref="GradientTape{T}"/> (constructed inside
    /// the helper); <paramref name="perExampleLoss"/> is invoked with the example
    /// index and MUST record its forward on the currently-active tape and return
    /// the scalar loss tensor for that example.
    /// </summary>
    /// <param name="batchSize">Number of examples to process.</param>
    /// <param name="perExampleLoss">Callback: given example index i, records the
    /// per-example forward on the currently-active tape and returns the scalar loss.</param>
    /// <param name="parameters">Trainable parameter tensors. The returned dictionary
    /// keys these by reference.</param>
    /// <param name="clipNorm">Per-example L2 norm clip threshold (C in Abadi 2016).</param>
    /// <param name="noiseMultiplier">Gaussian noise multiplier (σ in Abadi 2016). Set
    /// to 0 to disable noise (useful for validating the clip-and-aggregate contract
    /// without perturbing the gradient).</param>
    /// <param name="rng">RNG for the Gaussian noise draws. Callers pass a seeded
    /// instance for reproducibility.</param>
    /// <returns>Dictionary of averaged, noised, clipped gradients keyed by parameter
    /// tensor reference. Feed directly into an optimizer.Step / TapeStepContext.</returns>
    /// <exception cref="ArgumentOutOfRangeException">batchSize &lt;= 0 or clipNorm &lt;= 0.</exception>
    /// <exception cref="ArgumentNullException">Any callback or list argument is null.</exception>
    public static Dictionary<Tensor<T>, Tensor<T>> ComputeClippedAggregatedGradients(
        int batchSize,
        Func<int, Tensor<T>> perExampleLoss,
        IReadOnlyList<Tensor<T>> parameters,
        double clipNorm,
        double noiseMultiplier,
        Random rng)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize,
                "DP-SGD batch size must be positive.");
        if (clipNorm <= 0)
            throw new ArgumentOutOfRangeException(nameof(clipNorm), clipNorm,
                "DP-SGD clip norm must be positive (defines the L2-sensitivity bound).");
        if (perExampleLoss is null)
            throw new ArgumentNullException(nameof(perExampleLoss));
        if (parameters is null)
            throw new ArgumentNullException(nameof(parameters));
        if (rng is null)
            throw new ArgumentNullException(nameof(rng));

        var ops = MathHelper.GetNumericOperations<T>();

        // Zero-initialized per-parameter accumulators for the clipped sum.
        var sums = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        foreach (var p in parameters)
        {
            if (p is null)
                throw new ArgumentException("parameters must not contain null tensors.", nameof(parameters));
            sums[p] = new Tensor<T>(p._shape);
        }

        for (int example = 0; example < batchSize; example++)
        {
            // Per-example tape. Non-persistent because we consume the gradient
            // once and discard the tape — DP-SGD's per-example clip step is a
            // pure aggregation, not a repeated-replay pattern.
            using var tape = new GradientTape<T>();
            var loss = perExampleLoss(example);
            if (loss is null)
                throw new InvalidOperationException(
                    $"perExampleLoss({example}) returned null; must return the scalar loss tensor.");
            var grads = tape.ComputeGradients(loss, parameters);

            // GLOBAL L2 norm across ALL parameters (Abadi 2016 Algorithm 1, line 4):
            // "compute ∥g_t(x_i)∥_2 for each i" where g_t(x_i) is the FLATTENED
            // concatenation of every parameter's gradient. Clipping per-parameter
            // instead would break the L2-sensitivity bound and invalidate the
            // privacy proof.
            double normSquared = 0.0;
            foreach (var g in grads.Values)
            {
                if (g is null) continue;
                var span = g.AsSpan();
                for (int i = 0; i < span.Length; i++)
                {
                    double v = ops.ToDouble(span[i]);
                    normSquared += v * v;
                }
            }
            double clipFactor = Math.Min(1.0, clipNorm / Math.Sqrt(normSquared + 1e-12));

            // Accumulate clipped gradient (Abadi 2016 Algorithm 1, line 5):
            // "ḡ_t(x_i) = g_t(x_i) / max(1, ∥g_t(x_i)∥_2 / C)".
            foreach (var p in parameters)
            {
                if (!grads.TryGetValue(p, out var g) || g is null) continue;
                var sumSpan = sums[p].AsWritableSpan();
                var gSpan = g.AsSpan();
                for (int i = 0; i < gSpan.Length; i++)
                {
                    double v = ops.ToDouble(sumSpan[i]) + ops.ToDouble(gSpan[i]) * clipFactor;
                    sumSpan[i] = ops.FromDouble(v);
                }
            }
        }

        // Noise + average (Abadi 2016 Algorithm 1, line 6):
        // "g̃_t = (1/L)(Σ_i ḡ_t(x_i) + N(0, σ² C² I))".
        double invBatch = 1.0 / batchSize;
        double noiseStd = clipNorm * noiseMultiplier * invBatch;
        var result = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        foreach (var p in parameters)
        {
            var sum = sums[p];
            var averaged = new Tensor<T>(p._shape);
            var sumSpan = sum.AsSpan();
            var avgSpan = averaged.AsWritableSpan();
            for (int i = 0; i < sumSpan.Length; i++)
            {
                double noise = noiseStd > 0 ? SampleGaussian(rng) * noiseStd : 0.0;
                avgSpan[i] = ops.FromDouble(ops.ToDouble(sumSpan[i]) * invBatch + noise);
            }
            result[p] = averaged;
        }
        return result;
    }

    /// <summary>
    /// Box-Muller Gaussian sample from a uniform RNG. Kept private to avoid
    /// leaking a helper that callers might use with a wrong RNG source — DP-SGD's
    /// noise draws must come from a cryptographically-appropriate RNG for the
    /// privacy proof to hold; the caller's <c>Random</c> choice is theirs.
    /// </summary>
    private static double SampleGaussian(Random rng)
    {
        // Guard against u1 = 0 (Log(0) = -Infinity).
        double u1;
        do { u1 = rng.NextDouble(); } while (u1 < 1e-300);
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
