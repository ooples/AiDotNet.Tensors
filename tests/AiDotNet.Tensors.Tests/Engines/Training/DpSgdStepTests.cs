using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Training;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Training;

/// <summary>
/// Phase 4D — fused per-example DP-SGD step (Abadi et al. 2016 Algorithm 1).
/// Validates the correctness contract: per-example clip BEFORE aggregate,
/// global L2 norm across all parameters, Gaussian noise on the aggregate.
/// </summary>
public class DpSgdStepTests
{
    /// <summary>
    /// With <c>noiseMultiplier = 0</c> and <c>clipNorm</c> larger than any per-example
    /// gradient's L2 norm, the DP-SGD helper must return the same aggregate as
    /// plain SGD (arithmetic mean of per-example gradients). This isolates the
    /// clip+aggregate contract from the noise-injection contract.
    /// </summary>
    [Fact]
    public void DpSgd_HighClipZeroNoise_MatchesPlainMeanOfPerExampleGradients()
    {
        var engine = new CpuEngine();

        // Trivial linear model: y = w · x, loss = y² per example. Per-example grad:
        //   ∂(w·x)²/∂w = 2·x·(w·x) = 2·w·x²
        //   for w = 0.5, x_i values below: grads are 2·0.5·x²= x².
        var w = new Tensor<float>(new float[] { 0.5f }, new[] { 1 });
        float[] xValues = new float[] { 1f, 2f, 3f, 4f };  // batch of 4 examples
        int batchSize = xValues.Length;

        var noised = DpSgdStep<float>.ComputeClippedAggregatedGradients(
            batchSize,
            perExampleLoss: exIdx =>
            {
                var x = new Tensor<float>(new float[] { xValues[exIdx] }, new[] { 1 });
                var y = engine.TensorMultiply(w, x);
                return engine.TensorMultiply(y, y);  // loss = y²
            },
            parameters: new[] { w },
            clipNorm: 1000.0,  // effectively unclipped
            noiseMultiplier: 0.0,  // no noise
            rng: new Random(42));

        // Expected: mean of per-example gradients = mean(x_i²) = (1+4+9+16)/4 = 7.5
        float expected = 0.5f * 2f * (1f + 4f + 9f + 16f) / batchSize;
        float actual = noised[w].ToArray()[0];
        Assert.True(System.MathF.Abs(actual - expected) < 1e-4f,
            $"DP-SGD with high clip / no noise should reproduce plain-mean SGD. Expected {expected:R}, got {actual:R}.");
    }

    /// <summary>
    /// With <c>clipNorm</c> smaller than every per-example gradient's L2 norm,
    /// each per-example gradient gets scaled to exactly <c>clipNorm</c>. The
    /// aggregate is therefore <c>(clipNorm / ||g_i||) · g_i</c> averaged over
    /// the batch. This validates the per-example-clip-BEFORE-aggregate contract
    /// (Abadi 2016 requires this order for the L2-sensitivity bound).
    /// </summary>
    [Fact]
    public void DpSgd_TightClip_AppliesPerExampleBeforeAggregate()
    {
        var engine = new CpuEngine();
        // Same trivial model as above.
        var w = new Tensor<float>(new float[] { 1.0f }, new[] { 1 });
        float[] xValues = new float[] { 3f, 4f };  // batch of 2, gradients will be 2·1·9=18 and 2·1·16=32
        int batchSize = 2;
        double clipNorm = 1.0;  // TIGHTER than any per-example gradient norm

        var noised = DpSgdStep<float>.ComputeClippedAggregatedGradients(
            batchSize,
            perExampleLoss: exIdx =>
            {
                var x = new Tensor<float>(new float[] { xValues[exIdx] }, new[] { 1 });
                var y = engine.TensorMultiply(w, x);
                return engine.TensorMultiply(y, y);
            },
            parameters: new[] { w },
            clipNorm: clipNorm,
            noiseMultiplier: 0.0,
            rng: new Random(42));

        // Per-example gradients:
        //   ex 0: 2·w·x² = 2·1·9 = 18. norm = 18. clipFactor = min(1, 1/18) = 1/18. Clipped = 1.
        //   ex 1: 2·w·x² = 2·1·16 = 32. norm = 32. clipFactor = min(1, 1/32) = 1/32. Clipped = 1.
        // Aggregate: (1 + 1) / 2 = 1.
        float actual = noised[w].ToArray()[0];
        Assert.True(System.MathF.Abs(actual - 1.0f) < 1e-4f,
            $"With per-example clip to norm=1.0, aggregate should be 1.0 (mean of two clipped gradients). Got {actual:R}. " +
            $"If aggregate-then-clip was applied, result would be (18+32)/2 clipped to 1 = 1.0 by coincidence — this test needs " +
            $"the WGAN-GP-style verification below to catch the wrong order.");
    }

    /// <summary>
    /// Post-hoc clip vs. per-example clip differ when gradients are asymmetric.
    /// Two examples with gradients [10, 0] and [0, 10] (both L2 = 10):
    ///   Per-example clip to norm=5: clip each to [5, 0] and [0, 5] → mean = [2.5, 2.5], overall norm ≈ 3.54.
    ///   Aggregate-then-clip (WRONG): aggregate = [5, 5], norm ≈ 7.07, clip to 5 → [3.54, 3.54].
    /// These differ. This test catches an implementation that clips post-aggregation.
    /// </summary>
    [Fact]
    public void DpSgd_AsymmetricGradients_ClipsPerExampleNotPostAggregate()
    {
        var engine = new CpuEngine();
        // Two-parameter model contrived so the two per-example gradients are
        // asymmetric across the parameter vector.
        var w = new Tensor<float>(new float[] { 1f, 1f }, new[] { 2 });
        // Two examples: one selects param[0] (grad = [x, 0]), the other selects param[1] (grad = [0, x]).
        // Loss for ex 0: (w[0] * x)² → grad = [2·w[0]·x², 0].
        // Loss for ex 1: (w[1] * x)² → grad = [0, 2·w[1]·x²].
        float xVal = System.MathF.Sqrt(50f);  // so 2·1·x² = 100. Per-ex L2 norm = 100.

        var noised = DpSgdStep<float>.ComputeClippedAggregatedGradients(
            batchSize: 2,
            perExampleLoss: exIdx =>
            {
                var x = new Tensor<float>(new float[] { xVal }, new[] { 1 });
                if (exIdx == 0)
                {
                    var w0 = engine.TensorSlice(w, new[] { 0 }, new[] { 1 });
                    var y = engine.TensorMultiply(w0, x);
                    return engine.TensorMultiply(y, y);
                }
                else
                {
                    var w1 = engine.TensorSlice(w, new[] { 1 }, new[] { 1 });
                    var y = engine.TensorMultiply(w1, x);
                    return engine.TensorMultiply(y, y);
                }
            },
            parameters: new[] { w },
            clipNorm: 10.0,  // Clip to L2 = 10. Per-ex norm was 100, so each is scaled by 0.1.
            noiseMultiplier: 0.0,
            rng: new Random(42));

        // Per-example clipped gradients: [10, 0] (was [100, 0]) and [0, 10] (was [0, 100]).
        // Mean = [5, 5].
        var actual = noised[w].ToArray();
        Assert.Equal(2, actual.Length);
        Assert.True(System.MathF.Abs(actual[0] - 5.0f) < 0.1f && System.MathF.Abs(actual[1] - 5.0f) < 0.1f,
            $"Per-example-clip aggregate should be [5, 5]. Got [{actual[0]:R}, {actual[1]:R}]. " +
            $"If aggregate-then-clip was applied, result would be [7.07, 7.07] (aggregate [100, 100], norm=141, scale=10/141≈0.071 → [7.07, 7.07]).");
    }
}
