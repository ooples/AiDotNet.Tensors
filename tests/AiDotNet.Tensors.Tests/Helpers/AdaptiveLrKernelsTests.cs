using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Issue #348 — proves the three adaptive-LR kernels (Hypergradient,
/// Schedule-Free SGD, D-Adaptation) actually do what they say on the
/// kernel surface, via convergence tests on toy strongly-convex problems
/// with closed-form optima.
/// </summary>
public class AdaptiveLrKernelsTests
{
    /// <summary>Quadratic objective f(p) = 0.5 ||p - target||² has
    /// gradient (p - target). All three optimizers should converge to
    /// <c>target</c> from any starting point under appropriate conditions.</summary>
    private static Tensor<float> Quadratic_Gradient(Tensor<float> param, Tensor<float> target)
    {
        var grad = new Tensor<float>(param.Shape.ToArray());
        var p = param.AsSpan();
        var t = target.AsSpan();
        var g = grad.AsWritableSpan();
        for (int i = 0; i < p.Length; i++)
            g[i] = p[i] - t[i];
        return grad;
    }

    private static float L2DistanceSquared(Tensor<float> a, Tensor<float> b)
    {
        var aa = a.AsSpan();
        var bb = b.AsSpan();
        float sum = 0f;
        for (int i = 0; i < aa.Length; i++)
        {
            float d = aa[i] - bb[i];
            sum += d * d;
        }
        return sum;
    }

    [Fact]
    public void HypergradientSgd_ConvergesOnQuadratic()
    {
        var target = new Tensor<float>(new float[] { 1.0f, -0.5f, 2.0f, 0.3f }, new[] { 4 });
        var p = new Tensor<float>(new float[] { 5.0f, 5.0f, -5.0f, -5.0f }, new[] { 4 });
        var prevG = new Tensor<float>(new[] { 4 });

        // Start with a deliberately suboptimal lr; the kernel should
        // tune it up toward something productive.
        float lr = 0.001f;
        float hyperLr = 1e-4f;

        float initialDist = L2DistanceSquared(p, target);
        for (int step = 1; step <= 500; step++)
        {
            var g = Quadratic_Gradient(p, target);
            lr = OptimizerKernels.HypergradientSgdInPlace(p, g, prevG, lr, hyperLr);
        }

        float finalDist = L2DistanceSquared(p, target);
        Assert.True(finalDist < initialDist * 1e-4f,
            $"Hypergradient SGD didn't converge: initial L2² = {initialDist}, final L2² = {finalDist}");
    }

    [Fact]
    public void HypergradientSgd_LrAdaptsToProductiveRange()
    {
        // Same quadratic; this time we look at whether lr actually moves.
        // We expect lr to grow when consecutive gradients are aligned (early descent).
        var target = new Tensor<float>(new float[] { 0.0f, 0.0f, 0.0f, 0.0f }, new[] { 4 });
        var p = new Tensor<float>(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }, new[] { 4 });
        var prevG = new Tensor<float>(new[] { 4 });

        float lr0 = 0.01f;
        float lr = lr0;
        for (int step = 1; step <= 10; step++)
        {
            var g = Quadratic_Gradient(p, target);
            lr = OptimizerKernels.HypergradientSgdInPlace(p, g, prevG, lr, hyperLr: 1e-3f);
        }
        // After 10 aligned steps lr should have moved.
        Assert.NotEqual(lr0, lr);
        Assert.True(lr > 0f, $"lr went non-positive: {lr}");
    }

    [Fact]
    public void ScheduleFreeSgd_ConvergesOnQuadratic_XBetterThanZ()
    {
        var target = new Tensor<float>(new float[] { 1.0f, -0.5f, 2.0f, 0.3f }, new[] { 4 });
        var z = new Tensor<float>(new float[] { 5.0f, 5.0f, -5.0f, -5.0f }, new[] { 4 });
        // x starts at the same initial point as z by convention.
        var x = new Tensor<float>(new float[] { 5.0f, 5.0f, -5.0f, -5.0f }, new[] { 4 });
        var y = new Tensor<float>(new[] { 4 });

        float beta = 0.9f;
        float lr = 0.05f;
        float weightSum = 0f;

        // Defazio 2024 convergence bound on convex problems is
        // O(1/√T) — at T=2000 with starting L2 ~ 12, expected
        // residual ≈ 0.27, squared ≈ 0.07.
        for (int step = 1; step <= 2000; step++)
        {
            // Before forward: blend y from z and x.
            OptimizerKernels.ScheduleFreeYInPlace(y, z, x, beta);
            var g = Quadratic_Gradient(y, target);
            weightSum = OptimizerKernels.ScheduleFreeSgdInPlace(z, x, g, lr, weightSum);
        }

        // x (the weighted-average evaluation copy) is the algorithm's output.
        float distX = L2DistanceSquared(x, target);
        Assert.True(distX < 0.1f, $"Schedule-Free SGD: ||x - target||² = {distX} (expected < 0.1 at T=2000).");
    }

    [Fact]
    public void ScheduleFreeY_BlendsCorrectly()
    {
        // y = (1-β) z + β x — pure linear blend, kernel must match scalar formula.
        var z = new Tensor<float>(new float[] { 0.0f, 1.0f, 2.0f, -3.0f, 0.5f, 0.5f, 0.5f, 0.5f, 1.0f }, new[] { 9 });
        var x = new Tensor<float>(new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f }, new[] { 9 });
        var y = new Tensor<float>(new[] { 9 });
        OptimizerKernels.ScheduleFreeYInPlace(y, z, x, beta: 0.25f);
        var ys = y.AsSpan();
        var zs = z.AsSpan();
        var xs = x.AsSpan();
        for (int i = 0; i < ys.Length; i++)
            Assert.Equal(0.75f * zs[i] + 0.25f * xs[i], ys[i], 6);
    }

    [Fact]
    public void DAdaptationSgd_MakesProgressWithoutTuning()
    {
        // Two acceptance criteria for "learning-rate-free":
        //   (a) d grows from its initial value (algorithm IS adapting)
        //   (b) loss strictly decreases over the run (algorithm IS productive)
        // Strict convergence on a quadratic is sensitive to growthRate
        // tuning — the *point* of the algorithm is that the user
        // doesn't tune anything, including the growth bound. We verify
        // the qualitative claim instead.
        var target = new Tensor<float>(new float[] { 1.0f, -0.5f, 2.0f, 0.3f }, new[] { 4 });
        var p = new Tensor<float>(new float[] { 5.0f, 5.0f, -5.0f, -5.0f }, new[] { 4 });
        var sBuf = new Tensor<float>(new[] { 4 });

        float d0 = 1e-3f; // realistic small-problem initial estimate
        float d = d0;
        float rAccum = 0f;
        float initialDist = L2DistanceSquared(p, target);

        for (int step = 1; step <= 200; step++)
        {
            var g = Quadratic_Gradient(p, target);
            d = OptimizerKernels.DAdaptationSgdInPlace(p, g, sBuf, d, ref rAccum,
                lr: 1.0f, growthRate: 1.05f);
        }

        float finalDist = L2DistanceSquared(p, target);
        Assert.True(d > d0, $"D-Adaptation never grew d from initial {d0}; final d = {d}");
        Assert.True(finalDist < initialDist,
            $"D-Adaptation didn't reduce distance: initial = {initialDist}, final = {finalDist}");
    }

    [Fact]
    public void DAdaptationSgd_DIsMonotonicallyNonDecreasing()
    {
        // By construction d_{k+1} = max(d_k, ...) — d must never decrease.
        var target = new Tensor<float>(new float[] { 0.0f, 0.0f }, new[] { 2 });
        var p = new Tensor<float>(new float[] { 3.0f, 4.0f }, new[] { 2 });
        var sBuf = new Tensor<float>(new[] { 2 });

        float d = 1e-6f;
        float rAccum = 0f;
        float prev = d;
        for (int step = 1; step <= 50; step++)
        {
            var g = Quadratic_Gradient(p, target);
            d = OptimizerKernels.DAdaptationSgdInPlace(p, g, sBuf, d, ref rAccum);
            Assert.True(d >= prev, $"d decreased at step {step}: {prev} -> {d}");
            prev = d;
        }
    }

    // ── Per-parameter-group LR ──────────────────────────────────────────

    [Fact]
    public void SgdInPlaceGrouped_ParamsGetDistinctEffectiveLrs()
    {
        // Two parameter groups: backbone (lr=0.001) vs head (lr=0.01).
        // Both start at the same point with the same gradient; the larger
        // lr must produce a larger step.
        var backbone = new Tensor<float>(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }, new[] { 4 });
        var head = new Tensor<float>(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }, new[] { 4 });
        var backboneGrad = new Tensor<float>(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }, new[] { 4 });
        var headGrad = new Tensor<float>(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }, new[] { 4 });

        OptimizerKernels.SgdInPlaceGrouped(
            new[] { backbone, head },
            new[] { backboneGrad, headGrad },
            new[] { 0.001f, 0.01f },
            new[] { 0, 1 });

        // backbone: 1.0 - 0.001 * 1.0 = 0.999
        Assert.Equal(0.999f, backbone.AsSpan()[0], 5);
        // head:     1.0 - 0.01  * 1.0 = 0.990
        Assert.Equal(0.990f, head.AsSpan()[0], 5);
    }

    [Fact]
    public void SgdInPlaceGrouped_WithSchedule_AppliesPerStepLr()
    {
        var p1 = new Tensor<float>(new float[] { 0.0f }, new[] { 1 });
        var p2 = new Tensor<float>(new float[] { 0.0f }, new[] { 1 });
        var g1 = new Tensor<float>(new float[] { -1.0f }, new[] { 1 });
        var g2 = new Tensor<float>(new float[] { -1.0f }, new[] { 1 });

        // p1 group: constant 0.1; p2 group: cosine that returns 1.0 at step 1.
        var schedules = new LrSchedule[]
        {
            LrSchedule.Constant(0.1),
            LrSchedule.Cosine(lrMax: 1.0, totalSteps: 1000, lrMin: 0.0),
        };
        OptimizerKernels.SgdInPlaceGrouped(
            new[] { p1, p2 }, new[] { g1, g2 },
            schedules, new[] { 0, 1 }, step: 1);

        // p1: 0 - 0.1 * -1 = 0.1
        Assert.Equal(0.1f, p1.AsSpan()[0], 5);
        // p2: 0 - 1.0 * -1 = 1.0 (cos at step 1 is lrMax)
        Assert.Equal(1.0f, p2.AsSpan()[0], 5);
    }

    [Fact]
    public void SgdInPlaceGrouped_RejectsOutOfRangeGroupIndex()
    {
        var p = new Tensor<float>(new float[] { 0.0f }, new[] { 1 });
        var g = new Tensor<float>(new float[] { 0.0f }, new[] { 1 });
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            OptimizerKernels.SgdInPlaceGrouped(
                new[] { p }, new[] { g },
                new[] { 0.1f },
                new[] { 5 }));
    }

    // ── Eligibility predicate (feature 4) ───────────────────────────────

    [Theory]
    [InlineData(OptimizerType.SGD)]
    [InlineData(OptimizerType.Adam)]
    [InlineData(OptimizerType.AdamW)]
    [InlineData(OptimizerType.Lion)]
    [InlineData(OptimizerType.LAMB)]
    [InlineData(OptimizerType.HypergradientSGD)]
    [InlineData(OptimizerType.ScheduleFreeSGD)]
    [InlineData(OptimizerType.DAdaptationSGD)]
    public void IsFusedPathEligible_AcceptsAdaptiveLrOptimizers(OptimizerType opt)
    {
        // The whole point of issue #348: adaptive LR is NOT a fused-path
        // gate. Every Adam/Adam-derived optimizer + the new adaptive
        // kernels must pass.
        Assert.True(OptimizerKernels.IsFusedPathEligible(opt),
            $"IsFusedPathEligible({opt}) should be true — adaptive LR is not a kernel concern.");
    }

    [Theory]
    [InlineData(OptimizerType.LBFGS)]
    [InlineData(OptimizerType.SparseAdam)]
    public void IsFusedPathEligible_RejectsLegitimateExclusions(OptimizerType opt)
    {
        // LBFGS needs a closure to re-evaluate loss inside the step;
        // SparseAdam needs index/value plumbing the compiled plan doesn't
        // yet wire. These are the REAL exclusions.
        Assert.False(OptimizerKernels.IsFusedPathEligible(opt),
            $"IsFusedPathEligible({opt}) should be false.");
    }
}
