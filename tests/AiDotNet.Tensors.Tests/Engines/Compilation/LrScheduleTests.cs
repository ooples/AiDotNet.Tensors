using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Issue #348 — schedule formulas. Verified against the PyTorch
/// reference implementations of <c>torch.optim.lr_scheduler</c> so
/// users migrating from PyTorch hit identical numeric trajectories.
/// </summary>
public class LrScheduleTests
{
    private const double Tol = 1e-9;

    [Fact]
    public void Constant_ReturnsSameValueEveryStep()
    {
        var s = LrSchedule.Constant(0.05);
        Assert.Equal(0.05, s.GetLr(1), Tol);
        Assert.Equal(0.05, s.GetLr(100), Tol);
        Assert.Equal(0.05, s.GetLr(int.MaxValue), Tol);
    }

    [Fact]
    public void Cosine_StartsAtLrMaxEndsAtLrMin()
    {
        var s = LrSchedule.Cosine(lrMax: 1.0, totalSteps: 10, lrMin: 0.0);
        Assert.Equal(1.0, s.GetLr(1), Tol);
        Assert.Equal(0.0, s.GetLr(10), Tol);
    }

    [Fact]
    public void Cosine_IsMonotonicDecreasing()
    {
        var s = LrSchedule.Cosine(lrMax: 1.0, totalSteps: 100, lrMin: 0.0);
        double prev = double.PositiveInfinity;
        for (int t = 1; t <= 100; t++)
        {
            double cur = s.GetLr(t);
            Assert.True(cur <= prev + 1e-12, $"non-monotonic at step {t}: {prev} -> {cur}");
            prev = cur;
        }
    }

    [Fact]
    public void Cosine_HalfwayHitsHalfwayBetweenMaxAndMin()
    {
        // At progress = 0.5, cos(π/2) = 0, so lr = lrMin + (lrMax - lrMin) * 0.5 = (lrMax + lrMin)/2.
        var s = LrSchedule.Cosine(lrMax: 1.0, totalSteps: 99, lrMin: 0.1);
        // progress=0.5 ⇒ s = 1 + 0.5 * (99 - 1) = 50.
        Assert.Equal(0.55, s.GetLr(50), 1e-9);
    }

    [Fact]
    public void Cosine_ClampsBeyondEnd()
    {
        var s = LrSchedule.Cosine(lrMax: 1.0, totalSteps: 10, lrMin: 0.2);
        // Stepping past the end pins lr at lrMin (clamp on the right).
        Assert.Equal(0.2, s.GetLr(20), Tol);
        Assert.Equal(0.2, s.GetLr(1_000_000), Tol);
    }

    [Fact]
    public void OneCycle_LinearWarmupReachesLrMax()
    {
        var s = LrSchedule.OneCycle(lrMax: 1.0, totalSteps: 100, pctStart: 0.3,
            divFactor: 25.0, finalDivFactor: 1e4);
        // Warmup ends at step round(0.3 * 100) = 30.
        Assert.Equal(1.0 / 25.0, s.GetLr(1), 1e-9);
        Assert.Equal(1.0, s.GetLr(30), 1e-9);
    }

    [Fact]
    public void OneCycle_AnnealReachesPyTorchLrFinal()
    {
        // PyTorch's OneCycleLR: min_lr = initial_lr / final_div_factor
        //                            = (lrMax / divFactor) / finalDivFactor
        // NOT lrMax / finalDivFactor. With default (25, 1e4) the floor is
        // 1 / (25 * 1e4) = 4e-6, not 1e-4. PR #349 review #3.
        var s = LrSchedule.OneCycle(lrMax: 1.0, totalSteps: 100, pctStart: 0.3,
            divFactor: 25.0, finalDivFactor: 1e4);
        Assert.Equal((1.0 / 25.0) / 1e4, s.GetLr(100), 1e-12);
    }

    [Fact]
    public void LinearWarmupCosine_ZeroWarmup_StartsAtLrMax()
    {
        // With warmupSteps == 0 the cosine half must start AT lrMax on
        // step 1 — using (s - warmup) / (total - warmup) would give
        // progress = 1/total at step 1 (slightly below lrMax). PR #349
        // review #4.
        var s = LrSchedule.LinearWarmupCosine(lrMax: 1.0, warmupSteps: 0, totalSteps: 10, lrMin: 0.0);
        Assert.Equal(1.0, s.GetLr(1), 1e-12);
        Assert.Equal(0.0, s.GetLr(10), 1e-12);
    }

    [Fact]
    public void Exponential_DecaysGeometrically()
    {
        var s = LrSchedule.Exponential(lr0: 1.0, gamma: 0.9);
        Assert.Equal(1.0, s.GetLr(1), Tol);
        Assert.Equal(0.9, s.GetLr(2), Tol);
        Assert.Equal(0.81, s.GetLr(3), Tol);
        // 0.9^9 at step 10
        Assert.Equal(System.Math.Pow(0.9, 9), s.GetLr(10), Tol);
    }

    [Fact]
    public void Step_DropsAtBoundaries()
    {
        var s = LrSchedule.Step(lr0: 1.0, stepSize: 5, gamma: 0.1);
        Assert.Equal(1.0, s.GetLr(1), Tol);
        Assert.Equal(1.0, s.GetLr(5), Tol);  // floor((5-1)/5) = 0
        Assert.Equal(0.1, s.GetLr(6), Tol);  // floor(5/5) = 1
        Assert.Equal(0.01, s.GetLr(11), Tol); // floor(10/5) = 2
    }

    [Fact]
    public void Cyclic_RoundTripsBetweenBaseAndMax()
    {
        var s = LrSchedule.Cyclic(lrBase: 0.0, lrMax: 1.0, stepSize: 5);
        // Triangular: ascending phase 0..stepSize, then descending stepSize..2*stepSize.
        Assert.Equal(0.0, s.GetLr(1), Tol);   // phase=0
        Assert.Equal(1.0, s.GetLr(6), Tol);   // phase=5 -> t=1
        Assert.Equal(0.0, s.GetLr(11), Tol);  // phase=10 mod 10 = 0 -> t=0
        Assert.Equal(1.0, s.GetLr(16), Tol);  // back to peak — full cycle has wrapped
    }

    [Fact]
    public void LinearWarmupCosine_RisesLinearlyThenDescendsCosine()
    {
        var s = LrSchedule.LinearWarmupCosine(lrMax: 1.0, warmupSteps: 10, totalSteps: 110, lrMin: 0.0);
        // Linear warmup: step / warmupSteps * lrMax
        Assert.Equal(0.1, s.GetLr(1), 1e-9);
        Assert.Equal(0.5, s.GetLr(5), 1e-9);
        Assert.Equal(1.0, s.GetLr(10), 1e-9);
        // Cosine descent reaches lrMin at totalSteps.
        Assert.Equal(0.0, s.GetLr(110), 1e-9);
        // Monotonic decrease after warmup peak.
        double prev = 1.0;
        for (int t = 11; t <= 110; t++)
        {
            double cur = s.GetLr(t);
            Assert.True(cur <= prev + 1e-12, $"non-monotonic at step {t}");
            prev = cur;
        }
    }

    [Fact]
    public void LinearWarmup_Constant_MatchesAiDotNetBatchSequence()
    {
        var s = LrSchedule.LinearWarmup(
            lrMax: 0.2,
            warmupSteps: 4,
            totalSteps: 4,
            warmupInitLr: 0.01,
            decayMode: WarmupDecayMode.Constant);

        Assert.Equal(0.01, s.GetLr(1), 1e-12);
        Assert.Equal(0.0575, s.GetLr(2), 1e-12);
        Assert.Equal(0.105, s.GetLr(3), 1e-12);
        Assert.Equal(0.1525, s.GetLr(4), 1e-12);
        Assert.Equal(0.2, s.GetLr(5), 1e-12);
        Assert.Equal(0.2, s.GetLr(50), 1e-12);
    }

    [Fact]
    public void LinearWarmup_LinearDecay_MatchesAiDotNetBatchSequence()
    {
        var s = LrSchedule.LinearWarmup(
            lrMax: 0.1,
            warmupSteps: 2,
            totalSteps: 6,
            warmupInitLr: 0.02,
            decayMode: WarmupDecayMode.Linear,
            endLr: 0.001);

        Assert.Equal(0.02, s.GetLr(1), 1e-12);
        Assert.Equal(0.06, s.GetLr(2), 1e-12);
        Assert.Equal(0.1, s.GetLr(3), 1e-12);
        Assert.Equal(0.07525, s.GetLr(4), 1e-12);
        Assert.Equal(0.0505, s.GetLr(5), 1e-12);
        Assert.Equal(0.02575, s.GetLr(6), 1e-12);
        Assert.Equal(0.001, s.GetLr(7), 1e-12);
    }

    [Fact]
    public void LinearWarmup_CosineDecay_ReachesEndLrAndClamps()
    {
        var s = LrSchedule.LinearWarmup(
            lrMax: 0.05,
            warmupSteps: 3,
            totalSteps: 9,
            warmupInitLr: 0.005,
            decayMode: WarmupDecayMode.Cosine,
            endLr: 0.001);

        Assert.Equal(0.005, s.GetLr(1), 1e-12);
        Assert.Equal(0.02, s.GetLr(2), 1e-12);
        Assert.Equal(0.035, s.GetLr(3), 1e-12);
        Assert.Equal(0.05, s.GetLr(4), 1e-12);
        Assert.Equal(0.001, s.GetLr(10), 1e-12);
        Assert.Equal(0.001, s.GetLr(100), 1e-12);
    }

    [Theory]
    [InlineData(WarmupDecayMode.Constant)]
    [InlineData(WarmupDecayMode.Linear)]
    [InlineData(WarmupDecayMode.Cosine)]
    public void LinearWarmup_ZeroWarmup_StartsAtLrMax(WarmupDecayMode decayMode)
    {
        var s = LrSchedule.LinearWarmup(
            lrMax: 0.1,
            warmupSteps: 0,
            totalSteps: 4,
            warmupInitLr: 0.001,
            decayMode: decayMode,
            endLr: 0.01);

        Assert.Equal(0.1, s.GetLr(1), 1e-12);
    }

    [Theory]
    [InlineData(WarmupDecayMode.Linear, 0)]
    [InlineData(WarmupDecayMode.Linear, -1)]
    [InlineData(WarmupDecayMode.Cosine, 0)]
    [InlineData(WarmupDecayMode.Cosine, -1)]
    public void LinearWarmup_ZeroWarmupAndNormalizedTotal_StartsAtLrMax(
        WarmupDecayMode decayMode, int totalSteps)
    {
        var s = LrSchedule.LinearWarmup(
            lrMax: 0.1,
            warmupSteps: 0,
            totalSteps: totalSteps,
            warmupInitLr: 0.001,
            decayMode: decayMode,
            endLr: 0.01);

        Assert.Equal(0.1, s.GetLr(1), 1e-12);
    }

    [Fact]
    public void LinearWarmup_RejectsUndefinedDecayMode()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            LrSchedule.LinearWarmup(
                lrMax: 0.1,
                warmupSteps: 2,
                totalSteps: 4,
                decayMode: (WarmupDecayMode)999));
    }

    [Fact]
    public void LinearWarmup_DecayModesRejectTotalBeforeWarmup()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            LrSchedule.LinearWarmup(0.1, warmupSteps: 10, totalSteps: 9,
                decayMode: WarmupDecayMode.Linear));
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            LrSchedule.LinearWarmup(0.1, warmupSteps: 10, totalSteps: 9,
                decayMode: WarmupDecayMode.Cosine));

        var constant = LrSchedule.LinearWarmup(0.1, warmupSteps: 10, totalSteps: 9,
            decayMode: WarmupDecayMode.Constant);
        Assert.Equal(0.1, constant.GetLr(11), 1e-12);
    }

    [Theory]
    [InlineData(WarmupDecayMode.Linear, 0)]
    [InlineData(WarmupDecayMode.Linear, -1)]
    [InlineData(WarmupDecayMode.Cosine, 0)]
    [InlineData(WarmupDecayMode.Cosine, -1)]
    public void LinearWarmup_DecayModesDefaultNonPositiveTotalStepsToWarmup(
        WarmupDecayMode decayMode, int totalSteps)
    {
        var s = LrSchedule.LinearWarmup(
            lrMax: 0.1,
            warmupSteps: 3,
            totalSteps: totalSteps,
            warmupInitLr: 0.01,
            decayMode: decayMode,
            endLr: 0.001);

        Assert.Equal(0.01, s.GetLr(1), 1e-12);
        Assert.Equal(0.04, s.GetLr(2), 1e-12);
        Assert.Equal(0.07, s.GetLr(3), 1e-12);
        Assert.Equal(0.001, s.GetLr(4), 1e-12);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Cosine_RejectsBadTotalSteps(int total)
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() => LrSchedule.Cosine(1.0, total, 0.0));
    }

    [Fact]
    public void OneCycle_RejectsBadPctStart()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() => LrSchedule.OneCycle(1.0, 100, pctStart: 0.0));
        Assert.Throws<System.ArgumentOutOfRangeException>(() => LrSchedule.OneCycle(1.0, 100, pctStart: 1.0));
    }

    // Noam schedule (Vaswani 2017 §5.3): lr(t) = factor·d^(-0.5)·min(t^(-0.5), t·warmup^(-1.5)).
    // This is the schedule that lets the default Transformer (Adam β₂=0.98 + Noam)
    // run on the fused-compiled training path with a correct per-step ramp
    // instead of falling back to the eager tape (AiDotNet#1470).

    private static double NoamRef(int d, int warmup, double factor, int t)
        => factor * System.Math.Pow(d, -0.5)
                  * System.Math.Min(System.Math.Pow(t, -0.5), t * System.Math.Pow(warmup, -1.5));

    [Fact]
    public void Noam_MatchesPaperFormula_AcrossWarmupAndDecay()
    {
        const int d = 512, warmup = 4000;
        var s = LrSchedule.Noam(modelDimension: d, warmupSteps: warmup);
        foreach (int t in new[] { 1, 100, 2000, 3999, 4000, 4001, 8000, 50000 })
            Assert.Equal(NoamRef(d, warmup, 1.0, t), s.GetLr(t), Tol);
    }

    [Fact]
    public void Noam_RampsUpDuringWarmup_PeaksAtWarmup_ThenDecays()
    {
        const int d = 256, warmup = 100;
        var s = LrSchedule.Noam(modelDimension: d, warmupSteps: warmup);

        // Strictly increasing up to the warmup peak.
        for (int t = 1; t < warmup; t++)
            Assert.True(s.GetLr(t) < s.GetLr(t + 1),
                $"Noam must ramp up during warmup: GetLr({t}) < GetLr({t + 1})");

        // Peak at exactly t = warmup, value = factor·d^(-0.5)·warmup^(-0.5).
        double peak = s.GetLr(warmup);
        Assert.Equal(System.Math.Pow(d, -0.5) * System.Math.Pow(warmup, -0.5), peak, Tol);

        // Strictly decreasing after the peak.
        Assert.True(s.GetLr(warmup + 1) < peak);
        Assert.True(s.GetLr(2 * warmup) < s.GetLr(warmup + 1));

        // The frozen-at-warmup-step-1 symptom (#1470) would make GetLr(warmup)
        // ≈ GetLr(1); assert a real ramp of many ×.
        Assert.True(peak > s.GetLr(1) * 10,
            "Noam peak must be far above the warmup-step-1 value (the #1470 freeze symptom).");
    }

    [Fact]
    public void Noam_FactorScalesLinearly()
    {
        const int d = 128, warmup = 50;
        var s1 = LrSchedule.Noam(d, warmup, factor: 1.0);
        var s3 = LrSchedule.Noam(d, warmup, factor: 3.0);
        foreach (int t in new[] { 1, 25, 50, 200 })
            Assert.Equal(3.0 * s1.GetLr(t), s3.GetLr(t), Tol);
    }

    [Fact]
    public void Noam_ClampsNonPositiveStepToWarmupStart()
    {
        var s = LrSchedule.Noam(modelDimension: 64, warmupSteps: 10);
        // step < 1 clamps to t = 1 (no 0^(-0.5) blowup).
        Assert.Equal(s.GetLr(1), s.GetLr(0), Tol);
        Assert.Equal(s.GetLr(1), s.GetLr(-5), Tol);
    }

    [Fact]
    public void Noam_RejectsBadArgs()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() => LrSchedule.Noam(modelDimension: 0, warmupSteps: 10));
        Assert.Throws<System.ArgumentOutOfRangeException>(() => LrSchedule.Noam(modelDimension: 64, warmupSteps: 0));
        Assert.Throws<System.ArgumentOutOfRangeException>(() => LrSchedule.Noam(modelDimension: 64, warmupSteps: 10, factor: 0.0));
    }
}
