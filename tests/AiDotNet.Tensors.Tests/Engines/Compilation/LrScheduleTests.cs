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
}
