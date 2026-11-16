using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The schedules that mirror AiDotNet's per-batch schedulers: each must restore from its checkpoint to the identical
/// sequence (bit for bit), reject invalid arguments and payloads, and follow the documented step convention - step 1
/// at the start rate, step N at the rate after N - 1 completed steps. (Bit-equality with the eager schedulers
/// themselves is tested in AiDotNet, which owns them.)
/// </summary>
public class CompletedStepLrScheduleTests
{
    [Fact]
    public void Large_steps_preserve_hold_and_constant_restart_periods()
    {
        Assert.Equal(1.0, LrSchedule.NoamHoldAnnealing(1.0, 1500000000, 1500000000, 0.5).GetLr(2000000000));
        var restarts = LrSchedule.CosineWarmRestarts(0.1, 2);
        Assert.Equal(restarts.GetLr(1), restarts.GetLr(int.MaxValue));
    }

    public static IEnumerable<object[]> Schedules()
    {
        yield return new object[] { "wsd", LrSchedule.WarmupStableDecay(3e-4, 10, 50, 30, StableDecayShape.OneMinusSqrt, 1e-5) };
        yield return new object[] { "poly", LrSchedule.Polynomial(0.1, 60, 2.0, 1e-3) };
        yield return new object[] { "multistep", LrSchedule.MultiStep(0.1, new[] { 5, 20, 40 }, 0.5, 1e-3) };
        yield return new object[] { "tristage", LrSchedule.TriStage(0.1, 8, 10, 60, 1e-4, 2.0) };
        yield return new object[] { "noamhold", LrSchedule.NoamHoldAnnealing(1e-3, 10, 5, 0.5) };
        yield return new object[] { "sgdr", LrSchedule.CosineWarmRestarts(0.1, 7, 2, 1e-4) };
        yield return new object[] { "cosine-hold", LrSchedule.CosineAnnealing(0.1, 40, 1e-4) };
        yield return new object[] { "onecycle-lin", LrSchedule.OneCycleWithAnneal(0.1, 50, 0.3, 25, 1e4, OneCycleAnneal.Linear) };
        yield return new object[] { "cyclic-exp", LrSchedule.Cyclic(1e-3, 1e-2, 6, 9, CyclicLrMode.ExponentialRange, 0.97) };
        yield return new object[] { "cyclic-tri2", LrSchedule.Cyclic(1e-3, 1e-2, 4, 4, CyclicLrMode.Triangular2) };
        yield return new object[] { "sequential", LrSchedule.Sequential(
            new[] { LrSchedule.WarmupStableDecay(0.1, 5, 5, 0), LrSchedule.Sequential(new[] { LrSchedule.Constant(0.05), LrSchedule.Polynomial(0.05, 30) }, new[] { 10 }) },
            new[] { 12 }) };
    }

    [Theory]
    [MemberData(nameof(Schedules))]
    public void Checkpoint_RestoresTheIdenticalSequence(string name, LrSchedule schedule)
    {
        var checkpoint = schedule.TryCaptureCheckpoint();
        Assert.NotNull(checkpoint);
        if (checkpoint is null) return;
        var restored = checkpoint.ToSchedule();
        for (int step = 0; step <= 200; step++)
            Assert.True(BitConverter.DoubleToInt64Bits(schedule.GetLr(step)) == BitConverter.DoubleToInt64Bits(restored.GetLr(step)),
                $"{name} step {step}: {schedule.GetLr(step):R} vs restored {restored.GetLr(step):R}");
    }

    [Theory]
    [MemberData(nameof(Schedules))]
    public void Rates_AreFiniteAndNotNegative(string name, LrSchedule schedule)
    {
        for (int step = 1; step <= 500; step++)
        {
            double lr = schedule.GetLr(step);
            Assert.True(!double.IsNaN(lr) && !double.IsInfinity(lr) && lr >= 0, $"{name} step {step}: {lr}");
        }
    }

    [Fact]
    public void Sequential_StartsEachLaterScheduleAtItsOwnStepOne_AfterItsMilestone()
    {
        var a = LrSchedule.Polynomial(1.0, 10);
        var b = LrSchedule.MultiStep(0.5, new[] { 2 }, 0.1);
        var s = LrSchedule.Sequential(new[] { a, b }, new[] { 4 });
        // After four completed steps, schedule b begins at its own step one.
        for (int step = 1; step <= 4; step++) Assert.Equal(a.GetLr(step), s.GetLr(step));
        for (int step = 5; step <= 20; step++) Assert.Equal(b.GetLr(step - 4), s.GetLr(step));
    }

    [Fact]
    public void Conventions_FirstStepIsTheStartRate()
    {
        Assert.Equal(0.0, LrSchedule.WarmupStableDecay(1.0, 4, 8, 4).GetLr(1));
        Assert.Equal(0.25, LrSchedule.WarmupStableDecay(1.0, 4, 8, 4).GetLr(2));
        Assert.Equal(0.1, LrSchedule.MultiStep(0.1, new[] { 1 }, 0.5).GetLr(1));
        Assert.Equal(0.05, LrSchedule.MultiStep(0.1, new[] { 1 }, 0.5).GetLr(2));
        Assert.Equal(0.25, LrSchedule.TriStage(1.0, 4, 0, 8).GetLr(1));   // the ramp's first point, not the peak
        Assert.Equal(1.0 / 25, LrSchedule.OneCycleWithAnneal(1.0, 100).GetLr(1));
        Assert.Equal(0.1, LrSchedule.CosineWarmRestarts(0.1, 5).GetLr(1));
        Assert.Equal(0.1, LrSchedule.CosineWarmRestarts(0.1, 5).GetLr(6));   // restart after 5 completed steps
    }

    [Fact]
    public void InvalidArguments_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.WarmupStableDecay(1.0, 10, 5, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.WarmupStableDecay(1.0, 0, 0, 1, endLr: 2.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.Polynomial(1.0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.MultiStep(1.0, new[] { 5, 5 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.MultiStep(1.0, Array.Empty<int>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.TriStage(1.0, 6, 6, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.NoamHoldAnnealing(1.0, 0, 0, 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.CosineWarmRestarts(1.0, 5, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.OneCycleWithAnneal(1.0, 10, 1.0));
        // Valid factors whose quotient overflows: the starting rate would be infinite.
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.OneCycleWithAnneal(1.0, 10, 0.3, double.Epsilon, 1e4, OneCycleAnneal.Cosine));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.OneCycleWithAnneal(1.0, 10, 0.3, 1e-200, 1e-200, OneCycleAnneal.Linear));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.Cyclic(1.0, 0.5, 2, 2, CyclicLrMode.Triangular));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.Sequential(new[] { LrSchedule.Constant(1) }, new[] { 3 }));
    }

    [Fact]
    public void MalformedCheckpoints_FailAtRestore()
    {
        var cp = LrSchedule.TriStage(1.0, 2, 2, 10).TryCaptureCheckpoint();
        Assert.NotNull(cp);
        if (cp is null) return;
        cp.Ints[0] = 9;   // warmup + hold now exceed the run
        Assert.Throws<InvalidDataException>(() => cp.ToSchedule());

        var seq = LrSchedule.Sequential(new List<LrSchedule> { LrSchedule.Constant(1), LrSchedule.Polynomial(1, 5) }, new List<int> { 3 }).TryCaptureCheckpoint();
        Assert.NotNull(seq);
        if (seq is null) return;
        var truncated = new FusedLrScheduleCheckpoint(seq.Kind, seq.Doubles, seq.Ints.Take(seq.Ints.Length - 1).ToArray());
        Assert.Throws<InvalidDataException>(() => truncated.ToSchedule());

        var nan = LrSchedule.Polynomial(1, 5).TryCaptureCheckpoint();
        Assert.NotNull(nan);
        if (nan is null) return;
        nan.Doubles[1] = double.NaN;
        Assert.Throws<InvalidDataException>(() => nan.ToSchedule());
    }

    [Fact]
    public void CosineWarmRestarts_LargeStepAndMultiplier_DoesNotOverflowOrHang()
    {
        foreach (int tMult in new[] { 2, 10 })
        {
            var s = LrSchedule.CosineWarmRestarts(0.1, 7, tMult, 1e-4);
            foreach (int step in new[] { 1_000_000, 1_000_000_000, int.MaxValue })
            {
                double lr = s.GetLr(step);
                Assert.True(!double.IsNaN(lr) && !double.IsInfinity(lr) && lr >= 1e-4 && lr <= 0.1,
                    $"tMult={tMult} step={step}: lr={lr}");
            }
        }
        var ms = LrSchedule.MultiStep(0.1, new List<int> { 3, 7 }, 0.5, 1e-4);
        Assert.Equal(0.025, ms.GetLr(8), 12);
    }

    [Fact]
    public void Sequential_WithACustomChild_HasNoCheckpoint()
    {
        var s = LrSchedule.Sequential(new LrSchedule[] { new Custom(), LrSchedule.Constant(1) }, new[] { 3 });
        Assert.Null(s.TryCaptureCheckpoint());
    }

    [Fact]
    public void NoamHoldAnnealing_HoldEndingPastIntMax_HoldsThePeak()
    {
        // warmup + hold = 10 + int.MaxValue wrapped negative in int, so the hold was skipped and the decay ratio
        // warmup / (trainingStep - hold) went negative: Pow(negative, 0.5) = NaN.
        var s = LrSchedule.NoamHoldAnnealing(1.0, 10, int.MaxValue, 0.5);
        foreach (int step in new[] { 10, 11, 1_000, int.MaxValue })
            Assert.Equal(1.0, s.GetLr(step));
        Assert.Equal(0.5, s.GetLr(5));   // still on the warmup ramp: 5 / 10
    }

    [Fact]
    public void CosineWarmRestarts_FixedPeriod_IsClosedFormAndFastAtLargeSteps()
    {
        // tMult = 1 (the default) walked every period from 0: t / t0 iterations per call, so 1000 steps near int.MaxValue
        // with t0 = 1 took ~2e12 iterations. The closed form must also equal the per-period formula.
        const double lr0 = 0.1, etaMin = 1e-4;
        foreach (int t0 in new[] { 1, 7 })
        {
            var s = LrSchedule.CosineWarmRestarts(lr0, t0, 1, etaMin);
            var watch = System.Diagnostics.Stopwatch.StartNew();
            for (int step = int.MaxValue - 999; step > 0 && step <= int.MaxValue; step++)
            {
                int completed = step - 1;
                double position = completed % t0;
                double expected = etaMin + 0.5 * (lr0 - etaMin) * (1 + Math.Cos(Math.PI * position / t0));
                Assert.Equal(BitConverter.DoubleToInt64Bits(Math.Max(etaMin, expected)), BitConverter.DoubleToInt64Bits(s.GetLr(step)));
                if (step == int.MaxValue) break;
            }
            watch.Stop();
            Assert.True(watch.ElapsedMilliseconds < 5_000, $"t0={t0}: 1000 large steps took {watch.ElapsedMilliseconds} ms");
            for (int step = 1; step <= 50; step++)
            {
                double position = (step - 1) % t0;
                double expected = etaMin + 0.5 * (lr0 - etaMin) * (1 + Math.Cos(Math.PI * position / t0));
                Assert.Equal(BitConverter.DoubleToInt64Bits(step == 1 ? lr0 : Math.Max(etaMin, expected)),
                    BitConverter.DoubleToInt64Bits(s.GetLr(step)));
            }
        }
    }

    [Fact]
    public void Cyclic_CycleLongerThanIntMax_ReachesThePeak()
    {
        // up + down = 3e9 wrapped negative in int, which corrupted the cycle position (GetLr returned ~0.1367).
        var s = LrSchedule.Cyclic(0.0, 1.0, 1_500_000_000, 1_500_000_000, CyclicLrMode.Triangular);
        Assert.Equal(1.0, s.GetLr(1_500_000_001));                // 1.5e9 completed steps: the peak
        Assert.Equal(0.5, s.GetLr(750_000_001));                  // half way up
        Assert.Equal(1.0 - 1.0 / 1_500_000_000, s.GetLr(1_500_000_002));   // first step down
    }

    [Fact]
    public void Sequential_UsesEveryChildStartingAtItsFirstStep()
    {
        // Advance the child that just supplied a rate, then switch at the completed-step milestone.
        // The newly selected child has completed no steps and supplies its first rate next.
        // This deliberately corrects the older eager scheduler's delayed transition; it is not a parity claim for it.
        var children = new[] { LrSchedule.Polynomial(1.0, 10), LrSchedule.MultiStep(0.5, new[] { 2 }, 0.1), LrSchedule.CosineAnnealing(0.2, 5) };
        int[] milestones = { 4, 9 };
        var fused = LrSchedule.Sequential(children, milestones);

        int current = 0, index = 0;
        var childSteps = new int[children.Length];
        double rate = children[0].GetLr(1);
        for (int step = 1; step <= 40; step++)
        {
            Assert.True(BitConverter.DoubleToInt64Bits(rate) == BitConverter.DoubleToInt64Bits(fused.GetLr(step)),
                $"step {step}: eager {rate:R}, fused {fused.GetLr(step):R}");
            current++;
            childSteps[index]++;
            while (index < milestones.Length && current >= milestones[index]) index++;
            rate = children[index].GetLr(childSteps[index] + 1);
        }
    }

    [Fact]
    public void Sequential_DeeplyNestedCheckpoint_RestoresOncePerLevel()
    {
        // ToSchedule validated (one full restore) and then restored again, at every level: 2^depth leaf builds. At depth
        // 30 that is ~1e9 rebuilds; once per level it is instant.
        LrSchedule s = LrSchedule.Polynomial(0.1, 20);
        for (int d = 0; d < 30; d++) s = LrSchedule.Sequential(new[] { s }, Array.Empty<int>());
        var checkpoint = s.TryCaptureCheckpoint() ?? throw new InvalidOperationException("no checkpoint");
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var restored = checkpoint.ToSchedule();
        watch.Stop();
        Assert.True(watch.ElapsedMilliseconds < 5_000, $"restoring 30 nested levels took {watch.ElapsedMilliseconds} ms");
        for (int step = 1; step <= 30; step++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(s.GetLr(step)), BitConverter.DoubleToInt64Bits(restored.GetLr(step)));
    }

    [Fact]
    public void Sequential_CheckpointNestedPastTheLimit_IsRejectedAsInvalidData()
    {
        // Each level costs stack frames on restore, and a stack overflow cannot be caught: a crafted payload nesting
        // without bound must fail as invalid data instead.
        LrSchedule s = LrSchedule.Constant(0.1);
        for (int d = 0; d < 40; d++) s = LrSchedule.Sequential(new[] { s }, Array.Empty<int>());
        var checkpoint = s.TryCaptureCheckpoint() ?? throw new InvalidOperationException("no checkpoint");
        Assert.Throws<InvalidDataException>(() => checkpoint.ToSchedule());
        Assert.Throws<InvalidDataException>(() => checkpoint.Validate());
    }

    private sealed class Custom : LrSchedule
    {
        public override double GetLr(int step) => 0.5;
    }
}
