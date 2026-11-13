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
        var restored = schedule.TryCaptureCheckpoint()!.ToSchedule();
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
        // Steps 1-5 = 0-4 completed: schedule a. Step 6 = 5 completed, the first step of b after its start: b step 2.
        for (int step = 1; step <= 5; step++) Assert.Equal(a.GetLr(step), s.GetLr(step));
        for (int step = 6; step <= 20; step++) Assert.Equal(b.GetLr(step - 4), s.GetLr(step));
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
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.Cyclic(1.0, 0.5, 2, 2, CyclicLrMode.Triangular));
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.Sequential(new[] { LrSchedule.Constant(1) }, new[] { 3 }));
    }

    [Fact]
    public void MalformedCheckpoints_FailAtRestore()
    {
        var cp = LrSchedule.TriStage(1.0, 2, 2, 10).TryCaptureCheckpoint()!;
        cp.Ints[0] = 9;   // warmup + hold now exceed the run
        Assert.Throws<InvalidDataException>(() => cp.ToSchedule());

        var seq = LrSchedule.Sequential(new[] { LrSchedule.Constant(1), LrSchedule.Polynomial(1, 5) }, new[] { 3 }).TryCaptureCheckpoint()!;
        var truncated = new FusedLrScheduleCheckpoint(seq.Kind, seq.Doubles, seq.Ints[..^1]);
        Assert.Throws<InvalidDataException>(() => truncated.ToSchedule());

        var nan = LrSchedule.Polynomial(1, 5).TryCaptureCheckpoint()!;
        nan.Doubles[1] = double.NaN;
        Assert.Throws<InvalidDataException>(() => nan.ToSchedule());
    }

    [Fact]
    public void Sequential_WithACustomChild_HasNoCheckpoint()
    {
        var s = LrSchedule.Sequential(new LrSchedule[] { new Custom(), LrSchedule.Constant(1) }, new[] { 3 });
        Assert.Null(s.TryCaptureCheckpoint());
    }

    private sealed class Custom : LrSchedule
    {
        public override double GetLr(int step) => 0.5;
    }
}
