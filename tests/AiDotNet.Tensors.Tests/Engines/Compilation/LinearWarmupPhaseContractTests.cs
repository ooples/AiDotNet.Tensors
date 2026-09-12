using System.IO;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class LinearWarmupPhaseContractTests
{
    public enum MalformedSchedule { MissingDouble, MissingInteger, NegativeWarmup, UndefinedMode, DecayBeforeWarmup }

    [Theory]
    [InlineData(0, WarmupDecayMode.Constant)]
    [InlineData(0, WarmupDecayMode.Linear)]
    [InlineData(0, WarmupDecayMode.Cosine)]
    [InlineData(4, WarmupDecayMode.Constant)]
    [InlineData(4, WarmupDecayMode.Linear)]
    [InlineData(4, WarmupDecayMode.Cosine)]
    public void LegacyEagerWireKindPreservesTheSavedInitialRateEvenWithoutWarmup(int warmup, WarmupDecayMode mode)
    {
        // An explicit wire ID proves old binaries reject this new representation;
        // it must not be confused with kind 9, whose zero-warmup first rate is peak.
        const int legacyEagerWireKind = 11;
        var legacy = new FusedLrScheduleCheckpoint((FusedLrScheduleKind)legacyEagerWireKind,
            new[] { 1.0, 0.2, 0.9 }, new[] { warmup, 8, (int)mode });
        var restored = RoundTrip(legacy).ToSchedule();
#if !PUBLISHED_TENSORS
        var factory = LrSchedule.LegacyEagerLinearWarmup(1.0, warmup, 8, 0.2, mode, 0.9);
#endif
        for (int step = 0; step <= 10; step++)
        {
            double expected = step == 0 ? 0.2 : Math.Max(0.9, ExpectedRate(step, warmup, 8, mode, 0.9));
            Assert.Equal(expected, restored.GetLr(step + 1), 14);
#if !PUBLISHED_TENSORS
            Assert.Equal(expected, factory.GetLr(step + 1), 14);
#endif
        }
        var recaptured = Assert.IsType<FusedLrScheduleCheckpoint>(restored.TryCaptureCheckpoint());
        Assert.Equal(legacyEagerWireKind, (int)recaptured.Kind);
#if !PUBLISHED_TENSORS
        Assert.Equal(FusedLrScheduleKind.LinearWarmupLegacyEagerDecay,
            Assert.IsType<FusedLrScheduleCheckpoint>(factory.TryCaptureCheckpoint()).Kind);
#endif
        Assert.Equal(0.2, RoundTrip(recaptured).ToSchedule().GetLr(1));
    }

    [Theory]
    [InlineData(WarmupDecayMode.Constant)]
    [InlineData(WarmupDecayMode.Linear)]
    [InlineData(WarmupDecayMode.Cosine)]
    public void WarmupRemainsLinearWhenDecayEndsAboveTheInitialRate(WarmupDecayMode mode)
    {
        var schedule = LrSchedule.LinearWarmup(1.0, 4, 8, 0.2, mode, 0.9);
        for (int step = 0; step <= 10; step++)
            Assert.Equal(ExpectedRate(step, 4, 8, mode, 0.9), schedule.GetLr(step + 1), 14);
    }

    [Theory]
    [InlineData(WarmupDecayMode.Constant)]
    [InlineData(WarmupDecayMode.Linear)]
    [InlineData(WarmupDecayMode.Cosine)]
    public void CorrectedWarmupSurvivesActualBinaryOptimizerCheckpointRoundTrips(WarmupDecayMode mode)
    {
        var schedule = LrSchedule.LinearWarmup(1.0, 4, 8, 0.2, mode, 0.9);
        var captured = Assert.IsType<FusedLrScheduleCheckpoint>(schedule.TryCaptureCheckpoint());
        var restored = RoundTrip(captured).ToSchedule();
        for (int step = 0; step <= 10; step++)
            Assert.Equal(ExpectedRate(step, 4, 8, mode, 0.9), restored.GetLr(step + 1), 14);

        Assert.NotEqual(FusedLrScheduleKind.LinearWarmupDecay, captured.Kind);
        var recaptured = Assert.IsType<FusedLrScheduleCheckpoint>(restored.TryCaptureCheckpoint());
        Assert.Equal(captured.Kind, recaptured.Kind);
        Assert.Equal(captured.Doubles, recaptured.Doubles);
        Assert.Equal(captured.Ints, recaptured.Ints);
    }

    [Theory]
    [InlineData(WarmupDecayMode.Constant)]
    [InlineData(WarmupDecayMode.Linear)]
    [InlineData(WarmupDecayMode.Cosine)]
    public void LegacyCheckpointKeepsItsOriginalFlooredWarmupTrajectory(WarmupDecayMode mode)
    {
        var legacy = new FusedLrScheduleCheckpoint(FusedLrScheduleKind.LinearWarmupDecay,
            new[] { 1.0, 0.2, 0.9 }, new[] { 4, 8, (int)mode });
        var restored = RoundTrip(legacy).ToSchedule();
        for (int step = 0; step <= 10; step++)
        {
            double expected = step == 0 ? 0.2 : Math.Max(0.9, ExpectedRate(step, 4, 8, mode, 0.9));
            Assert.Equal(expected, restored.GetLr(step + 1), 14);
        }
        var recaptured = Assert.IsType<FusedLrScheduleCheckpoint>(restored.TryCaptureCheckpoint());
        Assert.Equal(FusedLrScheduleKind.LinearWarmupDecay, recaptured.Kind);
        Assert.Equal(0.9, RoundTrip(recaptured).ToSchedule().GetLr(2));
    }

    [Theory]
    [InlineData(WarmupDecayMode.Constant)]
    [InlineData(WarmupDecayMode.Linear)]
    [InlineData(WarmupDecayMode.Cosine)]
    public void DisabledWarmupStartsAtPeakAndDecaysAtTheSameStepBeforeAndAfterRestore(WarmupDecayMode mode)
    {
        var schedule = LrSchedule.LinearWarmup(1.0, 0, 8, 0.2, mode, 0.1);
        var restored = RoundTrip(Assert.IsType<FusedLrScheduleCheckpoint>(schedule.TryCaptureCheckpoint())).ToSchedule();
        for (int step = 0; step <= 10; step++)
        {
            double expected = ExpectedRate(step, 0, 8, mode, 0.1);
            Assert.Equal(expected, schedule.GetLr(step + 1), 14);
            Assert.Equal(expected, restored.GetLr(step + 1), 14);
        }
    }

    [Theory]
    [InlineData(WarmupDecayMode.Constant)]
    [InlineData(WarmupDecayMode.Linear)]
    [InlineData(WarmupDecayMode.Cosine)]
    public void OrdinaryWarmupDecayAndEndpointRetainTheirExistingTrajectory(WarmupDecayMode mode)
    {
        var schedule = LrSchedule.LinearWarmup(1.0, 4, 8, 0.2, mode, 0.1);
        for (int step = 0; step <= 10; step++)
            Assert.Equal(ExpectedRate(step, 4, 8, mode, 0.1), schedule.GetLr(step + 1), 14);
        Assert.Equal(mode == WarmupDecayMode.Constant ? 1.0 : 0.1, schedule.GetLr(int.MaxValue), 14);
    }

    [Theory]
    [InlineData(WarmupDecayMode.Constant)]
    [InlineData(WarmupDecayMode.Linear)]
    [InlineData(WarmupDecayMode.Cosine)]
    public void EmptyWarmupAndDecayKeepTheDocumentedFirstBatchPeak(WarmupDecayMode mode)
    {
        var schedule = LrSchedule.LinearWarmup(1.0, 0, 0, 0.2, mode, 0.1);
        var restored = RoundTrip(Assert.IsType<FusedLrScheduleCheckpoint>(schedule.TryCaptureCheckpoint())).ToSchedule();
        Assert.Equal(1.0, schedule.GetLr(1));
        Assert.Equal(1.0, restored.GetLr(1));
        double expected = mode == WarmupDecayMode.Constant ? 1.0 : 0.1;
        Assert.Equal(expected, schedule.GetLr(2));
        Assert.Equal(expected, restored.GetLr(2));
    }

    [Theory]
    [InlineData(WarmupDecayMode.Constant)]
    [InlineData(WarmupDecayMode.Linear)]
    [InlineData(WarmupDecayMode.Cosine)]
    public void WarmCompiledPlanReconfigurationAndRestoreDoNotReuseTheOtherSchedule(WarmupDecayMode mode)
    {
        var engine = new CpuEngine();
        var parameter = new Tensor<float>(new[] { 1 });
        parameter[0] = 1.0f;
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(engine.TensorMultiply(parameter, parameter), null);
            plan = scope.CompileTraining(new[] { parameter });
        }
        using (plan)
        {
            var legacy = new FusedLrScheduleCheckpoint(FusedLrScheduleKind.LinearWarmupDecay,
                new[] { 0.1, 0.02, 0.09 }, new[] { 4, 8, (int)mode });
            plan.ConfigureOptimizer(OptimizerType.SGD, legacy.ToSchedule(), extras: new FusedOptimizerExtras { Momentum = 0 });
            float expected = 1.0f;
            foreach (float rate in new[] { 0.02f, 0.09f })
            {
                plan.Step();
                expected -= rate * 2 * expected;
                Assert.Equal(expected, parameter[0], 6);
            }
            var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            var oldState = RoundTripOptimizer(Assert.IsType<FusedOptimizerCheckpoint>(concrete.CaptureFusedOptimizerCheckpoint()));
            Assert.Equal(FusedLrScheduleKind.LinearWarmupDecay, Assert.Single(oldState.Schedules).Kind);

            plan.ConfigureOptimizer(OptimizerType.SGD, LrSchedule.LinearWarmup(0.1, 4, 8, 0.02, mode, 0.09),
                extras: new FusedOptimizerExtras { Momentum = 0 });
            foreach (float rate in new[] { 0.02f, 0.04f })
            {
                plan.Step();
                expected -= rate * 2 * expected;
                Assert.Equal(expected, parameter[0], 6);
            }
            var newState = RoundTripOptimizer(Assert.IsType<FusedOptimizerCheckpoint>(concrete.CaptureFusedOptimizerCheckpoint()));
            Assert.NotEqual(Assert.Single(oldState.Schedules).Kind, Assert.Single(newState.Schedules).Kind);

            concrete.RestoreFusedOptimizerCheckpoint(oldState);
            plan.Step();
            expected -= 0.09f * 2 * expected;
            Assert.Equal(expected, parameter[0], 6);
            concrete.RestoreFusedOptimizerCheckpoint(newState);
            plan.Step();
            expected -= 0.06f * 2 * expected;
            Assert.Equal(expected, parameter[0], 6);
        }
    }

    [Theory]
    [InlineData(MalformedSchedule.MissingDouble)]
    [InlineData(MalformedSchedule.MissingInteger)]
    [InlineData(MalformedSchedule.NegativeWarmup)]
    [InlineData(MalformedSchedule.UndefinedMode)]
    [InlineData(MalformedSchedule.DecayBeforeWarmup)]
    public void NewlyCapturedScheduleRejectsMalformedCheckpointParameters(MalformedSchedule malformed)
    {
        var captured = Assert.IsType<FusedLrScheduleCheckpoint>(
            LrSchedule.LinearWarmup(1.0, 4, 8, 0.2, WarmupDecayMode.Linear, 0.9).TryCaptureCheckpoint());
        switch (malformed)
        {
            case MalformedSchedule.MissingDouble: captured.Doubles = new[] { 1.0 }; break;
            case MalformedSchedule.MissingInteger: captured.Ints = new[] { 4 }; break;
            case MalformedSchedule.NegativeWarmup: captured.Ints[0] = -1; break;
            case MalformedSchedule.UndefinedMode: captured.Ints[2] = int.MaxValue; break;
            case MalformedSchedule.DecayBeforeWarmup: captured.Ints[1] = 3; break;
            default: throw new ArgumentOutOfRangeException(nameof(malformed));
        }
        Assert.Throws<InvalidDataException>(() => RoundTrip(captured).ToSchedule());
    }

    [Fact]
    public void UnknownScheduleKindsFailClosed()
    {
        var unknown = new FusedLrScheduleCheckpoint((FusedLrScheduleKind)int.MaxValue,
            new[] { 1.0, 0.2, 0.9 }, new[] { 4, 8, 1 });
        Assert.Throws<InvalidDataException>(() => RoundTrip(unknown).ToSchedule());
    }

    private static FusedLrScheduleCheckpoint RoundTrip(FusedLrScheduleCheckpoint schedule)
        => Assert.Single(RoundTripOptimizer(new FusedOptimizerCheckpoint
        {
            OptimizerStep = 2,
            Schedules = new[] { schedule }
        }).Schedules);

    private static FusedOptimizerCheckpoint RoundTripOptimizer(FusedOptimizerCheckpoint checkpoint)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
            FusedOptimizerCheckpointSerializer.Write(writer, checkpoint);
        stream.Position = 0;
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        var restored = Assert.IsType<FusedOptimizerCheckpoint>(FusedOptimizerCheckpointSerializer.Read(reader));
        Assert.Equal(checkpoint.OptimizerStep, restored.OptimizerStep);
        Assert.Equal(stream.Length, stream.Position);
        return restored;
    }

    private static double ExpectedRate(int step, int warmup, int total, WarmupDecayMode mode, double end)
    {
        if (step < warmup) return 0.2 + 0.8 * step / warmup;
        if (mode == WarmupDecayMode.Constant) return 1.0;
        if (step >= total) return end;
        double progress = (double)(step - warmup) / (total - warmup);
        return mode switch
        {
            WarmupDecayMode.Linear => 1.0 - (1.0 - end) * progress,
            WarmupDecayMode.Cosine => end + (1.0 - end) * (1.0 + Math.Cos(Math.PI * progress)) / 2.0,
            _ => throw new ArgumentOutOfRangeException(nameof(mode))
        };
    }
}
