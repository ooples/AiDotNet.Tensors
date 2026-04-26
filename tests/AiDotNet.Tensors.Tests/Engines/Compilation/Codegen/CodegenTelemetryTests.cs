// Copyright (c) AiDotNet. All rights reserved.
// Phase G of issue #225: codegen observability — per-pass timing,
// emit outcomes, autotune cache stats, composite snapshot.

#nullable disable

using System.Threading;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Guards;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Telemetry;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

[Collection("CodegenSharedRegistry")]
public class CodegenTelemetryTests : System.IDisposable
{
    public CodegenTelemetryTests()
    {
        CodegenTelemetry.Reset();
        CodegenTelemetry.Enable();
        CodegenGuardRegistry.Clear();
    }

    public void Dispose()
    {
        CodegenTelemetry.Disable();
        CodegenTelemetry.Reset();
        CodegenGuardRegistry.Clear();
    }

    // ─── Enable / disable gate ───────────────────────────────────────

    [Fact]
    public void Disabled_TimePass_IsNoOp()
    {
        CodegenTelemetry.Disable();
        using (CodegenTelemetry.TimePass("Disabled"))
        {
            Thread.Sleep(1);
        }
        Assert.Empty(CodegenTelemetry.GetPassTimings());
    }

    [Fact]
    public void Disabled_RecordOutcome_IsNoOp()
    {
        CodegenTelemetry.Disable();
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.CpuDotNetJit, succeeded: true);
        Assert.Empty(CodegenTelemetry.GetEmitOutcomes());
    }

    [Fact]
    public void Disabled_RecordAutotune_IsNoOp()
    {
        CodegenTelemetry.Disable();
        CodegenTelemetry.RecordAutotuneHit();
        CodegenTelemetry.RecordAutotuneMiss();
        var stats = CodegenTelemetry.GetAutotuneStats();
        Assert.Equal(0, stats.Hits);
        Assert.Equal(0, stats.Misses);
    }

    // ─── Per-pass timing ─────────────────────────────────────────────

    [Fact]
    public void TimePass_AccumulatesCallCount()
    {
        for (int i = 0; i < 5; i++)
        {
            using var scope = CodegenTelemetry.TimePass("MyPass");
        }
        var timings = CodegenTelemetry.GetPassTimings();
        Assert.True(timings.ContainsKey("MyPass"));
        Assert.Equal(5, timings["MyPass"].CallCount);
    }

    [Fact]
    public void TimePass_MeanIsBetweenMinAndMax()
    {
        using (CodegenTelemetry.TimePass("A")) { Thread.Sleep(1); }
        using (CodegenTelemetry.TimePass("A")) { Thread.Sleep(5); }
        using (CodegenTelemetry.TimePass("A")) { Thread.Sleep(1); }
        var t = CodegenTelemetry.GetPassTimings()["A"];
        Assert.Equal(3, t.CallCount);
        Assert.True(t.MinTicks <= t.MeanTicks);
        Assert.True(t.MeanTicks <= t.MaxTicks);
        Assert.True(t.TotalElapsed.TotalMilliseconds >= 5); // Lower bound.
    }

    // ─── Emit outcomes ───────────────────────────────────────────────

    [Fact]
    public void RecordEmitOutcome_BucketsByTargetAndReason()
    {
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.CpuDotNetJit, succeeded: true);
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.CpuDotNetJit, succeeded: true);
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.CpuDotNetJit, succeeded: false, "unsupported op");
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.Triton, succeeded: true);

        // Decline reasons are normalised into a fixed taxonomy so the
        // emit-outcomes dictionary stays bounded — see
        // CodegenTelemetry.NormalizeDeclineReason.
        var outcomes = CodegenTelemetry.GetEmitOutcomes();
        Assert.Equal(2, outcomes[(CodegenTarget.CpuDotNetJit, "Succeeded")]);
        Assert.Equal(1, outcomes[(CodegenTarget.CpuDotNetJit, "UnsupportedOp")]);
        Assert.Equal(1, outcomes[(CodegenTarget.Triton, "Succeeded")]);
    }

    [Fact]
    public void RecordEmitOutcome_NormalizesFreeFormReasons_BoundedCardinality()
    {
        // Distinct free-form prose strings should fold into a small
        // fixed taxonomy so a long-running process can't accumulate
        // millions of decline buckets.
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.Triton, false,
            "Phase B CPU emitter does not yet handle Reduction ops (found ReduceSum)");
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.Triton, false,
            "Phase B CPU emitter does not yet handle Reduction ops (found Softmax)");
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.Hip, false,
            "Dtype Float64 not supported by this emitter.");
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.Wgsl, false,
            "Phase C pointwise emitter requires uniform element count across nodes; found 16 vs 24 at op Add");

        var outcomes = CodegenTelemetry.GetEmitOutcomes();
        // Two distinct UnsupportedOp prose strings → one bucket.
        Assert.Equal(2, outcomes[(CodegenTarget.Triton, "UnsupportedOp")]);
        // Dtype mention → UnsupportedDType bucket.
        Assert.Equal(1, outcomes[(CodegenTarget.Hip, "UnsupportedDType")]);
        // "element count" wording → ShapeMismatch.
        Assert.Equal(1, outcomes[(CodegenTarget.Wgsl, "ShapeMismatch")]);
    }

    [Fact]
    public void RecordEmitOutcome_NullDeclineReason_UsesDefault()
    {
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.Hip, succeeded: false, declineReason: null);
        var outcomes = CodegenTelemetry.GetEmitOutcomes();
        Assert.Equal(1, outcomes[(CodegenTarget.Hip, "DeclinedUnspecified")]);
    }

    // ─── Autotune stats ──────────────────────────────────────────────

    [Fact]
    public void Autotune_HitsAndMisses_AccumulateSeparately()
    {
        for (int i = 0; i < 7; i++) CodegenTelemetry.RecordAutotuneHit();
        for (int i = 0; i < 3; i++) CodegenTelemetry.RecordAutotuneMiss();

        var stats = CodegenTelemetry.GetAutotuneStats();
        Assert.Equal(7, stats.Hits);
        Assert.Equal(3, stats.Misses);
        Assert.Equal(10, stats.Total);
        Assert.Equal(0.7, stats.HitRatio, precision: 2);
    }

    [Fact]
    public void Autotune_HitRatio_ZeroWhenNoLookups()
    {
        var stats = CodegenTelemetry.GetAutotuneStats();
        Assert.Equal(0.0, stats.HitRatio);
    }

    // ─── Composite snapshot ──────────────────────────────────────────

    [Fact]
    public void Snapshot_AggregatesAllChannels()
    {
        using (CodegenTelemetry.TimePass("A")) { }
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.CpuDotNetJit, true);
        CodegenTelemetry.RecordAutotuneHit();
        CodegenGuardRegistry.TryReserveRecompile(0x1, "warmup");

        var snap = CodegenTelemetry.Snapshot();
        Assert.Contains("A", snap.PassTimings.Keys);
        Assert.True(snap.Autotune.Hits >= 1);
        Assert.Single(snap.RecompileLog);
        Assert.Contains(
            (CodegenTarget.CpuDotNetJit, "Succeeded"),
            snap.EmitOutcomes.Keys);
    }

    [Fact]
    public void Reset_ClearsAggregates_LeavesEnabledFlag()
    {
        CodegenTelemetry.RecordAutotuneHit();
        using (CodegenTelemetry.TimePass("B")) { }
        CodegenTelemetry.RecordEmitOutcome(CodegenTarget.Wgsl, true);

        CodegenTelemetry.Reset();

        Assert.Empty(CodegenTelemetry.GetPassTimings());
        Assert.Empty(CodegenTelemetry.GetEmitOutcomes());
        var stats = CodegenTelemetry.GetAutotuneStats();
        Assert.Equal(0, stats.Hits);
        Assert.Equal(0, stats.Misses);
        // Enabled flag survives Reset — next record still takes effect.
        Assert.True(CodegenTelemetry.IsEnabled);
        CodegenTelemetry.RecordAutotuneMiss();
        Assert.Equal(1, CodegenTelemetry.GetAutotuneStats().Misses);
    }
}
