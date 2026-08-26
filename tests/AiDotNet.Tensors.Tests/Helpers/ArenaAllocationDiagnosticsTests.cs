// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Reading the arena allocation counters must never hand back numbers that do not mean anything.
/// </summary>
/// <remarks>
/// The counters only tally when AIDOTNET_ALLOC_DIAG=1. Before this API the only way to read them
/// was the raw fields, so with recording off they read zero — indistinguishable from "no
/// allocations happened". That zero was actually misread as proof that TensorAllocator.Rent was
/// never called during a forward pass, when in truth 268 rents totalling 227.1 MB had gone through
/// it. TryGetArenaDiagnostics returns false rather than zeros so that cannot happen again.
///
/// The record/snapshot/reset logic lives on <see cref="TensorAllocator.ArenaAllocationCounters"/>
/// precisely so it can be asserted directly in EVERY run. An earlier version of these tests
/// returned early whenever the process had the other AIDOTNET_ALLOC_DIAG setting, which meant a
/// default run verified none of the recording behaviour while still reporting green.
/// </remarks>
public class ArenaAllocationDiagnosticsTests
{
    // ---- counter semantics: unconditional, independent of the env flag ----

    [Fact]
    public void Counters_TallyEachOutcomeSeparately()
    {
        var counters = new TensorAllocator.ArenaAllocationCounters();

        counters.RecordHit(100);
        counters.RecordHit(50);
        counters.RecordMiss(7);
        counters.RecordNull(3);
        counters.RecordNull(4);

        var snapshot = counters.Snapshot();

        Assert.Equal(2, snapshot.Hit);
        Assert.Equal(150, snapshot.HitBytes);
        Assert.Equal(1, snapshot.Miss);
        Assert.Equal(7, snapshot.MissBytes);
        Assert.Equal(2, snapshot.Null);
        Assert.Equal(7, snapshot.NullBytes);
    }

    [Fact]
    public void Counters_ResetZeroesEveryTally()
    {
        var counters = new TensorAllocator.ArenaAllocationCounters();
        counters.RecordHit(10);
        counters.RecordMiss(20);
        counters.RecordNull(30);

        counters.Reset();
        var snapshot = counters.Snapshot();

        Assert.Equal(0, snapshot.Hit);
        Assert.Equal(0, snapshot.Miss);
        Assert.Equal(0, snapshot.Null);
        Assert.Equal(0, snapshot.HitBytes);
        Assert.Equal(0, snapshot.MissBytes);
        Assert.Equal(0, snapshot.NullBytes);
    }

    [Fact]
    public void Counters_SnapshotIsAValueNotALiveView()
    {
        var counters = new TensorAllocator.ArenaAllocationCounters();
        counters.RecordHit(1);

        var before = counters.Snapshot();
        counters.RecordHit(1);

        Assert.Equal(1, before.Hit);
        Assert.Equal(2, counters.Snapshot().Hit);
    }

    // ---- public read contract ----

    [Fact]
    public void TryGet_AgreesWithTheEnabledFlag()
    {
        bool read = TensorAllocator.TryGetArenaDiagnostics(out _);
        Assert.Equal(TensorAllocator.ArenaDiagnosticsEnabled, read);
    }

    [Fact]
    public void ReadContract_HoldsForWhicheverConfigurationThisProcessIsIn()
    {
        // Both branches ASSERT. Neither returns early: whichever way AIDOTNET_ALLOC_DIAG is set,
        // this test verifies the contract that applies, so a default run is not a free pass.
        if (TensorAllocator.ArenaDiagnosticsEnabled)
        {
            Assert.True(TensorAllocator.TryGetArenaDiagnostics(out _));
            Assert.True(TensorAllocator.TryResetArenaDiagnostics());

            // Wiring: the instrumented entry point must feed the counters.
            var shape = new[] { 64, 64 };
            _ = TensorAllocator.RentUninitialized<double>(shape);
            Assert.True(TensorAllocator.TryGetArenaDiagnostics(out var noArena));
            Assert.True(
                noArena.Null > 0 && noArena.NullBytes > 0,
                $"a rent with no arena active must tally as Null; got Null={noArena.Null}, bytes={noArena.NullBytes}");

            Assert.True(TensorAllocator.TryResetArenaDiagnostics());
            using (var arena = TensorArena.Create())
            {
                _ = TensorAllocator.RentUninitialized<double>(shape);
            }

            Assert.True(TensorAllocator.TryGetArenaDiagnostics(out var withArena));
            Assert.True(
                withArena.Hit + withArena.Miss > 0,
                $"a rent inside an arena must tally as Hit or Miss, not Null; got Hit={withArena.Hit}, "
                    + $"Miss={withArena.Miss}, Null={withArena.Null}");
            Assert.Equal(0, withArena.Null);
        }
        else
        {
            Assert.False(
                TensorAllocator.TryGetArenaDiagnostics(out var diagnostics),
                "not recording, so the read must report that rather than succeed with zeros");
            Assert.Equal(0, diagnostics.Hit);
            Assert.Equal(0, diagnostics.Null);
            Assert.False(TensorAllocator.TryResetArenaDiagnostics());
        }
    }
}
