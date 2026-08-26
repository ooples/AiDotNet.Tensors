// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Reading the arena allocation counters must never hand back numbers that do not mean anything.
/// </summary>
/// <remarks>
/// The counters only record when AIDOTNET_ALLOC_DIAG=1. Before this API the only way to read them
/// was the raw fields, so with recording off they read zero — indistinguishable from "no
/// allocations happened". That zero was actually misread as proof that TensorAllocator.Rent was
/// never called during a forward pass, when in truth 268 rents totalling 227.1 MB had gone through
/// it. TryGetArenaDiagnostics returns false rather than zeros so that cannot happen again.
/// </remarks>
public class ArenaAllocationDiagnosticsTests
{
    [Fact]
    public void TryGet_AgreesWithTheEnabledFlag()
    {
        bool read = TensorAllocator.TryGetArenaDiagnostics(out _);
        Assert.Equal(TensorAllocator.ArenaDiagnosticsEnabled, read);
    }

    [Fact]
    public void TryGet_WhenNotRecording_ReturnsFalseAndNotZeros()
    {
        if (TensorAllocator.ArenaDiagnosticsEnabled)
        {
            // Running with AIDOTNET_ALLOC_DIAG=1: the disabled contract cannot be exercised here,
            // and the recording contract is covered by the test below.
            return;
        }

        bool read = TensorAllocator.TryGetArenaDiagnostics(out var diagnostics);

        Assert.False(read, "counters are not recording, so the read must report that rather than succeed");
        Assert.Equal(0, diagnostics.Hit);
        Assert.Equal(0, diagnostics.Null);
        Assert.False(TensorAllocator.TryResetArenaDiagnostics());
    }

    [Fact]
    public void WhenRecording_ArenaRentsAreCounted()
    {
        if (!TensorAllocator.ArenaDiagnosticsEnabled)
        {
            // Only meaningful with AIDOTNET_ALLOC_DIAG=1. Guarded rather than asserted so the suite
            // does not fail by default, and stated plainly so nobody reads a green run here as
            // proof that the counters work.
            return;
        }

        Assert.True(TensorAllocator.TryResetArenaDiagnostics());

        // Rent is the counted entry point, so call it directly rather than hoping some higher-level
        // operation routes through it -- an earlier version used Transpose/CloneDeepCopy, and another used Rent (whose arena tier is not the counted one)
        // and counted nothing, which would have shipped a green test that proved nothing.
        var shape = new[] { 64, 64 };
        _ = TensorAllocator.RentUninitialized<double>(shape);
        Assert.True(TensorAllocator.TryGetArenaDiagnostics(out var noArena));
        Assert.True(
            noArena.Null > 0 && noArena.NullBytes > 0,
            $"a rent with no arena active must count as Null; got Null={noArena.Null}, bytes={noArena.NullBytes}");

        Assert.True(TensorAllocator.TryResetArenaDiagnostics());
        using (var arena = TensorArena.Create())
        {
            _ = TensorAllocator.RentUninitialized<double>(shape);
        }

        Assert.True(TensorAllocator.TryGetArenaDiagnostics(out var withArena));
        Assert.True(
            withArena.Hit + withArena.Miss > 0,
            $"a rent inside an arena must count as Hit or Miss, not Null; got Hit={withArena.Hit}, "
                + $"Miss={withArena.Miss}, Null={withArena.Null}");
        Assert.Equal(0, withArena.Null);
    }
}
