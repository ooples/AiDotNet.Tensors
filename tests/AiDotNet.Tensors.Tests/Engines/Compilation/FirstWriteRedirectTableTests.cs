using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The first-write redirect rules, without a GPU: a pooled slot is redirected into a gradient accumulator only after an
/// executed run in which its output was that accumulator's first write and reached no other accumulation. A redirected
/// contribution IS the accumulator's buffer, so a second consumer would read whatever later contributions add to it.
/// </summary>
public class FirstWriteRedirectTableTests
{
    private sealed class Buffer : IGpuBuffer
    {
        public int Size => 64;
        public long SizeInBytes => 256;
        public IntPtr Handle { get; } = new(1);
        public void Dispose() { }
    }

    private const long Slot = (3L << 32) | 1;

    [Fact]
    public void A_single_use_first_write_is_redirected_after_an_executed_run()
    {
        var table = new FirstWriteRedirectTable();
        var scratch = new Buffer();
        var accumulator = new Buffer();
        table.BeginRun();
        table.NoteContribution(scratch);
        table.Offer(Slot, scratch, accumulator);
        Assert.False(table.TryGetTarget(Slot, out _));   // nothing is redirected mid-run
        table.EndRun(executed: true);

        Assert.True(table.TryGetTarget(Slot, out var target));
        Assert.Same(accumulator, target);
    }

    [Fact]
    public void A_contribution_that_also_feeds_a_second_accumulator_is_never_redirected()
    {
        var table = new FirstWriteRedirectTable();
        var scratch = new Buffer();
        var accumulatorA = new Buffer();
        table.BeginRun();
        table.NoteContribution(scratch);                 // first write of A
        table.Offer(Slot, scratch, accumulatorA);
        table.NoteContribution(scratch);                 // the same contribution also reaches B
        table.EndRun(executed: true);
        Assert.False(table.TryGetTarget(Slot, out _));

        // Blocked for good: a later run in which the second consumer happens not to show up does not lift it.
        table.BeginRun();
        table.NoteContribution(scratch);
        table.Offer(Slot, scratch, accumulatorA);
        table.EndRun(executed: true);
        Assert.False(table.TryGetTarget(Slot, out _));
    }

    [Fact]
    public void A_second_use_seen_while_redirected_blocks_the_slot()
    {
        var table = new FirstWriteRedirectTable();
        var scratch = new Buffer();
        var accumulator = new Buffer();
        table.BeginRun();
        table.NoteContribution(scratch);
        table.Offer(Slot, scratch, accumulator);
        table.EndRun(executed: true);
        Assert.True(table.TryGetTarget(Slot, out _));

        // Redirected: the contribution is the accumulator's own buffer, consumed twice this run.
        table.BeginRun();
        table.NoteContribution(accumulator);
        table.NoteContribution(accumulator);
        table.EndRun(executed: true);
        Assert.False(table.TryGetTarget(Slot, out _));
    }

    [Fact]
    public void A_captured_or_failed_run_is_not_evidence()
    {
        var table = new FirstWriteRedirectTable();
        var scratch = new Buffer();
        var accumulator = new Buffer();
        table.BeginRun();
        table.NoteContribution(scratch);
        table.Offer(Slot, scratch, accumulator);
        table.EndRun(executed: false);
        Assert.False(table.TryGetTarget(Slot, out _));
    }
}
