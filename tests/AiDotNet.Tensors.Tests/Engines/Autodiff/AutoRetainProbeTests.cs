using System;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// <see cref="AutoRetainProbe"/>'s window rules, on host objects (the probe keys on storages, not on a device):
/// the first backward run of a graph probes; its quarantine closes at that same graph's next run, which frees what
/// nothing read and retains the producer of what something did; another graph's run leaves it open; a storage nothing
/// holds drops out; a failing release surfaces after the rest of the window was released.
/// </summary>
// The probe is process-wide: run alone, like the GPU tests that drive it through real tapes.
[Collection("DirectGpuSerial")]
public sealed class AutoRetainProbeTests : IDisposable
{
    private const long GraphA = 0x5A5A_0001;
    private const long GraphB = 0x5A5A_0002;

    public AutoRetainProbeTests() => AutoRetainProbe.ResetForTests();

    public void Dispose() => AutoRetainProbe.ResetForTests();

    [Fact]
    public void The_first_run_of_a_graph_probes_and_later_runs_do_not()
    {
        Assert.True(AutoRetainProbe.BeginRun(GraphA));
        Assert.False(AutoRetainProbe.BeginRun(GraphA));
        Assert.True(AutoRetainProbe.BeginRun(GraphB));
        Assert.False(AutoRetainProbe.Active);
    }

    [Fact]
    public void An_unread_quarantined_storage_is_released_at_the_same_graphs_next_run()
    {
        AutoRetainProbe.BeginRun(GraphA);
        var vector = new object();
        var array = new object();
        int released = 0;
        AutoRetainProbe.Quarantine(GraphA, vector, array, key: 11, () => released++);
        Assert.True(AutoRetainProbe.Active);
        Assert.Equal(1, AutoRetainProbe.OpenWindowCount);

        Assert.False(AutoRetainProbe.BeginRun(GraphA));
        Assert.Equal(1, released);
        Assert.False(AutoRetainProbe.IsRetained(11));
        Assert.Equal(0, AutoRetainProbe.OpenWindowCount);
        Assert.False(AutoRetainProbe.Active);
        GC.KeepAlive(vector);
        GC.KeepAlive(array);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_read_storage_is_kept_and_its_producer_retained(bool readThroughArray)
    {
        AutoRetainProbe.BeginRun(GraphA);
        var vector = new object();
        var array = new object();
        int released = 0;
        AutoRetainProbe.Quarantine(GraphA, vector, array, key: 22, () => released++);

        AutoRetainProbe.NoteAccess(readThroughArray ? array : vector);
        AutoRetainProbe.NoteAccess(vector);   // a second read changes nothing
        Assert.True(AutoRetainProbe.IsRetained(22));
        Assert.Equal(1, AutoRetainProbe.RetainedCount);

        AutoRetainProbe.BeginRun(GraphA);
        Assert.Equal(0, released);
        GC.KeepAlive(vector);
        GC.KeepAlive(array);
    }

    [Fact]
    public void A_read_of_a_storage_that_was_never_quarantined_retains_nothing()
    {
        AutoRetainProbe.NoteAccess(new object());
        Assert.Equal(0, AutoRetainProbe.RetainedCount);
        AutoRetainProbe.Retain(33);
        Assert.True(AutoRetainProbe.IsRetained(33));
        Assert.False(AutoRetainProbe.IsRetained(34));
    }

    [Fact]
    public void Another_graphs_run_leaves_the_window_open()
    {
        AutoRetainProbe.BeginRun(GraphA);
        var vector = new object();
        int released = 0;
        AutoRetainProbe.Quarantine(GraphA, vector, array: null, key: 44, () => released++);

        // Keep A's output, run B's backward, then read A's output: ordinary code that must still work.
        AutoRetainProbe.BeginRun(GraphB);
        Assert.Equal(0, released);
        Assert.Equal(1, AutoRetainProbe.OpenWindowCount);
        Assert.True(AutoRetainProbe.Active);

        AutoRetainProbe.BeginRun(GraphA);
        Assert.Equal(1, released);
        GC.KeepAlive(vector);
    }

    [Fact]
    public void A_storage_quarantined_under_two_keys_is_kept_for_both_once_read()
    {
        // The same storage freed under two producer keys in one window (a view's producer and its base's). The second
        // quarantine used to rebind the storage to its own probe, so a read marked only that one, and closing the window
        // ran the first entry's release on storage that had been read.
        AutoRetainProbe.BeginRun(GraphA);
        var vector = new object();
        var array = new object();
        int released55 = 0, released56 = 0;
        AutoRetainProbe.Quarantine(GraphA, vector, array, key: 55, () => released55++);
        AutoRetainProbe.Quarantine(GraphA, vector, array, key: 56, () => released56++);
        AutoRetainProbe.NoteAccess(vector);
        Assert.True(AutoRetainProbe.IsRetained(56));
        Assert.True(AutoRetainProbe.IsRetained(55));

        AutoRetainProbe.BeginRun(GraphA);   // closes the window
        Assert.Equal(0, released55);
        Assert.Equal(0, released56);
        Assert.Equal(0, AutoRetainProbe.OpenWindowCount);
        GC.KeepAlive(vector);
        GC.KeepAlive(array);
    }

    [Fact]
    public void A_storage_quarantined_under_two_keys_and_never_read_is_released()
    {
        AutoRetainProbe.BeginRun(GraphA);
        var vector = new object();
        int released = 0;
        AutoRetainProbe.Quarantine(GraphA, vector, null, key: 57, () => released++);
        AutoRetainProbe.Quarantine(GraphA, vector, null, key: 58, () => released++);
        AutoRetainProbe.BeginRun(GraphA);
        Assert.Equal(2, released);
        Assert.False(AutoRetainProbe.IsRetained(57));
        Assert.False(AutoRetainProbe.IsRetained(58));
        // The closed window forgot the storage: a later read retains nothing.
        AutoRetainProbe.NoteAccess(vector);
        Assert.Equal(0, AutoRetainProbe.RetainedCount);
        GC.KeepAlive(vector);
    }

    [Fact]
    public void Producer_keys_compare_their_whole_structure()
    {
        // [1, 32] and [2, 1] folded to the same 31-based hash, so two different graphs shared one signature and one probe.
        var method = typeof(AutoRetainProbeTests).GetMethod(nameof(Producer_keys_compare_their_whole_structure))
            ?? throw new InvalidOperationException("method not found");
        var a = new ProducerKey(method, new[] { 1, 32 });
        var b = new ProducerKey(method, new[] { 2, 1 });
        Assert.NotEqual(a, b);
        Assert.Equal(a, new ProducerKey(method, new[] { 1, 32 }));
        Assert.NotEqual(a, a.WithOrdinal(1));
        Assert.Equal(a.WithOrdinal(1), new ProducerKey(method, new[] { 1, 32 }).WithOrdinal(1));
        Assert.NotEqual(a, new ProducerKey(null, new[] { 1, 32 }));
        Assert.NotEqual(new GraphSignature(3, new[] { a }), new GraphSignature(3, new[] { b }));
        Assert.NotEqual(new GraphSignature(3, new[] { a }), new GraphSignature(4, new[] { a }));
        Assert.Equal(new GraphSignature(3, new[] { a, b }), new GraphSignature(3, new[] { new ProducerKey(method, new[] { 1, 32 }), b }));

        // The probe tells the two graphs apart: each one's first run probes.
        Assert.True(AutoRetainProbe.BeginRun(new GraphSignature(3, new[] { a })));
        Assert.True(AutoRetainProbe.BeginRun(new GraphSignature(3, new[] { b })));
        Assert.False(AutoRetainProbe.BeginRun(new GraphSignature(3, new[] { new ProducerKey(method, new[] { 1, 32 }) })));
    }

    [Fact]
    public void Probe_history_is_bounded_and_a_forgotten_graph_probes_again()
    {
        // Every signature stayed remembered until ResetForTests: a run over many shapes grew the set without bound.
        for (long g = 0; g < AutoRetainProbe.MaxProbedGraphs + 100; g++)
            Assert.True(AutoRetainProbe.BeginRun(0x7000_0000L + g));
        Assert.Equal(AutoRetainProbe.MaxProbedGraphs, AutoRetainProbe.ProbedGraphCount);
        Assert.True(AutoRetainProbe.BeginRun(0x7000_0000L));    // the oldest was forgotten: it probes again
        Assert.False(AutoRetainProbe.BeginRun(0x7000_0000L + AutoRetainProbe.MaxProbedGraphs + 99));   // a recent one does not
    }

    [Fact]
    public void A_storage_nothing_holds_drops_out_of_another_graphs_window()
    {
        AutoRetainProbe.BeginRun(GraphA);
        int released = 0;
        QuarantineUnreferenced(GraphA, key: 66, () => released++);
        Assert.Equal(1, AutoRetainProbe.OpenWindowCount);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // B's run prunes A's dead entry; A's window is then empty and closes without releasing anything.
        AutoRetainProbe.BeginRun(GraphB);
        Assert.Equal(0, AutoRetainProbe.OpenWindowCount);
        Assert.False(AutoRetainProbe.Active);
        Assert.Equal(0, released);
    }

    [Fact]
    public void A_dead_storage_in_the_closing_window_is_not_released()
    {
        AutoRetainProbe.BeginRun(GraphA);
        int released = 0;
        QuarantineUnreferenced(GraphA, key: 77, () => released++);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        AutoRetainProbe.BeginRun(GraphA);
        Assert.Equal(0, released);
    }

    [Fact]
    public void A_failing_release_is_rethrown_after_the_rest_of_the_window_is_released()
    {
        AutoRetainProbe.BeginRun(GraphA);
        var first = new object();
        var second = new object();
        var third = new object();
        int released = 0;
        AutoRetainProbe.Quarantine(GraphA, first, null, 1, () => throw new InvalidOperationException("first"));
        AutoRetainProbe.Quarantine(GraphA, second, null, 2, () => throw new InvalidOperationException("second"));
        AutoRetainProbe.Quarantine(GraphA, third, null, 3, () => released++);

        var ex = Assert.Throws<InvalidOperationException>(() => AutoRetainProbe.BeginRun(GraphA));
        Assert.Equal("first", ex.Message);
        Assert.Equal(1, released);
        Assert.Equal(0, AutoRetainProbe.OpenWindowCount);
        GC.KeepAlive(first);
        GC.KeepAlive(second);
        GC.KeepAlive(third);
    }

    [Fact]
    public void ResetForTests_closes_every_window()
    {
        AutoRetainProbe.BeginRun(GraphA);
        var vector = new object();
        AutoRetainProbe.Quarantine(GraphA, vector, new object(), 88, () => { });
        AutoRetainProbe.Retain(89);
        AutoRetainProbe.ResetForTests();
        Assert.Equal(0, AutoRetainProbe.OpenWindowCount);
        Assert.Equal(0, AutoRetainProbe.RetainedCount);
        Assert.False(AutoRetainProbe.Active);
        Assert.True(AutoRetainProbe.BeginRun(GraphA));   // probed graphs were forgotten too
        GC.KeepAlive(vector);
    }

    // In its own frame, so no local keeps the storage alive in a Debug build.
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void QuarantineUnreferenced(long graph, long key, Action release)
        => AutoRetainProbe.Quarantine(graph, new object(), new object(), key, release);
}
