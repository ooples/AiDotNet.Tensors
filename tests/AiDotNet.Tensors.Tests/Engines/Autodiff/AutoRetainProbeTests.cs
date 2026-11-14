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
    public void Requarantining_a_storage_rebinds_it_to_the_new_probe()
    {
        AutoRetainProbe.BeginRun(GraphA);
        var vector = new object();
        var array = new object();
        AutoRetainProbe.Quarantine(GraphA, vector, array, key: 55, () => { });
        AutoRetainProbe.Quarantine(GraphA, vector, array, key: 56, () => { });
        AutoRetainProbe.NoteAccess(vector);
        Assert.True(AutoRetainProbe.IsRetained(56));
        Assert.False(AutoRetainProbe.IsRetained(55));
        GC.KeepAlive(vector);
        GC.KeepAlive(array);
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
