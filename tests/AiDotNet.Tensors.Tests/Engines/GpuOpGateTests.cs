// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The lease gate a GPU engine's Dispose drains: no device needed, so these run on every lane.
/// </summary>
public class GpuOpGateTests
{
    private static readonly TimeSpan Long = TimeSpan.FromSeconds(30);

    [Fact]
    public void A_drain_waits_for_a_lease_held_on_another_thread()
    {
        var gate = new GpuOpGate();
        using var held = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var holder = Task.Run(() =>
        {
            using var lease = gate.Enter();
            Assert.True(lease.Held);
            held.Set();
            release.Wait();
        });
        held.Wait();

        var drain = Task.Run(() => gate.CloseAndDrain(Long));
        Assert.False(drain.Wait(200), "the drain finished while another thread still held a lease");
        release.Set();
        Assert.True(drain.Wait(Long));
        Assert.True(drain.Result);
        holder.Wait();
    }

    [Fact]
    public void A_closed_gate_grants_no_lease()
    {
        var gate = new GpuOpGate();
        Assert.True(gate.CloseAndDrain(Long));
        Assert.False(gate.IsOpen);
        using var lease = gate.Enter();
        Assert.False(lease.Held);
        Assert.False(gate.HeldByCurrentThread);
    }

    [Fact]
    public void A_drain_that_times_out_returns_false_and_the_gate_reopens()
    {
        var gate = new GpuOpGate();
        using var held = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var holder = Task.Run(() =>
        {
            using var lease = gate.Enter();
            held.Set();
            release.Wait();
        });
        held.Wait();

        Assert.False(gate.CloseAndDrain(TimeSpan.FromMilliseconds(50)));
        gate.Reopen();
        using (var lease = gate.Enter()) Assert.True(lease.Held);
        release.Set();
        holder.Wait();
    }

    // The closing thread cannot return its own leases while it waits, so they are not waited for.
    [Fact]
    public void A_drain_does_not_wait_for_the_closing_threads_own_lease()
    {
        var gate = new GpuOpGate();
        using var lease = gate.Enter();
        Assert.True(gate.HeldByCurrentThread);
        Assert.True(gate.CloseAndDrain(TimeSpan.FromSeconds(5)));
    }

    [Fact]
    public void Nested_leases_are_counted_until_the_outermost_is_returned()
    {
        var gate = new GpuOpGate();
        var outer = gate.Enter();
        var inner = gate.Enter();
        inner.Dispose();
        Assert.True(gate.HeldByCurrentThread);
        outer.Dispose();
        Assert.False(gate.HeldByCurrentThread);
    }

    // An async op can finish on another thread: its lease is still returned to the thread that took it.
    [Fact]
    public async Task A_lease_returned_on_another_thread_releases_the_drain()
    {
        var gate = new GpuOpGate();
        var lease = await Task.Run(() => gate.Enter());
        Assert.True(lease.Held);
        await Task.Run(() => lease.Dispose());
        Assert.True(gate.CloseAndDrain(TimeSpan.FromSeconds(5)));
    }

    // Many threads taking and returning leases while one closes: the drain completes, and after it no lease is held.
    [Fact]
    public void A_close_under_contention_completes_and_admits_no_new_lease()
    {
        var gate = new GpuOpGate();
        using var stop = new CancellationTokenSource();
        int heldAfterClose = 0;
        var workers = new Task[Environment.ProcessorCount];
        for (int w = 0; w < workers.Length; w++)
        {
            workers[w] = Task.Run(() =>
            {
                while (!stop.IsCancellationRequested)
                {
                    using var lease = gate.Enter();
                    if (lease.Held && !gate.IsOpen) Interlocked.Increment(ref heldAfterClose);
                    Thread.SpinWait(20);
                }
            });
        }
        Thread.Sleep(50);
        Assert.True(gate.CloseAndDrain(Long));
        // A lease granted before the close may still see it closed while it finishes; none may start after the drain.
        int afterDrain = Volatile.Read(ref heldAfterClose);
        Thread.Sleep(50);
        Assert.Equal(afterDrain, Volatile.Read(ref heldAfterClose));
        stop.Cancel();
        Task.WaitAll(workers);
    }
}
