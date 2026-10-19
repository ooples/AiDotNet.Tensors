using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The OpenCL flush cadence: right after the host waits on the device, the next command is flushed at once, then the
/// interval doubles to <see cref="DirectOpenClFlushPolicy.MaxCommandsPerFlush"/> and stays there.
/// </summary>
public class DirectOpenClFlushPolicyTests
{
    private static List<int> FlushPoints(int commands)
    {
        var points = new List<int>();
        for (int i = 1; i <= commands; i++)
            if (DirectOpenClFlushPolicy.OnSubmitted())
            {
                points.Add(i);
                DirectOpenClFlushPolicy.OnFlushed();
            }
        return points;
    }

    [Fact]
    public void After_a_host_wait_the_interval_ramps_from_one_to_the_cap()
    {
        DirectOpenClFlushPolicy.OnHostWaited();
        // Intervals 1, 2, 4, 8, 16, 16, 16.
        Assert.Equal(new[] { 1, 3, 7, 15, 31, 47, 63 }, FlushPoints(63));
    }

    [Fact]
    public void A_host_wait_mid_batch_restarts_the_ramp()
    {
        DirectOpenClFlushPolicy.OnHostWaited();
        FlushPoints(20);
        DirectOpenClFlushPolicy.OnHostWaited();
        Assert.Equal(new[] { 1, 3, 7 }, FlushPoints(7));
    }

    [Fact]
    public void A_flush_that_failed_is_due_again_and_does_not_advance_the_ramp()
    {
        DirectOpenClFlushPolicy.OnHostWaited();
        Assert.True(DirectOpenClFlushPolicy.OnSubmitted());
        // clFlush failed: OnFlushed is not called, so the next command is due again at the same interval.
        Assert.True(DirectOpenClFlushPolicy.OnSubmitted());
        DirectOpenClFlushPolicy.OnFlushed();
        Assert.Equal(new[] { 2, 6 }, FlushPoints(6));
    }

    [Fact]
    public void A_thread_that_never_waited_flushes_its_first_command()
    {
        var points = new List<int>();
        var thread = new System.Threading.Thread(() => points.AddRange(FlushPoints(3)));
        thread.Start();
        thread.Join();
        Assert.Equal(new[] { 1, 3 }, points);
    }
}
