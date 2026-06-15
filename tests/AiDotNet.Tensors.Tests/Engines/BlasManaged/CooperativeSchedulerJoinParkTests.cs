using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.BlasManaged.Pool;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Guards the spin-then-park join in <see cref="CooperativeGemmScheduler"/> (issue
/// #589). When the participating caller drains the queue but its job's last chunks
/// are still running on other threads, it now PARKS on the job's done event after a
/// bounded spin instead of busy-spinning a core (which stole ~47% of a core from a
/// conv kernel on the parallel path). These tests force that park path — chunks that
/// out-live the caller's spin budget — and verify correctness with no lost wakeup or
/// hang.
/// </summary>
public class CooperativeSchedulerJoinParkTests
{
    [Fact]
    public void Dispatch_LongChunks_ForceCallerPark_CompleteExactlyOnce()
    {
        // Each chunk sleeps well past the caller's spin budget, so after the caller
        // has drained what it can it must park and be woken by the thread that
        // finishes the last chunk. A lost wakeup would hang here (caught by Timeout).
        const int chunks = 16;
        var counts = new int[chunks];

        var done = Task.Run(() =>
            CooperativeGemmScheduler.Dispatch(chunks, c =>
            {
                Thread.Sleep(15); // >> CoopJoinSpinBeforePark (~µs) → caller parks
                Interlocked.Increment(ref counts[c]);
            }));

        Assert.True(done.Wait(TimeSeconds(30)), "Dispatch hung — likely a lost wakeup in the join-park path");
        for (int c = 0; c < chunks; c++)
            Assert.Equal(1, counts[c]);
    }

    [Fact]
    public void Dispatch_RepeatedParkWake_NoHang_NoDoubleRun()
    {
        // Hammer the park/wake transition across many dispatches to surface any
        // lost-wakeup or double-execution race in the ParkPending handshake.
        var sw = Stopwatch.StartNew();
        for (int iter = 0; iter < 200 && sw.Elapsed < TimeSeconds(60); iter++)
        {
            const int chunks = 8;
            var counts = new int[chunks];
            var done = Task.Run(() =>
                CooperativeGemmScheduler.Dispatch(chunks, c =>
                {
                    // Mix: some chunks fast, some slow, so the caller sometimes drains
                    // them all (no park) and sometimes must park for the slow tail.
                    if ((c & 1) == 0) Thread.SpinWait(50);
                    else Thread.Sleep(3);
                    Interlocked.Increment(ref counts[c]);
                }));
            Assert.True(done.Wait(TimeSeconds(20)), $"Dispatch hung at iteration {iter}");
            for (int c = 0; c < chunks; c++)
                Assert.Equal(1, counts[c]);
        }
    }

    private static TimeSpan TimeSeconds(int s) => TimeSpan.FromSeconds(s);
}
