using System;
using System.Diagnostics;
using System.Threading;
using AiDotNet.Tensors.Engines.BlasManaged.Pool;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Issue #589 regression guard: idle StreamingWorkerPool workers must PARK between
/// dispatches rather than busy-spin. The old loop spent iterations 1000-5000 in
/// Thread.Yield(), which under a co-running workload keeps the idle worker runnable and
/// oversubscribes the cores the real work needs (~47% of busy CPU wasted during conv-
/// dominated diffusion sampling, a 55-minute CI shard hang).
///
/// The test measures the EXTRA process CPU the pool's idle workers add across a series of
/// (dispatch → single-threaded busy gap) cycles, versus the busy gaps alone. A parking
/// pool adds only µs-scale wake overhead; a spinning pool adds ~cores × gap time. Marked
/// Performance (CI excludes it) because it reads wall/CPU timing; run manually to verify.
/// </summary>
[Collection("BlasManaged-Perf-Serial")]
[Trait("Category", "Performance")]
public class StreamingWorkerPoolIdleSpinTests
{
    private readonly ITestOutputHelper _o;
    public StreamingWorkerPoolIdleSpinTests(ITestOutputHelper o) => _o = o;

    // Deterministic single-threaded CPU work (the stand-in for the "conv on the other
    // parallel path" that the idle GEMM-pool spinners were stealing cycles from).
    private static long BusyWork(int iterations)
    {
        long s = 1;
        for (int i = 0; i < iterations; i++)
            s = unchecked(s * 6364136223846793005L + 1442695040888963407L) ^ (s >> 7);
        return s;
    }

    [Fact]
    public void IdleWorkers_ParkBetweenDispatches_DoNotOversubscribe()
    {
        int chunks = Math.Max(2, Environment.ProcessorCount);
        Action<int> tiny = _ => { };
        const int cycles = 150;
        const int gapIters = 3_000_000;

        // Warm up: initialize the pool + JIT both paths, then let workers park.
        long sink = 0;
        for (int i = 0; i < 30; i++) { StreamingWorkerPool.Dispatch(chunks, 1 << 20, tiny); sink += BusyWork(gapIters); }
        Thread.Sleep(150);

        var proc = Process.GetCurrentProcess();

        // Baseline: busy gaps only. The pool gets no dispatches, so its workers stay parked.
        var t0 = proc.TotalProcessorTime;
        for (int i = 0; i < cycles; i++) sink += BusyWork(gapIters);
        double baseline = (proc.TotalProcessorTime - t0).TotalMilliseconds;

        // With pool: each gap is preceded by a dispatch that wakes the workers; they then
        // go idle for the duration of the single-threaded gap. Parked workers add ≈0 CPU;
        // workers spinning across all cores add roughly cores × gap time.
        var t1 = proc.TotalProcessorTime;
        for (int i = 0; i < cycles; i++) { StreamingWorkerPool.Dispatch(chunks, 1 << 20, tiny); sink += BusyWork(gapIters); }
        double withPool = (proc.TotalProcessorTime - t1).TotalMilliseconds;

        double excess = withPool - baseline;
        _o.WriteLine($"cores={Environment.ProcessorCount} baseline={baseline:F0}ms withPool={withPool:F0}ms excess={excess:F0}ms sink={sink}");

        // The idle pool must add LESS than the real work itself. A spinning pool adds many
        // multiples of `baseline` (one busy core of real work vs ~cores of idle spinning).
        Assert.True(excess < baseline,
            $"idle StreamingWorkerPool workers oversubscribed (#589): baseline {baseline:F0}ms vs withPool {withPool:F0}ms " +
            $"(excess {excess:F0}ms must be < {baseline:F0}ms). Workers spinning instead of parking?");
    }
}
