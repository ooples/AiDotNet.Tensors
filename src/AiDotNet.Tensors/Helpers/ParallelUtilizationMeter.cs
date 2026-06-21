using System;
using System.Diagnostics;
using System.Threading;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// #653 Phase 0 diagnostic: a sampling meter for cooperative-pool core utilization.
/// <para>
/// While active it samples <c>CooperativeGemmScheduler.ActiveWorkers</c> (the number of
/// fanned-out chunks executing concurrently) at a fixed interval and reports the mean and
/// peak. This directly measures the issue #642 symptom — a near-serial forward leaves the
/// worker pool parked, so <see cref="MeanActiveWorkers"/> reads ~0; a saturated workload
/// reads ~<c>MaxDegreeOfParallelism</c>. Pair it with the probe's speedup/N efficiency
/// (which captures serial orchestrator time) for a full picture.
/// </para>
/// <para>
/// The meter is opt-in: constructing it flips the scheduler's <c>MeasureUtilization</c>
/// flag on (so the per-chunk Interlocked pair is only paid during measurement) and
/// <see cref="Dispose"/> flips it back off. The sampler runs on a dedicated background
/// thread; on a fully saturated box that thread competes for one core, so treat the mean
/// as a close lower bound rather than an exact figure.
/// </para>
/// </summary>
public sealed class ParallelUtilizationMeter : IDisposable
{
    // Ref-counts active meters so overlapping windows don't disable each other's measurement and a
    // constructor failure can't leave the global gauge stuck on. Only the 0→1 transition enables the
    // scheduler gauge; only the →0 transition disables it.
    private static int s_meterRefCount;

    private readonly Thread _sampler;
    private readonly int _sleepMs;        // sampler sleep between samples, derived from sampleMicros
    private volatile bool _running;
    private long _sum;       // sum of per-sample active-worker counts (sampler-thread only)
    private long _count;     // number of samples taken
    private int _peak;       // max active-worker count observed
    private bool _disposed;

    // Pool-agnostic CPU-time accounting: parked worker threads consume ~0 CPU, so
    // (process CPU-time delta / wall) is the true mean number of busy cores across ALL
    // parallel pools (cooperative GEMM scheduler, streaming/GEMM pool, Parallel.For, ...),
    // not just the one the gauge instruments. This is the headline utilization figure.
    private readonly Stopwatch _wall;
    private readonly TimeSpan _cpuAtStart;
    private TimeSpan _cpuAtStop;

    private ParallelUtilizationMeter(int sampleMicros)
    {
        if (sampleMicros < 1) sampleMicros = 1;
        // Thread.Sleep granularity is milliseconds, so the sub-ms request is rounded up to >= 1 ms.
        _sleepMs = Math.Max(1, sampleMicros / 1000);

        // Enable the scheduler gauge only on the first concurrent meter (ref-counted).
        if (Interlocked.Increment(ref s_meterRefCount) == 1)
        {
            Engines.BlasManaged.Pool.CooperativeGemmScheduler.ResetUtilization();
            Engines.BlasManaged.Pool.CooperativeGemmScheduler.MeasureUtilization = true;
        }

        // If anything below throws, the ref-count must be undone so the gauge isn't stuck on.
        try
        {
            _cpuAtStart = Process.GetCurrentProcess().TotalProcessorTime;
            _wall = Stopwatch.StartNew();

            _running = true;
            _sampler = new Thread(SampleLoop)
            {
                IsBackground = true,
                Name = "AiDotNet-UtilSampler",
                Priority = ThreadPriority.AboveNormal,
            };
            _sampler.Start();
        }
        catch
        {
            if (Interlocked.Decrement(ref s_meterRefCount) == 0)
                Engines.BlasManaged.Pool.CooperativeGemmScheduler.MeasureUtilization = false;
            throw;
        }
    }

    /// <summary>Start a new utilization measurement window.</summary>
    /// <param name="sampleMicros">Target sampling interval in microseconds (default 200).</param>
    public static ParallelUtilizationMeter Start(int sampleMicros = 200) => new(sampleMicros);

    private void SampleLoop()
    {
        // Sleep (not busy-spin) between samples: a spin loop would itself burn ~1 core and
        // inflate the process-CPU-time headline by ~1. The interval honors the configured
        // sampleMicros (rounded up to Thread.Sleep's ms granularity) and wakes often enough
        // (hundreds of samples over a typical multi-hundred-ms window) to give a representative
        // mean of the cooperative-pool gauge while consuming ~0 CPU.
        while (_running)
        {
            int a = Engines.BlasManaged.Pool.CooperativeGemmScheduler.ActiveWorkers;
            _sum += a;
            _count++;
            if (a > _peak) _peak = a;
            Thread.Sleep(_sleepMs);
        }
    }

    /// <summary>Mean number of pool chunks executing concurrently over the window.</summary>
    public double MeanActiveWorkers => Volatile.Read(ref _count) == 0
        ? 0.0
        : (double)Volatile.Read(ref _sum) / Volatile.Read(ref _count);

    /// <summary>Peak number of pool chunks observed executing concurrently.</summary>
    public int PeakActiveWorkers => Volatile.Read(ref _peak);

    /// <summary>Number of samples taken (sanity check that the sampler ran).</summary>
    public long SampleCount => Volatile.Read(ref _count);

    /// <summary>
    /// Mean number of busy CPU cores over the window (process CPU-time delta / wall-clock).
    /// Pool-agnostic: captures parallelism through every pool, not just the cooperative one.
    /// Subtract ~1 sampler core when interpreting near-saturation figures.
    /// </summary>
    public double MeanBusyCores
    {
        get
        {
            double wallSec = _wall.Elapsed.TotalSeconds;
            if (wallSec <= 0) return 0.0;
            double cpuSec = (_cpuAtStop == default ? _cpuAtStart : _cpuAtStop).Subtract(_cpuAtStart).TotalSeconds;
            return cpuSec / wallSec;
        }
    }

    /// <summary>Stop sampling and disengage the scheduler gauge.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _running = false;
        _sampler.Join();
        _wall.Stop();
        _cpuAtStop = Process.GetCurrentProcess().TotalProcessorTime;
        // Disable the gauge only when the last concurrent meter disposes (ref-counted).
        if (Interlocked.Decrement(ref s_meterRefCount) == 0)
            Engines.BlasManaged.Pool.CooperativeGemmScheduler.MeasureUtilization = false;
    }
}
