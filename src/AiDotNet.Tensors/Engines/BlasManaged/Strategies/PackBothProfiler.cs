using System.Diagnostics;
using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// #409: opt-in phase profiler for <see cref="PackBothStrategy"/> — splits the
/// GEMM time into pack-B / pack-A / microkernel so the managed-vs-OpenBLAS gap can
/// be attributed. Off by default; <see cref="Enabled"/> is a single bool check on
/// the (few) pack/kernel-block boundaries, so it adds nothing in production and
/// only coarse timestamp deltas when on. Intended for single-thread profiling
/// (run with NumThreads=1), but the counters are accumulated via
/// <see cref="Interlocked"/> so that enabling it on the parallel path still yields
/// correct totals (lost-update-free) rather than silently-wrong timings.
/// </summary>
internal static class PackBothProfiler
{
    /// <summary>Set true by the profiling harness; leave false in production.</summary>
    internal static bool Enabled;

    internal static long PackBTicks;
    internal static long PackATicks;
    internal static long KernelTicks;

    /// <summary>Atomically add elapsed ticks to the pack-B counter.</summary>
    internal static void AddPackB(long ticks) => Interlocked.Add(ref PackBTicks, ticks);
    /// <summary>Atomically add elapsed ticks to the pack-A counter.</summary>
    internal static void AddPackA(long ticks) => Interlocked.Add(ref PackATicks, ticks);
    /// <summary>Atomically add elapsed ticks to the microkernel counter.</summary>
    internal static void AddKernel(long ticks) => Interlocked.Add(ref KernelTicks, ticks);

    internal static void Reset()
    {
        Interlocked.Exchange(ref PackBTicks, 0);
        Interlocked.Exchange(ref PackATicks, 0);
        Interlocked.Exchange(ref KernelTicks, 0);
    }

    private static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    internal static double PackBMs => Ms(PackBTicks);
    internal static double PackAMs => Ms(PackATicks);
    internal static double KernelMs => Ms(KernelTicks);
}
