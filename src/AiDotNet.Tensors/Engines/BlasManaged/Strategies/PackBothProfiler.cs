using System.Diagnostics;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// #409: opt-in phase profiler for <see cref="PackBothStrategy"/> — splits the
/// serial GEMM time into pack-B / pack-A / microkernel so the managed-vs-OpenBLAS
/// gap can be attributed. Off by default; <see cref="Enabled"/> is a single bool
/// check on the (few) pack/kernel-block boundaries, so it adds nothing in
/// production and only coarse timestamp deltas when on. Single-thread profiling
/// (run with NumThreads=1) so no cross-thread aggregation is needed.
/// </summary>
internal static class PackBothProfiler
{
    /// <summary>Set true by the profiling harness; leave false in production.</summary>
    internal static bool Enabled;

    internal static long PackBTicks;
    internal static long PackATicks;
    internal static long KernelTicks;

    internal static void Reset()
    {
        PackBTicks = 0;
        PackATicks = 0;
        KernelTicks = 0;
    }

    private static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    internal static double PackBMs => Ms(PackBTicks);
    internal static double PackAMs => Ms(PackATicks);
    internal static double KernelMs => Ms(KernelTicks);
}
