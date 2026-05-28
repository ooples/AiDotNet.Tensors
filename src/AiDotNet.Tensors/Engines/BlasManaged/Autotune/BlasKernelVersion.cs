namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Version stamp tagging every persisted autotune entry (#375 G2/G8). A cache entry
/// measured under a different kernel version is ignored on read, so a kernel/strategy/
/// blocking change can't serve a stale tuning. Combines the assembly informational
/// version (auto-bumped per build/release) with a manually-tracked kernel epoch.
///
/// <para>
/// The epoch is the one human-owned knob — bump it whenever a strategy/microkernel/
/// blocking change should invalidate learned tunings. Combining it with the assembly
/// version means even a forgotten epoch bump still invalidates across releases (so the
/// staleness window is bounded to a single dev build). A future source-generator can
/// replace the epoch with a content hash of the kernel sources for full automation.
/// </para>
/// </summary>
internal static class BlasKernelVersion
{
    /// <summary>Bump when a kernel/strategy/blocking change should invalidate learned tunings.</summary>
    private const int KernelEpoch = 1;

    private static readonly string _current =
        $"{typeof(BlasKernelVersion).Assembly.GetName().Version}-k{KernelEpoch}";

    /// <summary>The current kernel-version token (stable for the process lifetime).</summary>
    public static string Current => _current;
}
