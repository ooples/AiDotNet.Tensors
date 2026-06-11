using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Cross-platform process-memory metrics for benchmarking. Reports the operating-system
/// resident set size (RSS / working set) of the whole process — the same quantity PyTorch's
/// benchmark harness records via <c>psutil</c> as <c>rss_mb_peak</c> — so .NET and PyTorch
/// memory numbers are directly comparable.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A program's "RSS" (Resident Set Size) is how much physical RAM
/// the operating system currently has assigned to it. PyTorch reports this number. .NET's
/// usual <see cref="GC.GetTotalMemory(bool)"/> reports only the <i>managed heap</i> — a subset
/// of RSS that excludes the runtime, JIT code, native buffers, and unused-but-reserved memory.
/// Comparing managed-heap to PyTorch's RSS understates .NET's footprint; this helper measures
/// the comparable process-wide RSS instead.</para>
/// <para>On Linux, RSS is read from <c>/proc/self/status</c> (<c>VmRSS</c> current, <c>VmHWM</c>
/// peak-since-start). On other platforms it falls back to <see cref="Process.WorkingSet64"/> /
/// <see cref="Process.PeakWorkingSet64"/>.</para>
/// </remarks>
public static class MemoryMetrics
{
    private const double BytesPerMb = 1024.0 * 1024.0;

    /// <summary>Current process resident set size (RSS / working set) in bytes.</summary>
    public static long CurrentProcessRssBytes
    {
        get
        {
            if (TryReadProcStatus("VmRSS:", out long linuxBytes))
                return linuxBytes;
            using var p = Process.GetCurrentProcess();
            return p.WorkingSet64;
        }
    }

    /// <summary>Current process RSS in megabytes (directly comparable to PyTorch <c>rss_mb</c>).</summary>
    public static double CurrentProcessRssMb => CurrentProcessRssBytes / BytesPerMb;

    /// <summary>
    /// Peak process RSS since process start, in bytes. On Linux this is the kernel's
    /// <c>VmHWM</c> high-water mark; elsewhere it is <see cref="Process.PeakWorkingSet64"/>.
    /// Note this is process-lifetime peak and cannot be reset — to measure the peak of a
    /// specific window (one epoch, one phase) use a <see cref="PeakRssSampler"/>.
    /// </summary>
    public static long PeakProcessRssBytes
    {
        get
        {
            if (TryReadProcStatus("VmHWM:", out long linuxBytes))
                return linuxBytes;
            using var p = Process.GetCurrentProcess();
            p.Refresh();
            return p.PeakWorkingSet64;
        }
    }

    /// <summary>Peak process RSS since process start in megabytes (comparable to PyTorch <c>rss_mb_peak</c>).</summary>
    public static double PeakProcessRssMb => PeakProcessRssBytes / BytesPerMb;

    /// <summary>Managed-heap size in bytes (<see cref="GC.GetTotalMemory(bool)"/>). A subset of RSS;
    /// kept for contrast, not for cross-framework comparison.</summary>
    public static long ManagedHeapBytes => GC.GetTotalMemory(forceFullCollection: false);

    /// <summary>Managed-heap size in megabytes. Reported alongside RSS so the managed-vs-RSS
    /// gap is visible, but only <see cref="PeakProcessRssMb"/> is comparable to PyTorch.</summary>
    public static double ManagedHeapMb => ManagedHeapBytes / BytesPerMb;

    // Reads a numeric kB field (e.g. "VmRSS:   12345 kB") from /proc/self/status on Linux.
    private static bool TryReadProcStatus(string key, out long bytes)
    {
        bytes = 0;
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return false;
        try
        {
            foreach (var rawLine in File.ReadLines("/proc/self/status"))
            {
                if (!rawLine.StartsWith(key, StringComparison.Ordinal))
                    continue;
                var rest = rawLine.Substring(key.Length).Trim();
                int space = rest.IndexOf(' ');
                var num = space > 0 ? rest.Substring(0, space) : rest;
                if (long.TryParse(num, NumberStyles.Integer, CultureInfo.InvariantCulture, out long kb))
                {
                    bytes = kb * 1024L;
                    return true;
                }
                return false;
            }
        }
        catch (IOException)
        {
            // /proc unreadable in this sandbox — fall back to the Process API.
        }
        catch (UnauthorizedAccessException)
        {
        }
        return false;
    }

    /// <summary>
    /// Samples current process RSS on a background thread and tracks the maximum, so callers can
    /// measure the peak RSS of a specific window (e.g. one training epoch) rather than the
    /// non-resettable process-lifetime peak. Dispose to stop sampling and read <see cref="PeakMb"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The OS only tracks a single "highest ever" peak for the whole
    /// program's life, which you can't reset between phases. This sampler instead checks the
    /// current RSS many times a second and remembers the largest value it saw while it was
    /// running — mirroring how PyTorch's harness samples RSS during a run.</para>
    /// </remarks>
    public sealed class PeakRssSampler : IDisposable
    {
        private readonly Timer _timer;
        private long _peakBytes;
        private int _disposed;

        /// <summary>Starts sampling RSS every <paramref name="sampleIntervalMs"/> milliseconds.</summary>
        /// <param name="sampleIntervalMs">Sampling period; smaller catches sharper spikes at higher cost. Default 5 ms.</param>
        public PeakRssSampler(int sampleIntervalMs = 5)
        {
            if (sampleIntervalMs < 1) sampleIntervalMs = 1;
            _peakBytes = CurrentProcessRssBytes;
            _timer = new Timer(_ => Sample(), state: null, dueTime: 0, period: sampleIntervalMs);
        }

        private void Sample()
        {
            long now = CurrentProcessRssBytes;
            // Single-writer (the timer's thread pool callback) high-water update; reads are via
            // Volatile in PeakBytes. A torn long is impossible on 64-bit; Interlocked keeps 32-bit safe.
            long current = Interlocked.Read(ref _peakBytes);
            while (now > current)
            {
                long prior = Interlocked.CompareExchange(ref _peakBytes, now, current);
                if (prior == current) break;
                current = prior;
            }
        }

        /// <summary>Highest RSS (bytes) observed since this sampler was created.</summary>
        public long PeakBytes => Interlocked.Read(ref _peakBytes);

        /// <summary>Highest RSS (MB) observed since this sampler was created — comparable to PyTorch <c>rss_mb_peak</c>.</summary>
        public double PeakMb => PeakBytes / BytesPerMb;

        /// <summary>Stops sampling. Takes one final sample so a spike just before disposal is captured.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Sample();
            _timer.Dispose();
        }
    }
}
