using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Production GPU device-memory tracker and leak detector. A built-in, zero-dependency alternative to an
/// external tool like <c>compute-sanitizer</c> for answering "what is holding GPU memory right now?".
/// </summary>
/// <remarks>
/// <para>
/// Records every <b>device</b> allocation (the actual <c>cuMemAlloc</c> / <c>cuMemAllocAsync</c>, not the
/// software buffer-pool rent/return layer above it) together with its byte size and the managed call stack
/// at the allocation site, and clears it on the matching free. <see cref="Report"/> returns the LIVE set
/// grouped by call site and sorted by bytes — i.e. precisely the retention/leak breakdown a memcheck would
/// surface, but in-process and on any driver (no toolkit install required).
/// </para>
/// <para>
/// <b>Opt-in and zero-cost when off.</b> All hooks early-out on <see cref="Enabled"/> (env
/// <c>AIDOTNET_GPU_MEMTRACK=1</c>) before doing any work, so a production build with tracking disabled pays
/// only a single boolean check per allocation. When enabled, each allocation additionally captures a short
/// managed stack trace (a few frames) — cheap relative to a device allocation but not free, hence opt-in.
/// </para>
/// <para>
/// Thread-safe: the live map is a <see cref="ConcurrentDictionary{TKey,TValue}"/> and the running counters
/// are updated with interlocked operations, so it is safe under the multi-stream / BLAS-pool concurrency the
/// GPU backends use.
/// </para>
/// </remarks>
public static class GpuMemoryTracker
{
    /// <summary>True when tracking is enabled (env <c>AIDOTNET_GPU_MEMTRACK=1</c>). Checked first on every hook.</summary>
    public static readonly bool Enabled =
        Environment.GetEnvironmentVariable("AIDOTNET_GPU_MEMTRACK") == "1";

    private sealed class Rec
    {
        public long Bytes;
        public string Site = "";
        public long Seq;
    }

    private static readonly ConcurrentDictionary<long, Rec> s_live = new();
    private static long s_seq;
    private static long s_liveBytes;
    private static long s_peakBytes;
    private static long s_totalAllocs;
    private static long s_totalFrees;

    // Capture-trace: log every device allocation that fires INSIDE a CUDA-graph capture window. Such an
    // allocation is a graph-purity violation — it either aborts capture (CUDA 906 STREAM_CAPTURE_UNSUPPORTED)
    // or pins device pointers the replay can't honour — so listing the offending ops (by call site) is exactly
    // what's needed to make a training step graph-capturable. Opt-in via AIDOTNET_GPU_CAPTURE_TRACE=1 and
    // independent of the full tracker (Enabled): the capture window is one step, so the per-alloc stack capture
    // here is bounded even with the heavyweight tracker off.
    public static readonly bool CaptureTrace =
        Environment.GetEnvironmentVariable("AIDOTNET_GPU_CAPTURE_TRACE") == "1";
    private static int s_capturing;
    private static readonly System.Collections.Generic.List<string> s_captureAllocs = new();
    private static readonly object s_captureLock = new();

    // Download-trace (PR #638 Phase A0): attribute every device→host copy (cuMemcpyDtoH) to the op that
    // triggered it, ranked by frequency. A DtoH inside a CUDA-graph capture window aborts the capture, so the
    // backward ops that still download are exactly the work-list for making the backward capture-resident. Run
    // EAGER (AIDOTNET_CUDA_GRAPH_STEP=0) so the copies are legal and ALL of them are observed in one pass (a
    // captured run would abort on the first). Opt-in via AIDOTNET_GPU_DOWNLOAD_TRACE=1; armed only inside a
    // BeginDownloadTrace window so the forward's legitimate scalar-loss download isn't attributed to backward.
    public static readonly bool DownloadTrace =
        Environment.GetEnvironmentVariable("AIDOTNET_GPU_DOWNLOAD_TRACE") == "1";
    private static int s_dlTracing;
    private static readonly Dictionary<string, (long count, long bytes)> s_dlSites = new();
    private static readonly object s_dlLock = new();
    private static int s_dlDumped;

    /// <summary>Bytes currently allocated on the device and not yet freed (sum over the live set).</summary>
    public static long LiveBytes => Interlocked.Read(ref s_liveBytes);

    /// <summary>The high-water mark of <see cref="LiveBytes"/> observed since process start.</summary>
    public static long PeakBytes => Interlocked.Read(ref s_peakBytes);

    /// <summary>
    /// Reset the peak watermark down to the CURRENT live bytes, so a subsequent <see cref="PeakBytes"/> read
    /// reflects the peak of just the next measurement window (e.g. one forward). Used to A/B the device-memory
    /// footprint of two configurations (FP32 vs FP16 activation storage) within one process. No-op when off.
    /// </summary>
    public static void ResetPeak()
    {
        if (!Enabled) return;
        Interlocked.Exchange(ref s_peakBytes, Interlocked.Read(ref s_liveBytes));
    }

    /// <summary>Number of live (un-freed) device allocations.</summary>
    public static int LiveCount => s_live.Count;

    /// <summary>
    /// Begins a CUDA-graph capture window: every <see cref="OnAlloc"/> until the returned scope is disposed is
    /// recorded as a capture-time allocation (a graph-purity violation). On dispose the collected list is
    /// appended to the capture-trace file. No-op unless <see cref="CaptureTrace"/>. Re-entrant-safe.
    /// </summary>
    public static IDisposable BeginCapture(string label)
    {
        if (!CaptureTrace) return NoopScope.Instance;
        Interlocked.Increment(ref s_capturing);
        return new CaptureScope(label);
    }

    private sealed class NoopScope : IDisposable { public static readonly NoopScope Instance = new(); public void Dispose() { } }

    private sealed class CaptureScope : IDisposable
    {
        private readonly string _label;
        public CaptureScope(string label) => _label = label;
        public void Dispose()
        {
            Interlocked.Decrement(ref s_capturing);
            string[] snapshot;
            lock (s_captureLock) { snapshot = s_captureAllocs.ToArray(); s_captureAllocs.Clear(); }
            try
            {
                string path = Environment.GetEnvironmentVariable("AIDOTNET_GPU_MEMTRACK_FILE")
                              ?? System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_capture_allocs.txt");
                var sb = new StringBuilder();
                sb.AppendLine($"==== capture window [{_label}] — {snapshot.Length} device alloc(s) DURING capture (graph-purity violations) ====");
                foreach (var s in snapshot) sb.AppendLine("  " + s);
                System.IO.File.AppendAllText(path, sb.ToString());
            }
            catch { /* diagnostics must never destabilize the caller */ }
        }
    }

    /// <summary>
    /// Begins a download-trace window (PR #638 A0): every <see cref="OnDownload"/> until the returned scope is
    /// disposed is attributed (by managed call site) to the op that issued the DtoH copy. Use it to wrap the
    /// backward pass of one eager step so the ranked list names exactly the backward ops still downloading
    /// gradient intermediates. No-op unless <see cref="DownloadTrace"/>. Re-entrant-safe.
    /// </summary>
    public static IDisposable BeginDownloadTrace()
    {
        if (!DownloadTrace) return NoopScope.Instance;
        Interlocked.Increment(ref s_dlTracing);
        return new DownloadScope();
    }

    private sealed class DownloadScope : IDisposable
    {
        public void Dispose() => Interlocked.Decrement(ref s_dlTracing);
    }

    /// <summary>Records a device→host copy. Call from the backend's DtoH primitive. Cheap no-op when the trace is
    /// off or no <see cref="BeginDownloadTrace"/> window is active.</summary>
    public static void OnDownload(long bytes)
    {
        if (!DownloadTrace || Volatile.Read(ref s_dlTracing) <= 0) return;
        string site = CaptureSite();
        lock (s_dlLock)
        {
            s_dlSites.TryGetValue(site, out var cur);
            s_dlSites[site] = (cur.count + 1, cur.bytes + bytes);
        }
    }

    /// <summary>Writes the download-trace sites ranked by call count (then bytes) to a file, then clears the
    /// accumulator. One-shot per process by default (the op set repeats every step), so call it after the FIRST
    /// eager backward. No-op when the trace is off or already dumped.</summary>
    public static void DumpDownloadTrace(string label)
    {
        if (!DownloadTrace) return;
        if (Interlocked.Exchange(ref s_dlDumped, 1) != 0) return;
        List<KeyValuePair<string, (long count, long bytes)>> sorted;
        lock (s_dlLock)
        {
            sorted = new List<KeyValuePair<string, (long count, long bytes)>>(s_dlSites);
            s_dlSites.Clear();
        }
        sorted.Sort((a, b) => a.Value.count != b.Value.count
            ? b.Value.count.CompareTo(a.Value.count)
            : b.Value.bytes.CompareTo(a.Value.bytes));
        try
        {
            string path = Environment.GetEnvironmentVariable("AIDOTNET_GPU_DOWNLOAD_TRACE_FILE")
                          ?? System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_download_trace.txt");
            var sb = new StringBuilder();
            long totalCount = 0, totalBytes = 0;
            foreach (var kv in sorted) { totalCount += kv.Value.count; totalBytes += kv.Value.bytes; }
            sb.AppendLine($"==== download trace [{label}] — {sorted.Count} site(s), {totalCount} DtoH copies, " +
                          $"{totalBytes / 1048576.0:F1} MB total. Ranked by copy count (the backward-residency work-list): ====");
            foreach (var kv in sorted)
                sb.AppendLine($"  x{kv.Value.count,-5} {kv.Value.bytes / 1024.0,10:F1} KB  {kv.Key}");
            System.IO.File.AppendAllText(path, sb.ToString());
        }
        catch { /* diagnostics must never destabilize the caller */ }
    }

    /// <summary>Records a device allocation. Call right after a successful <c>cuMemAlloc</c>/<c>cuMemAllocAsync</c>.</summary>
    public static void OnAlloc(IntPtr ptr, long bytes)
    {
        if (ptr == IntPtr.Zero || bytes <= 0) return;
        if (CaptureTrace && Volatile.Read(ref s_capturing) > 0)
        {
            string site = CaptureSite();
            lock (s_captureLock) s_captureAllocs.Add($"{bytes / 1024.0,10:F1} KB  {site}");
        }
        if (!Enabled) return;
        var rec = new Rec { Bytes = bytes, Site = CaptureSite(), Seq = Interlocked.Increment(ref s_seq) };
        long key = (long)ptr;
        long delta;
        while (true)
        {
            if (s_live.TryAdd(key, rec))
            {
                delta = bytes;
                break;
            }
            if (s_live.TryGetValue(key, out var prior) && s_live.TryUpdate(key, rec, prior))
            {
                delta = bytes - prior.Bytes;
                break;
            }
        }
        Interlocked.Increment(ref s_totalAllocs);
        long lb = Interlocked.Add(ref s_liveBytes, delta);
        // Best-effort peak update (CAS loop; a benign miss under contention only undercounts the peak slightly).
        long peak;
        while (lb > (peak = Interlocked.Read(ref s_peakBytes)))
            if (Interlocked.CompareExchange(ref s_peakBytes, lb, peak) == peak) break;
        // Peak-composition dump (AIDOTNET_GPU_MEMTRACK_PEAKDUMP=1): each time the live high-water grows by a
        // big margin, append the current live-set Report so the PEAK's allocation breakdown is captured (the
        // post-run Report only shows the small residual). Throttled to big jumps so it doesn't spam.
        if (s_peakDump && lb > Interlocked.Read(ref s_lastPeakDump) + PeakDumpStepBytes)
        {
            Interlocked.Exchange(ref s_lastPeakDump, lb);
            Dump($"peak-grow {lb / (double)BytesPerMiB:F1} MB");
        }
    }

    private const long BytesPerMiB = 1024L * 1024L;
    // Only re-dump the peak composition once the live high-water grows by this
    // much, so the throttled dump doesn't spam on every small allocation.
    private const long PeakDumpStepBytes = 64L * BytesPerMiB;

    private static readonly bool s_peakDump =
        Environment.GetEnvironmentVariable("AIDOTNET_GPU_MEMTRACK_PEAKDUMP") == "1";
    private static long s_lastPeakDump;

    /// <summary>Clears a device allocation. Call right before the matching <c>cuMemFree</c>/<c>cuMemFreeAsync</c>.</summary>
    public static void OnFree(IntPtr ptr)
    {
        if (!Enabled || ptr == IntPtr.Zero) return;
        if (s_live.TryRemove((long)ptr, out var rec))
        {
            Interlocked.Add(ref s_liveBytes, -rec.Bytes);
            Interlocked.Increment(ref s_totalFrees);
        }
    }

    private static string CaptureSite()
    {
        // Skip the tracker frame + the immediate backend alloc wrapper; show the next several callers so the
        // owning op (e.g. FusedLinear / StepEager / ConfigureOptimizer) is visible without the noise.
        var st = new StackTrace(2, false);
        var sb = new StringBuilder();
        int shown = 0;
        for (int i = 0; i < st.FrameCount && shown < 7; i++)
        {
            var m = st.GetFrame(i)?.GetMethod();
            if (m is null) continue;
            string name = (m.DeclaringType?.Name ?? "?") + "." + m.Name;
            if (name.Contains("GpuMemoryTracker")) continue;
            if (shown > 0) sb.Append(" < ");
            sb.Append(name);
            shown++;
        }
        return sb.Length == 0 ? "(no managed frames)" : sb.ToString();
    }

    /// <summary>
    /// A human-readable snapshot of the LIVE device allocations, grouped by call site and sorted by bytes
    /// (largest first). The header line carries totals (live bytes/count, peak, lifetime alloc/free counts).
    /// Returns a short "disabled" note when <see cref="Enabled"/> is false.
    /// </summary>
    public static string Report(int topN = 40)
    {
        if (!Enabled) return "[GpuMemoryTracker] disabled (set AIDOTNET_GPU_MEMTRACK=1)";
        var groups = new Dictionary<string, (long bytes, int count)>();
        foreach (var kv in s_live)
        {
            groups.TryGetValue(kv.Value.Site, out var cur);
            groups[kv.Value.Site] = (cur.bytes + kv.Value.Bytes, cur.count + 1);
        }
        var sorted = new List<KeyValuePair<string, (long bytes, int count)>>(groups);
        sorted.Sort((a, b) => b.Value.bytes.CompareTo(a.Value.bytes));

        var sb = new StringBuilder();
        sb.AppendLine($"[GpuMemoryTracker] live={LiveBytes / 1048576.0:F1} MB across {LiveCount} allocs " +
                      $"(peak={PeakBytes / 1048576.0:F1} MB; lifetime allocs={Interlocked.Read(ref s_totalAllocs)}, " +
                      $"frees={Interlocked.Read(ref s_totalFrees)}). Top {Math.Min(topN, sorted.Count)} sites by live bytes:");
        for (int i = 0; i < sorted.Count && i < topN; i++)
            sb.AppendLine($"  {sorted[i].Value.bytes / 1048576.0,9:F1} MB  x{sorted[i].Value.count,-5}  {sorted[i].Key}");
        return sb.ToString();
    }

    /// <summary>
    /// Appends <see cref="Report"/> to a file (default: <c>%TEMP%/aidotnet_gpumem.txt</c>) with a label.
    /// Called by the backends right before they surface an out-of-memory failure so the OOM is accompanied by
    /// the live-retention breakdown. No-op when disabled. Never throws (best-effort diagnostics).
    /// </summary>
    public static void Dump(string label)
    {
        if (!Enabled) return;
        try
        {
            string path = Environment.GetEnvironmentVariable("AIDOTNET_GPU_MEMTRACK_FILE")
                          ?? System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_gpumem.txt");
            System.IO.File.AppendAllText(path, $"==== {label} ====" + Environment.NewLine + Report() + Environment.NewLine);
        }
        catch { /* diagnostics must never destabilize the caller */ }
    }
}
