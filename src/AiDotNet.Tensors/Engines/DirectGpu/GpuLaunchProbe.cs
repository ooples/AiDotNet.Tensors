using System.Diagnostics;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Process-wide GPU work counter. Its only purpose is to let the op-parity backend-completeness
/// guard EMPIRICALLY tell which ops actually execute on the GPU from those that silently fall back
/// to the CPU: every real GPU dispatch — a compute kernel OR a device-to-device buffer copy (the
/// resident data movement behind concat/narrow/gather) — calls <see cref="OnLaunch"/>. Host I/O
/// (write/read buffer) is deliberately NOT counted, since a CPU fallback op that never touches the
/// device makes zero of ALL these calls. A test can <see cref="Reset"/> the counter, run one op on
/// the GPU engine, and read <see cref="Count"/> back — an op that does ZERO GPU work ran entirely on
/// the host, a genuine coverage gap no amount of reflection guesswork can hide.
/// </summary>
/// <remarks>
/// Deliberately backend-agnostic: each GPU backend increments it at its single kernel-dispatch choke
/// point (OpenCL: <c>DirectOpenClKernel.Execute*</c>). The cost is one interlocked add per launch,
/// dwarfed by the launch itself, and the counter is inert unless a test reads it. It is NOT a
/// correctness or scheduling signal — do not gate kernel behaviour on it.
/// </remarks>
internal static class GpuLaunchProbe
{
    private static long _count;
    private static long _totalEver;   // DIAGNOSTIC: never reset, distinguishes "no launch" from "count zeroed"
    private static long _kernelMisses;
    private static long _readbacks;
    private static long _readbackBytes;
    private static long _uploads;
    private static long _uploadBytes;
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> _uploadSites = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, long> _uploadSiteBytes = new();
    private static int _captureReadbackSites;
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> _missedNames = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> _readbackSites = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, long> _readbackSiteBytes = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> _fallbacks = new();

    /// <summary>Total kernel launches observed since the last <see cref="Reset"/> (lock-free read).</summary>
    public static long Count => Interlocked.Read(ref _count);

    /// <summary>Called once per GPU kernel dispatch. Cheap; safe to call from any thread.</summary>
    public static void OnLaunch() { Interlocked.Increment(ref _count); Interlocked.Increment(ref _totalEver); }

    /// <summary>DIAGNOSTIC: launches since process start; <see cref="Reset"/> does NOT clear it.</summary>
    public static long TotalEver => Interlocked.Read(ref _totalEver);

    /// <summary>Device-to-host transfers observed since the last <see cref="Reset"/>.</summary>
    public static long Readbacks => Interlocked.Read(ref _readbacks);

    /// <summary>Total bytes transferred device-to-host since the last <see cref="Reset"/>.</summary>
    public static long ReadbackBytes => Interlocked.Read(ref _readbackBytes);

    /// <summary>Enables call-site collection for residency diagnostics. Disabled outside targeted tests.</summary>
    public static bool CaptureReadbackSites
    {
        get => Volatile.Read(ref _captureReadbackSites) != 0;
        set => Volatile.Write(ref _captureReadbackSites, value ? 1 : 0);
    }

    /// <summary>Distinct readback call sites and their invocation counts since the last reset.</summary>
    public static string[] ReadbackSites => System.Linq.Enumerable.ToArray(
        System.Linq.Enumerable.Select(
            System.Linq.Enumerable.OrderBy(_readbackSites, entry => entry.Key),
            entry => $"{entry.Value}x {(_readbackSiteBytes.TryGetValue(entry.Key, out long b) ? b : 0):N0} B {entry.Key}"));

    private static long _readbackSyncTicks;
    private static long _readbackCopyTicks;

    /// <summary>Stopwatch ticks the (site-capturing) readbacks spent waiting for the stream, then copying.</summary>
    public static (double SyncSeconds, double CopySeconds) ReadbackSeconds =>
        (Interlocked.Read(ref _readbackSyncTicks) / (double)Stopwatch.Frequency,
         Interlocked.Read(ref _readbackCopyTicks) / (double)Stopwatch.Frequency);

    /// <summary>Accumulates a readback's stream-wait and copy time (only measured while sites are captured).</summary>
    public static void OnReadbackTiming(long syncTicks, long copyTicks)
    {
        Interlocked.Add(ref _readbackSyncTicks, syncTicks);
        Interlocked.Add(ref _readbackCopyTicks, copyTicks);
    }

    /// <summary>Records one device-to-host transfer at a backend download choke point.</summary>
    public static void OnReadback(long byteCount)
    {
        Interlocked.Increment(ref _readbacks);
        Interlocked.Add(ref _readbackBytes, byteCount);
        if (CaptureReadbackSites)
        {
            var site = CallSite();
            _readbackSites.AddOrUpdate(site, 1, static (_, count) => count + 1);
            _readbackSiteBytes.AddOrUpdate(site, byteCount, (_, total) => total + byteCount);
        }
    }

    /// <summary>The engine/backend frame that issued a transfer, plus the first caller outside this assembly.</summary>
    private static string CallSite()
    {
        var frames = new StackTrace(2, true).GetFrames();
        var frame = frames is null ? null : System.Linq.Enumerable.FirstOrDefault(frames, candidate =>
        {
            var method = candidate.GetMethod();
            if (method is null) return false;
            // The counted driver-copy wrappers are the transfer itself, not who issued it.
            if (method.Name.StartsWith("cuMemcpy", System.StringComparison.Ordinal)) return false;
            var declaringType = method.DeclaringType;
            if (declaringType == typeof(DirectGpuTensorEngine))
            {
                return method.Name is not "DeferTensorResult" and not "FinishGpuOp"
                and not "GetOrAllocateBuffer" and not "UploadTensorRaw"
                and not "MaterializeIfDeferred";
            }
            string? typeName = declaringType?.FullName;
            return typeName is not null
                && typeName.StartsWith("AiDotNet.Tensors.Engines.DirectGpu.", System.StringComparison.Ordinal)
                && method.Name != "DownloadBuffer";
        });
        var method = frame?.GetMethod();
        var site = frame is null || method is null
            ? "unknown"
            : $"{method.DeclaringType?.FullName}.{method.Name}:{frame.GetFileLineNumber()}";
        // A deferred download is triggered by whoever first reads the host array, often far from the op that
        // produced the tensor (the engine frame is then just the materializer callback). Name the first caller
        // outside this assembly too, so a site says both which op's result and which consumer forced it.
        var external = frames is null ? null : System.Linq.Enumerable.FirstOrDefault(frames, candidate =>
            candidate.GetMethod()?.DeclaringType?.Assembly is { } asm && asm != typeof(GpuLaunchProbe).Assembly
            && asm != typeof(object).Assembly);
        var externalMethod = external?.GetMethod();
        if (externalMethod is not null)
            site += $" <- {externalMethod.DeclaringType?.Name}.{externalMethod.Name}";
        return site;
    }

    /// <summary>Host-to-device transfers observed since the last <see cref="Reset"/>.</summary>
    public static long Uploads => Interlocked.Read(ref _uploads);

    /// <summary>Total bytes transferred host-to-device since the last <see cref="Reset"/>.</summary>
    public static long UploadBytes => Interlocked.Read(ref _uploadBytes);

    /// <summary>Distinct upload call sites, formatted like <see cref="ReadbackSites"/>.</summary>
    public static string[] UploadSites => System.Linq.Enumerable.ToArray(
        System.Linq.Enumerable.Select(
            System.Linq.Enumerable.OrderBy(_uploadSites, entry => entry.Key),
            entry => $"{entry.Value}x {(_uploadSiteBytes.TryGetValue(entry.Key, out var b) ? b : 0):N0} B {entry.Key}"));

    /// <summary>Records one host-to-device transfer at a driver copy (every cuMemcpyHtoD / HtoDAsync).</summary>
    public static void OnUpload(long byteCount)
    {
        Interlocked.Increment(ref _uploads);
        Interlocked.Add(ref _uploadBytes, byteCount);
        if (CaptureReadbackSites)
        {
            var site = CallSite();
            _uploadSites.AddOrUpdate(site, 1, static (_, count) => count + 1);
            _uploadSiteBytes.AddOrUpdate(site, byteCount, (_, total) => total + byteCount);
        }
    }

    /// <summary>Kernel-not-found lookups (throwing indexer) since the last <see cref="Reset"/>. A op that
    /// does ZERO launches but has a miss recorded is a HOLLOW override: it asked for an unregistered kernel
    /// and its caller silently fell back to the CPU. (Checked TryGetValue misses are NOT counted here.)</summary>
    public static long KernelMisses => Interlocked.Read(ref _kernelMisses);

    /// <summary>Distinct kernel names that were looked up and missing since the last <see cref="Reset"/>.</summary>
    public static string[] MissedKernelNames => System.Linq.Enumerable.ToArray(_missedNames.Keys);

    /// <summary>Records a throwing kernel-cache miss (called from the kernel cache indexer only).</summary>
    public static void OnKernelMiss(string name)
    {
        Interlocked.Increment(ref _kernelMisses);
        if (name is not null) _missedNames.TryAdd(name, 0);
    }

    /// <summary>Reasons GPU overrides silently routed to the CPU since the last <see cref="Reset"/>.</summary>
    /// <remarks>
    /// The launch counter tells you an op did zero GPU work; it cannot tell you WHY. Most overrides in
    /// <c>DirectGpuTensorEngine</c> wrap their device path in <c>catch (Exception) { return base.Op(...); }</c>,
    /// which turns a genuine kernel defect into a silent, correct-looking CPU result — invisible to both
    /// parity (the CPU answer is right) and the hollow-override check (no kernel-cache miss is recorded).
    /// This channel makes that class observable: a fallback records the op and the exception that caused it.
    /// </remarks>
    public static string[] Fallbacks => System.Linq.Enumerable.ToArray(
        System.Linq.Enumerable.Select(
            System.Linq.Enumerable.OrderBy(_fallbacks, entry => entry.Key),
            entry => $"{entry.Value}x {entry.Key}"));

    /// <summary>Records one silent GPU-to-CPU fallback. <paramref name="reason"/> is the caught exception, or
    /// null when a guard or route declined the device path before attempting it.</summary>
    public static void OnFallback(string op, System.Exception? reason)
    {
        // Diagnostic keys must have bounded cardinality. Callers may append tensor shapes or route details
        // after a colon, and exception messages often contain input-specific values; retaining either in the
        // process-wide dictionary would allow a long-running workload to create an unbounded number of keys.
        int detailSeparator = op.IndexOf(':');
        string operation = detailSeparator >= 0 ? op.Substring(0, detailSeparator) : op;
        string key = reason is null
            ? $"{operation}: guard or route declined"
            : $"{operation}: {reason.GetType().Name}";
        _fallbacks.AddOrUpdate(key, 1, static (_, count) => count + 1);
    }

    /// <summary>Zeroes launches AND misses before a measured region. Returns the pre-reset launch count.</summary>
    public static long Reset()
    {
        Interlocked.Exchange(ref _kernelMisses, 0);
        Interlocked.Exchange(ref _readbacks, 0);
        Interlocked.Exchange(ref _readbackBytes, 0);
        Interlocked.Exchange(ref _uploads, 0);
        Interlocked.Exchange(ref _uploadBytes, 0);
        _uploadSites.Clear();
        _uploadSiteBytes.Clear();
        _missedNames.Clear();
        _readbackSites.Clear();
        _readbackSiteBytes.Clear();
        Interlocked.Exchange(ref _readbackSyncTicks, 0);
        Interlocked.Exchange(ref _readbackCopyTicks, 0);
        _fallbacks.Clear();
        return Interlocked.Exchange(ref _count, 0);
    }
}
