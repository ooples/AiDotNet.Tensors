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
    private static int _captureReadbackSites;
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> _missedNames = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> _readbackSites = new();
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
            entry => $"{entry.Value}x {entry.Key}"));

    /// <summary>Records one device-to-host transfer at a backend download choke point.</summary>
    public static void OnReadback(long byteCount)
    {
        Interlocked.Increment(ref _readbacks);
        Interlocked.Add(ref _readbackBytes, byteCount);
        if (CaptureReadbackSites)
        {
            var frames = new StackTrace(1, true).GetFrames();
            var frame = frames is null ? null : System.Linq.Enumerable.FirstOrDefault(frames, candidate =>
            {
                var method = candidate.GetMethod();
                if (method is null) return false;
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
            _readbackSites.AddOrUpdate(site, 1, static (_, count) => count + 1);
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
        _missedNames.Clear();
        _readbackSites.Clear();
        _fallbacks.Clear();
        return Interlocked.Exchange(ref _count, 0);
    }
}
