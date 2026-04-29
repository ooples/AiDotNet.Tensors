// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;

namespace AiDotNet.Tensors.Engines.Profiling.Memory;

/// <summary>
/// Records the lifetime of every <see cref="Tensor{T}"/> allocated through
/// <c>TensorAllocator</c> (and other allocators that opt in via
/// <see cref="RecordAllocation"/> / <see cref="RecordFree"/>) during a
/// recording window. Used to:
///
/// <list type="bullet">
///   <item>Build a memory timeline for the chrome-trace UI.</item>
///   <item>Surface "largest live allocations" during an OOM-adjacent run.</item>
///   <item>Report peak / current / total bytes per allocator.</item>
/// </list>
///
/// <para>Modelled on <c>torch.cuda.memory._record_memory_history</c> +
/// <c>_dump_snapshot</c>. The profiler is OFF by default — toggle with
/// <see cref="RecordHistory"/>; the call-site instrumentation is a single
/// volatile read when off.</para>
/// </summary>
public static class MemoryProfiler
{
    /// <summary>Modes for the memory recorder.</summary>
    public enum RecordMode
    {
        /// <summary>Recording disabled.</summary>
        Off,

        /// <summary>Record allocation events only — no per-event stack capture.
        /// Cheapest meaningful mode; ~50 ns per allocation.</summary>
        State,

        /// <summary>Record allocation events plus a managed stack trace at
        /// each allocation site. Useful for "who allocated this 800 MB
        /// tensor right before the OOM" forensics. Heavier (~5 µs per
        /// allocation) — turn on only when needed.</summary>
        All,
    }

    private static volatile RecordMode _mode = RecordMode.Off;
    private static readonly ConcurrentQueue<MemoryEvent> _events = new();
    private static readonly ConcurrentDictionary<long, LiveAllocation> _live = new();
    private static long _nextAllocId;
    private static long _peakBytes;
    private static long _currentBytes;
    private static long _totalBytes;
    private static long _historyStartTicks;
    private static readonly long _ticksPerSecond = Stopwatch.Frequency;

    /// <summary>Current recording mode.</summary>
    public static RecordMode Mode => _mode;

    /// <summary>Bytes currently held alive by tracked allocators.</summary>
    public static long CurrentBytes => System.Threading.Volatile.Read(ref _currentBytes);

    /// <summary>Maximum <see cref="CurrentBytes"/> observed since the last
    /// <see cref="ResetPeakStats"/> call.</summary>
    public static long PeakBytes => System.Threading.Volatile.Read(ref _peakBytes);

    /// <summary>Lifetime sum of allocated bytes (never reset by free).
    /// Useful for churn analysis — a large delta over a steady CurrentBytes
    /// means the workload is allocator-thrashing.</summary>
    public static long TotalAllocatedBytes => System.Threading.Volatile.Read(ref _totalBytes);

    /// <summary>Begin recording. <paramref name="mode"/> = Off stops recording
    /// without dropping the existing event log; call <see cref="Reset"/> to
    /// drop the log explicitly.</summary>
    public static void RecordHistory(RecordMode mode)
    {
        _mode = mode;
        if (mode != RecordMode.Off && _historyStartTicks == 0)
            System.Threading.Interlocked.CompareExchange(ref _historyStartTicks, Stopwatch.GetTimestamp(), 0);
    }

    /// <summary>Reset the recorded event log + counters. Mode unchanged.</summary>
    public static void Reset()
    {
        while (_events.TryDequeue(out _)) { }
        _live.Clear();
        System.Threading.Interlocked.Exchange(ref _peakBytes, 0);
        System.Threading.Interlocked.Exchange(ref _currentBytes, 0);
        System.Threading.Interlocked.Exchange(ref _totalBytes, 0);
        System.Threading.Interlocked.Exchange(ref _historyStartTicks, 0);
    }

    /// <summary>Reset only the peak counter to the current bytes value.
    /// Mirrors <c>torch.cuda.reset_peak_memory_stats</c>.</summary>
    public static void ResetPeakStats()
    {
        System.Threading.Interlocked.Exchange(ref _peakBytes, _currentBytes);
    }

    /// <summary>
    /// Allocator hook — call after a successful allocation. Returns an
    /// allocation id that the matching <see cref="RecordFree"/> call must
    /// pass back. Returns -1 when recording is off (and the allocator can
    /// skip the matching free hook).
    /// </summary>
    public static long RecordAllocation(string allocator, long bytes, int[]? shape = null, string? dtypeName = null)
    {
        if (_mode == RecordMode.Off) return -1;
        long id = System.Threading.Interlocked.Increment(ref _nextAllocId);
        long now = Stopwatch.GetTimestamp();
        long current = System.Threading.Interlocked.Add(ref _currentBytes, bytes);
        System.Threading.Interlocked.Add(ref _totalBytes, bytes);
        UpdatePeak(current);

        string? stack = _mode == RecordMode.All ? CaptureStack() : null;
        var live = new LiveAllocation(id, allocator, bytes, shape, dtypeName, now, stack);
        _live[id] = live;
        _events.Enqueue(MemoryEvent.Alloc(id, allocator, bytes, ToMicrosFromStart(now), stack, shape, dtypeName));
        return id;
    }

    /// <summary>Allocator hook — call when freeing the allocation that
    /// <see cref="RecordAllocation"/> assigned <paramref name="allocationId"/>.
    /// Silent no-op when <paramref name="allocationId"/> is -1 (recording was
    /// off at allocation time) or when recording was reset between alloc/free.</summary>
    public static void RecordFree(long allocationId)
    {
        // Even when recording is currently Off we still need to retire allocations that
        // were tracked while it was on — otherwise CurrentBytes stays permanently
        // inflated and live-allocation snapshots leak. We just suppress the new Free
        // event in Off mode.
        if (allocationId < 0) return;
        if (!_live.TryRemove(allocationId, out var live)) return;

        long now = Stopwatch.GetTimestamp();
        System.Threading.Interlocked.Add(ref _currentBytes, -live.Bytes);
        if (_mode != RecordMode.Off)
            _events.Enqueue(MemoryEvent.Free(allocationId, live.Allocator, live.Bytes, ToMicrosFromStart(now)));
    }

    /// <summary>
    /// Snapshot of every live allocation, ordered largest-first. Useful for
    /// "what's holding 80% of the heap right before the OOM" debugging.
    /// </summary>
    public static IReadOnlyList<LiveAllocation> GetLargestLiveAllocations(int top = 10)
    {
        var arr = new LiveAllocation[_live.Count];
        int i = 0;
        foreach (var kv in _live)
        {
            if (i >= arr.Length) break;
            arr[i++] = kv.Value;
        }
        Array.Sort(arr, 0, i, LiveAllocationByBytesDescending.Instance);
        if (top < i) i = top;
        var result = new LiveAllocation[i];
        Array.Copy(arr, result, i);
        return result;
    }

    /// <summary>Snapshot of every event recorded since <see cref="RecordHistory"/>
    /// last switched on (or <see cref="Reset"/> was called).</summary>
    public static IReadOnlyList<MemoryEvent> Events => _events.ToArray();

    /// <summary>Writes a textual snapshot summary to <paramref name="path"/>.
    /// Format: peak/current/total counters, top-N live allocations, recent events.</summary>
    public static void DumpSnapshot(string path, int topLiveAllocations = 25, int recentEvents = 200)
    {
        using var sw = new System.IO.StreamWriter(path);
        sw.WriteLine("# AiDotNet.Tensors Memory Snapshot");
        sw.WriteLine($"Mode:           {_mode}");
        sw.WriteLine($"PeakBytes:      {_peakBytes:N0}");
        sw.WriteLine($"CurrentBytes:   {_currentBytes:N0}");
        sw.WriteLine($"TotalAllocated: {_totalBytes:N0}");
        sw.WriteLine($"LiveAllocs:     {_live.Count:N0}");
        sw.WriteLine();
        sw.WriteLine($"## Top {topLiveAllocations} live allocations");
        var top = GetLargestLiveAllocations(topLiveAllocations);
        foreach (var a in top)
        {
            string shape = a.Shape is null ? "(?)" : "[" + string.Join(",", a.Shape) + "]";
            string dt = a.DtypeName ?? "?";
            sw.WriteLine($"  id={a.Id} {a.Allocator,-16} {a.Bytes,12:N0} B  shape={shape} dtype={dt}");
            if (a.Stack is not null) sw.WriteLine(a.Stack);
        }
        sw.WriteLine();
        sw.WriteLine($"## Last {recentEvents} events");
        var ev = _events.ToArray();
        int start = System.Math.Max(0, ev.Length - recentEvents);
        for (int i = start; i < ev.Length; i++)
        {
            var e = ev[i];
            sw.WriteLine($"  +{e.TimestampMicros:N0}us {(e.Kind == MemoryEventKind.Alloc ? "ALLOC" : "FREE ")} id={e.AllocationId} {e.Allocator,-16} {e.Bytes,12:N0} B");
        }
    }

    private static void UpdatePeak(long current)
    {
        long peak;
        do
        {
            peak = _peakBytes;
            if (current <= peak) return;
        } while (System.Threading.Interlocked.CompareExchange(ref _peakBytes, current, peak) != peak);
    }

    private static long ToMicrosFromStart(long ticks)
    {
        long start = _historyStartTicks;
        if (start == 0) return 0;
        return ((ticks - start) * 1_000_000L) / _ticksPerSecond;
    }

    private static string? CaptureStack()
    {
        try { return new StackTrace(skipFrames: 2, fNeedFileInfo: true).ToString(); }
        catch { return null; }
    }

    private sealed class LiveAllocationByBytesDescending : IComparer<LiveAllocation>
    {
        public static readonly LiveAllocationByBytesDescending Instance = new();
        public int Compare(LiveAllocation? x, LiveAllocation? y)
            => (y?.Bytes ?? 0).CompareTo(x?.Bytes ?? 0);
    }
}

/// <summary>Snapshot of one live allocation.</summary>
public sealed class LiveAllocation
{
    /// <summary>Monotonic id assigned at <c>RecordAllocation</c> time.</summary>
    public long Id { get; }

    /// <summary>Allocator label — typically <c>"TensorAllocator"</c>,
    /// <c>"ArrayPool"</c>, <c>"NativeMemory"</c>.</summary>
    public string Allocator { get; }

    /// <summary>Allocation size in bytes.</summary>
    public long Bytes { get; }

    /// <summary>Tensor shape, when known.</summary>
    public int[]? Shape { get; }

    /// <summary>Element type name, when known.</summary>
    public string? DtypeName { get; }

    /// <summary>Stopwatch-tick timestamp at allocation.</summary>
    public long AllocationTicks { get; }

    /// <summary>Captured stack at allocation site, or null if mode != All.</summary>
    public string? Stack { get; }

    internal LiveAllocation(long id, string allocator, long bytes, int[]? shape, string? dtypeName,
        long allocationTicks, string? stack)
    {
        Id = id;
        Allocator = allocator;
        Bytes = bytes;
        Shape = shape;
        DtypeName = dtypeName;
        AllocationTicks = allocationTicks;
        Stack = stack;
    }
}

/// <summary>Allocator event kind.</summary>
public enum MemoryEventKind
{
    /// <summary>Allocation event.</summary>
    Alloc,
    /// <summary>Free event.</summary>
    Free,
}

/// <summary>One recorded memory event.</summary>
public readonly struct MemoryEvent
{
    /// <summary>Allocation/free distinction.</summary>
    public MemoryEventKind Kind { get; }

    /// <summary>Id assigned at allocation; same id appears on the matching free.</summary>
    public long AllocationId { get; }

    /// <summary>Allocator label.</summary>
    public string Allocator { get; }

    /// <summary>Bytes allocated (or freed — always the original alloc size).</summary>
    public long Bytes { get; }

    /// <summary>Microseconds since recording began.</summary>
    public long TimestampMicros { get; }

    /// <summary>Stack trace at allocation site, or null. Always null for Free events.</summary>
    public string? Stack { get; }

    /// <summary>Tensor shape at allocation site (null for Free events or unknown).</summary>
    public int[]? Shape { get; }

    /// <summary>Element type name (null for Free events or unknown).</summary>
    public string? DtypeName { get; }

    private MemoryEvent(MemoryEventKind kind, long id, string allocator, long bytes, long ts,
        string? stack, int[]? shape, string? dtype)
    {
        Kind = kind; AllocationId = id; Allocator = allocator; Bytes = bytes;
        TimestampMicros = ts; Stack = stack; Shape = shape; DtypeName = dtype;
    }

    internal static MemoryEvent Alloc(long id, string allocator, long bytes, long ts, string? stack, int[]? shape, string? dtype)
        => new(MemoryEventKind.Alloc, id, allocator, bytes, ts, stack, shape, dtype);

    internal static MemoryEvent Free(long id, string allocator, long bytes, long ts)
        => new(MemoryEventKind.Free, id, allocator, bytes, ts, null, null, null);
}
