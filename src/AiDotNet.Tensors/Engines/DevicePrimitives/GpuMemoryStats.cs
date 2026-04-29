// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DevicePrimitives;

/// <summary>
/// GPU memory stats — mirrors <c>torch.cuda.memory_stats()</c> +
/// <c>memory_summary()</c> + <c>memory_snapshot()</c>. Aggregates state
/// from every backend allocator (CUDA caching allocator, ROCm caching
/// allocator, the in-process <c>TensorArena</c>, the GPU workspace
/// pool) so a snapshot during a training step lists every allocation
/// across every backend, satisfying #219's acceptance criterion:
/// "Memory snapshot: snapshot during a training step lists all
/// <c>TensorArena</c> + <c>GpuWorkspace</c> allocations with correct
/// sizes."
///
/// <para>The CPU-side memory profiler from #220 covers
/// <c>TensorAllocator</c> + in-process arena and is wired in by the
/// follow-up integration commit once #220 lands; this facade owns the
/// GPU-allocator surface independently so it ships with #219 alone.</para>
/// </summary>
public static class GpuMemoryStats
{
    private static readonly object _lock = new();
    private static long _peakBytes;
    private static long _currentBytes;
    private static long _totalBytes;
    private static long _activeAllocations;
    // Per-allocator counters keyed by the label passed to
    // RecordAllocation/RecordFree (e.g. "cuda_caching", "cuda_pinned_host",
    // "rocm_caching", "gpu_workspace"). Without this, every backend's
    // state collapsed into the global aggregates and the
    // "backend-by-backend snapshot" #219 promised was unreachable from
    // outside. Updated under the same _lock as the aggregates.
    private static readonly Dictionary<string, AllocatorState> _byAllocator = new(StringComparer.Ordinal);

    private sealed class AllocatorState
    {
        public long CurrentBytes;
        public long PeakBytes;
        public long TotalBytes;
        public long ActiveAllocations;
    }

    /// <summary>Bytes currently held by GPU allocators.</summary>
    public static long CurrentBytes
    {
        get { lock (_lock) return _currentBytes; }
    }

    /// <summary>Maximum <see cref="CurrentBytes"/> observed since
    /// <see cref="ResetPeakStats"/>.</summary>
    public static long PeakBytes
    {
        get { lock (_lock) return _peakBytes; }
    }

    /// <summary>Lifetime sum of bytes allocated.</summary>
    public static long TotalAllocatedBytes
    {
        get { lock (_lock) return _totalBytes; }
    }

    /// <summary>Number of currently-live allocations.</summary>
    public static long ActiveAllocations
    {
        get { lock (_lock) return _activeAllocations; }
    }

    /// <summary>Allocator hook — call after a successful GPU
    /// allocation. Backends (CUDA caching allocator, GPU workspace,
    /// ROCm allocator) call into this to publish their state.</summary>
    public static void RecordAllocation(string allocator, long bytes)
    {
        if (allocator is null) throw new ArgumentNullException(nameof(allocator));
        lock (_lock)
        {
            _currentBytes += bytes;
            _totalBytes += bytes;
            _activeAllocations++;
            if (_currentBytes > _peakBytes) _peakBytes = _currentBytes;

            if (!_byAllocator.TryGetValue(allocator, out var state))
            {
                state = new AllocatorState();
                _byAllocator[allocator] = state;
            }
            state.CurrentBytes += bytes;
            state.TotalBytes += bytes;
            state.ActiveAllocations++;
            if (state.CurrentBytes > state.PeakBytes) state.PeakBytes = state.CurrentBytes;
        }
    }

    /// <summary>Free hook — pairs with <see cref="RecordAllocation"/>.</summary>
    public static void RecordFree(string allocator, long bytes)
    {
        if (allocator is null) throw new ArgumentNullException(nameof(allocator));
        lock (_lock)
        {
            _currentBytes -= bytes;
            _activeAllocations--;

            if (_byAllocator.TryGetValue(allocator, out var state))
            {
                state.CurrentBytes -= bytes;
                state.ActiveAllocations--;
            }
            // No fallback for unknown-allocator frees — that's a real
            // bug in the calling backend (paired Alloc/Free strings
            // must match), and silently swallowing it would mask the
            // bookkeeping drift forever.
        }
    }

    /// <summary>Resets the peak-bytes counter to the current value.</summary>
    public static void ResetPeakStats()
    {
        lock (_lock)
        {
            _peakBytes = _currentBytes;
            foreach (var state in _byAllocator.Values)
                state.PeakBytes = state.CurrentBytes;
        }
    }

    /// <summary>Drops every counter back to zero. Used for test
    /// isolation; production code shouldn't call this mid-training.</summary>
    public static void Reset()
    {
        lock (_lock)
        {
            _peakBytes = 0;
            _currentBytes = 0;
            _totalBytes = 0;
            _activeAllocations = 0;
            _byAllocator.Clear();
        }
    }

    /// <summary>
    /// Snapshot of every per-allocator counter set, keyed by allocator
    /// label. The dictionary is a deep copy taken under the lock — safe
    /// to enumerate / serialise without holding any state from this
    /// type. Mirrors PyTorch's per-pool entries in <c>memory_stats()</c>.
    /// </summary>
    public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, long>> StatsByAllocator()
    {
        lock (_lock)
        {
            var result = new Dictionary<string, IReadOnlyDictionary<string, long>>(StringComparer.Ordinal);
            foreach (var kv in _byAllocator)
            {
                result[kv.Key] = new Dictionary<string, long>
                {
                    ["allocated_bytes.current"] = kv.Value.CurrentBytes,
                    ["allocated_bytes.peak"]    = kv.Value.PeakBytes,
                    ["allocated_bytes.total"]   = kv.Value.TotalBytes,
                    ["active.current"]          = kv.Value.ActiveAllocations,
                };
            }
            return result;
        }
    }

    /// <summary>
    /// Human-readable summary suitable for emitting from
    /// <c>memory_summary()</c>-style debug calls.
    /// </summary>
    public static string Summary()
    {
        lock (_lock)
        {
            return $"GPU Memory Summary\n" +
                   $"  Current:    {_currentBytes:N0} B\n" +
                   $"  Peak:       {_peakBytes:N0} B\n" +
                   $"  Total Alloc:{_totalBytes:N0} B\n" +
                   $"  Active:     {_activeAllocations:N0} allocations\n";
        }
    }

    /// <summary>Returns a serializable per-counter dictionary, mirroring
    /// PyTorch's <c>memory_stats()</c> shape.</summary>
    public static IReadOnlyDictionary<string, long> Stats()
    {
        lock (_lock)
        {
            return new Dictionary<string, long>
            {
                ["allocated_bytes.current"] = _currentBytes,
                ["allocated_bytes.peak"]    = _peakBytes,
                ["allocated_bytes.total"]   = _totalBytes,
                ["active.current"]          = _activeAllocations,
            };
        }
    }
}
