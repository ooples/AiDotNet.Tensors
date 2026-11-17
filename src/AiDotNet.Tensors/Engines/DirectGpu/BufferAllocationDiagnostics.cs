using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Opt-in successful primitive8/primitive32 backend buffer requests, including pool hits and framework workspaces.
/// These are allocator-interface requests, not Tensor wrappers, driver allocations, or reserved device bytes.
/// </summary>
internal static class BufferAllocationDiagnostics
{
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("AIDOTNET_BUFFER_ALLOC_DIAG") == "1";
    private static readonly RequestRecorder Recorder = new();
    internal static bool IsInsideRequest => RequestRecorder.IsInside;

    internal static T Observe<T>(string backend, long bytes, Func<T> allocate)
        => Recorder.Observe(backend, bytes, allocate);

    internal static TResult Observe<TState, TResult>(string backend, long bytes, TState state, Func<TState, TResult> allocate)
        => Recorder.Observe(backend, bytes, state, allocate);

    // The benchmark accesses this internal diagnostic reflectively, so the public library API need not grow.
    internal static Dictionary<string, object> Snapshot()
    {
        var totals = Recorder.Snapshot();
        long bytes = 0, count = 0;
        var backends = new Dictionary<string, object>();
        foreach (var pair in totals)
        {
            bytes += pair.Value.Bytes;
            count += pair.Value.Count;
            backends[pair.Key] = new Dictionary<string, object>
            {
                ["bytes"] = pair.Value.Bytes,
                ["count"] = pair.Value.Count,
            };
        }
        return new Dictionary<string, object>
        {
            ["enabled"] = Enabled,
            ["gpu_buffer_requested"] = new Dictionary<string, object>
            {
                ["bytes"] = bytes,
                ["count"] = count,
                ["coverage"] = Enabled ? "primitive8_32_backend_buffer_api_only" : "disabled",
                ["scope"] = "Successful backend buffer requests including framework workspaces and pool hits; excludes views and delegated duplicate calls.",
                ["backends"] = backends,
            },
            ["cpu_storage_requested"] = Helpers.CpuStorageAllocationDiagnostics.Snapshot(),
            // Explicitly unknown: these categories cannot be inferred from buffer requests or GC object bytes.
            ["cpu_framework_requested"] = new Dictionary<string, object> { ["coverage"] = "unverified" },
            ["gpu_framework_requested"] = new Dictionary<string, object> { ["coverage"] = "unverified_api_coverage" },
            ["cpu_backing"] = new Dictionary<string, object> { ["coverage"] = "unverified" },
            ["gpu_backing"] = new Dictionary<string, object> { ["coverage"] = "unverified" },
        };
    }

    internal sealed class Totals
    {
        private long _bytes, _count;
        internal long Bytes => Interlocked.Read(ref _bytes);
        internal long Count => Interlocked.Read(ref _count);
        internal void Add(long bytes)
        {
            Interlocked.Add(ref _bytes, bytes);
            Interlocked.Increment(ref _count);
        }
    }

    internal sealed class RequestRecorder
    {
        [ThreadStatic] private static int t_depth;
        private readonly ConcurrentDictionary<string, Totals> _totals = new();
        internal static bool IsInside => t_depth != 0;

        internal T Observe<T>(string backend, long bytes, Func<T> allocate)
            => Observe(backend, bytes, allocate, static action => action());

        internal TResult Observe<TState, TResult>(string backend, long bytes, TState state, Func<TState, TResult> allocate)
        {
            bool outermost = t_depth++ == 0;
            try
            {
                TResult result = allocate(state);
                // Only successful, positive-size requests count. An OOM/invalid argument leaves no allocation.
                if (outermost && bytes > 0) _totals.GetOrAdd(backend, _ => new Totals()).Add(bytes);
                return result;
            }
            finally { t_depth--; }
        }

        internal Dictionary<string, Totals> Snapshot() => new(_totals);
    }
}
