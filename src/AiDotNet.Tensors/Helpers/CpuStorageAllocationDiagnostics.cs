using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Opt-in successful CPU numeric-storage requests, including pool/arena hits. The tags distinguish
/// tensor results from framework workspaces. These are not CLR object bytes or OS heap allocations.
/// Coverage is partial until every active family route is independently accounted for.
/// </summary>
internal static class CpuStorageAllocationDiagnostics
{
    internal static readonly bool Enabled = Engines.DirectGpu.BufferAllocationDiagnostics.Enabled;
    private static readonly RequestRecorder Recorder = new();
    internal static bool IsInsideRequest => RequestRecorder.IsInside;

    internal static long ShapeBytes<T>(int[] shape)
    {
        // Let the allocator retain its own validation/exception contract; invalid requests never count.
        long length = 1;
        foreach (int dimension in shape)
        {
            if (dimension <= 0 || length > int.MaxValue / dimension) return 0;
            length *= dimension;
        }
        return length * Unsafe.SizeOf<T>();
    }

    internal static TResult Observe<TState, TResult>(string tag, long bytes, TState state, Func<TState, TResult> allocate)
        => Recorder.Observe(tag, bytes, state, allocate);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Record<T>(string tag, int length)
    {
        if (Enabled && !IsInsideRequest && length > 0)
            Recorder.Record(tag, (long)length * Unsafe.SizeOf<T>());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static T[] RentWorkspace<T>(int minimumLength, string tag)
    {
        T[] array = ArrayPool<T>.Shared.Rent(minimumLength);
        Record<T>(tag, minimumLength);
        return array;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static T[] NewNumericArray<T>(int length, string tag)
    {
        T[] array = new T[length];
        Record<T>(tag, length);
        return array;
    }

    internal static Dictionary<string, object> Snapshot()
    {
        long bytes = 0, count = 0;
        var tags = new Dictionary<string, object>();
        foreach (var pair in Recorder.Snapshot())
        {
            bytes += pair.Value.Bytes;
            count += pair.Value.Count;
            tags[pair.Key] = new Dictionary<string, object> { ["bytes"] = pair.Value.Bytes, ["count"] = pair.Value.Count };
        }
        return new Dictionary<string, object>
        {
            ["bytes"] = bytes, ["count"] = count,
            ["coverage"] = Enabled ? "partial_numeric_storage_request_boundaries" : "disabled",
            ["scope"] = "Successful instrumented CPU storage requests, including tensor allocator pool/arena hits and tagged framework workspaces; excludes wrappers/views and nested delegation.",
            ["tags"] = tags,
            ["unverified"] = "Direct array creation, native/vendor workspaces and uninstrumented private pools may bypass these boundaries. Not complete ATen allocator coverage.",
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
        internal void Record(string tag, long bytes)
        {
            if (bytes > 0) _totals.GetOrAdd(tag, static _ => new Totals()).Add(bytes);
        }
        internal TResult Observe<TState, TResult>(string tag, long bytes, TState state, Func<TState, TResult> allocate)
        {
            bool outermost = t_depth++ == 0;
            try
            {
                TResult result = allocate(state);
                if (outermost) Record(tag, bytes);
                return result;
            }
            finally { t_depth--; }
        }
        internal Dictionary<string, Totals> Snapshot() => new(_totals);
    }
}
