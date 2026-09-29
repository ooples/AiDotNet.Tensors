using System;
using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    /// <summary>
    /// The memory a captured CUDA graph bakes in, kept alive for as long as the graph can replay (PyTorch's
    /// graph-private pool). A graph replays the device addresses it saw during capture, so no buffer the capture
    /// used, allocated or released may be freed or handed to another allocation until the graph is destroyed.
    /// Otherwise a replay writes into memory someone else now owns, or reads memory the allocator has returned to
    /// the driver (CUDA 700 on whichever thread touches the device next).
    /// </summary>
    /// <remarks>
    /// While a thread captures, every buffer whose device pointer it reads (a kernel argument, copy, memset or
    /// library call) or allocates is PINNED to the capture: the pool holds it strongly, so GC cannot finalize it,
    /// and its release (by any thread, e.g. an activation-cache eviction) is parked here instead of run. A release
    /// the capturing thread issues is parked as well. Retiring the pool (graph destroyed, capture failed, or graph
    /// replaced by an update) re-issues the parked releases, which then wait on any other graph still pinning the
    /// buffer. Frees are stream-ordered on the compute stream, where replays run, so they follow a replay in flight.
    /// </remarks>
    internal sealed class CaptureMemoryPool
    {
        private sealed class ByReference : IEqualityComparer<object>
        {
            internal static readonly ByReference Instance = new();
            public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
            public int GetHashCode(object obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
        }

        private readonly object _lock = new();
        private List<Action>? _parked = new();
        private HashSet<object>? _pinned = new(ByReference.Instance);

        /// <summary>The pool of the capture running on this thread, if any.</summary>
        [ThreadStatic] internal static CaptureMemoryPool? Current;

        /// <summary>
        /// Identifies the backend whose graph owns this pool (retired on that backend's disposal). A token, not the
        /// backend: the static graph registry must not keep an abandoned backend and its caches reachable.
        /// </summary>
        internal object? OwnerToken;

        internal bool IsRetired { get { lock (_lock) return _parked is null; } }

        /// <summary>True when the capture used no buffer at all (nothing to keep alive).</summary>
        internal bool IsEmpty { get { lock (_lock) return _parked is { Count: 0 } && _pinned is { Count: 0 }; } }

        /// <summary>
        /// Pins <paramref name="buffer"/> to this capture: adds this pool to the buffer's pin list (dropping pools
        /// already retired) and holds the buffer until retirement.
        /// </summary>
        internal void Pin(ref CaptureMemoryPool[]? pins, object buffer)
        {
            var current = Volatile.Read(ref pins);
            if (current is not null && Array.IndexOf(current, this) >= 0) return;
            lock (_lock)
            {
                if (_pinned is null) return;
                _pinned.Add(buffer);
            }
            while (true)
            {
                int live = 0;
                if (current is not null)
                    foreach (var p in current) if (!p.IsRetired) live++;
                var next = new CaptureMemoryPool[live + 1];
                int i = 0;
                if (current is not null)
                    foreach (var p in current) if (!p.IsRetired) next[i++] = p;
                next[i] = this;
                var seen = Interlocked.CompareExchange(ref pins, next, current);
                if (ReferenceEquals(seen, current)) return;
                current = seen;
                if (current is not null && Array.IndexOf(current, this) >= 0) return;
            }
        }

        /// <summary>Queues <paramref name="release"/> until retirement; false once retired (release now).</summary>
        internal bool TryPark(Action release)
        {
            lock (_lock)
            {
                if (_parked is null) return false;
                _parked.Add(release);
                return true;
            }
        }

        /// <summary>
        /// Parks a buffer's release on the first live capture pinning it, or on this thread's running capture.
        /// The parked action is the buffer's full release, so on retirement it re-checks the remaining pins.
        /// </summary>
        internal static bool TryDeferRelease(CaptureMemoryPool[]? pins, Action release)
        {
            if (pins is not null)
                foreach (var pool in pins)
                    if (pool.TryPark(release)) return true;
            return Current is { } capturing && capturing.TryPark(release);
        }

        /// <summary>Runs every parked release; later releases of this pool's buffers run immediately.</summary>
        internal void Retire()
        {
            List<Action>? parked;
            lock (_lock)
            {
                parked = _parked;
                _parked = null;
                _pinned = null;
            }
            if (parked is null) return;
            foreach (var release in parked)
            {
                try { release(); } catch { /* one buffer's free must not strand the rest */ }
            }
        }
    }

    // Graph exec -> the pool keeping its memory alive. Static: exec handles are process-unique while they live, and
    // the direct-PTX single-kernel graphs are destroyed through the static DestroyCapturedGraphCurrentContext.
    private static readonly object s_graphMemoryLock = new();
    private static readonly Dictionary<IntPtr, CaptureMemoryPool> s_graphMemoryPools = new();

    private readonly object _graphOwnerToken = new();

    // Graphs whose owner was collected without disposing them (a plan's finalizer). The driver must not be called
    // on the finalizer thread, so the destroy runs on the next op thread (EnsureContextCurrent), which then retires
    // the graph's pool -- PyTorch frees a graph's private pool when the graph object dies, the same way.
    internal static readonly System.Collections.Concurrent.ConcurrentQueue<IntPtr> PendingGraphDestroys = new();

    /// <summary>Destroys a captured graph from a finalizer: queued, run on the next GPU op thread.</summary>
    internal static void DestroyCapturedGraphFromFinalizer(IntPtr graphExec)
    {
        if (graphExec != IntPtr.Zero && !IsRuntimeTearingDown && !ProcessExiting)
            PendingGraphDestroys.Enqueue(graphExec);
    }

    private static void DrainPendingGraphDestroys()
    {
        while (PendingGraphDestroys.TryDequeue(out var graphExec))
        {
            if (IsRuntimeTearingDown || ProcessExiting) continue;
            CudaNativeBindings.cuGraphExecDestroy(graphExec);
            RetireGraphMemoryPool(graphExec);
        }
    }


    /// <summary>Hands a capture's pool to the graph it produced; replaces (and retires) a prior pool.</summary>
    private void AttachGraphMemoryPool(IntPtr graphExec, CaptureMemoryPool pool)
    {
        pool.OwnerToken = _graphOwnerToken;
        CaptureMemoryPool? previous;
        lock (s_graphMemoryLock)
        {
            s_graphMemoryPools.TryGetValue(graphExec, out previous);
            s_graphMemoryPools[graphExec] = pool;
        }
        previous?.Retire();
    }

    private static void RetireGraphMemoryPool(IntPtr graphExec)
    {
        CaptureMemoryPool? pool;
        lock (s_graphMemoryLock)
        {
            if (!s_graphMemoryPools.TryGetValue(graphExec, out pool)) return;
            s_graphMemoryPools.Remove(graphExec);
        }
        pool.Retire();
    }

    /// <summary>Retires the pools of every graph this backend captured (backend disposal).</summary>
    private void RetireAllGraphMemoryPools()
    {
        var pools = new List<CaptureMemoryPool>();
        lock (s_graphMemoryLock)
        {
            foreach (var kv in new List<KeyValuePair<IntPtr, CaptureMemoryPool>>(s_graphMemoryPools))
            {
                if (!ReferenceEquals(kv.Value.OwnerToken, _graphOwnerToken)) continue;
                pools.Add(kv.Value);
                s_graphMemoryPools.Remove(kv.Key);
            }
        }
        foreach (var pool in pools) pool.Retire();
    }

    /// <summary>
    /// Runs a capture body with a fresh graph-private pool, then gives the pool to the graph the body produced
    /// (<paramref name="graphOf"/> returns it), or retires the pool at once when no graph survives.
    /// </summary>
    private TResult CaptureWithPrivateMemory<TResult>(Func<TResult> body, Func<TResult, IntPtr> graphOf)
    {
        var pool = new CaptureMemoryPool();
        bool attached = false;
        try
        {
            var previous = CaptureMemoryPool.Current;
            CaptureMemoryPool.Current = pool;
            TResult result;
            try
            {
                // The body opens the capture itself (EnterCapture): the side stream, its ordering after the main stream,
                // this thread's redirection and the per-context gate all live there.
                result = body();
            }
            finally
            {
                CaptureMemoryPool.Current = previous;
            }
            IntPtr graphExec = graphOf(result);
            if (graphExec != IntPtr.Zero && !pool.IsEmpty)
            {
                AttachGraphMemoryPool(graphExec, pool);
                attached = true;
            }
            return result;
        }
        finally
        {
            if (!attached) pool.Retire();
        }
    }
}
