// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Lets a GPU engine release its native resources while other threads may still be running ops on it. Every op takes
/// a lease (<see cref="Enter"/>) for as long as it uses the backend; <see cref="CloseAndDrain"/> stops new leases and
/// waits until every thread's leases are returned, so teardown never frees a handle or buffer under a running op (the
/// cudnnSetStream access violation a disposal under a concurrent convolution caused).
/// </summary>
/// <remarks>
/// <para>Per-op cost is one uncontended interlocked increment and decrement on a counter only the current thread
/// takes leases on, plus a volatile read: no shared cache line bounces between threads running ops in parallel, as a
/// single shared in-flight counter would. The increment is a full fence and so is the close's state write, so either
/// the op sees the gate closed or the drain sees its lease (Dekker's handshake); no process-wide barrier is needed and
/// the scheme is the same on every target framework.</para>
/// <para>A closed gate refuses new leases rather than blocking: the engine's ops take their CPU fallbacks, as they do
/// whenever the GPU is unavailable. A thread already holding a lease keeps it until the op ends, nested ops included.
/// </para>
/// </remarks>
internal sealed class GpuOpGate
{
    // One per thread that ever took a lease here; padded so two threads' counters never share a cache line.
    internal sealed class Counter
    {
#pragma warning disable CS0169, IDE0051 // padding only
        private long _p0, _p1, _p2, _p3, _p4, _p5, _p6;
#pragma warning restore CS0169, IDE0051
        internal int Depth;
#pragma warning disable CS0169, IDE0051
        private long _q0, _q1, _q2, _q3, _q4, _q5, _q6;
#pragma warning restore CS0169, IDE0051
    }

    /// <summary>A held lease; disposing it returns it. The default value holds nothing (the gate was closed).</summary>
    internal readonly struct Lease : IDisposable
    {
        private readonly Counter? _counter;

        internal Lease(Counter counter) => _counter = counter;

        /// <summary>True when the lease was granted; false when the gate was closed and the op must not use the GPU.</summary>
        public bool Held => _counter is not null;

        // Interlocked, not a plain write: an async op can return its lease from another thread.
        public void Dispose()
        {
            if (_counter is not null) Interlocked.Decrement(ref _counter.Depth);
        }
    }

    private const int Open = 0, Closed = 1;
    private int _state;
    private readonly ThreadLocal<Counter> _counters = new(() => new Counter(), trackAllValues: true);

    // The last gate this thread leased on and its counter: a field read instead of a ThreadLocal lookup on the
    // common path (one engine per thread).
    [ThreadStatic] private static GpuOpGate? t_lastGate;
    [ThreadStatic] private static Counter? t_lastCounter;

    /// <summary>
    /// Whether <see cref="DirectGpuTensorEngine"/>'s backend accessors require the calling thread to hold a lease
    /// (AIDOTNET_GPU_LEASE_STRICT=1): a test-suite check that every op leases, so a missed site fails instead of
    /// racing a release. Settable so a test can turn it on around the ops it drives.
    /// </summary>
    internal static bool Strict { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_GPU_LEASE_STRICT") == "1";

    /// <summary>Strict mode for this gate only (a test's own engine), alongside the process-wide <see cref="Strict"/>.</summary>
    internal bool StrictLeases { get; set; }

    /// <summary>False once <see cref="CloseAndDrain"/> began: new ops do not use the GPU.</summary>
    public bool IsOpen => Volatile.Read(ref _state) == Open;

    /// <summary>
    /// Takes a lease for the calling op. When the gate is closed the returned lease holds nothing (check
    /// <see cref="Lease.Held"/> or the engine's availability, which reads the same state).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Lease Enter()
    {
        var counter = ReferenceEquals(t_lastGate, this) ? t_lastCounter! : CounterForThisThread(this);
        // Full fence: the increment is visible before the state is read, the order the drain relies on.
        Interlocked.Increment(ref counter.Depth);
        if (Volatile.Read(ref _state) != Open)
        {
            Interlocked.Decrement(ref counter.Depth);
            return default;
        }
        return new Lease(counter);
    }

    // Static: it fills the thread's one-entry cache, which is per thread rather than per gate.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static Counter CounterForThisThread(GpuOpGate gate)
    {
        var counter = gate._counters.Value!;
        t_lastGate = gate;
        t_lastCounter = counter;
        return counter;
    }

    /// <summary>Whether the calling thread holds a lease on this gate (the strict-mode check).</summary>
    public bool HeldByCurrentThread
    {
        get
        {
            var counter = ReferenceEquals(t_lastGate, this) ? t_lastCounter : null;
            if (counter is null)
            {
                if (!_counters.IsValueCreated) return false;
                counter = _counters.Value;
            }
            return counter is not null && Volatile.Read(ref counter.Depth) > 0;
        }
    }

    /// <summary>
    /// Closes the gate (no new leases) and waits up to <paramref name="timeout"/> for every other thread's leases to
    /// be returned. Leases the calling thread itself holds are not waited for: it cannot return them while it waits.
    /// Returns false on timeout; the gate stays closed and the caller must not free what the leases protect.
    /// </summary>
    public bool CloseAndDrain(TimeSpan timeout)
    {
        // Full fence: the closed state is visible before any counter is read (the other half of the handshake).
        Interlocked.Exchange(ref _state, Closed);
        Counter? own = _counters.IsValueCreated ? _counters.Value : null;
        int ownDepth = own is null ? 0 : Volatile.Read(ref own.Depth);
        var clock = Stopwatch.StartNew();
        var spin = new SpinWait();
        while (true)
        {
            bool busy = false;
            foreach (var counter in _counters.Values)
            {
                int depth = Volatile.Read(ref counter.Depth) - (ReferenceEquals(counter, own) ? ownDepth : 0);
                if (depth > 0) { busy = true; break; }
            }
            if (!busy) return true;
            if (clock.Elapsed >= timeout) return false;
            spin.SpinOnce();
        }
    }

    /// <summary>Reopens a gate whose drain timed out, so the engine it guards can be used (and released) again.</summary>
    public void Reopen() => Interlocked.Exchange(ref _state, Open);
}
