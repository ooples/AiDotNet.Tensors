using System;
using System.Buffers;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

// A GEMM participant owns this lease until its joined local-finally. It is not thread-static:
// a TPL local can move between threads without lending its storage to another participant.
internal sealed class PackBothParticipantWorkspace : IDisposable
{
    private readonly ArrayPool<byte> _pool;
    private readonly string _tag;
    private byte[]? _buffer;
    private int _capacity, _disposed;
    private bool _tracked;
    private readonly bool _participantTracked;

    internal PackBothParticipantWorkspace(string tag, ArrayPool<byte>? pool = null)
    {
        _tag = tag;
        _pool = pool ?? ArrayPool<byte>.Shared;
        if (DiagnosticsEnabled)
        {
            lock (DiagnosticGate) s_ownersCreated++;
            _participantTracked = true;
        }
    }

    internal Span<byte> ForPanel(int bytes)
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(PackBothParticipantWorkspace));
        if (bytes < 0) throw new ArgumentOutOfRangeException(nameof(bytes));
        if (bytes == 0) return Span<byte>.Empty;
        if (bytes <= _capacity) return _buffer.AsSpan(0, bytes);

        // A previous panel has finished before the next body calls this method. Return its lease first,
        // so capacity growth never holds both arrays and an allocation failure cannot strand the old one.
        ReleaseBuffer();
        _buffer = _pool.Rent(bytes);
        _capacity = bytes;
        _tracked = RecordRent(bytes, _buffer.Length);
        CpuStorageAllocationDiagnostics.Record<byte>(_tag, bytes);
        return _buffer.AsSpan(0, bytes);
    }

    private void ReleaseBuffer()
    {
        var buffer = _buffer;
        if (buffer is null) return;
        int requested = _capacity;
        bool tracked = _tracked;
        _buffer = null;
        _capacity = 0;
        _tracked = false;
        try { _pool.Return(buffer); }
        finally { if (tracked) RecordReturn(requested, buffer.Length); }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        try { ReleaseBuffer(); }
        finally { if (_participantTracked) lock (DiagnosticGate) s_ownersDisposed++; }
    }

    // Opt-in diagnostics for focused allocation/rounding proofs, separate from storage-request counters.
    // These report this operator's successfully recorded participant leases, not process backing acquisitions.
    internal static bool DiagnosticsEnabled;
    private static readonly object DiagnosticGate = new();
    private static long s_rents, s_returns, s_requested, s_lentBacking, s_liveRequested, s_liveBacking;
    private static long s_peakRequested, s_peakBacking;
    private static long s_ownersCreated, s_ownersDisposed;

    private static bool RecordRent(int requested, int backing)
    {
        if (!DiagnosticsEnabled) return false;
        lock (DiagnosticGate)
        {
            s_rents++;
            s_requested += requested;
            s_lentBacking += backing;
            s_liveRequested += requested;
            s_liveBacking += backing;
            s_peakRequested = Math.Max(s_peakRequested, s_liveRequested);
            s_peakBacking = Math.Max(s_peakBacking, s_liveBacking);
        }
        return true;
    }

    private static void RecordReturn(int requested, int backing)
    {
        lock (DiagnosticGate)
        {
            s_returns++;
            s_liveRequested -= requested;
            s_liveBacking -= backing;
        }
    }

    internal static void ResetDiagnostics()
    {
        lock (DiagnosticGate)
        {
            if (s_liveBacking != 0 || s_ownersCreated != s_ownersDisposed)
                throw new InvalidOperationException("Cannot reset active participants or leases.");
            s_rents = s_returns = s_requested = s_lentBacking = s_liveRequested = s_liveBacking = 0;
            s_peakRequested = s_peakBacking = 0;
            s_ownersCreated = s_ownersDisposed = 0;
        }
    }

    internal static Dictionary<string, long> SnapshotDiagnostics()
    {
        lock (DiagnosticGate)
            return new Dictionary<string, long>
            {
                ["rents"] = s_rents, ["returns"] = s_returns,
                ["owners_created"] = s_ownersCreated, ["owners_disposed"] = s_ownersDisposed,
                ["requested_bytes"] = s_requested, ["lent_backing_bytes_including_reuse"] = s_lentBacking,
                ["live_requested_bytes"] = s_liveRequested, ["live_backing_bytes"] = s_liveBacking,
                ["peak_recorded_live_requested_bytes"] = s_peakRequested,
                ["peak_recorded_live_backing_bytes"] = s_peakBacking,
            };
    }
}
