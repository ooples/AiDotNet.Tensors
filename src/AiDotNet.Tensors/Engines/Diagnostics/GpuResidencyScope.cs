using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Diagnostics;

/// <summary>The direction of a host/device crossing, or a forced wait on the device.</summary>
public enum GpuTransferKind
{
    /// <summary>Host memory copied to the device.</summary>
    HostToDevice,

    /// <summary>Device memory copied back to the host.</summary>
    DeviceToHost,

    /// <summary>The host blocked until the device finished its queued work.</summary>
    Synchronize,
}

/// <summary>One host/device crossing observed inside a <see cref="GpuResidencyScope"/>.</summary>
/// <param name="Kind">What crossed.</param>
/// <param name="Bytes">How many bytes crossed; zero for a synchronization.</param>
/// <param name="Backend">The backend that performed it, e.g. <c>CudaBackend</c>.</param>
/// <param name="Operation">
/// The engine operation that caused it, e.g. <c>DirectGpuTensorEngine.TensorMatMul</c>, when the scope captures
/// operations; otherwise null.
/// </param>
public readonly record struct GpuTransferEvent(GpuTransferKind Kind, long Bytes, string Backend, string? Operation);

/// <summary>
/// Counts every host/device transfer and device synchronization made while it is open, so a test can prove a GPU
/// workload stays resident (issue #1058).
/// </summary>
/// <remarks>
/// <para>
/// A GPU operation with no device kernel falls back to the CPU engine, which reads its inputs on the host and
/// writes a host result that the next GPU operation uploads again. That round trip, and the synchronization it
/// forces, costs more than most kernels, yet nothing observed it. Every backend now reports its transfer primitives
/// here, so the round trip is counted where it happens, whichever of the engine's call sites caused it.
/// </para>
/// <para>
/// With no scope open the cost is one volatile read per transfer. A scope started with
/// <c>captureOperations: true</c> also walks the stack on each transfer to name the engine operation responsible;
/// that is for diagnosis, not for timed runs.
/// </para>
/// <para>
/// A scope is thread-local by default, so parallel tests do not see each other's transfers. Pass
/// <c>processWide: true</c> when the work runs on other threads.
/// </para>
/// </remarks>
public sealed class GpuResidencyScope : IDisposable
{
    private readonly List<GpuTransferEvent> _events = new();
    private readonly object _gate = new();
    private readonly bool _processWide;
    private readonly int _threadId;
    private GpuResidencyScope? _previous;
    private int _disposed;
    // Set under _gate once the scope is unlinked. Add checks it under the same lock, so a recording that
    // raced Dispose either lands before Dispose returns or is dropped: the report never changes afterwards.
    private bool _closed;

    private GpuResidencyScope(bool captureOperations, bool processWide)
    {
        CaptureOperations = captureOperations;
        _processWide = processWide;
        _threadId = Environment.CurrentManagedThreadId;
    }

    /// <summary>Opens a scope that records transfers until it is disposed.</summary>
    /// <param name="captureOperations">Name the engine operation behind each transfer (walks the stack).</param>
    /// <param name="processWide">Record transfers from every thread, not just this one.</param>
    public static GpuResidencyScope Begin(bool captureOperations = false, bool processWide = false)
    {
        var scope = new GpuResidencyScope(captureOperations, processWide);
        DirectGpu.GpuLaunchProbe.EnterScope(scope);
        return scope;
    }

    /// <summary>Whether each event names the engine operation that caused it.</summary>
    public bool CaptureOperations { get; }

    internal bool ProcessWide => _processWide;

    internal GpuResidencyScope? Previous
    {
        get => _previous;
        set => _previous = value;
    }

    internal int ThreadId => _threadId;

    /// <summary>Every transfer recorded so far, in order.</summary>
    public IReadOnlyList<GpuTransferEvent> Events
    {
        get { lock (_gate) return _events.ToArray(); }
    }

    /// <summary>Host-to-device transfers.</summary>
    public int Uploads => Count(GpuTransferKind.HostToDevice);

    /// <summary>Device-to-host transfers.</summary>
    public int Downloads => Count(GpuTransferKind.DeviceToHost);

    /// <summary>Forced device synchronizations.</summary>
    public int Synchronizations => Count(GpuTransferKind.Synchronize);

    /// <summary>Bytes copied host to device.</summary>
    public long BytesUploaded => Bytes(GpuTransferKind.HostToDevice);

    /// <summary>Bytes copied device to host.</summary>
    public long BytesDownloaded => Bytes(GpuTransferKind.DeviceToHost);

    internal void Add(in GpuTransferEvent e)
    {
        lock (_gate)
        {
            if (!_closed) _events.Add(e);
        }
    }

    private int Count(GpuTransferKind kind)
    {
        lock (_gate)
        {
            int n = 0;
            foreach (var e in _events) if (e.Kind == kind) n++;
            return n;
        }
    }

    private long Bytes(GpuTransferKind kind)
    {
        lock (_gate)
        {
            long n = 0;
            foreach (var e in _events) if (e.Kind == kind) n += e.Bytes;
            return n;
        }
    }

    /// <summary>Stops recording.</summary>
    /// <exception cref="InvalidOperationException">A thread scope is disposed on another thread, or while a scope
    /// opened after it is still open. Scopes nest: unlinking an outer one first would leave the inner one's
    /// restore pointing back at a disposed scope, which would then collect transfers again.</exception>
    public void Dispose()
    {
        if (System.Threading.Volatile.Read(ref _disposed) != 0) return;
        if (!_processWide)
        {
            if (Environment.CurrentManagedThreadId != _threadId)
                throw new InvalidOperationException("A thread GpuResidencyScope must be disposed on the thread that opened it.");
            if (!DirectGpu.GpuLaunchProbe.IsInnermostThreadScope(this))
                throw new InvalidOperationException(
                    "GpuResidencyScopes nest: dispose the scope opened most recently on this thread first.");
        }

        // One caller unlinks the scope; a concurrent second Dispose of a process-wide scope must not decrement
        // the active-scope count again (that could reach zero while another scope is open and drop its events).
        if (System.Threading.Interlocked.Exchange(ref _disposed, 1) != 0) return;
        DirectGpu.GpuLaunchProbe.ExitScope(this);
        lock (_gate) _closed = true;
    }
}
