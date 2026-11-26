using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Between-step loans of a compiled plan's idle step memory (<see cref="IIdleDeviceMemoryLender"/>).
/// </summary>
/// <remarks>
/// A loan is a non-owning <see cref="CudaGpuBuffer"/> over part of a lent buffer: it is not counted as allocated (the
/// lender's buffer already is), and releasing it hands the range back to the lent buffer instead of to the driver. When
/// its owner reclaims the memory while the loan is still alive, the loan is moved: fresh memory, a device copy queued on
/// the compute stream, and the same buffer object then addresses the copy and owns it. Every capture moves every live
/// loan before it begins and takes none while it is open, so no graph ever bakes in a lent address; an export to
/// another library (DLPack) moves its loan first for the same reason.
/// </remarks>
public sealed partial class CudaBackend : IIdleDeviceMemoryLender, IIdleLoanCaptureGuard
{
    IDisposable IIdleLoanCaptureGuard.SuspendLoansForCapture()
    {
        SuspendIdleLoans();
        return new LoanSuspension(this);
    }

    private sealed class LoanSuspension : IDisposable
    {
        private CudaBackend? _backend;
        public LoanSuspension(CudaBackend backend) => _backend = backend;
        public void Dispose() => System.Threading.Interlocked.Exchange(ref _backend, null)?.ResumeIdleLoans();
    }

    void IIdleDeviceMemoryLender.LendIdle(object owner, IReadOnlyCollection<IGpuBuffer> buffers) => LendIdle(owner, buffers);
    void IIdleDeviceMemoryLender.ReclaimIdle(object owner) => ReclaimIdle(owner);
    void IIdleDeviceMemoryLender.MoveLiveLoans() => MoveLiveIdleLoans();
    void IIdleDeviceMemoryLender.LendIdleOnDemand(object owner, Func<IReadOnlyCollection<IGpuBuffer>?> collect)
        => LendIdleOnDemand(owner, collect);

    private const long IdleLoanAlignment = 256;
    private readonly object _idleLock = new();
    private readonly List<IdleRegion> _idleRegions = new();
    private readonly List<IdleLoan> _idleLoans = new();
    // Owners that lent on demand and whose buffers are not collected yet.
    private readonly List<(object Owner, Func<IReadOnlyCollection<IGpuBuffer>?> Collect)> _idlePending = new();
    private int _idleSuspended;
    [ThreadStatic] private static int t_noIdleLoan;

    private static bool s_idleLoanDiag => global::AiDotNet.Tensors.Engines.Compilation.CompiledTrainingDiagnostics.TraceIdleLoans;

    /// <summary>Loans served, bytes served, requests not served while memory was lent, loans moved and bytes moved.</summary>
    internal (long Loans, long LoanBytes, long Misses, long MissBytes, long Moves, long MoveBytes) IdleLoanCounters
    {
        get { lock (_idleLock) return (_idleLoanCount, _idleLoanBytes, _idleMissCount, _idleMissBytes, _idleMoveCount, _idleMoveBytes); }
    }
    private long _idleLoanCount, _idleLoanBytes, _idleMissCount, _idleMissBytes, _idleMoveCount, _idleMoveBytes, _idleLargestMiss;

    /// <summary>Bytes currently lent to this backend by every owner.</summary>
    internal long IdleLentBytes
    {
        get
        {
            lock (_idleLock)
            {
                long bytes = 0;
                foreach (var region in _idleRegions) bytes += region.Bytes;
                return bytes;
            }
        }
    }

    private sealed class IdleRegion
    {
        internal IdleRegion(object owner, CudaGpuBuffer source, IntPtr basePtr, long bytes)
        {
            Owner = owner;
            Source = source;
            Base = basePtr;
            Bytes = bytes;
            Free.Add((0, bytes));
        }

        internal readonly object Owner;
        // Held so the lent memory outlives every loan of it, whatever happens to the owner.
        internal readonly CudaGpuBuffer Source;
        internal readonly IntPtr Base;
        internal readonly long Bytes;
        // Free ranges, sorted by offset and never adjacent.
        internal readonly List<(long Offset, long Length)> Free = new();
        internal bool Withdrawn;
    }

    /// <summary>One loan: a range of a lent region and the buffer object that uses it.</summary>
    internal sealed class IdleLoan
    {
        internal IdleLoan(CudaBackend lender, object region, long offset, long length, CudaGpuBuffer borrower)
        {
            Lender = lender;
            Region = region;
            Offset = offset;
            Length = length;
            Borrower = new WeakReference<CudaGpuBuffer>(borrower);
        }

        internal readonly CudaBackend Lender;
        internal readonly object Region;
        internal readonly long Offset;
        internal readonly long Length;
        internal readonly WeakReference<CudaGpuBuffer> Borrower;
        // CompiledTrainingDiagnostics.TraceIdleLoans: where the loan was taken, reported if it has to be moved.
        internal string? Origin;
        private int _state;   // 0 lent, 1 returned, 2 moved to memory of its own

        internal bool TryMarkReturned() => Interlocked.CompareExchange(ref _state, 1, 0) == 0;
        internal bool TryMarkMoved() => Interlocked.CompareExchange(ref _state, 2, 0) == 0;
        internal bool IsLent => Volatile.Read(ref _state) == 0;
    }

    /// <summary>Allocations inside this scope never borrow: their address is handed to a library that keeps it.</summary>
    internal static NoIdleLoanScope NoIdleLoans() => new(0);

    internal readonly struct NoIdleLoanScope : IDisposable
    {
        internal NoIdleLoanScope(int _) => t_noIdleLoan++;
        public void Dispose() => t_noIdleLoan--;
    }

    private void LendIdle(object owner, IReadOnlyCollection<IGpuBuffer> buffers)
    {
        if (owner is null) throw new ArgumentNullException(nameof(owner));
        if (buffers is null) throw new ArgumentNullException(nameof(buffers));
        ReclaimIdle(owner);
        using var _ = PushContext();
        lock (_idleLock) AddIdleRegionsLocked(owner, buffers);
    }

    private void LendIdleOnDemand(object owner, Func<IReadOnlyCollection<IGpuBuffer>?> collect)
    {
        if (owner is null) throw new ArgumentNullException(nameof(owner));
        if (collect is null) throw new ArgumentNullException(nameof(collect));
        ReclaimIdle(owner);
        lock (_idleLock) _idlePending.Add((owner, collect));
    }

    private void AddIdleRegionsLocked(object owner, IReadOnlyCollection<IGpuBuffer> buffers)
    {
        foreach (var buffer in buffers)
        {
            if (buffer is not CudaGpuBuffer cuda || cuda.IsArenaSlot || cuda.IsIdleLoan || cuda.IsReleased) continue;
            var ptr = cuda.RawDevicePointer;
            long bytes = cuda.SizeInBytes / IdleLoanAlignment * IdleLoanAlignment;
            if (ptr == IntPtr.Zero || bytes <= 0) continue;
            _idleRegions.Add(new IdleRegion(owner, cuda, ptr, bytes));
            // Debug switch: lent memory reads as NaN, so a lender that reads a buffer it declared idle fails loudly.
            if (PoisonUninitializedBuffers)
                CuBlasNative.CheckCudaResult(
                    CudaNativeBindings.cuMemsetD8Async(ptr, 0xFF, (ulong)bytes, _stream), "cuMemsetD8Async(lent poison)");
        }
    }

    // Collects what the on-demand owners lend: an allocation needs memory the pool could not serve. Under _idleLock,
    // so an owner's reclaim (the start of its next step) waits for its collection and finds it registered.
    // A collector that throws lends nothing (its owner keeps its memory); the others still lend, and the failure does
    // not surface in the allocation that happened to ask (it is traced instead).
    private void CollectPendingIdleLocked()
    {
        var pending = _idlePending.ToArray();
        _idlePending.Clear();
        foreach (var (owner, collect) in pending)
        {
            IReadOnlyCollection<IGpuBuffer>? buffers;
            try
            {
                buffers = collect();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    "Idle step memory: a lender's collection failed, so it lends nothing this time: " + ex.Message);
                continue;
            }
            if (buffers is { Count: > 0 }) AddIdleRegionsLocked(owner, buffers);
        }
    }

    private void ReclaimIdle(object owner)
    {
        if (owner is null) throw new ArgumentNullException(nameof(owner));
        List<IdleLoan>? live = null;
        long lent = 0;
        lock (_idleLock)
        {
            for (int i = _idlePending.Count - 1; i >= 0; i--)
                if (ReferenceEquals(_idlePending[i].Owner, owner)) _idlePending.RemoveAt(i);
            for (int i = _idleRegions.Count - 1; i >= 0; i--)
            {
                var region = _idleRegions[i];
                if (!ReferenceEquals(region.Owner, owner)) continue;
                region.Withdrawn = true;
                lent += region.Bytes;
                _idleRegions.RemoveAt(i);
            }
            if (lent == 0) return;
            for (int i = _idleLoans.Count - 1; i >= 0; i--)
            {
                var loan = _idleLoans[i];
                if (!((IdleRegion)loan.Region).Withdrawn) continue;
                _idleLoans.RemoveAt(i);
                if (loan.IsLent) (live ??= new()).Add(loan);
            }
        }
        // Every live loan is moved even when one cannot be: a loan left behind would keep reading memory the plan's
        // next step overwrites.
        List<Exception>? failures = null;
        if (live is not null)
            foreach (var loan in live)
            {
                try { MoveIdleLoan(loan); }
                catch (Exception ex)   // OOM included: the remaining loans still have to move; rethrown below
                {
                    (failures ??= new()).Add(ex);
                }
            }
        if (failures is not null)
            throw failures.Count == 1 ? failures[0] : new AggregateException(
                "Some buffers borrowed from a compiled plan's idle step memory could not be moved.", failures);
        if (s_idleLoanDiag)
        {
            var c = IdleLoanCounters;
            AiDotNetEngine.LogDiagnostic($"[IDLE-LOAN] reclaimed lent_bytes={lent} loans={c.Loans} loan_bytes={c.LoanBytes} " +
                $"misses={c.Misses} miss_bytes={c.MissBytes} largest_miss={Interlocked.Read(ref _idleLargestMiss)} " +
                $"moves={c.Moves} move_bytes={c.MoveBytes}");
        }
    }

    /// <summary>
    /// Moves every live loan to memory of its own and stops lending until <see cref="ResumeIdleLoans"/>: a capture is
    /// about to begin, and a graph must never bake in an address its lender will overwrite.
    /// </summary>
    private void SuspendIdleLoans()
    {
        lock (_idleLock) _idleSuspended++;
        try
        {
            MoveLiveIdleLoans();
        }
        catch
        {
            ResumeIdleLoans();   // the capture never opens
            throw;
        }
    }

    /// <summary>
    /// Moves every live loan to memory of its own, so that no address read from here on is a lent one. A capture's
    /// preparation calls this before it reads device addresses into tables a graph keeps (the step I/O's
    /// multi-tensor tables): a loan moved later, when the capture itself begins, would leave those tables naming
    /// memory its lender overwrites and lends again.
    /// </summary>
    internal void MoveLiveIdleLoans()
    {
        List<IdleLoan>? live = null;
        lock (_idleLock)
        {
            for (int i = _idleLoans.Count - 1; i >= 0; i--)
            {
                var loan = _idleLoans[i];
                if (loan.IsLent) (live ??= new()).Add(loan);
            }
        }
        if (live is null) return;
        // Every live loan is moved even when one cannot be, as a reclaim does.
        List<Exception>? failures = null;
        foreach (var loan in live)
        {
            try { MoveIdleLoan(loan); }
            catch (Exception ex) { (failures ??= new()).Add(ex); }
        }
        if (failures is not null)
            throw failures.Count == 1 ? failures[0] : new AggregateException(
                "Some buffers borrowed from a compiled plan's idle step memory could not be moved.", failures);
    }

    private void ResumeIdleLoans()
    {
        lock (_idleLock) _idleSuspended--;
    }

    /// <summary>A loan of idle step memory for <paramref name="size"/> floats, or null.</summary>
    private CudaGpuBuffer? TryRentIdleLoan(int size)
    {
        if ((_idleRegions.Count == 0 && _idlePending.Count == 0) || t_noIdleLoan > 0 || IsCapturingOnThisThread
            || _backendStreamCaptureActive
            || CaptureMemoryPool.Current is not null || AiDotNet.Tensors.Engines.Gpu.GpuBufferReleaseDeferral.IsActive)
            return null;
        long need = ((long)size * sizeof(float) + IdleLoanAlignment - 1) / IdleLoanAlignment * IdleLoanAlignment;
        lock (_idleLock)
        {
            if (_idleSuspended > 0) return null;
            if (_idlePending.Count > 0) CollectPendingIdleLocked();
            if (_idleRegions.Count == 0) return null;
            IdleRegion? best = null;
            int bestRange = -1;
            long bestLength = long.MaxValue;
            foreach (var region in _idleRegions)
            {
                var free = region.Free;
                for (int r = 0; r < free.Count; r++)
                    if (free[r].Length >= need && free[r].Length < bestLength)
                    {
                        best = region;
                        bestRange = r;
                        bestLength = free[r].Length;
                    }
            }
            if (best is null)
            {
                _idleMissCount++;
                _idleMissBytes += need;
                if (need > _idleLargestMiss) _idleLargestMiss = need;
                return null;
            }
            var (offset, length) = best.Free[bestRange];
            if (length == need) best.Free.RemoveAt(bestRange);
            else best.Free[bestRange] = (offset + need, length - need);
            var borrower = CudaGpuBuffer.ForIdleLoan(_cudaContext, best.Base + (nint)offset, size);
            var loan = new IdleLoan(this, best, offset, need, borrower);
            if (s_idleLoanDiag)
                loan.Origin = string.Join(" < ", CallerFrames(new System.Diagnostics.StackTrace(2, false), 5, "CudaBackend"));
            borrower.AttachIdleLoan(loan);
            _idleLoans.Add(loan);
            _idleLoanCount++;
            _idleLoanBytes += need;
            return borrower;
        }
    }

    /// <summary>A loan ended (released, disposed or finalized): its range is free again, unless reclaimed meanwhile.
    /// No driver call: this can run on the finalizer thread.</summary>
    internal void ReturnIdleLoan(IdleLoan loan)
    {
        lock (_idleLock)
        {
            _idleLoans.Remove(loan);
            var region = (IdleRegion)loan.Region;
            if (region.Withdrawn) return;
            var free = region.Free;
            int at = 0;
            while (at < free.Count && free[at].Offset < loan.Offset) at++;
            long start = loan.Offset, end = loan.Offset + loan.Length;
            if (at > 0 && free[at - 1].Offset + free[at - 1].Length == start)
            {
                start = free[at - 1].Offset;
                free.RemoveAt(--at);
            }
            if (at < free.Count && free[at].Offset == end)
            {
                end = free[at].Offset + free[at].Length;
                free.RemoveAt(at);
            }
            free.Insert(at, (start, end - start));
        }
    }

    /// <summary>
    /// Moves a live loan to memory of its own: a fresh owned allocation, the contents copied on the compute stream (after
    /// every earlier use of the loan, before the lender's next write), and the borrower's address switched to it.
    /// </summary>
    // The copy is queued on the backend's stream and the borrower switches to its own memory after it. A borrower is
    // used between the lending plan's steps; a kernel another thread queues on it while the plan starts its next step
    // (the only caller of a reclaim) races that step the same way as any buffer used by two threads at once.
    /// <summary>Test hook: runs with the fresh memory just before a loan's contents are copied into it.</summary>
    internal static Action<IGpuBuffer>? TestFaultBeforeIdleLoanMoveCopy;

    internal void MoveIdleLoan(IdleLoan loan)
    {
        if (!loan.IsLent) return;
        if (!loan.Borrower.TryGetTarget(out var borrower))
        {
            // Collected, its finalizer not yet run: nothing can use the memory again.
            if (loan.TryMarkReturned()) ReturnIdleLoan(loan);
            return;
        }
        if (borrower.HasCapturePins)
            throw new InvalidOperationException(
                "A buffer borrowed from a compiled plan's idle step memory was baked into a captured graph or exported; "
                + "it cannot be moved before the plan's next step overwrites that memory.");
        int size = borrower.Size;
        if (s_idleLoanDiag) AiDotNetEngine.LogDiagnostic($"[IDLE-MOVE] bytes={loan.Length} at {loan.Origin}");
        using var _ = PushContext();
        CudaGpuBuffer owned;
        using (NoIdleLoans())
        {
            owned = _asyncAlloc
                ? TryRentAsyncBuffer(size) ?? new CudaGpuBuffer(_cudaContext, AllocDeviceMemoryAsync((ulong)size * sizeof(float)),
                    size, returnToPool: ReturnAsyncBuffer, asyncFreeStream: _mainStream, account: _memoryAccount)
                : (CudaGpuBuffer)AllocateBufferFromPool(size);
        }
        var from = borrower.RawDevicePointer;
        try
        {
            TestFaultBeforeIdleLoanMoveCopy?.Invoke(owned);
            if (from != IntPtr.Zero)
                CuBlasNative.CheckCudaResult(
                    CudaNativeBindings.cuMemcpyDtoDAsync(owned.Handle, from, (ulong)size * sizeof(float), _stream),
                    "cuMemcpyDtoDAsync(idle loan move)");
        }
        catch
        {
            owned.Dispose();   // the move did not happen: the fresh memory goes back, the borrower keeps its loan
            throw;
        }
        if (!borrower.MoveIdleLoanTo(owned))
        {
            owned.Dispose();   // released meanwhile: its range went back through ReturnIdleLoan
            return;
        }
        // The range is free again in a region still lent (a capture moved it); a reclaimed region keeps nothing.
        ReturnIdleLoan(loan);
        lock (_idleLock)
        {
            _idleMoveCount++;
            _idleMoveBytes += loan.Length;
        }
    }

    /// <summary>A loan is about to be handed to another library by address: give it memory of its own first.</summary>
    private void EnsureNotIdleLoan(IGpuBuffer buffer)
    {
        if (GpuBufferViewRoot.Of(buffer) is CudaGpuBuffer { IdleLoanState: { } loan }) MoveIdleLoan(loan);
    }
}
