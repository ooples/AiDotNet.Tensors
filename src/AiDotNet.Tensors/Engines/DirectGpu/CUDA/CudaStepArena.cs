using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// A static memory plan for training steps, beyond a caching allocator. One step's device allocations are recorded
/// (size, order, and when each went back), packed offline into one device block by their recorded lifetimes, and every
/// later step serves allocation <c>i</c> at its planned offset: a pointer, no pool lookup, no driver call, no reclaim
/// charge, and a peak no larger than the packed plan. Buffers the recorded step did not return before the next step
/// (results it kept, buffers the GC freed) stay on the pool.
/// </summary>
/// <remarks>
/// <para>Safety does not rest on the replayed step matching the recording. A slot is handed out only when every slot
/// sharing its memory is dead (disposed or finalized): a buffer that lives longer than it did when recorded makes the
/// overlapping allocation fall back to the pool, never alias it. A size or count mismatch ends the replay for the rest
/// of the step and the next step records a new plan; a backend that keeps mismatching stops planning. Reuse is stream
/// ordered exactly as the pool's: every slot is used on the backend's compute stream.</para>
/// <para>The first step of a thread is not recorded (it allocates the step's one-time state: optimizer moments, lazily
/// uploaded weights); the second is, the third and later replay.</para>
/// </remarks>
internal sealed class CudaStepArena : IDisposable
{
    /// <summary>Process switch for tests that need the plain pool.</summary>
    internal static bool Enabled = true;

    private const long Alignment = 256;
    private const int MaxReplans = 4;
    // Multi-gigabyte training steps are already served in O(1) by the stream-ordered GpuBufferPool;
    // carving a single multi-GB contiguous cuMemAlloc block risks double-buffering VRAM and WDDM
    // PCIe paging whenever a step dynamically branches or replans.
    internal readonly long MaxPlannedBytes = 512L * 1024 * 1024;

    private readonly IntPtr _context;
    private readonly long _contextGeneration;
    private readonly Func<ulong, IntPtr> _allocateBlock;
    private readonly Action<IntPtr> _freeBlock;
    private readonly object _lock = new();
    private bool _disposed;

    private enum Mode { WarmUp, Recording, Replaying, Off }
    private Mode _mode = Mode.WarmUp;
    private bool _warmUpSeen;
    private int _thread = -1;
    private int _step = -1;

    // Recording.
    private readonly List<int> _sizes = new();
    private readonly List<long> _allocatedAt = new();
    private readonly List<long> _returnedAt = new();
    // Keyed by device pointer, not by buffer: a buffer key kept a buffer its caller dropped undisposed reachable, so its
    // finalizer never ran and its memory stayed out for as long as the arena kept recording. A pool pointer names one
    // live buffer at a time, and its entry goes when the buffer is returned.
    private readonly Dictionary<IntPtr, int> _recorded = new();
    private long _clock;
    private int _pendingRecord = -1;

    // Replay.
    private Plan? _plan;
    private Block? _block;
    private readonly List<Block> _retiring = new();
    private int _next;
    private bool _aborted;
    private int _replans;

    /// <summary>Slots handed out, pool fallbacks and replans, for tests and diagnostics.</summary>
    internal long SlotHits, PoolFallbacks;
    internal int Replans => _replans;
    internal long PlannedBytes => _plan?.TotalBytes ?? 0;

    internal CudaStepArena(IntPtr context, long contextGeneration, Func<ulong, IntPtr> allocateBlock, Action<IntPtr> freeBlock)
    {
        _context = context;
        _contextGeneration = contextGeneration;
        _allocateBlock = allocateBlock;
        _freeBlock = freeBlock;
    }

    private sealed class Plan
    {
        internal readonly int[] Size;
        internal readonly long[] Offset;     // -1: pool
        internal readonly long[] Bytes;      // each slot's aligned extent
        internal readonly long TotalBytes;

        internal Plan(int[] size, long[] offset, long[] bytes, long totalBytes)
        {
            Size = size;
            Offset = offset;
            Bytes = bytes;
            TotalBytes = totalBytes;
        }
    }

    internal sealed class Block
    {
        internal IntPtr Ptr;
        internal readonly IntPtr Context;
        internal readonly long ContextGeneration;
        internal readonly int[] SlotLive;
        internal int Live;
        internal int ArenaDisposed;

        // The extents of the slots handed out and not yet dead, sorted by start. They never overlap: a slot is handed
        // out only when its extent meets none of them, so the one starting last before a slot's end is the only one
        // that can overlap it. One live set for the block, in place of a list per slot of every slot sharing its
        // memory, which grows with the square of a step's same-size temporaries.
        private const long UnclaimedStart = long.MinValue;
        private readonly object _gate = new();
        private readonly List<long> _liveStart = new();
        private readonly List<long> _liveEnd = new();
        private readonly List<int> _liveSlot = new();
        private readonly long[] _slotStart;

        internal Block(IntPtr ptr, IntPtr context, long contextGeneration, int slots)
        {
            Ptr = ptr;
            Context = context;
            ContextGeneration = contextGeneration;
            SlotLive = new int[slots];
            _slotStart = new long[slots];
            for (int i = 0; i < slots; i++) _slotStart[i] = UnclaimedStart;
        }

        /// <summary>Marks <paramref name="slot"/> live over [start, end) unless it is live or a live slot meets it.</summary>
        internal bool TryClaim(int slot, long start, long end)
        {
            lock (_gate)
            {
                if (SlotLive[slot] != 0) return false;
                if (end > start)
                {
                    int before = FirstStartAtOrAfter(end);   // live extents starting before `end`: [0, before)
                    if (before > 0 && _liveEnd[before - 1] > start) return false;
                    int insert = FirstStartAtOrAfter(start);
                    _liveStart.Insert(insert, start);
                    _liveEnd.Insert(insert, end);
                    _liveSlot.Insert(insert, slot);
                    _slotStart[slot] = start;
                }
                else
                {
                    _slotStart[slot] = UnclaimedStart;
                }
                Volatile.Write(ref SlotLive[slot], 1);
                return true;
            }
        }

        /// <summary>Drops a dead slot's extent from the live set.</summary>
        internal void Release(int slot)
        {
            lock (_gate)
            {
                long start = _slotStart[slot];
                if (start == UnclaimedStart) return;
                int i = FirstStartAtOrAfter(start);
                while (i < _liveStart.Count && _liveStart[i] == start)
                {
                    if (_liveSlot[i] == slot)
                    {
                        _slotStart[slot] = UnclaimedStart;
                        _liveStart.RemoveAt(i);
                        _liveEnd.RemoveAt(i);
                        _liveSlot.RemoveAt(i);
                        return;
                    }
                    i++;
                }
            }
        }

        private int FirstStartAtOrAfter(long value)
        {
            int lo = 0, hi = _liveStart.Count;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (_liveStart[mid] < value) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        }
    }

    /// <summary>
    /// The planned slot for this allocation, or null for the pool. Call on the allocating thread only inside a
    /// training step (<see cref="TrainingStep.InStep"/>); when it returns null while recording, pass the pool's buffer to
    /// <see cref="Track"/>.
    /// </summary>
    internal CudaBackend.CudaGpuBuffer? TryRent(int size)
    {
        // One training thread per arena, claimed atomically by the first. Another thread touches no state, the pending
        // record included: clearing it between the owner's TryRent and Track lost the owner's record.
        if (!OwnedByThisThread(claim: true)) return null;
        _pendingRecord = -1;
        if (Volatile.Read(ref _disposed)) return null;
        int step = TrainingStep.Current;
        if (step != _step) BeginStep(step);

        switch (_mode)
        {
            case Mode.Recording:
                // Under the lock OnReturned takes: a buffer can go back from another thread (a finalizer) mid-record.
                lock (_lock)
                {
                    _pendingRecord = _sizes.Count;
                    _sizes.Add(size);
                    _allocatedAt.Add(_clock++);
                    _returnedAt.Add(-1);
                }
                return null;
            case Mode.Replaying:
                return Replay(size);
            default:
                return null;
        }
    }

    /// <summary>Records the pool buffer an allocation <see cref="TryRent"/> declined while recording.</summary>
    internal void Track(CudaBackend.CudaGpuBuffer buffer)
    {
        if (!OwnedByThisThread(claim: false)) return;
        int pending = _pendingRecord;
        _pendingRecord = -1;
        if (pending < 0) return;
        lock (_lock)
            if (pending < _sizes.Count && buffer.RawDevicePointer is var ptr && ptr != IntPtr.Zero) _recorded[ptr] = pending;
    }

    /// <summary>Test seam: runs on a claiming thread after it read the arena as unowned and before it claims it.</summary>
    internal Action? BeforeClaimForTests;

    /// <summary>Test seam: runs in a replay after the slot is claimed and before it counts as live in its block.</summary>
    internal Action? BeforeSlotGoesLiveForTests;

    private bool OwnedByThisThread(bool claim)
    {
        int thread = Environment.CurrentManagedThreadId;
        int owner = Volatile.Read(ref _thread);
        if (owner == thread) return true;
        if (!claim || owner != -1) return false;
        BeforeClaimForTests?.Invoke();
        int prior = Interlocked.CompareExchange(ref _thread, thread, -1);
        return prior == -1 || prior == thread;
    }

    /// <summary>A pool buffer is going back: if recorded this step, that is the end of its lifetime.</summary>
    internal void OnReturned(CudaBackend.CudaGpuBuffer buffer)
    {
        if (_mode != Mode.Recording) return;
        lock (_lock)
        {
            var ptr = buffer.RawDevicePointer;
            if (ptr != IntPtr.Zero && _recorded.TryGetValue(ptr, out int index))
            {
                _recorded.Remove(ptr);
                if (index < _returnedAt.Count) _returnedAt[index] = _clock++;
            }
        }
    }

    /// <summary>A slot buffer was disposed, released or finalized: its memory may be handed out again.</summary>
    internal static void OnSlotDead(Block block, int slot)
    {
        if (Interlocked.Exchange(ref block.SlotLive[slot], 0) == 0) return;
        block.Release(slot);
        if (Interlocked.Decrement(ref block.Live) == 0 && Volatile.Read(ref block.ArenaDisposed) != 0)
        {
            var ptr = Interlocked.Exchange(ref block.Ptr, IntPtr.Zero);
            if (ptr != IntPtr.Zero)
                CudaBackend.CompleteCudaBufferReclaimOrFinalizer(
                    0, -1, ptr, block.Context, block.ContextGeneration, IntPtr.Zero);
        }
    }

    private CudaBackend.CudaGpuBuffer? Replay(int size)
    {
        if (_aborted || _plan is not { } plan || _block is not { } block) { PoolFallbacks++; return null; }
        int i = _next++;
        if (i >= plan.Size.Length || plan.Size[i] != size)
        {
            // This step is not the recorded one: the rest of it uses the pool, and the next step records again.
            _aborted = true;
            PoolFallbacks++;
            return null;
        }
        long offset = plan.Offset[i];
        var blockPtr = Volatile.Read(ref block.Ptr);
        if (offset < 0 || blockPtr == IntPtr.Zero) { PoolFallbacks++; return null; }
        // A live slot sharing this one's memory outlived its recorded lifetime: never alias it.
        if (!block.TryClaim(i, offset, offset + plan.Bytes[i])) { PoolFallbacks++; return null; }
        BeforeSlotGoesLiveForTests?.Invoke();
        Interlocked.Increment(ref block.Live);
        // Dispose may have run on another thread since the pointer was read: it marks the block disposed, then frees it
        // if nothing is live. Both sides fence, so either it saw this slot live and left the block, or this sees the mark:
        // then the slot is undone, and the block freed here when this was all that held it.
        if (Volatile.Read(ref block.ArenaDisposed) != 0 || Volatile.Read(ref block.Ptr) == IntPtr.Zero)
        {
            Interlocked.Exchange(ref block.SlotLive[i], 0);
            block.Release(i);
            if (Interlocked.Decrement(ref block.Live) == 0 && Volatile.Read(ref block.ArenaDisposed) != 0)
            {
                var ptr = Interlocked.Exchange(ref block.Ptr, IntPtr.Zero);
                if (ptr != IntPtr.Zero) _freeBlock(ptr);
            }
            PoolFallbacks++;
            return null;
        }
        SlotHits++;
        return CudaBackend.CudaGpuBuffer.ForArenaSlot(_context, new IntPtr(blockPtr.ToInt64() + offset), size, block, i);
    }

    private void BeginStep(int step)
    {
        _step = step;
        FreeRetiredBlocks();
        switch (_mode)
        {
            case Mode.WarmUp:
                // The first step this arena sees allocates one-time state; the second is recorded.
                if (_warmUpSeen) StartRecording();
                else _warmUpSeen = true;
                break;
            case Mode.Recording:
                BuildPlan();
                break;
            case Mode.Replaying:
                if (_aborted || _plan is null || _next != _plan.Size.Length)
                {
                    if (++_replans > MaxReplans) { Retire(); _mode = Mode.Off; return; }
                    Retire();
                    StartRecording();
                    return;
                }
                _next = 0;
                break;
        }
    }

    private void StartRecording()
    {
        lock (_lock)
        {
            _recorded.Clear();
            _sizes.Clear();
            _allocatedAt.Clear();
            _returnedAt.Clear();
            _clock = 0;
            _pendingRecord = -1;
            _mode = Mode.Recording;
        }
    }

    private void BuildPlan()
    {
        long clockSnapshot;
        lock (_lock)
        {
            _recorded.Clear();   // still out at the step's end: they stay on the pool
            _mode = Mode.Off;    // no more returns are recorded; Replaying once the plan is in place below
            clockSnapshot = _clock;
        }
        int n = _sizes.Count;
        if (n == 0 || clockSnapshot <= 0 || clockSnapshot > int.MaxValue) { _mode = Mode.Off; return; }

        var bytes = new long[n];
        var offset = new long[n];
        var events = new int[(int)clockSnapshot];
        int placedCount = 0;
        for (int i = 0; i < n; i++)
        {
            offset[i] = -1;
            bytes[i] = ((long)_sizes[i] * sizeof(float) + Alignment - 1) / Alignment * Alignment;
            long allocAt = _allocatedAt[i];
            long retAt = _returnedAt[i];
            if (retAt < 0) continue;
            if (bytes[i] <= 0)
            {
                offset[i] = 0;
                placedCount++;
                continue;
            }
            if (allocAt >= 0 && allocAt < retAt && retAt < clockSnapshot)
            {
                events[(int)allocAt] = i + 1;
                events[(int)retAt] = -(i + 1);
                placedCount++;
            }
        }
        if (placedCount == 0) { _mode = Mode.Off; return; }

        // Chronological sweep-line allocator with immediate free-interval coalescing and frontier
        // retraction: replaces the O(n^2 log n) pairwise overlap scan with O(n * activeGaps),
        // keeping every overlapping [allocatedAt, returnedAt) pair strictly disjoint.
        var freeGaps = new List<(long Start, long End)>(32);
        long frontier = 0;
        long total = 0;
        for (int t = 0; t < events.Length; t++)
        {
            int ev = events[t];
            if (ev == 0) continue;
            if (ev > 0)
            {
                int slot = ev - 1;
                long need = bytes[slot];
                int bestIdx = -1;
                long bestLen = long.MaxValue;
                for (int g = 0; g < freeGaps.Count; g++)
                {
                    long len = freeGaps[g].End - freeGaps[g].Start;
                    if (len >= need && len < bestLen)
                    {
                        bestLen = len;
                        bestIdx = g;
                        if (len == need) break;
                    }
                }
                if (bestIdx >= 0)
                {
                    var gap = freeGaps[bestIdx];
                    offset[slot] = gap.Start;
                    if (bestLen == need) freeGaps.RemoveAt(bestIdx);
                    else freeGaps[bestIdx] = (gap.Start + need, gap.End);
                }
                else
                {
                    offset[slot] = frontier;
                    frontier += need;
                    if (frontier > total) total = frontier;
                }
            }
            else
            {
                int slot = (-ev) - 1;
                long start = offset[slot];
                long end = start + bytes[slot];
                int idx = FirstGapAtOrAfter(freeGaps, start);
                bool leftAdj = idx > 0 && freeGaps[idx - 1].End == start;
                bool rightAdj = idx < freeGaps.Count && end == freeGaps[idx].Start;
                if (leftAdj && rightAdj)
                {
                    freeGaps[idx - 1] = (freeGaps[idx - 1].Start, freeGaps[idx].End);
                    freeGaps.RemoveAt(idx);
                }
                else if (leftAdj)
                {
                    freeGaps[idx - 1] = (freeGaps[idx - 1].Start, end);
                }
                else if (rightAdj)
                {
                    freeGaps[idx] = (start, freeGaps[idx].End);
                }
                else if (end == frontier)
                {
                    frontier = start;
                }
                else
                {
                    freeGaps.Insert(idx, (start, end));
                }

                while (freeGaps.Count > 0 && freeGaps[freeGaps.Count - 1].End == frontier)
                {
                    frontier = freeGaps[freeGaps.Count - 1].Start;
                    freeGaps.RemoveAt(freeGaps.Count - 1);
                }
            }
        }
        if (total == 0) total = Alignment;
        if (total > MaxPlannedBytes) { _mode = Mode.Off; return; }

        // A slot that outlives its recorded lifetime (into a later step, or past its recorded return) must not be
        // aliased: the block's live set (Block.TryClaim) refuses any slot whose memory a live slot still holds.
        IntPtr ptr;
        try { ptr = _allocateBlock((ulong)total); }
        catch (InvalidOperationException noRoom)
        {
            // No room for the plan (GpuOutOfMemoryException is one): the pool serves the steps from now on. Recorded,
            // so a workload that silently lost its arena shows it.
            GpuLaunchProbe.OnFallback("CudaStepArena.BuildPlan", noRoom);
            _mode = Mode.Off;
            return;
        }
        _plan = new Plan(_sizes.ToArray(), offset, bytes, total);
        _block = new Block(ptr, _context, _contextGeneration, n);
        _next = 0;
        _aborted = false;
        _mode = Mode.Replaying;
    }

    private static int FirstGapAtOrAfter(List<(long Start, long End)> gaps, long value)
    {
        int lo = 0, hi = gaps.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (gaps[mid].Start < value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    // The current block stops serving; its memory is freed once every slot it handed out is dead.
    private void Retire()
    {
        lock (_lock)
        {
            if (_block is { } block) _retiring.Add(block);
            _block = null;
            _plan = null;
        }
    }

    internal void FreeRetiredBlocks()
    {
        List<IntPtr>? toFree = null;
        lock (_lock)
        {
            for (int i = _retiring.Count - 1; i >= 0; i--)
            {
                if (Volatile.Read(ref _retiring[i].Live) != 0) continue;
                var ptr = Interlocked.Exchange(ref _retiring[i].Ptr, IntPtr.Zero);
                if (ptr != IntPtr.Zero) (toFree ??= new List<IntPtr>()).Add(ptr);
                _retiring.RemoveAt(i);
            }
        }
        if (toFree is not null)
            foreach (var ptr in toFree) _freeBlock(ptr);
    }

    public void Dispose()
    {
        Block[] blocks;
        lock (_lock)
        {
            if (_disposed) return;
            Volatile.Write(ref _disposed, true);
            _mode = Mode.Off;
            _recorded.Clear();
            if (_block is { } active)
            {
                _retiring.Add(active);
                _block = null;
            }
            _plan = null;
            blocks = _retiring.ToArray();
            _retiring.Clear();
        }
        foreach (var block in blocks)
        {
            Interlocked.Exchange(ref block.ArenaDisposed, 1);
            if (Volatile.Read(ref block.Live) == 0)
            {
                var ptr = Interlocked.Exchange(ref block.Ptr, IntPtr.Zero);
                if (ptr != IntPtr.Zero) _freeBlock(ptr);
            }
        }
    }
}
