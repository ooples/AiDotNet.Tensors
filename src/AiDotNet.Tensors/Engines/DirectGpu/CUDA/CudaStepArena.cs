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
    private ConditionalWeakTable<CudaBackend.CudaGpuBuffer, RecordedSlot> _recorded = new();
    private sealed class RecordedSlot { internal int Index; internal RecordedSlot(int index) => Index = index; }
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
    internal long PlannedBytes => _plan?.Bytes ?? 0;

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
        internal readonly long Bytes;

        internal Plan(int[] size, long[] offset, long bytes)
        {
            Size = size;
            Offset = offset;
            Bytes = bytes;
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

        internal RangeOccupancy? Occupancy;
        internal Block(IntPtr ptr, IntPtr context, long contextGeneration, int slots)
        {
            Ptr = ptr;
            Context = context;
            ContextGeneration = contextGeneration;
            SlotLive = new int[slots];
        }
    }

    /// <summary>
    // Compressed range occupancy: O(n) metadata, O(log n) admission/return, including identical offsets.
    internal sealed class RangeOccupancy
    {
        private readonly int[] _start, _end, _max, _lazy;
        private readonly int _segments;

        internal RangeOccupancy(long[] offsets, long[] bytes)
        {
            var endpoints = new SortedSet<long>();
            for (int i = 0; i < offsets.Length; i++)
                if (offsets[i] >= 0) { endpoints.Add(offsets[i]); endpoints.Add(offsets[i] + bytes[i]); }
            var points = endpoints.ToArray();
            _segments = Math.Max(1, points.Length - 1);
            _start = new int[offsets.Length];
            _end = new int[offsets.Length];
            for (int i = 0; i < offsets.Length; i++)
            {
                _start[i] = Array.BinarySearch(points, offsets[i]);
                _end[i] = Array.BinarySearch(points, offsets[i] + bytes[i]) - 1;
            }
            _max = new int[4 * _segments];
            _lazy = new int[4 * _segments];
        }

        internal bool TryAcquire(int slot)
        {
            if (_start[slot] < 0 || Query(1, 0, _segments - 1, _start[slot], _end[slot]) != 0) return false;
            Add(1, 0, _segments - 1, _start[slot], _end[slot], 1);
            return true;
        }

        internal void Release(int slot) => Add(1, 0, _segments - 1, _start[slot], _end[slot], -1);

        private int Query(int node, int lo, int hi, int from, int to)
        {
            if (from <= lo && hi <= to) return _max[node];
            int mid = lo + (hi - lo) / 2, value = 0;
            if (from <= mid) value = Query(node * 2, lo, mid, from, to);
            if (to > mid) value = Math.Max(value, Query(node * 2 + 1, mid + 1, hi, from, to));
            return value + _lazy[node];
        }

        private void Add(int node, int lo, int hi, int from, int to, int delta)
        {
            if (from <= lo && hi <= to) { _max[node] += delta; _lazy[node] += delta; return; }
            int mid = lo + (hi - lo) / 2;
            if (from <= mid) Add(node * 2, lo, mid, from, to, delta);
            if (to > mid) Add(node * 2 + 1, mid + 1, hi, from, to, delta);
            _max[node] = _lazy[node] + Math.Max(_max[node * 2], _max[node * 2 + 1]);
        }
    }

    /// The planned slot for this allocation, or null for the pool. Call on the allocating thread only inside a
    /// training step (<see cref="TrainingStep.InStep"/>); when it returns null while recording, pass the pool's buffer to
    /// <see cref="Track"/>.
    /// </summary>
    internal CudaBackend.CudaGpuBuffer? TryRent(int size)
    {
        _pendingRecord = -1;
        if (Volatile.Read(ref _disposed)) return null;
        int thread = Environment.CurrentManagedThreadId;
        if (_thread != thread)
        {
            if (_thread != -1) return null;   // one training thread per arena
            _thread = thread;
        }
        int step = TrainingStep.Current;
        if (step != _step) BeginStep(step);

        switch (_mode)
        {
            case Mode.Recording:
                _pendingRecord = _sizes.Count;
                _sizes.Add(size);
                _allocatedAt.Add(_clock++);
                _returnedAt.Add(-1);
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
        if (_pendingRecord < 0) return;
        lock (_lock) { _recorded.Remove(buffer); _recorded.Add(buffer, new RecordedSlot(_pendingRecord)); }
        _pendingRecord = -1;
    }

    /// <summary>A pool buffer is going back: if recorded this step, that is the end of its lifetime.</summary>
    internal void OnReturned(CudaBackend.CudaGpuBuffer buffer)
    {
        if (_mode != Mode.Recording) return;
        lock (_lock)
        {
            if (_recorded.TryGetValue(buffer, out var recorded))
            {
                _recorded.Remove(buffer);
                _returnedAt[recorded.Index] = _clock++;
            }
        }
    }

    /// <summary>A slot buffer was disposed, released or finalized: its memory may be handed out again.</summary>
    internal static void OnSlotDead(Block block, int slot)
    {
        lock (block)
        {
            if (Interlocked.Exchange(ref block.SlotLive[slot], 0) == 0) return;
            block.Occupancy?.Release(slot);
        }
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
        if (offset < 0 || blockPtr == IntPtr.Zero || Volatile.Read(ref block.SlotLive[i]) != 0) { PoolFallbacks++; return null; }
        lock (block)
        {
            if (block.SlotLive[i] != 0 || !block.Occupancy!.TryAcquire(i)) { PoolFallbacks++; return null; }
            Volatile.Write(ref block.SlotLive[i], 1);
            Interlocked.Increment(ref block.Live);
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
        _sizes.Clear();
        _allocatedAt.Clear();
        _returnedAt.Clear();
        lock (_lock) _recorded = new();
        _clock = 0;
        _mode = Mode.Recording;
    }

    private void BuildPlan()
    {
        lock (_lock) _recorded = new();   // still out at the step's end: they stay on the pool
        int n = _sizes.Count;
        var bytes = new long[n];
        var order = new List<int>(n);
        for (int i = 0; i < n; i++)
        {
            bytes[i] = ((long)_sizes[i] * sizeof(float) + Alignment - 1) / Alignment * Alignment;
            if (_returnedAt[i] >= 0) order.Add(i);
        }
        // Largest first, each at the lowest offset free of every placed slot whose lifetime overlaps its own.
        order.Sort((a, b) => bytes[b].CompareTo(bytes[a]));
        var offset = new long[n];
        for (int i = 0; i < n; i++) offset[i] = -1;
        var placed = new List<int>(order.Count);
        var busy = new List<(long Start, long End)>();
        long total = 0;
        foreach (int i in order)
        {
            busy.Clear();
            foreach (int j in placed)
                if (_allocatedAt[j] < _returnedAt[i] && _allocatedAt[i] < _returnedAt[j])
                    busy.Add((offset[j], offset[j] + bytes[j]));
            busy.Sort((a, b) => a.Start.CompareTo(b.Start));
            long at = 0;
            foreach (var (start, end) in busy)
            {
                if (at + bytes[i] <= start) break;
                if (end > at) at = end;
            }
            offset[i] = at;
            placed.Add(i);
            if (at + bytes[i] > total) total = at + bytes[i];
        }
        if (placed.Count == 0) { _mode = Mode.Off; return; }

        IntPtr ptr;
        try { ptr = _allocateBlock((ulong)total); }
        catch (InvalidOperationException) { _mode = Mode.Off; return; }   // no room for the plan: the pool serves the steps
        _plan = new Plan(_sizes.ToArray(), offset, total);
        _block = new Block(ptr, _context, _contextGeneration, n) { Occupancy = new RangeOccupancy(offset, bytes) };
        _next = 0;
        _aborted = false;
        _mode = Mode.Replaying;
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
            _recorded = new();
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
