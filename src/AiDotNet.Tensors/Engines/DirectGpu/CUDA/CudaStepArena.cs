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
internal sealed class CudaStepArena
{
    /// <summary>Process switch for tests that need the plain pool.</summary>
    internal static bool Enabled = true;

    private const long Alignment = 256;
    private const int MaxReplans = 4;

    private readonly IntPtr _context;
    private readonly Func<ulong, IntPtr> _allocateBlock;
    private readonly Action<IntPtr> _freeBlock;
    private readonly object _lock = new();

    private enum Mode { WarmUp, Recording, Replaying, Off }
    private Mode _mode = Mode.WarmUp;
    private int _thread = -1;
    private int _step = -1;

    // Recording.
    private readonly List<int> _sizes = new();
    private readonly List<long> _allocatedAt = new();
    private readonly List<long> _returnedAt = new();
    private readonly Dictionary<CudaBackend.CudaGpuBuffer, int> _recorded = new(Autodiff.ReferenceEqualityComparer<CudaBackend.CudaGpuBuffer>.Instance);
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

    internal CudaStepArena(IntPtr context, Func<ulong, IntPtr> allocateBlock, Action<IntPtr> freeBlock)
    {
        _context = context;
        _allocateBlock = allocateBlock;
        _freeBlock = freeBlock;
    }

    private sealed class Plan
    {
        internal readonly int[] Size;
        internal readonly long[] Offset;     // -1: pool
        internal readonly int[][] Conflicts; // slots sharing memory: all must be dead to hand this one out
        internal readonly long Bytes;

        internal Plan(int[] size, long[] offset, int[][] conflicts, long bytes)
        {
            Size = size;
            Offset = offset;
            Conflicts = conflicts;
            Bytes = bytes;
        }
    }

    internal sealed class Block
    {
        internal readonly IntPtr Ptr;
        internal readonly bool[] SlotLive;
        internal int Live;

        internal Block(IntPtr ptr, int slots)
        {
            Ptr = ptr;
            SlotLive = new bool[slots];
        }
    }

    /// <summary>
    /// The planned slot for this allocation, or null for the pool. Call on the allocating thread only inside a
    /// training step (<see cref="TrainingStep.InStep"/>); when it returns null while recording, pass the pool's buffer to
    /// <see cref="Track"/>.
    /// </summary>
    internal CudaBackend.CudaGpuBuffer? TryRent(int size)
    {
        _pendingRecord = -1;
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
        lock (_lock) _recorded[buffer] = _pendingRecord;
        _pendingRecord = -1;
    }

    /// <summary>A pool buffer is going back: if recorded this step, that is the end of its lifetime.</summary>
    internal void OnReturned(CudaBackend.CudaGpuBuffer buffer)
    {
        if (_mode != Mode.Recording) return;
        lock (_lock)
        {
            if (_recorded.TryGetValue(buffer, out int index))
            {
                _recorded.Remove(buffer);
                _returnedAt[index] = _clock++;
            }
        }
    }

    /// <summary>A slot buffer was disposed, released or finalized: its memory may be handed out again.</summary>
    internal static void OnSlotDead(Block block, int slot)
    {
        if (!block.SlotLive[slot]) return;
        Volatile.Write(ref block.SlotLive[slot], false);
        Interlocked.Decrement(ref block.Live);
    }

    private CudaBackend.CudaGpuBuffer? Replay(int size)
    {
        var plan = _plan!;
        if (_aborted) { PoolFallbacks++; return null; }
        int i = _next++;
        if (i >= plan.Size.Length || plan.Size[i] != size)
        {
            // This step is not the recorded one: the rest of it uses the pool, and the next step records again.
            _aborted = true;
            PoolFallbacks++;
            return null;
        }
        long offset = plan.Offset[i];
        var block = _block!;
        if (offset < 0 || Volatile.Read(ref block.SlotLive[i])) { PoolFallbacks++; return null; }
        foreach (int j in plan.Conflicts[i])
            if (Volatile.Read(ref block.SlotLive[j])) { PoolFallbacks++; return null; }   // outlived its recorded lifetime
        block.SlotLive[i] = true;
        Interlocked.Increment(ref block.Live);
        SlotHits++;
        return CudaBackend.CudaGpuBuffer.ForArenaSlot(_context, new IntPtr(block.Ptr.ToInt64() + offset), size, block, i);
    }

    private void BeginStep(int step)
    {
        _step = step;
        FreeRetiredBlocks();
        switch (_mode)
        {
            case Mode.WarmUp:
                StartRecording();
                break;
            case Mode.Recording:
                BuildPlan();
                break;
            case Mode.Replaying:
                if (_aborted || _next != _plan!.Size.Length)
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
        lock (_lock) _recorded.Clear();
        _clock = 0;
        _mode = Mode.Recording;
    }

    private void BuildPlan()
    {
        lock (_lock) _recorded.Clear();   // still out at the step's end: they stay on the pool
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

        // Conflicts: every other slot sharing memory. Within a step the plan keeps them apart in time, but one that
        // outlives its recorded lifetime (into a later step, or past its recorded return) must not be aliased.
        var conflicts = new int[n][];
        var scratch = new List<int>();
        for (int i = 0; i < n; i++)
        {
            scratch.Clear();
            if (offset[i] >= 0)
                foreach (int j in placed)
                    if (j != i && offset[j] < offset[i] + bytes[i] && offset[i] < offset[j] + bytes[j])
                        scratch.Add(j);
            conflicts[i] = scratch.Count == 0 ? Array.Empty<int>() : scratch.ToArray();
        }

        IntPtr ptr;
        try { ptr = _allocateBlock((ulong)total); }
        catch (Exception) { _mode = Mode.Off; return; }   // no room for the plan: the pool serves the steps
        _plan = new Plan(_sizes.ToArray(), offset, conflicts, total);
        _block = new Block(ptr, n);
        _next = 0;
        _aborted = false;
        _mode = Mode.Replaying;
    }

    // The current block stops serving; its memory is freed once every slot it handed out is dead.
    private void Retire()
    {
        if (_block is { } block) _retiring.Add(block);
        _block = null;
        _plan = null;
    }

    private void FreeRetiredBlocks()
    {
        for (int i = _retiring.Count - 1; i >= 0; i--)
        {
            if (Volatile.Read(ref _retiring[i].Live) != 0) continue;
            try { _freeBlock(_retiring[i].Ptr); } catch (Exception) { /* the context may be gone */ }
            _retiring.RemoveAt(i);
        }
    }
}
