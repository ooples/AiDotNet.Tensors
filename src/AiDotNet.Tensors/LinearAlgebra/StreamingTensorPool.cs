// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;
using K4os.Compression.LZ4;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Cross-backend streaming weight pool (issue #276 sub-feature 2). Tracks
/// resident weight tensors by last-access timestamp; when the pool's total
/// resident bytes exceeds <see cref="GpuOffloadOptions.StreamingPoolMaxResidentBytes"/>,
/// the LRU entry is paged out to a memory-mapped backing store and the
/// caller's reference is invalidated. Subsequent access through
/// <see cref="Rehydrate{T}"/> brings the tensor back into the resident set.
///
/// <para>Used by every direct-GPU backend (CUDA / HIP / Metal / OpenCL /
/// Vulkan / WebGPU): each backend's dispatch layer calls
/// <see cref="MarkAccessed"/> on the weight before kernel launch and
/// <see cref="Rehydrate{T}"/> if the weight has been evicted. Same shape
/// as DeepSpeed's ZeRO-Offload eviction policy.</para>
/// </summary>
public sealed class StreamingTensorPool : IDisposable
{
    private readonly object _lock = new();
    private readonly Dictionary<long, Entry> _entries = new();
    private readonly LinkedList<long> _lruOrder = new(); // most recent at head
    private readonly Dictionary<long, LinkedListNode<long>> _lruIndex = new();

    // Schedule-aware (Belady-optimal) paging. When the consumer supplies the
    // repeating per-step handle access order (a transformer's forward 0..N then
    // backward N..0 is fully deterministic), eviction picks the resident entry
    // whose NEXT scheduled use is furthest in the future — the provably minimal
    // page-fault policy — instead of LRU. Crucially, for a cyclic/scan pattern LRU
    // is near-PESSIMAL: it evicts the entry about to be reused next. _schedule is
    // the access order; _scheduleOccurrences maps handle → its sorted positions in
    // it; _schedulePos is the cursor into the current step. Null _schedule ⇒ LRU.
    private long[]? _schedule;
    private Dictionary<long, int[]>? _scheduleOccurrences;
    private int _schedulePos;
    private long _residentBytes;
    private long _residentBytesPeak;
    private long _nextHandleId = 1;
    // Pre-allocated headroom the pool has promised to outstanding
    // AllocateStreaming callers but that hasn't yet landed in
    // _residentBytes (the caller is still initializing the GC byte[]
    // before calling RegisterWeight). Eviction's free-bytes calculation
    // adds this to _residentBytes so a second concurrent
    // AllocateStreaming can't pass the same headroom-check the first
    // one did. Decremented on RegisterWeight (or on direct
    // ReleaseReservation when the caller bails before Register).
    // Always written under _lock.
    private long _reservedBytes;
    private readonly long _maxResidentBytes;
    private readonly string _backingDir;
    // Single contiguous backing file (append-only). Opened lazily on the first
    // page-out. All access is under _lock, so one shared FileStream + Seek is safe
    // and avoids a per-rehydrate file open/mmap/close.
    private FileStream? _backingFile;
    private long _backingLength;
    private readonly bool _enableCompression;
    private readonly bool _transparentAutoEviction;
    private bool _disposed;

    // Telemetry counters (#1222 PR-A task #181). All written under _lock.
    private long _diskReadCount;
    private long _diskReadBytes;
    private long _diskWriteBytes;
    private long _evictionCount;
    // Clean/dirty-eviction telemetry: evictions that skipped the disk
    // write+compress because the backing file already held the entry's
    // current bytes, and the on-disk bytes that re-write would have cost.
    private long _cleanEvictionCount;
    private long _cleanEvictionBytesSkipped;
    private long _compressedBytesTotal;
    private long _uncompressedBytesTotal;
    // Prefetch effectiveness — distinguishes background prefetch reads
    // (good — they overlapped with foreground compute) from on-demand
    // reads (bad — the prefetch window was too short or the LRU evicted
    // the entry before Materialize hit it).
    //
    // Hit/Miss counters are gated on a prefetch having been issued for
    // the handle. Without this gate, every register-then-materialize
    // pair would count as a "hit" (because the bytes are still resident
    // from register), making the hit rate misleading for tuning the
    // prefetch window. The _prefetchPending set tracks "I prefetched
    // this; the next foreground read either uses it (hit) or had to
    // re-read disk (miss)". Cleared on consumption so a stale prefetch
    // doesn't keep counting future reads as hits.
    private long _prefetchHitCount;     // Foreground Materialize after a prefetch found bytes resident
    private long _prefetchMissCount;    // Foreground Materialize after a prefetch still had to read disk
    private long _prefetchIssueCount;   // Total Rehydrate calls flagged as isPrefetch=true
    private readonly HashSet<long> _prefetchPending = new();

    // Transparent-streaming owner-drop notification (issue #430 follow-up).
    // When the pool pages an entry out to disk under budget pressure it frees
    // its OWN byte[] snapshot — but the owning Tensor's GC-heap `_data` (if it
    // was made resident by a prior Materialize) is invisible to the pool: this
    // class is type-erased (handles + byte[], no Tensor<T>). Transparent
    // auto-rehydrate (TensorBase.EnsureMaterialized) re-materialises a weight's
    // `_data` on access; without dropping that copy when the pool evicts the
    // entry, every accessed weight would stay GC-resident and the resident set
    // would grow unbounded — defeating the bound the pool enforces on its own
    // snapshots. So each paged-out handle is recorded here and WeightRegistry
    // drains it (dropping the owning tensor via a weak back-reference). A queue
    // rather than an immediate callback keeps the pool free of registry
    // knowledge AND avoids running arbitrary drop logic under _lock: the
    // registry drains lazily on its next Materialize, so evictions from any
    // source (Register, prefetch, Materialize) are reconciled there. Written
    // under _lock.
    //
    // A SET, not a queue: a handle is "pending owner-drop" while its bytes are
    // paged out and not yet reconciled. If the same handle is re-paged-in
    // (transparent auto-rehydrate on access) BEFORE the registry drains, its
    // resident _data is valid again and MUST NOT be dropped — Rehydrate removes
    // it from this set on page-in. Without that cancellation, a stale
    // register-time eviction would make the very next drain drop the weight the
    // caller just materialized, emptying its storage mid-read.
    private readonly HashSet<long> _pendingOwnerDrops = new();

    public StreamingTensorPool(GpuOffloadOptions? options = null)
    {
        var opts = options ?? new GpuOffloadOptions();
        _maxResidentBytes = opts.StreamingPoolMaxResidentBytes;
        _enableCompression = opts.EnableCompression;
        _transparentAutoEviction = opts.TransparentAutoEviction;
        // Always append a Guid sub-directory — even when the caller
        // supplies StreamingBackingStorePath. Two pools sharing the same
        // base path would otherwise collide on streaming-{id}.bin
        // filenames and either pool's Dispose() would delete the other's
        // live files (the recursive-delete is unconditional). The Guid
        // sub-dir makes Dispose safe to scope to this pool's lifetime.
        string baseDir = opts.StreamingBackingStorePath ?? Path.GetTempPath();
        _backingDir = Path.Combine(baseDir, "aidotnet-streaming-pool-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_backingDir);
    }

    /// <summary>Total resident bytes — pool does not exceed
    /// <see cref="GpuOffloadOptions.StreamingPoolMaxResidentBytes"/> before
    /// triggering eviction.</summary>
    public long ResidentBytes => Interlocked.Read(ref _residentBytes);

    /// <summary>Number of entries currently resident in the working set.
    /// Excludes entries that have been paged out to the backing store.</summary>
    public int ResidentEntryCount
    {
        get
        {
            lock (_lock)
            {
                ThrowIfDisposed();
                int count = 0;
                foreach (var entry in _entries.Values)
                {
                    if (entry.Data is not null) count++;
                }
                return count;
            }
        }
    }

    /// <summary>Total registered entries (resident + paged out).</summary>
    public int RegisteredEntryCount
    {
        get { lock (_lock) { ThrowIfDisposed(); return _entries.Count; } }
    }

    /// <summary>
    /// Returns true when the entry's bytes are currently in the resident
    /// set (no disk read needed on next <see cref="Rehydrate"/>). Used by
    /// the LRU-bridge in <c>NeuralNetworkBase.Backpropagate</c> to skip
    /// unnecessary materialize calls when forward already left the bytes
    /// at LRU head.
    /// </summary>
    public bool IsResident(long handleId)
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            return IsResidentUnlocked(handleId);
        }
    }

    private bool IsResidentUnlocked(long handleId)
        => _entries.TryGetValue(handleId, out var entry) && entry.Data is not null;

    /// <summary>
    /// Schedule-driven prefetch targeting: returns up to <paramref name="lookahead"/>
    /// DISTINCT handles the access schedule says will be read soonest and that are
    /// NOT currently resident — exactly what to background-prefetch, with no guessing
    /// and no wasted prefetch of already-resident entries. Empty if no schedule is
    /// set (<see cref="SetAccessSchedule"/>). The consumer (or registry) feeds these
    /// to its background reader so the next layers' weights stream in while the
    /// current layer computes.
    /// </summary>
    public long[] GetScheduledPrefetchTargets(int lookahead)
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            if (_schedule is null || lookahead <= 0) return Array.Empty<long>();
            int len = _schedule.Length;
            var targets = new System.Collections.Generic.List<long>(lookahead);
            var seen = new System.Collections.Generic.HashSet<long>();
            for (int step = 0; step < len && targets.Count < lookahead; step++)
            {
                long h = _schedule[(_schedulePos + step) % len];
                if (!seen.Add(h)) continue;          // distinct only
                if (IsResidentUnlocked(h)) continue; // already in memory — don't prefetch
                targets.Add(h);
            }
            return targets.ToArray();
        }
    }

    /// <summary>Registers a weight buffer with the streaming pool. Returns a
    /// handle the caller stores on its <see cref="Tensor{T}"/> and uses for
    /// access through <see cref="MarkAccessed"/> / <see cref="Rehydrate{T}"/>.</summary>
    public long Register(byte[] data)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        lock (_lock)
        {
            ThrowIfDisposed();
            long id = _nextHandleId++;
            var entry = new Entry { Data = data, ResidentBytes = data.Length };
            _entries[id] = entry;
            var node = _lruOrder.AddFirst(id);
            _lruIndex[id] = node;
            // Use Interlocked for the long write so 32-bit hosts (net471
            // x86) don't see torn reads from concurrent ResidentBytes
            // accesses. Reads use Interlocked.Read for the same reason.
            // Inside _lock, atomicity is already guaranteed for
            // serialization, but the property accessor is lock-free.
            Interlocked.Add(ref _residentBytes, data.Length);
            long peak = Interlocked.Read(ref _residentBytesPeak);
            long current = Interlocked.Read(ref _residentBytes);
            if (current > peak) Interlocked.Exchange(ref _residentBytesPeak, current);
            EvictIfOverBudget();
            return id;
        }
    }

    /// <summary>Bumps the last-access timestamp for a weight so the LRU
    /// eviction policy keeps it resident. Surfaces use-after-dispose as
    /// <see cref="ObjectDisposedException"/> rather than silently
    /// no-op'ing so callers don't keep operating on a dead pool.</summary>
    public void MarkAccessed(long handleId)
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            AdvanceSchedule(handleId);
            if (!_lruIndex.TryGetValue(handleId, out var node)) return;
            _lruOrder.Remove(node);
            _lruOrder.AddFirst(node);
        }
    }

    /// <summary>
    /// Marks an entry's backing file STALE so the next eviction re-writes it.
    /// The pool's normal lifecycle never needs this — callers can't mutate the
    /// bytes returned by <see cref="Rehydrate(long)"/> (ReadOnlySpan) or
    /// <see cref="RehydrateInto"/> (copy), so a written backing file always
    /// matches <c>entry.Data</c>. It exists for any future path that replaces an
    /// entry's content in place under the SAME handle (rather than the normal
    /// Unregister + Register), so clean/dirty eviction stays correct. No-op for
    /// an unknown handle.
    /// </summary>
    public void MarkDirty(long handleId)
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            if (_entries.TryGetValue(handleId, out var entry))
                entry.BackingFileCurrent = false;
        }
    }

    /// <summary>
    /// Supplies the repeating per-step handle access order so eviction can use
    /// Belady-optimal selection (evict the entry whose next use is furthest) instead
    /// of LRU. For a transformer this is forward layers 0..N then backward N..0 — a
    /// known static schedule. Crucially LRU is near-PESSIMAL on such cyclic patterns
    /// (it evicts the entry about to be reused); Belady is provably minimal-fault.
    /// Pass an empty span (or never call this) to keep LRU. The order may contain a
    /// handle multiple times (forward + backward uses); it's treated as one cycle.
    /// </summary>
    public void SetAccessSchedule(ReadOnlySpan<long> order)
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            if (order.Length == 0)
            {
                _schedule = null; _scheduleOccurrences = null; _schedulePos = 0;
                return;
            }
            var sched = order.ToArray();
            var occ = new Dictionary<long, System.Collections.Generic.List<int>>();
            for (int i = 0; i < sched.Length; i++)
            {
                if (!occ.TryGetValue(sched[i], out var list)) { list = new System.Collections.Generic.List<int>(); occ[sched[i]] = list; }
                list.Add(i);
            }
            var occArr = new Dictionary<long, int[]>(occ.Count);
            foreach (var kv in occ) occArr[kv.Key] = kv.Value.ToArray(); // already sorted (ascending i)
            _schedule = sched;
            _scheduleOccurrences = occArr;
            _schedulePos = 0;
        }
    }

    /// <summary>
    /// Reads back a weight, bringing it back from the backing store if it
    /// has been evicted. Returned span aliases the pool's internal byte[];
    /// the array is rooted by the returned reference, so the bytes remain
    /// valid for the span's lifetime even if the entry is later evicted
    /// (eviction nulls <c>entry.Data</c> but the byte[] survives as long
    /// as the caller holds it).
    /// </summary>
    /// <remarks>
    /// Callers that need a stable independent copy should use
    /// <see cref="RehydrateInto"/> instead — that path copies under the
    /// pool lock, freeing the caller from coordinating with concurrent
    /// pool mutations.
    /// </remarks>
    public ReadOnlySpan<byte> Rehydrate(long handleId) => Rehydrate(handleId, isPrefetch: false);

    /// <summary>
    /// Rehydrate variant that flags whether the call originated from
    /// <see cref="WeightRegistry.PrefetchAsync{T}"/> (background) or from
    /// <see cref="WeightRegistry.Materialize{T}"/> (foreground). Used to
    /// derive prefetch effectiveness counters (hit / miss / issue) in
    /// <see cref="StreamingPoolReport"/>.
    /// </summary>
    public ReadOnlySpan<byte> Rehydrate(long handleId, bool isPrefetch)
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            if (!_entries.TryGetValue(handleId, out var entry))
                throw new InvalidOperationException($"Streaming pool: handle {handleId} is unknown.");

            // A real (foreground) read advances the schedule cursor so Belady
            // eviction predicts the next use from the true access stream. Prefetch
            // is speculative — it must NOT move the cursor.
            if (!isPrefetch) AdvanceSchedule(handleId);

            if (isPrefetch)
            {
                // Mark this handle as "prefetch was issued"; the next
                // foreground read on this handle will resolve to either
                // hit or miss based on whether the bytes are still
                // resident at that point. Also bumps the issue counter.
                _prefetchIssueCount++;
                _prefetchPending.Add(handleId);
            }
            else
            {
                // Foreground read. Only count as hit/miss if a prefetch
                // was actually issued for this handle — i.e., we're
                // measuring prefetch effectiveness, not raw hot-cache
                // residency. Reads that never had a prefetch (e.g.,
                // register-then-immediately-materialize) don't enter
                // either counter.
                if (_prefetchPending.Remove(handleId))
                {
                    if (entry.Data is null)
                    {
                        // Prefetch was issued but the bytes aren't
                        // resident anymore — either the prefetch never
                        // completed in time, or LRU evicted the entry
                        // between the prefetch and this foreground read.
                        _prefetchMissCount++;
                    }
                    else
                    {
                        // Prefetch warmed the cache; foreground found the
                        // bytes resident → real prefetch effectiveness.
                        _prefetchHitCount++;
                    }
                }
            }

            if (entry.Data is null)
            {
                // Paged out — read from the entry's slice of the contiguous backing file.
                if (entry.FileOffset < 0)
                    throw new InvalidOperationException($"Streaming pool: handle {handleId} has no backing offset (never paged out).");

                int onDiskBytes = (int)entry.PagedOutBytes;
                var diskBuffer = ReadFromBacking(entry.FileOffset, onDiskBytes);
                _diskReadCount++;
                _diskReadBytes += onDiskBytes;

                byte[] buffer;
                if (entry.IsCompressed)
                {
                    // Decompress LZ4 block — UncompressedBytes was recorded
                    // during eviction. LZ4Codec.Decode returns the byte
                    // count actually written; mismatch means corrupt data.
                    buffer = new byte[entry.UncompressedBytes];
                    int decoded = LZ4Codec.Decode(diskBuffer, 0, onDiskBytes, buffer, 0, buffer.Length);
                    if (decoded != buffer.Length)
                        throw new InvalidOperationException(
                            $"Streaming pool: LZ4 decode produced {decoded} bytes, expected {buffer.Length} for handle {handleId}.");
                }
                else
                {
                    buffer = diskBuffer;
                }

                entry.Data = buffer;
                // entry.Data now holds exactly what's in the backing file, which
                // is still on disk — so a subsequent eviction can drop this resident
                // copy without re-writing it. (Cleared only if the content is later
                // replaced out-of-band via MarkDirty.)
                entry.BackingFileCurrent = true;
                entry.ResidentBytes = buffer.Length;
                Interlocked.Add(ref _residentBytes, buffer.Length);
                long peak = Interlocked.Read(ref _residentBytesPeak);
                long current = Interlocked.Read(ref _residentBytes);
                if (current > peak) Interlocked.Exchange(ref _residentBytesPeak, current);
                // Re-add to LRU — the prior eviction removed it from
                // _lruIndex/_lruOrder, so MarkAccessed and future eviction
                // walks would silently skip the rehydrated entry.
                if (!_lruIndex.ContainsKey(handleId))
                {
                    var freshNode = _lruOrder.AddFirst(handleId);
                    _lruIndex[handleId] = freshNode;
                }
                // This entry is resident again — cancel any pending owner-drop
                // queued when it was previously evicted. Otherwise the next
                // registry drain would drop the owning tensor's _data right
                // after the caller materialised it (e.g. a register-time
                // eviction that was never reconciled), emptying storage mid-read.
                _pendingOwnerDrops.Remove(handleId);
                // Skip the just-rehydrated entry during eviction. If this
                // tensor alone exceeds the budget, evicting it here would
                // page it back out before Rehydrate returns and the caller
                // would see stale/empty bytes.
                EvictIfOverBudget(protectedHandleId: handleId);
            }

            // Refresh LRU on resident hit.
            if (_lruIndex.TryGetValue(handleId, out var node))
            {
                _lruOrder.Remove(node);
                _lruOrder.AddFirst(node);
            }
            return entry.Data!;
        }
    }

    /// <summary>
    /// Reads a weight back into a freshly-allocated byte[] under the pool
    /// lock. Faster path for callers who'd otherwise call Rehydrate then
    /// copy out — this version avoids the second lock acquire and gives
    /// the caller a buffer that's independent of the pool's resident set.
    /// Used by <see cref="WeightRegistry.Materialize{T}"/> so the registry
    /// lock can be released before the deserialize-into-tensor memcpy.
    /// </summary>
    public byte[] RehydrateInto(long handleId)
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            // Reuse Rehydrate to consolidate the resident-/paged-out
            // codepaths, then snapshot the bytes into a caller-owned
            // array before releasing the lock.
            var span = Rehydrate(handleId);
            var copy = new byte[span.Length];
            span.CopyTo(copy);
            return copy;
        }
    }

    /// <summary>Frees a weight + its backing-store file. Called at model
    /// dispose.</summary>
    public void Unregister(long handleId)
    {
        lock (_lock)
        {
            if (!_entries.TryGetValue(handleId, out var entry)) return;
            Interlocked.Add(ref _residentBytes, -entry.ResidentBytes);
            _entries.Remove(handleId);
            if (_lruIndex.TryGetValue(handleId, out var node))
            {
                _lruOrder.Remove(node);
                _lruIndex.Remove(handleId);
            }
            // Drop any pending prefetch state — handle IDs are monotonic
            // so this is defensive, but it keeps the pending set bounded
            // by live entries even when callers register/unregister at
            // high churn.
            _prefetchPending.Remove(handleId);
            // The entry's slice of the contiguous backing file is left in place
            // (reclaimed when the whole file is deleted at Dispose). Weight handles
            // are register-once for a model, so per-unregister compaction would be
            // churn for no benefit; the file is bounded by total bytes paged out.
        }
    }

    /// <summary>
    /// Pages LRU entries to disk until either the requested
    /// <paramref name="byteCount"/> bytes of headroom are available
    /// under <see cref="_maxResidentBytes"/> OR the LRU is empty.
    /// Best-effort: returns whether the requested headroom was secured.
    /// Internal plumbing for <see cref="ReserveBytes"/> and direct
    /// test-coverage of the eviction loop; production callers should use
    /// <see cref="ReserveBytes"/> instead so the headroom they secured
    /// is held against concurrent allocators.
    /// </summary>
    /// <param name="byteCount">Bytes of headroom to make available.</param>
    /// <returns>True if the pool now has at least
    /// <paramref name="byteCount"/> bytes of free budget; false if it
    /// emptied its LRU and still doesn't (caller's allocation may push
    /// the pool past budget; <see cref="EvictIfOverBudget"/> on the
    /// next register will recover).</returns>
    /// <remarks>
    /// <para>byteCount &lt;= 0 is a no-op. The free-bytes calculation
    /// includes <see cref="_reservedBytes"/> so a concurrent
    /// AllocateStreaming (whose reservation hasn't yet been committed
    /// via Register) can't double-spend the same headroom.</para>
    /// </remarks>
    internal bool EvictUntilFreeBytes(long byteCount)
    {
        if (byteCount <= 0) return true;
        lock (_lock)
        {
            ThrowIfDisposed();
            return EvictUntilFreeBytesUnlocked(byteCount);
        }
    }

    /// <summary>Caller already holds <see cref="_lock"/>. Same contract as
    /// <see cref="EvictUntilFreeBytes"/>.</summary>
    private bool EvictUntilFreeBytesUnlocked(long byteCount)
    {
        // Oversize-request short-circuit: the caller is asking for more
        // headroom than the entire pool budget. Even draining every
        // resident entry to disk wouldn't satisfy them, so the loop
        // below would do a full LRU flush only to return false at the
        // end — flushing every warm weight for nothing and forcing a
        // round of foreground rehydrates afterward. Bail out before we
        // touch the LRU: the caller's allocation will overshoot budget
        // by (byteCount - max), and EvictIfOverBudget on the next
        // Register will catch up the same way it would have without
        // this call.
        if (byteCount > _maxResidentBytes) return false;

        // "Free bytes available" = budget - resident - reserved. We loop
        // until either we have at least byteCount free, OR the LRU is
        // empty. Including _reservedBytes is what closes the TOCTOU
        // race between concurrent AllocateStreaming calls — without
        // it, both callers would observe the same headroom and
        // overshoot.
        while (_maxResidentBytes - _residentBytes - _reservedBytes < byteCount && _lruOrder.Count > 0)
        {
            if (!EvictOneLruEntry(protectedHandleId: null)) break;
        }
        return _maxResidentBytes - _residentBytes - _reservedBytes >= byteCount;
    }

    /// <summary>
    /// Atomically pages out LRU entries to free at least
    /// <paramref name="byteCount"/> bytes of headroom AND records the
    /// reservation against <see cref="_reservedBytes"/> so a concurrent
    /// caller can't pass the same eviction gate. Internal plumbing for
    /// <see cref="WeightRegistry.AllocateStreaming{T}"/>; release happens
    /// through <see cref="WeightRegistry.RegisterWeight{T}"/> (commits
    /// reserved → resident) or <see cref="WeightRegistry.UnregisterWeight{T}"/>
    /// (returns headroom to the budget). External code should never
    /// call this directly: the byteCount-based release path takes no
    /// opaque token and so can't validate that the right amount is
    /// being released; routing all calls through WeightRegistry keeps
    /// the bookkeeping coherent.
    /// </summary>
    /// <param name="byteCount">Bytes of headroom to reserve. Non-positive
    /// is a no-op (returns true).</param>
    /// <returns>True if the reservation was secured under budget; false
    /// when the request exceeds the entire budget. False reservations
    /// are still recorded — the caller will cause a transient overshoot
    /// until the next Register's <see cref="EvictIfOverBudget"/> walk
    /// catches up.</returns>
    internal bool ReserveBytes(long byteCount)
    {
        if (byteCount <= 0) return true;
        lock (_lock)
        {
            ThrowIfDisposed();
            bool secured = EvictUntilFreeBytesUnlocked(byteCount);
            _reservedBytes += byteCount;
            return secured;
        }
    }

    /// <summary>
    /// Releases bytes previously reserved via <see cref="ReserveBytes"/>.
    /// Internal plumbing — called by
    /// <see cref="WeightRegistry.RegisterWeight{T}"/> just before the
    /// committed bytes land in <see cref="_residentBytes"/>, or by
    /// <see cref="WeightRegistry.UnregisterWeight{T}"/> when the caller
    /// bails before Register. External callers must go through
    /// UnregisterWeight rather than calling this directly: the raw
    /// byteCount API has no opaque token to prove the release amount
    /// matches the prior reservation, and a mismatch would silently
    /// floor at zero (see floor logic below).
    /// </summary>
    /// <param name="byteCount">Bytes to release. Non-positive is a no-op.</param>
    internal void ReleaseReservation(long byteCount)
    {
        if (byteCount <= 0) return;
        lock (_lock)
        {
            // No ThrowIfDisposed here: a tensor that was reserved against
            // an old pool but never registered may be released after a
            // Configure swap. The reservation accounting is best-effort
            // bookkeeping — it doesn't affect resident bytes once the
            // pool is disposed.
            if (_disposed) return;
            _reservedBytes -= byteCount;
            // Floor at zero — defensive against double-release. Bumping
            // a "release without reserve" exception would turn a
            // bookkeeping mistake into a tensor leak, which is worse.
            if (_reservedBytes < 0) _reservedBytes = 0;
        }
    }

    /// <summary>Outstanding reservations from in-flight
    /// <see cref="WeightRegistry.AllocateStreaming{T}"/> calls. Internal
    /// — exposed for tests via InternalsVisibleTo, not public API.</summary>
    internal long ReservedBytes
    {
        get { lock (_lock) return _reservedBytes; }
    }

    /// <summary>
    /// Evicts a single LRU entry to disk. Returns true if an entry was
    /// evicted, false if the LRU is empty (or only contains the
    /// protected handle). Extracted so <see cref="EvictIfOverBudget"/>
    /// and <see cref="EvictUntilFreeBytes"/> share the same eviction
    /// machinery without duplicating the LZ4-encode + stream-write
    /// path.
    /// </summary>
    /// <remarks>Caller already holds <see cref="_lock"/>.</remarks>
    private bool EvictOneLruEntry(long? protectedHandleId)
    {
        // Caller already holds _lock.
        if (_lruOrder.Count == 0) return false;
        var victim = _schedule is null
            ? SelectLruVictim(protectedHandleId)
            : SelectBeladyVictim(protectedHandleId);
        if (victim is null) return false;
        return EvictNodeInternal(victim);
    }

    // LRU: the tail (least-recently-used), skipping the protected handle.
    private LinkedListNode<long>? SelectLruVictim(long? protectedHandleId)
    {
        var oldest = _lruOrder.Last;
        while (oldest is not null && protectedHandleId.HasValue && oldest.Value == protectedHandleId.Value)
            oldest = oldest.Previous;
        return oldest;
    }

    // Belady: among resident entries, the one whose NEXT scheduled access is
    // furthest in the future (or never) — provably the minimal-fault choice. O(R)
    // over the resident set (bounded by budget); ties resolved toward LRU tail.
    private LinkedListNode<long>? SelectBeladyVictim(long? protectedHandleId)
    {
        LinkedListNode<long>? best = null;
        long bestDist = long.MinValue;
        // Walk LRU tail→head so equal-distance ties pick the less-recently-used one.
        for (var node = _lruOrder.Last; node is not null; node = node.Previous)
        {
            long h = node.Value;
            if (protectedHandleId.HasValue && h == protectedHandleId.Value) continue;
            long dist = NextAccessDistance(h);
            if (dist > bestDist) { bestDist = dist; best = node; }
            if (dist == long.MaxValue) break; // never used again → evict immediately
        }
        // Fall back to LRU if the schedule somehow knows none of them (shouldn't happen).
        return best ?? SelectLruVictim(protectedHandleId);
    }

    // Distance (in accesses) from the current schedule cursor to handle h's next
    // occurrence, wrapping to the next cycle. long.MaxValue if h is not scheduled.
    private long NextAccessDistance(long handle)
    {
        if (_scheduleOccurrences is null || _schedule is null) return long.MaxValue;
        if (!_scheduleOccurrences.TryGetValue(handle, out var positions)) return long.MaxValue;
        // Smallest position >= _schedulePos via binary search.
        int lo = 0, hi = positions.Length;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (positions[mid] >= _schedulePos) hi = mid; else lo = mid + 1; }
        if (lo < positions.Length) return positions[lo] - _schedulePos;
        // Wrap: first occurrence in the next cycle.
        return (long)_schedule.Length - _schedulePos + positions[0];
    }

    // Advance the schedule cursor to just past handle h's next occurrence, so the
    // cursor tracks the real access stream even if it skips/reorders slightly.
    private void AdvanceSchedule(long handle)
    {
        if (_scheduleOccurrences is null || _schedule is null) return;
        if (!_scheduleOccurrences.TryGetValue(handle, out var positions)) return;
        int lo = 0, hi = positions.Length;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (positions[mid] >= _schedulePos) hi = mid; else lo = mid + 1; }
        int next = lo < positions.Length ? positions[lo] : positions[0];
        _schedulePos = (next + 1) % _schedule.Length;
    }

    private void EvictIfOverBudget(long? protectedHandleId = null)
    {
        // Caller already holds _lock.
        while (_residentBytes > _maxResidentBytes && _lruOrder.Count > 0)
        {
            if (!EvictOneLruEntry(protectedHandleId)) break;
        }
    }

    /// <summary>
    /// Returns and clears the handles the pool has paged out to disk since the
    /// last drain, so <see cref="WeightRegistry"/> can drop the owning tensors'
    /// resident <c>_data</c> (see <c>_pendingOwnerDrops</c>). The pool frees
    /// only its own byte[] snapshot on eviction; the tensor's GC-heap copy —
    /// re-materialised by transparent auto-rehydrate — is freed by the registry
    /// after this drain, keeping the resident set bounded. Returns
    /// <see cref="Array.Empty{T}"/> when nothing was evicted. Snapshot is taken
    /// under <see cref="_lock"/> so it's internally consistent; the registry
    /// performs the actual drops (tensor-storage CAS only, no pool/registry
    /// lock) after this returns.
    /// </summary>
    internal long[] DrainEvictedHandles()
    {
        lock (_lock)
        {
            if (_pendingOwnerDrops.Count == 0) return Array.Empty<long>();
            long[] result = new long[_pendingOwnerDrops.Count];
            _pendingOwnerDrops.CopyTo(result);
            _pendingOwnerDrops.Clear();
            return result;
        }
    }

    /// <summary>
    /// Pages a single LRU entry to disk. Extracted from
    /// <see cref="EvictIfOverBudget"/> so
    /// <see cref="EvictUntilFreeBytes"/> can drive the same eviction
    /// machinery via a different stop condition (free bytes vs. budget).
    /// Returns true if an entry was evicted; false if the LRU is empty
    /// or contains only the protected handle.
    /// </summary>
    /// <remarks>Caller already holds <see cref="_lock"/>.</remarks>
    private bool EvictNodeInternal(LinkedListNode<long> oldest)
    {
        long id = oldest.Value;

        if (!_entries.TryGetValue(id, out var entry) || entry.Data is null)
        {
            // Stale LRU node (entry already evicted/unregistered) — drop.
            _lruOrder.Remove(oldest);
            _lruIndex.Remove(id);
            return true; // we did make progress — try the next iteration
        }

        // Page out: write to backing store, drop resident reference,
        // remove from LRU index (Rehydrate re-adds it). When
        // EnableCompression is true, LZ4-compress before writing —
        // ~30-40% disk-footprint reduction on near-Gaussian fp32
        // weights. UncompressedBytes is the rehydration target size;
        // PagedOutBytes is what's actually on disk.
        //
        // Memory-peak optimisation: stream-write the encoded bytes
        // directly from the rented LZ4 buffer (no intermediate
        // `new byte[encoded]` copy) and null out entry.Data
        // immediately after the write. The previous code kept
        // entry.Data + encodeBuf + toWrite all live across the
        // disk write — for a 268 MB tensor that peaked at ~716 MB
        // resident during eviction, reintroducing OOMs on the
        // memory-bound models this feature is meant to help. After
        // this change peak is entry.Data + encodeBuf ≈ ~536 MB,
        // and entry.Data is freed BEFORE the write returns.
        int uncompressed = entry.Data.Length;

        // Clean/dirty eviction. Once an entry's backing file has been written it
        // holds that entry's exact bytes for the rest of its life: callers can
        // never mutate entry.Data (Rehydrate hands out a ReadOnlySpan and
        // RehydrateInto returns a copy), so entry.Data is only ever (a) the bytes
        // passed to Register — dirty until first written — or (b) bytes just read
        // back from this same file by Rehydrate — clean. Re-writing and
        // re-LZ4-compressing a CLEAN entry on every subsequent eviction is pure
        // waste, and it is the dominant cost of a read-only forward/backward pass
        // (every layer's rehydrate evicts a clean peer). A clean entry is simply
        // dropped; its file, PagedOutBytes, UncompressedBytes and IsCompressed all
        // stay valid for the next Rehydrate. Only genuinely-new content (a freshly
        // Registered entry, or one explicitly marked dirty by MarkDirty) is written.
        if (entry.BackingFileCurrent)
        {
            entry.Data = null;
            _cleanEvictionCount++;
            _cleanEvictionBytesSkipped += entry.PagedOutBytes;
        }
        else
        {
            int paged;
            bool compressed = false;
            if (_enableCompression)
            {
                // LZ4 worst-case bound is uncompressed + (uncompressed/255) + 16.
                // Rent from ArrayPool to keep the encode buffer pooled
                // across evictions instead of allocating a fresh worst-
                // case buffer each time.
                int maxOut = LZ4Codec.MaximumOutputSize(uncompressed);
                byte[] encodeBuf = ArrayPool<byte>.Shared.Rent(maxOut);
                try
                {
                    int encoded = LZ4Codec.Encode(entry.Data, 0, uncompressed, encodeBuf, 0, maxOut);
                    if (encoded > 0 && encoded < uncompressed)
                    {
                        // Append the encoded slice to the single contiguous backing
                        // file (no intermediate byte[encoded] copy — the dominant
                        // peak-memory win is preserved).
                        entry.FileOffset = AppendToBacking(encodeBuf, encoded);
                        paged = encoded;
                        compressed = true;
                    }
                    else
                    {
                        // Compression didn't shrink the payload (entropy too
                        // high or below LZ4's break-even) — append entry.Data raw
                        // and flag IsCompressed=false so Rehydrate doesn't decode.
                        entry.FileOffset = AppendToBacking(entry.Data, uncompressed);
                        paged = uncompressed;
                        compressed = false;
                    }
                }
                finally
                {
                    // clearArray:true so the next renter doesn't observe the
                    // previous tenant's serialized weight bytes — these are
                    // model parameters, often proprietary, and a downstream
                    // consumer renting the same buffer would see them
                    // unzeroed. The clear cost is amortized against the
                    // disk write latency anyway.
                    ArrayPool<byte>.Shared.Return(encodeBuf, clearArray: true);
                }
            }
            else
            {
                entry.FileOffset = AppendToBacking(entry.Data, uncompressed);
                paged = uncompressed;
            }
            // Free entry.Data IMMEDIATELY now that the bytes are durably
            // on disk. Any further work (counter updates, LRU bookkeeping)
            // doesn't need the resident copy. Holding it alive across
            // the rest of this method (or worse, until the next eviction)
            // would defeat the whole eviction.
            entry.Data = null;
            entry.PagedOutBytes = paged;
            entry.UncompressedBytes = uncompressed;
            entry.IsCompressed = compressed;
            entry.BackingFileCurrent = true;
            _diskWriteBytes += paged;
            _compressedBytesTotal += paged;
            _uncompressedBytesTotal += uncompressed;
        }
        _evictionCount++;
        Interlocked.Add(ref _residentBytes, -entry.ResidentBytes);
        entry.ResidentBytes = 0;
        _lruOrder.Remove(oldest);
        _lruIndex.Remove(id);
        // Record the page-out so WeightRegistry can drop the owning tensor's
        // resident _data on its next drain — but ONLY in transparent mode.
        // With explicit orchestration the model owns residency (and a
        // concurrent reader could be mid-read on the very weight we'd drop),
        // so leave the owner copy alone and keep DrainEvictedHandles empty.
        // Only the real page-out path records — the stale-node branch above
        // (entry.Data already null) was paged out earlier and recorded then.
        if (_transparentAutoEviction)
            _pendingOwnerDrops.Add(id);
        return true;
    }

    // Backing files are flat raw bytes (or LZ4-compressed bytes when
    // EnableCompression is on). No magic header / version prefix —
    // backing files are tied to a single pool's lifetime (deleted on
    // Dispose), so cross-version compatibility doesn't apply. If the
    // pool is ever extended to support persistent state across process
    // restarts, add `[u32 magic][u32 version]` here and adjust Rehydrate
    // to validate before LZ4Codec.Decode.
    // Lazily opens the single contiguous backing file (read-write, exclusive).
    // Caller holds _lock.
    private FileStream BackingFile()
        => _backingFile ??= new FileStream(
            BackingFilePath,
            // FileShare.Read (not None) so a zero-copy reader can open a
            // read-only memory-mapping of the same file concurrently with the
            // pool's append handle. The pool stays the only WRITER (no other
            // handle is granted write), so the append-only contract is intact.
            FileMode.Create, FileAccess.ReadWrite, FileShare.Read);

    // Absolute path of the single contiguous backing file. Stable for the
    // pool's lifetime; used by the zero-copy mmap path.
    private string BackingFilePath => Path.Combine(_backingDir, "backing.bin");

    /// <summary>
    /// For the zero-copy mmap path: if <paramref name="handleId"/> is currently
    /// paged out to a stable, uncompressed slice of the backing file whose bytes
    /// are the tensor's NATIVE element bytes, returns that slice (file + offset +
    /// length) so the caller can alias it read-only. Pure lookup — does NOT touch
    /// LRU heat, resident accounting, or eviction. Returns false when the entry is
    /// resident, compressed, or never paged out (no zero-copy is possible).
    /// </summary>
    public bool TryGetZeroCopyByteRange(long handleId, out string path, out long fileOffset, out int byteLength)
    {
        path = string.Empty;
        fileOffset = -1;
        byteLength = 0;
        lock (_lock)
        {
            ThrowIfDisposed();
            if (!_entries.TryGetValue(handleId, out var entry)) return false;
            // Must be paged out to a current, uncompressed file slice. Compressed
            // (LZ4) or bf16/int8/lossless-encoded bytes can't be aliased — they
            // require a decode pass, which is a copy by definition.
            if (entry.Data is not null) return false;            // resident — caller's fast path already avoids a copy
            if (entry.IsCompressed) return false;                // LZ4 needs decode
            if (!entry.BackingFileCurrent) return false;         // slice not guaranteed to hold current bytes
            if (entry.FileOffset < 0) return false;              // never paged out
            if (entry.PagedOutBytes <= 0 || entry.PagedOutBytes > int.MaxValue) return false;
            path = BackingFilePath;
            fileOffset = entry.FileOffset;
            byteLength = (int)entry.PagedOutBytes;
            return true;
        }
    }

    // Appends `count` bytes from `buf` to the backing file and returns the offset
    // the slice was written at. Append-only — entries land contiguously in
    // first-eviction order. Caller holds _lock.
    private long AppendToBacking(byte[] buf, int count)
    {
        var fs = BackingFile();
        long offset = _backingLength;
        fs.Seek(offset, SeekOrigin.Begin);
        fs.Write(buf, 0, count);
        fs.Flush();
        _backingLength += count;
        return offset;
    }

    // Reads exactly `count` bytes at `offset` from the backing file. Caller holds _lock.
    private byte[] ReadFromBacking(long offset, int count)
    {
        var fs = BackingFile();
        fs.Seek(offset, SeekOrigin.Begin);
        var buf = new byte[count];
        int read = 0;
        while (read < count)
        {
            int r = fs.Read(buf, read, count - read);
            if (r <= 0) break;
            read += r;
        }
        if (read != count)
            throw new InvalidOperationException(
                $"Streaming backing file: short read at offset {offset} ({read}/{count} bytes).");
        return buf;
    }

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            // Free resident byte[]s + LRU bookkeeping eagerly so Dispose
            // doesn't leave the dictionary populated until GC. Without this,
            // a long-lived host that swaps pools via Configure() would hold
            // every old pool's resident bytes in memory until the next GC.
            _entries.Clear();
            _lruOrder.Clear();
            _lruIndex.Clear();
            _pendingOwnerDrops.Clear();
            // Atomic write so the lock-free ResidentBytes property doesn't
            // see torn values on 32-bit hosts (net471 x86 still ships).
            Interlocked.Exchange(ref _residentBytes, 0);
            // Close the single backing-file handle before the directory delete.
            try { _backingFile?.Dispose(); } catch { /* best-effort */ }
            _backingFile = null;
        }
        // Backing-store deletion outside the lock — Directory.Delete on a
        // large pool can be slow, and there's no in-memory state to
        // protect anymore.
        try { if (Directory.Exists(_backingDir)) Directory.Delete(_backingDir, recursive: true); }
        catch { /* best-effort */ }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(StreamingTensorPool));
    }

    private sealed class Entry
    {
        public byte[]? Data;
        public long ResidentBytes;
        public long PagedOutBytes;
        // Set during eviction so Rehydrate knows the decompress target
        // size + whether to invoke LZ4Codec.Decode at all.
        public long UncompressedBytes;
        public bool IsCompressed;
        // True iff a backing file exists on disk holding this entry's CURRENT
        // bytes — so a subsequent eviction can drop the resident copy without
        // re-writing. Set after a page-out write and after a rehydrate-from-disk
        // (entry.Data then equals the file); cleared by Register (no file yet)
        // and MarkDirty (content replaced out-of-band). See EvictNodeInternal.
        public bool BackingFileCurrent;
        // Byte offset of this entry's PagedOutBytes in the single contiguous
        // backing file (-1 until first paged out). With clean/dirty eviction each
        // entry is written once, so the file is append-only and entries land
        // contiguously in first-eviction order → large sequential reads + one
        // persistent file handle instead of a file open/close per rehydrate.
        public long FileOffset = -1;
    }

    /// <summary>
    /// Snapshot of pool counters for telemetry. Read at end of inference /
    /// training pass; consumed by AiDotNet's
    /// <c>PredictionModelResult.StreamingReport</c>.
    /// </summary>
    public StreamingPoolReport GetReport()
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            double ratio = _uncompressedBytesTotal > 0
                ? (double)_compressedBytesTotal / _uncompressedBytesTotal
                : 1.0;
            return new StreamingPoolReport
            {
                ResidentBytes = _residentBytes,
                ResidentBytesPeak = _residentBytesPeak,
                RegisteredEntryCount = _entries.Count,
                DiskReadCount = _diskReadCount,
                DiskReadBytes = _diskReadBytes,
                DiskWriteBytes = _diskWriteBytes,
                EvictionCount = _evictionCount,
                CleanEvictionCount = _cleanEvictionCount,
                CleanEvictionBytesSkipped = _cleanEvictionBytesSkipped,
                CompressionRatio = ratio,
                CompressionEnabled = _enableCompression,
                PrefetchHitCount = _prefetchHitCount,
                PrefetchMissCount = _prefetchMissCount,
                PrefetchIssueCount = _prefetchIssueCount,
            };
        }
    }
}

/// <summary>
/// Snapshot of streaming pool counters. Returned by
/// <see cref="StreamingTensorPool.GetReport"/> and surfaced in
/// AiDotNet's <c>PredictionModelResult.StreamingReport</c>.
///
/// <para><b>Default-instance contract:</b> <c>default(StreamingPoolReport)</c>
/// and <c>new StreamingPoolReport()</c> both report
/// <see cref="CompressionRatio"/> = <c>1.0</c> (the documented "no
/// compression has run" value), not the all-zero-struct's <c>0.0</c>.
/// This matters for consumers that surface telemetry before any
/// streaming registration has happened — a 0.0 ratio would imply
/// "perfect compression" which is misleading. The normalization is
/// implemented by storing the raw ratio in a private field and
/// returning <c>1.0</c> when it's <c>&lt;= 0</c>.</para>
/// </summary>
public readonly record struct StreamingPoolReport
{
    /// <summary>Currently resident bytes (uncompressed).</summary>
    public long ResidentBytes { get; init; }

    /// <summary>Peak resident bytes observed since the pool was
    /// created. Helpful for tuning
    /// <see cref="GpuOffloadOptions.StreamingPoolMaxResidentBytes"/>
    /// — set the budget just above the peak to avoid eviction churn.</summary>
    public long ResidentBytesPeak { get; init; }

    /// <summary>Total entries in the pool (resident + paged out).</summary>
    public int RegisteredEntryCount { get; init; }

    /// <summary>Number of times the backing store had to be read to
    /// rehydrate an entry. High values relative to entry count
    /// indicate the resident budget is too small / the prefetch
    /// window is too small relative to the access pattern.</summary>
    public long DiskReadCount { get; init; }

    /// <summary>Total bytes read from the backing store.</summary>
    public long DiskReadBytes { get; init; }

    /// <summary>Total bytes written to the backing store across all
    /// evictions (compressed size when compression is on).</summary>
    public long DiskWriteBytes { get; init; }

    /// <summary>Number of eviction events. Each event may page out
    /// one entry.</summary>
    public long EvictionCount { get; init; }

    /// <summary>Of <see cref="EvictionCount"/>, the evictions that skipped the
    /// disk write+compress because the backing file already held the entry's
    /// current bytes (clean/dirty eviction). High relative to EvictionCount on a
    /// read-heavy workload (forward/backward) is the expected, healthy case.</summary>
    public long CleanEvictionCount { get; init; }

    /// <summary>Total on-disk bytes NOT re-written thanks to clean-eviction
    /// skips — the write I/O (and LZ4 compression) saved versus re-persisting
    /// every eviction.</summary>
    public long CleanEvictionBytesSkipped { get; init; }

    // Backing field for CompressionRatio. The default value of zero
    // (from default(StreamingPoolReport)) is interpreted as "no
    // compression has run" and surfaced as 1.0 by the property's
    // getter. Real values from GetReport() are always > 0 (LZ4 never
    // emits zero-byte output for non-empty input).
    private readonly double _compressionRatioRaw;

    /// <summary>Average compressed/uncompressed ratio across all
    /// evictions, weighted by uncompressed byte count. <c>1.0</c> when
    /// no compression has run; lower is better. Typical for fp32
    /// weights: 0.6–0.7. Reading this on a default-constructed
    /// instance returns <c>1.0</c> (not <c>0.0</c>) so consumers that
    /// surface telemetry before any streaming registration don't
    /// report a misleading "perfect compression" ratio.</summary>
    public double CompressionRatio
    {
        get => _compressionRatioRaw <= 0.0 ? 1.0 : _compressionRatioRaw;
        init => _compressionRatioRaw = value;
    }

    /// <summary>Whether <see cref="GpuOffloadOptions.EnableCompression"/>
    /// was set when the pool was constructed.</summary>
    public bool CompressionEnabled { get; init; }

    /// <summary>Number of foreground
    /// <see cref="WeightRegistry.Materialize{T}"/> calls on a handle
    /// that had a prefetch in flight and found the bytes already
    /// resident — a prefetch (or LRU heat following one) saved a disk
    /// read. High hit rate = prefetch window is well-tuned. Reads on
    /// handles that never had a prefetch issued are NOT counted here
    /// — those are normal hot-cache reads, not prefetch hits.</summary>
    public long PrefetchHitCount { get; init; }

    /// <summary>Foreground Materialize calls that had a prefetch in
    /// flight but had to read from the backing store anyway. High
    /// miss rate = prefetch window is too short, or LRU evicted the
    /// entry before foreground reached it. Reads on handles that
    /// never had a prefetch issued are NOT counted here.</summary>
    public long PrefetchMissCount { get; init; }

    /// <summary>Total <see cref="WeightRegistry.PrefetchAsync{T}"/>
    /// calls that resulted in actual prefetch work. PrefetchIssueCount
    /// far exceeding PrefetchHitCount indicates wasted prefetch work
    /// (rehydrated entries got evicted before being read).</summary>
    public long PrefetchIssueCount { get; init; }
}
