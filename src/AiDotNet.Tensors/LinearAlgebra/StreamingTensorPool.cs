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
    private readonly bool _enableCompression;
    private bool _disposed;

    // Telemetry counters (#1222 PR-A task #181). All written under _lock.
    private long _diskReadCount;
    private long _diskReadBytes;
    private long _diskWriteBytes;
    private long _evictionCount;
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

    public StreamingTensorPool(GpuOffloadOptions? options = null)
    {
        var opts = options ?? new GpuOffloadOptions();
        _maxResidentBytes = opts.StreamingPoolMaxResidentBytes;
        _enableCompression = opts.EnableCompression;
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
            return _entries.TryGetValue(handleId, out var entry) && entry.Data is not null;
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
            if (!_lruIndex.TryGetValue(handleId, out var node)) return;
            _lruOrder.Remove(node);
            _lruOrder.AddFirst(node);
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
                // Paged out — read from backing file.
                string path = BackingPathFor(handleId);
                if (!File.Exists(path))
                    throw new InvalidOperationException($"Streaming pool: handle {handleId} backing file missing at {path}.");

                int onDiskBytes = (int)entry.PagedOutBytes;
                var diskBuffer = new byte[onDiskBytes];
                using (var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read))
                using (var view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read))
                {
                    view.ReadArray(0, diskBuffer, 0, onDiskBytes);
                }
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
            string path = BackingPathFor(handleId);
            try { if (File.Exists(path)) File.Delete(path); } catch { /* best-effort */ }
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
        var oldest = _lruOrder.Last;
        while (oldest is not null && protectedHandleId.HasValue && oldest.Value == protectedHandleId.Value)
            oldest = oldest.Previous;
        if (oldest is null) return false;
        return EvictNodeInternal(oldest);
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
        string path = BackingPathFor(id);
        int uncompressed = entry.Data.Length;
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
                    // Stream-write only the encoded slice. FileStream
                    // .Write copies (uncompressed + overhead) bytes
                    // straight to disk — no intermediate byte[encoded]
                    // copy. This is the dominant peak-memory win:
                    // before this, we'd have allocated a third
                    // copy-out buffer here.
                    using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
                    {
                        fs.Write(encodeBuf, 0, encoded);
                    }
                    paged = encoded;
                    compressed = true;
                }
                else
                {
                    // Compression didn't shrink the payload (entropy too
                    // high or below LZ4's break-even) — write entry.Data
                    // raw and flag IsCompressed=false so Rehydrate
                    // doesn't try to decode. Same File.WriteAllBytes
                    // path as the no-compression branch below.
                    File.WriteAllBytes(path, entry.Data);
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
            File.WriteAllBytes(path, entry.Data);
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
        _diskWriteBytes += paged;
        _evictionCount++;
        _compressedBytesTotal += paged;
        _uncompressedBytesTotal += uncompressed;
        Interlocked.Add(ref _residentBytes, -entry.ResidentBytes);
        entry.ResidentBytes = 0;
        _lruOrder.Remove(oldest);
        _lruIndex.Remove(id);
        return true;
    }

    // Backing files are flat raw bytes (or LZ4-compressed bytes when
    // EnableCompression is on). No magic header / version prefix —
    // backing files are tied to a single pool's lifetime (deleted on
    // Dispose), so cross-version compatibility doesn't apply. If the
    // pool is ever extended to support persistent state across process
    // restarts, add `[u32 magic][u32 version]` here and adjust Rehydrate
    // to validate before LZ4Codec.Decode.
    private string BackingPathFor(long id) => Path.Combine(_backingDir, $"streaming-{id}.bin");

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
            // Atomic write so the lock-free ResidentBytes property doesn't
            // see torn values on 32-bit hosts (net471 x86 still ships).
            Interlocked.Exchange(ref _residentBytes, 0);
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
