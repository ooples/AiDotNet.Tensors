// Copyright (c) AiDotNet. All rights reserved.

using System;
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

    public StreamingTensorPool(GpuOffloadOptions? options = null)
    {
        var opts = options ?? new GpuOffloadOptions();
        _maxResidentBytes = opts.StreamingPoolMaxResidentBytes;
        _enableCompression = opts.EnableCompression;
        _backingDir = opts.StreamingBackingStorePath
            ?? Path.Combine(Path.GetTempPath(), "aidotnet-streaming-pool-" + Guid.NewGuid().ToString("N"));
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
    public int RegisteredEntryCount { get { lock (_lock) return _entries.Count; } }

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
            long id = _nextHandleId++;
            var entry = new Entry { Data = data, ResidentBytes = data.Length };
            _entries[id] = entry;
            var node = _lruOrder.AddFirst(id);
            _lruIndex[id] = node;
            _residentBytes += data.Length;
            if (_residentBytes > _residentBytesPeak) _residentBytesPeak = _residentBytes;
            EvictIfOverBudget();
            return id;
        }
    }

    /// <summary>Bumps the last-access timestamp for a weight so the LRU
    /// eviction policy keeps it resident.</summary>
    public void MarkAccessed(long handleId)
    {
        lock (_lock)
        {
            if (!_lruIndex.TryGetValue(handleId, out var node)) return;
            _lruOrder.Remove(node);
            _lruOrder.AddFirst(node);
        }
    }

    /// <summary>Reads back a weight, bringing it back from the backing store
    /// if it has been evicted. Returned span is valid until the next
    /// eviction; callers should copy out before kernel dispatch.</summary>
    public ReadOnlySpan<byte> Rehydrate(long handleId)
    {
        lock (_lock)
        {
            if (!_entries.TryGetValue(handleId, out var entry))
                throw new InvalidOperationException($"Streaming pool: handle {handleId} is unknown.");

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
                _residentBytes += buffer.Length;
                if (_residentBytes > _residentBytesPeak) _residentBytesPeak = _residentBytes;
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

    /// <summary>Frees a weight + its backing-store file. Called at model
    /// dispose.</summary>
    public void Unregister(long handleId)
    {
        lock (_lock)
        {
            if (!_entries.TryGetValue(handleId, out var entry)) return;
            _residentBytes -= entry.ResidentBytes;
            _entries.Remove(handleId);
            if (_lruIndex.TryGetValue(handleId, out var node))
            {
                _lruOrder.Remove(node);
                _lruIndex.Remove(handleId);
            }
            string path = BackingPathFor(handleId);
            try { if (File.Exists(path)) File.Delete(path); } catch { /* best-effort */ }
        }
    }

    private void EvictIfOverBudget(long? protectedHandleId = null)
    {
        // Caller already holds _lock.
        while (_residentBytes > _maxResidentBytes && _lruOrder.Count > 0)
        {
            // Tail of LRU is least-recently-used. Walk forward (toward more
            // recent) past the protected entry so a single tensor exceeding
            // the budget doesn't page itself back out during rehydrate.
            var oldest = _lruOrder.Last;
            while (oldest is not null && protectedHandleId.HasValue && oldest.Value == protectedHandleId.Value)
                oldest = oldest.Previous;
            if (oldest is null) break; // only the protected entry remains
            long id = oldest.Value;

            if (!_entries.TryGetValue(id, out var entry) || entry.Data is null)
            {
                // Stale LRU node (entry already evicted/unregistered) — drop.
                _lruOrder.Remove(oldest);
                _lruIndex.Remove(id);
                continue;
            }

            // Page out: write to backing store, drop resident reference,
            // remove from LRU index (Rehydrate re-adds it). When
            // EnableCompression is true, LZ4-compress before writing —
            // ~30-40% disk-footprint reduction on near-Gaussian fp32
            // weights. UncompressedBytes is the rehydration target size;
            // PagedOutBytes is what's actually on disk.
            string path = BackingPathFor(id);
            int uncompressed = entry.Data.Length;
            byte[] toWrite;
            bool compressed = false;
            if (_enableCompression)
            {
                // LZ4 worst-case bound is uncompressed + (uncompressed/255) + 16.
                int maxOut = LZ4Codec.MaximumOutputSize(uncompressed);
                var encodeBuf = new byte[maxOut];
                int encoded = LZ4Codec.Encode(entry.Data, 0, uncompressed, encodeBuf, 0, maxOut);
                if (encoded > 0 && encoded < uncompressed)
                {
                    toWrite = new byte[encoded];
                    Array.Copy(encodeBuf, 0, toWrite, 0, encoded);
                    compressed = true;
                }
                else
                {
                    // Compression didn't help (entropy too high or below
                    // LZ4's break-even) — store raw to skip the decompress
                    // cost on rehydrate.
                    toWrite = entry.Data;
                }
            }
            else
            {
                toWrite = entry.Data;
            }
            File.WriteAllBytes(path, toWrite);
            entry.PagedOutBytes = toWrite.Length;
            entry.UncompressedBytes = uncompressed;
            entry.IsCompressed = compressed;
            _diskWriteBytes += toWrite.Length;
            _evictionCount++;
            _compressedBytesTotal += toWrite.Length;
            _uncompressedBytesTotal += uncompressed;
            _residentBytes -= entry.ResidentBytes;
            entry.ResidentBytes = 0;
            entry.Data = null;
            _lruOrder.Remove(oldest);
            _lruIndex.Remove(id);
        }
    }

    private string BackingPathFor(long id) => Path.Combine(_backingDir, $"streaming-{id}.bin");

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try { if (Directory.Exists(_backingDir)) Directory.Delete(_backingDir, recursive: true); }
        catch { /* best-effort */ }
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
            double ratio = _uncompressedBytesTotal > 0
                ? (double)_compressedBytesTotal / _uncompressedBytesTotal
                : 1.0;
            return new StreamingPoolReport(
                ResidentBytes: _residentBytes,
                ResidentBytesPeak: _residentBytesPeak,
                RegisteredEntryCount: _entries.Count,
                DiskReadCount: _diskReadCount,
                DiskReadBytes: _diskReadBytes,
                DiskWriteBytes: _diskWriteBytes,
                EvictionCount: _evictionCount,
                CompressionRatio: ratio,
                CompressionEnabled: _enableCompression);
        }
    }
}

/// <summary>
/// Snapshot of streaming pool counters. Returned by
/// <see cref="StreamingTensorPool.GetReport"/> and surfaced in
/// AiDotNet's <c>PredictionModelResult.StreamingReport</c>.
/// </summary>
/// <param name="ResidentBytes">Currently resident bytes (uncompressed).</param>
/// <param name="ResidentBytesPeak">Peak resident bytes observed since the
/// pool was created. Helpful for tuning <see cref="GpuOffloadOptions.StreamingPoolMaxResidentBytes"/>
/// — set the budget just above the peak to avoid eviction churn.</param>
/// <param name="RegisteredEntryCount">Total entries in the pool (resident
/// + paged out).</param>
/// <param name="DiskReadCount">Number of times the backing store had to
/// be read to rehydrate an entry. High values relative to entry count
/// indicate the resident budget is too small / the prefetch window is
/// too small relative to the access pattern.</param>
/// <param name="DiskReadBytes">Total bytes read from the backing store.</param>
/// <param name="DiskWriteBytes">Total bytes written to the backing store
/// across all evictions (compressed size when compression is on).</param>
/// <param name="EvictionCount">Number of eviction events. Each event may
/// page out one entry.</param>
/// <param name="CompressionRatio">Average compressed/uncompressed ratio
/// across all evictions, weighted by uncompressed byte count. 1.0 when no
/// compression has run; lower is better. Typical for fp32 weights:
/// 0.6–0.7.</param>
/// <param name="CompressionEnabled">Whether <see cref="GpuOffloadOptions.EnableCompression"/>
/// was set when the pool was constructed.</param>
public readonly record struct StreamingPoolReport(
    long ResidentBytes,
    long ResidentBytesPeak,
    int RegisteredEntryCount,
    long DiskReadCount,
    long DiskReadBytes,
    long DiskWriteBytes,
    long EvictionCount,
    double CompressionRatio,
    bool CompressionEnabled);
