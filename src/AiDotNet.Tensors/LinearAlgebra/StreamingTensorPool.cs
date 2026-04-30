// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;

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
    private long _nextHandleId = 1;
    private readonly long _maxResidentBytes;
    private readonly string _backingDir;
    private bool _disposed;

    public StreamingTensorPool(GpuOffloadOptions? options = null)
    {
        var opts = options ?? new GpuOffloadOptions();
        _maxResidentBytes = opts.StreamingPoolMaxResidentBytes;
        _backingDir = opts.StreamingBackingStorePath
            ?? Path.Combine(Path.GetTempPath(), "aidotnet-streaming-pool-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_backingDir);
    }

    /// <summary>Total resident bytes — pool does not exceed
    /// <see cref="GpuOffloadOptions.StreamingPoolMaxResidentBytes"/> before
    /// triggering eviction.</summary>
    public long ResidentBytes => Interlocked.Read(ref _residentBytes);

    /// <summary>Number of entries currently resident in the working set.</summary>
    public int ResidentEntryCount { get { lock (_lock) return _entries.Count; } }

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
                using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
                using var view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
                var buffer = new byte[entry.PagedOutBytes];
                view.ReadArray(0, buffer, 0, (int)entry.PagedOutBytes);
                entry.Data = buffer;
                entry.ResidentBytes = buffer.Length;
                _residentBytes += buffer.Length;
                EvictIfOverBudget();
            }

            // Refresh LRU.
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

    private void EvictIfOverBudget()
    {
        // Caller already holds _lock.
        while (_residentBytes > _maxResidentBytes && _lruOrder.Count > 0)
        {
            // Tail of LRU is least-recently-used.
            var oldest = _lruOrder.Last!;
            long id = oldest.Value;
            _lruOrder.RemoveLast();
            _lruIndex.Remove(id);

            if (_entries.TryGetValue(id, out var entry) && entry.Data is not null)
            {
                // Page out: write to backing-store file then drop the
                // resident reference. Eviction is preserved by the
                // memory-mapped file so Rehydrate can pull it back.
                string path = BackingPathFor(id);
                File.WriteAllBytes(path, entry.Data);
                entry.PagedOutBytes = entry.Data.Length;
                _residentBytes -= entry.ResidentBytes;
                entry.ResidentBytes = 0;
                entry.Data = null;
            }
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
    }
}
