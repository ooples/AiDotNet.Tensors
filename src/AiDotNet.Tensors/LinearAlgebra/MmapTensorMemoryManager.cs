// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Backs a <see cref="Memory{T}"/> with a slice of a read-only memory-mapped file, so a
/// streaming weight tensor can alias its bytes on the backing store directly (zero-copy)
/// instead of being copied into a fresh array on rehydrate. The OS page cache manages
/// residency; reads fault pages in on demand and the kernel evicts cold ones — no
/// explicit copy, no managed allocation.
///
/// <para>Lifetime: owns the <see cref="MemoryMappedFile"/> + view + acquired pointer and
/// releases them on <see cref="Dispose(bool)"/>. The owning tensor holds this manager and
/// disposes it when it drops/replaces its storage, so the mapping lives exactly as long as
/// the tensor aliases it.</para>
///
/// <para>Read-only by default (inference / forward-read — a write would fault). Pass
/// <c>writable: true</c> (#1715 param-IO round-trip) to map the slice <c>MemoryMappedFileAccess.ReadWrite</c>
/// so writes through <see cref="GetSpan"/> hit the mapped pages and persist to the backing file
/// (MAP_SHARED) — the OS flushes dirty pages under memory pressure and faults them back on the next
/// read, with the mutation intact. A writable alias must back an all-mmap store (no concurrent
/// FileStream writer on the same file) to stay page-cache-coherent.</para>
/// </summary>
internal sealed unsafe class MmapTensorMemoryManager<T> : MemoryManager<T>
{
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _view;
    private byte* _basePtr;            // start of the acquired view mapping (page-aligned)
    private readonly long _sliceByteOffset; // PointerOffset + the requested intra-view offset
    private readonly int _count;       // element count of T
    private readonly bool _writable;
    private bool _disposed;

    /// <summary>Whether this mapping is writable (writes persist) vs read-only (writes fault).</summary>
    public bool IsWritable => _writable;

    /// <summary>
    /// Maps <paramref name="elementCount"/> elements of <typeparamref name="T"/> starting at
    /// <paramref name="byteOffset"/> in the file at <paramref name="path"/>. Read-only unless
    /// <paramref name="writable"/> is set, in which case writes persist to the file (MAP_SHARED).
    /// </summary>
    public MmapTensorMemoryManager(string path, long byteOffset, int byteLength, int elementCount, bool writable = false)
    {
        _count = elementCount;
        _writable = writable;
        // Each step opens an additional handle on top of any prior one; if a
        // later step throws (e.g. CreateViewAccessor on a corrupt file,
        // AcquirePointer on an out-of-memory address space), the earlier
        // handles leak until finalization. TryAliasZeroCopy on the caller
        // side already falls back to the copy path on failure, so the
        // mapping failure is recoverable — but under repeated zero-copy
        // fallback a transient mapping failure would silently leak file
        // descriptors / view handles. Dispose what we've already opened
        // before rethrowing.
        System.IO.FileStream? stream = null;
        try
        {
            // Read-only or read-write per `_writable`. The FileStream access must match the
            // mapping access (ReadWrite needs write permission on the file handle too).
            var fileAccess = _writable ? System.IO.FileAccess.ReadWrite : System.IO.FileAccess.Read;
            var mapAccess = _writable ? MemoryMappedFileAccess.ReadWrite : MemoryMappedFileAccess.Read;
            // Hold the stream in a local: if CreateFromFile throws BEFORE it takes ownership (leaveOpen:
            // false transfers ownership only on success), the catch must dispose the stream itself —
            // otherwise the file handle leaks on a recoverable mapping failure.
            stream = new System.IO.FileStream(path, System.IO.FileMode.Open, fileAccess, System.IO.FileShare.ReadWrite);
            _mmf = MemoryMappedFile.CreateFromFile(
                stream, mapName: null, capacity: 0,
                mapAccess, HandleInheritability.None, leaveOpen: false);
            stream = null; // ownership transferred to _mmf (leaveOpen: false) — don't double-dispose
            _view = _mmf.CreateViewAccessor(byteOffset, byteLength, mapAccess);
            byte* p = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
            _basePtr = p;
            // CreateViewAccessor rounds the start down to a page boundary; PointerOffset is the
            // distance from that boundary to the requested byteOffset.
            _sliceByteOffset = _view.PointerOffset;
        }
        catch
        {
            try { stream?.Dispose(); } catch { /* best-effort: CreateFromFile didn't take ownership */ }
            _view?.Dispose();
            _view = null;
            _mmf?.Dispose();
            _mmf = null;
            throw;
        }
    }

    public override Span<T> GetSpan()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MmapTensorMemoryManager<T>));
        // Span<T>(void*, int) (vs MemoryMarshal.CreateSpan) so this compiles on net471,
        // where CreateSpan is unavailable. The pointer ctor throws for reference Ts; the
        // streaming pool only routes blittable element types (float/double/int/long) here.
        return new Span<T>(_basePtr + _sliceByteOffset, _count);
    }

    public override MemoryHandle Pin(int elementIndex = 0)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MmapTensorMemoryManager<T>));
        if ((uint)elementIndex > (uint)_count) throw new ArgumentOutOfRangeException(nameof(elementIndex));
        // The mapped pointer is already pinned by the OS; no GC handle needed.
        void* p = _basePtr + _sliceByteOffset + (long)elementIndex * Unsafe.SizeOf<T>();
        return new MemoryHandle(p, default, this);
    }

    public override void Unpin() { /* OS-pinned mapping; nothing to release per-pin. */ }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (_view is not null)
        {
            // Push dirty pages to the file before tearing the mapping down so a writable
            // alias's mutations are durable even if the OS hasn't flushed them yet.
            if (_writable) { try { _view.Flush(); } catch { /* best-effort */ } }
            try { _view.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { /* best-effort */ }
            _view.Dispose();
            _view = null;
        }
        _mmf?.Dispose();
        _mmf = null;
        _basePtr = null;
    }
}
