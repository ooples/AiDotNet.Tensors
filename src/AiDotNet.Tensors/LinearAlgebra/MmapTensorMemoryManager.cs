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
/// <para>READ-ONLY: the view is mapped read-only, so this must only back weights that are
/// not written while aliased (inference / forward-read). A write would fault.</para>
/// </summary>
internal sealed unsafe class MmapTensorMemoryManager<T> : MemoryManager<T>
{
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _view;
    private byte* _basePtr;            // start of the acquired view mapping (page-aligned)
    private readonly long _sliceByteOffset; // PointerOffset + the requested intra-view offset
    private readonly int _count;       // element count of T
    private bool _disposed;

    /// <summary>
    /// Maps <paramref name="elementCount"/> elements of <typeparamref name="T"/> starting at
    /// <paramref name="byteOffset"/> in the file at <paramref name="path"/>, read-only.
    /// </summary>
    public MmapTensorMemoryManager(string path, long byteOffset, int byteLength, int elementCount)
    {
        _count = elementCount;
        // Each step opens an additional handle on top of any prior one; if a
        // later step throws (e.g. CreateViewAccessor on a corrupt file,
        // AcquirePointer on an out-of-memory address space), the earlier
        // handles leak until finalization. TryAliasZeroCopy on the caller
        // side already falls back to the copy path on failure, so the
        // mapping failure is recoverable — but under repeated zero-copy
        // fallback a transient mapping failure would silently leak file
        // descriptors / view handles. Dispose what we've already opened
        // before rethrowing.
        try
        {
            // Open a read-only mapping. The owning FileStream must share read access.
            _mmf = MemoryMappedFile.CreateFromFile(
                new System.IO.FileStream(path, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.ReadWrite),
                mapName: null, capacity: 0,
                MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
            _view = _mmf.CreateViewAccessor(byteOffset, byteLength, MemoryMappedFileAccess.Read);
            byte* p = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
            _basePtr = p;
            // CreateViewAccessor rounds the start down to a page boundary; PointerOffset is the
            // distance from that boundary to the requested byteOffset.
            _sliceByteOffset = _view.PointerOffset;
        }
        catch
        {
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
            try { _view.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { /* best-effort */ }
            _view.Dispose();
            _view = null;
        }
        _mmf?.Dispose();
        _mmf = null;
        _basePtr = null;
    }
}
