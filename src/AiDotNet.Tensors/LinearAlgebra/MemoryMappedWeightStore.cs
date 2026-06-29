using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Memory-mapped backing store for streamed weights (#1715 redesign). A single growable
/// file is memory-mapped with a writable view; each weight is a 64-byte-aligned
/// <c>(offset, length)</c> slice. The OS page cache provides residency, eviction, AND
/// write-back FOR FREE — so unlike the hand-rolled streaming pool there is no resident-byte
/// accounting, no LRU, no explicit page-out/rehydrate IO, and no write-back-before-drop
/// (the bug class the bespoke pool needed). Under host memory pressure the OS flushes dirty
/// slices to the file and reclaims their pages; the next access faults them back in with the
/// mutations intact. This is exactly the model <c>torch.load(mmap=True)</c> / safetensors use
/// for loading and offloading models that exceed RAM.
/// </summary>
/// <remarks>
/// SPAN LIFETIME CONTRACT: <see cref="GetSpan"/> returns a span aliasing the live mapped
/// region. A grow (triggered only by <see cref="Allocate"/> when capacity is exceeded)
/// re-maps the file and moves the base pointer, invalidating any span obtained earlier. The
/// streaming round-trip registers all weights (Allocate) before reading any back (GetSpan),
/// so a grow never races a held span; callers that interleave must re-fetch the span after an
/// Allocate. All public methods are serialized under an internal lock.
/// </remarks>
internal sealed class MemoryMappedWeightStore : IDisposable
{
    // Slices are 64-byte aligned (cache-line) so adjacent writable aliases never share a cache line;
    // the file capacity is page-aligned (4 KiB) so the OS maps whole pages; the default initial capacity
    // is 64 MiB (sparse — costs no disk until written).
    private const long SliceAlignmentBytes = 64;
    private const long PageAlignmentBytes = 4096;
    private const long DefaultInitialCapacityBytes = 64L * 1024 * 1024;

    private readonly object _lock = new();
    private readonly string _path;
    private FileStream _file;
    private MemoryMappedFile _mmf = null!;
    private MemoryMappedViewAccessor _view = null!;
    private unsafe byte* _base;
    private bool _pointerAcquired;
    private long _capacity;
    private long _used;          // bump-allocation high-water mark
    private bool _disposed;

    /// <summary>Total bytes handed out by <see cref="Allocate"/> (the logical size; the file
    /// capacity may be larger and is sparse where the OS supports it).</summary>
    public long UsedBytes { get { lock (_lock) { return _used; } } }

    /// <summary>Current mapped file capacity in bytes (grows by doubling).</summary>
    public long Capacity { get { lock (_lock) { return _capacity; } } }

    /// <summary>Backing file path — callers that writably-alias a slice (the #1715 param-IO path) open
    /// their own independent MAP_SHARED view of this file at the slice's offset; an alias's mapping
    /// stays valid across a store grow (the file only ever grows, never moves existing bytes).</summary>
    public string Path => _path;

    public MemoryMappedWeightStore(string path, long initialCapacity = DefaultInitialCapacityBytes)
    {
        _path = path ?? throw new ArgumentNullException(nameof(path));
        if (initialCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(initialCapacity));
        _capacity = AlignUp(Math.Max(initialCapacity, PageAlignmentBytes), PageAlignmentBytes);
        // FileShare.ReadWrite so a slice can be writably aliased through an independent MAP_SHARED view
        // (the #1715 param-IO round-trip) concurrently with the store's own mapping. The two mappings
        // touch the same page-cache pages, so writes to disjoint byte ranges stay coherent.
        // Clean up the file + handle if any step after CreateNew throws (CreateViewAccessor /
        // AcquirePointer on a recoverable failure) — the ctor never reaches Dispose() on throw, so a
        // partially-created mapping + the backing temp file would otherwise leak.
        try
        {
            _file = new FileStream(_path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.ReadWrite);
            _file.SetLength(_capacity);
            TryMarkSparse(_file); // unwritten capacity costs no disk where supported (Linux: always; Windows: NTFS)
            Map();
        }
        catch
        {
            try { _file?.Dispose(); } catch { /* best-effort */ }
            try { if (File.Exists(_path)) File.Delete(_path); } catch { /* best-effort */ }
            throw;
        }
    }

    private unsafe void Map()
    {
        // leaveOpen: true — we own _file and dispose it explicitly; the MMF must not close it
        // so a later Grow can SetLength the same handle. Build into locals and publish only after the
        // pointer is acquired, so a throw from CreateViewAccessor / AcquirePointer disposes the partial
        // MMF/view instead of leaking it (the ctor's catch deletes the file).
        MemoryMappedFile? mmf = null;
        MemoryMappedViewAccessor? view = null;
        bool acquired = false;
        byte* p = null;
        try
        {
            mmf = MemoryMappedFile.CreateFromFile(
                _file, mapName: null, _capacity, MemoryMappedFileAccess.ReadWrite,
                HandleInheritability.None, leaveOpen: true);
            view = mmf.CreateViewAccessor(0, _capacity, MemoryMappedFileAccess.ReadWrite);
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
            acquired = true;
            _mmf = mmf;
            _view = view;
            _base = p;
            _pointerAcquired = true;
        }
        catch
        {
            if (acquired && view is not null) { try { view.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { /* best-effort */ } }
            view?.Dispose();
            mmf?.Dispose();
            throw;
        }
    }

    private unsafe void Unmap()
    {
        if (_pointerAcquired)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _pointerAcquired = false;
        }
        _view?.Flush();           // push dirty pages to the file before tearing the mapping down
        _view?.Dispose();
        _mmf?.Dispose();
        _base = null;
    }

    /// <summary>Bump-allocate a 64-byte-aligned slice, copy <paramref name="data"/> into the
    /// mapped region, and return the byte offset (the handle for <see cref="GetSpan"/>). Grows
    /// the file (doubling) when the current capacity is exceeded.</summary>
    public unsafe long Allocate(ReadOnlySpan<byte> data)
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            long offset = AlignUp(_used, SliceAlignmentBytes);
            long need = offset + data.Length;
            if (need > _capacity) Grow(need);
            data.CopyTo(new Span<byte>(_base + offset, data.Length));
            _used = offset + data.Length;
            return offset;
        }
    }

    /// <summary>Returns a writable span over the slice at <paramref name="offset"/>. Reading
    /// faults the pages in; writing through the span is preserved on the next read and flushed
    /// to disk under memory pressure (MAP_SHARED semantics). See the span-lifetime contract.</summary>
    public unsafe Span<byte> GetSpan(long offset, int length)
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            // Validate offset first, then length against the remaining span — `offset + length` could
            // overflow before the comparison and then `_base + offset` would form a span from a bad address.
            if (offset < 0 || length < 0 || offset > _used || length > _used - offset)
            {
                long end = offset >= 0 && length >= 0 && offset <= long.MaxValue - length
                    ? offset + length
                    : long.MaxValue;
                throw new ArgumentOutOfRangeException(nameof(offset),
                    $"slice [{offset},{end}) is outside the allocated region [0,{_used}).");
            }
            return new Span<byte>(_base + offset, length);
        }
    }

    private unsafe void Grow(long minCapacity)
    {
        long newCap = _capacity;
        while (newCap < minCapacity) newCap = checked(newCap * 2);
        newCap = AlignUp(newCap, PageAlignmentBytes);

        // Tear the mapping down (flushing dirty pages to the file), grow the file, re-map. The
        // file bytes already written persist across SetLength, so offsets stay valid.
        Unmap();
        _file.SetLength(newCap);
        _capacity = newCap;
        Map();
    }

    private static long AlignUp(long value, long alignment) => (value + (alignment - 1)) & ~(alignment - 1);

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MemoryMappedWeightStore));
    }

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            try { Unmap(); } catch { /* best-effort teardown */ }
            try { _file?.Dispose(); } catch { /* best-effort */ }
            // Delete the backing file so a killed/long-running process doesn't leak tens of GB
            // (the orphaned-backing-file failure mode observed with the bespoke pool).
            try { if (File.Exists(_path)) File.Delete(_path); } catch { /* best-effort */ }
        }
    }

    /// <summary>Best-effort sparse-file marking so unwritten capacity costs no disk. Linux/macOS
    /// make truncated files sparse automatically; on Windows (NTFS) we must request it via
    /// FSCTL_SET_SPARSE. Failure is non-fatal — the file is simply fully allocated.</summary>
    private static void TryMarkSparse(FileStream file)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return; // POSIX truncate is already sparse
        try
        {
            const uint FSCTL_SET_SPARSE = 0x000900C4;
            DeviceIoControl(file.SafeFileHandle.DangerousGetHandle(), FSCTL_SET_SPARSE,
                IntPtr.Zero, 0, IntPtr.Zero, 0, out _, IntPtr.Zero);
        }
        catch { /* sparse is an optimization, not a correctness requirement */ }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool DeviceIoControl(
        IntPtr hDevice, uint dwIoControlCode, IntPtr lpInBuffer, uint nInBufferSize,
        IntPtr lpOutBuffer, uint nOutBufferSize, out uint lpBytesReturned, IntPtr lpOverlapped);
}
