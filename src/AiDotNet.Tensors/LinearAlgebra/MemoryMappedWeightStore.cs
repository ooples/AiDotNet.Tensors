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

    public MemoryMappedWeightStore(string path, long initialCapacity = 64L * 1024 * 1024)
    {
        _path = path ?? throw new ArgumentNullException(nameof(path));
        if (initialCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(initialCapacity));
        _capacity = AlignUp(Math.Max(initialCapacity, 4096), 4096);
        // FileShare.ReadWrite so a slice can be writably aliased through an independent MAP_SHARED view
        // (the #1715 param-IO round-trip) concurrently with the store's own mapping. The two mappings
        // touch the same page-cache pages, so writes to disjoint byte ranges stay coherent.
        _file = new FileStream(_path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.ReadWrite);
        _file.SetLength(_capacity);
        TryMarkSparse(_file);     // unwritten capacity costs no disk where supported (Linux: always; Windows: NTFS)
        Map();
    }

    private unsafe void Map()
    {
        // leaveOpen: true — we own _file and dispose it explicitly; the MMF must not close it
        // so a later Grow can SetLength the same handle.
        _mmf = MemoryMappedFile.CreateFromFile(
            _file, mapName: null, _capacity, MemoryMappedFileAccess.ReadWrite,
            HandleInheritability.None, leaveOpen: true);
        _view = _mmf.CreateViewAccessor(0, _capacity, MemoryMappedFileAccess.ReadWrite);
        byte* p = null;
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
        _base = p;
        _pointerAcquired = true;
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
            long offset = AlignUp(_used, 64);
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
            if (offset < 0 || length < 0 || offset + length > _used)
                throw new ArgumentOutOfRangeException(nameof(offset),
                    $"slice [{offset},{offset + length}) is outside the allocated region [0,{_used}).");
            return new Span<byte>(_base + offset, length);
        }
    }

    private unsafe void Grow(long minCapacity)
    {
        long newCap = _capacity;
        while (newCap < minCapacity) newCap = checked(newCap * 2);
        newCap = AlignUp(newCap, 4096);

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
