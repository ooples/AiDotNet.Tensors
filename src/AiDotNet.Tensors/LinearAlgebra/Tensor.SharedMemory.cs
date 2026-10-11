using System;
using System.Buffers;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Cross-process shared memory (torch <c>Tensor.share_memory_</c> / <c>is_shared</c>): the tensor's storage moves into
/// a shared mapping that another process maps with <see cref="FromSharedMemory"/>, and writes through either are seen
/// by both.
/// </summary>
/// <remarks>
/// <para>The mapping is a file mapped MAP_SHARED: under <c>/dev/shm</c> (memory, no disk) where it exists, otherwise
/// in the temporary directory, where the OS keeps the pages in its cache. The file is private to the creating user:
/// on Unix it is created with mode 0600 (no window where it is readable by others), and on Windows it lives in the
/// user's temporary directory, whose ACL admits only that user and the system.</para>
/// <para>The creating process deletes the file when the shared storage is released (its last tensor or view is
/// disposed), or at process exit; a process that maps it keeps its own mapping valid.</para>
/// <para>Supported element types are the fixed-size primitives (and <see cref="Half"/>). The tensor must be a dense,
/// contiguous tensor that owns its storage: call <c>Contiguous()</c> on a view first (a sparse tensor has no dense
/// storage and is rejected). A GPU-resident tensor is downloaded to the host first, as torch moves a CUDA tensor's
/// storage only once it is on the host.</para>
/// </remarks>
public partial class Tensor<T>
{
    // The path is tied to the shared STORAGE, not to one tensor: every storage-sharing view (Reshape, Transpose, a
    // slice) reports the same mapping, and the entry goes away with the storage.
    private static readonly ConditionalWeakTable<TensorStorage<T>, string> s_sharedPaths = new();

    /// <summary>True when this tensor's storage is a cross-process shared mapping (torch <c>is_shared</c>).</summary>
    public bool IsShared => SharedMemoryPath is not null;

    /// <summary>The path another process passes to <see cref="FromSharedMemory"/>, or null when not shared.</summary>
    public string? SharedMemoryPath
        => IsWritableMmapAliased && s_sharedPaths.TryGetValue(_storage, out var path) ? path : null;

    /// <summary>
    /// Moves this tensor's storage into cross-process shared memory and returns this tensor (torch
    /// <c>share_memory_</c>). A no-op when it is already shared.
    /// </summary>
    public Tensor<T> ShareMemory()
    {
        if (IsShared) return this;
        int elementSize = SharedElementSize();
        if (IsSparse) throw new NotSupportedException("A sparse tensor has no dense storage to share.");
        if (!IsContiguous || _storageOffset != 0 || IsView)
            throw new InvalidOperationException("share_memory_ needs a contiguous tensor that owns its storage; call Contiguous() first.");
        long bytes = (long)Length * elementSize;
        if (bytes > int.MaxValue) throw new NotSupportedException("A shared tensor is limited to 2 GiB.");

        string path = Path.Combine(SharedMemoryDirectory(), "aidotnet-shm-" + Guid.NewGuid().ToString("N") + ".bin");
        try
        {
            using (var file = CreatePrivateFile(path))
            {
                if (bytes > 0) WriteInChunks(file, AsSpan(), (int)bytes);
                else file.SetLength(1);   // an empty file cannot be mapped
            }
            AliasSharedFile(this, path, (int)bytes, deleteWhenReleased: true);
        }
        catch
        {
            TryDelete(path);   // never leave a file behind for a share that did not happen
            throw;
        }
        return this;
    }

    /// <summary>
    /// Maps the shared memory another process created with <see cref="ShareMemory"/>: a tensor of
    /// <paramref name="shape"/> over the same bytes, so writes by either process are seen by both.
    /// </summary>
    public static Tensor<T> FromSharedMemory(string path, int[] shape)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        int elementSize = SharedElementSize();
        long count = 1;
        foreach (int extent in shape)
        {
            if (extent < 0) throw new ArgumentException($"Shape [{string.Join(", ", shape)}] has a negative extent.", nameof(shape));
            count *= extent;
            if (count > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(shape), "A shared tensor holds at most int.MaxValue elements.");
        }
        long bytes = count * elementSize;
        // The mapping and its span are sized in int bytes: a larger request would map a truncated view and expose
        // elements past its end.
        if (bytes > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(shape), "Shared-memory mappings cannot exceed int.MaxValue bytes.");
        if (!File.Exists(path)) throw new FileNotFoundException("No shared tensor at this path.", path);
        long fileLength = new FileInfo(path).Length;
        if (fileLength < bytes)
            throw new ArgumentException($"The shared memory holds {fileLength} bytes; shape [{string.Join(", ", shape)}] needs {bytes}.", nameof(shape));

        // Map first, then build the tensor over the mapping: no full-size buffer is allocated and thrown away.
        var mapping = new SharedMapping(path, (int)bytes, (int)count, deleteWhenReleased: false);
        try
        {
            var tensor = new Tensor<T>((int[])shape.Clone(), Vector<T>.WrapMemory(mapping.Manager.Memory));
            tensor.AliasStorageFromMmap(Vector<T>.WrapMemory(mapping.Manager.Memory), mapping, writable: true);
            RecordSharedPath(tensor, path);
            return tensor;
        }
        catch
        {
            mapping.Dispose();
            throw;
        }
    }

    private static void AliasSharedFile(Tensor<T> tensor, string path, int bytes, bool deleteWhenReleased)
    {
        var mapping = new SharedMapping(path, bytes, tensor.Length, deleteWhenReleased);
        try
        {
            tensor.AliasStorageFromMmap(Vector<T>.WrapMemory(mapping.Manager.Memory), mapping, writable: true);
        }
        catch
        {
            mapping.Dispose();
            throw;
        }
        RecordSharedPath(tensor, path);
    }

    private static void RecordSharedPath(Tensor<T> tensor, string path)
    {
        // ConditionalWeakTable.AddOrUpdate is not on .NET Framework.
        s_sharedPaths.Remove(tensor._storage);
        s_sharedPaths.Add(tensor._storage, path);
    }

    /// <summary>
    /// The writable mapping of one shared file, owned by the shared storage: disposed when the storage's last tensor
    /// or view releases it, which also deletes the file when this process created it. Process exit deletes it as a
    /// fallback for storage that is never released.
    /// </summary>
    private sealed class SharedMapping : IDisposable
    {
        private readonly string? _deletePath;
        private EventHandler? _exitHandler;
        private int _disposed;

        public SharedMapping(string path, int bytes, int elements, bool deleteWhenReleased)
        {
            Manager = new MmapTensorMemoryManager<T>(path, 0, Math.Max(bytes, 1), elements, writable: true);
            if (deleteWhenReleased)
            {
                _deletePath = path;
                _exitHandler = (_, _) => TryDelete(path);
                AppDomain.CurrentDomain.ProcessExit += _exitHandler;
            }
        }

        public MmapTensorMemoryManager<T> Manager { get; }

        public void Dispose()
        {
            if (System.Threading.Interlocked.Exchange(ref _disposed, 1) != 0) return;
            ((IDisposable)Manager).Dispose();
            if (_exitHandler is { } handler)
            {
                AppDomain.CurrentDomain.ProcessExit -= handler;
                _exitHandler = null;
            }
            if (_deletePath is not null) TryDelete(_deletePath);
        }
    }

    // Creates the file readable and writable by its owner only. On Unix the mode is set at creation, so there is no
    // window where the umask-derived default (often 0644) exposes it; on Windows the per-user temporary directory's
    // inherited ACL already restricts it to the user.
    private static FileStream CreatePrivateFile(string path)
    {
#if NET7_0_OR_GREATER
        if (!OperatingSystem.IsWindows())
            return new FileStream(path, new FileStreamOptions
            {
                Mode = FileMode.CreateNew,
                Access = FileAccess.Write,
                Share = FileShare.ReadWrite,
                UnixCreateMode = UnixFileMode.UserRead | UnixFileMode.UserWrite,
            });
#endif
        return new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.ReadWrite);
    }

    // Writes the tensor's bytes through a bounded pooled buffer instead of a tensor-sized copy.
    private static void WriteInChunks(FileStream file, ReadOnlySpan<T> span, int bytes)
    {
        const int ChunkBytes = 1 << 20;
        var chunk = ArrayPool<byte>.Shared.Rent(Math.Min(ChunkBytes, bytes));
        try
        {
            ref byte source = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(span));
            for (int offset = 0; offset < bytes; offset += chunk.Length)
            {
                int n = Math.Min(chunk.Length, bytes - offset);
                Unsafe.CopyBlockUnaligned(ref chunk[0], ref Unsafe.Add(ref source, offset), (uint)n);
                file.Write(chunk, 0, n);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(chunk);
        }
    }

    private static int SharedElementSize()
    {
        var t = typeof(T);
        if (t == typeof(float) || t == typeof(int) || t == typeof(uint)) return 4;
        if (t == typeof(double) || t == typeof(long) || t == typeof(ulong)) return 8;
        if (t == typeof(short) || t == typeof(ushort) || t == typeof(char)) return 2;
        if (t == typeof(byte) || t == typeof(sbyte) || t == typeof(bool)) return 1;
#if NET5_0_OR_GREATER
        if (t == typeof(Half)) return 2;
#endif
        throw new NotSupportedException($"Shared memory holds fixed-size primitive elements; {t.Name} is not one.");
    }

    private static string SharedMemoryDirectory()
        => Directory.Exists("/dev/shm") ? "/dev/shm" : Path.GetTempPath();

    // Best effort: another process may still hold the file open (Windows refuses the delete then; process exit retries).
    private static void TryDelete(string path)
    {
        try { File.Delete(path); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }
}
