using System;
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
/// in the temporary directory, where the OS keeps the pages in its cache. The creating process deletes the file when
/// it exits; a process that maps it keeps its mapping valid until then.</para>
/// <para>Supported element types are the fixed-size primitives (and <see cref="Half"/>); views, sparse and GPU-only
/// tensors are made contiguous host tensors first, as torch moves a CUDA tensor's storage only once it is on the
/// host.</para>
/// </remarks>
public partial class Tensor<T>
{
    private string? _sharedMemoryPath;

    /// <summary>True when this tensor's storage is a cross-process shared mapping (torch <c>is_shared</c>).</summary>
    public bool IsShared => _sharedMemoryPath is not null && IsWritableMmapAliased;

    /// <summary>The path another process passes to <see cref="FromSharedMemory"/>, or null when not shared.</summary>
    public string? SharedMemoryPath => IsShared ? _sharedMemoryPath : null;

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
        var span = AsSpan();
        using (var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.ReadWrite))
        {
            if (bytes > 0)
            {
                var buffer = new byte[bytes];
                Unsafe.CopyBlockUnaligned(ref buffer[0], ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(span)), (uint)bytes);
                file.Write(buffer, 0, buffer.Length);
            }
            else file.SetLength(1);   // an empty file cannot be mapped
        }
        DeleteAtExit(path);
        AliasSharedFile(this, path, (int)bytes);
        _sharedMemoryPath = path;
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
        var tensor = new Tensor<T>(shape);
        long bytes = (long)tensor.Length * elementSize;
        if (!File.Exists(path)) throw new FileNotFoundException("No shared tensor at this path.", path);
        if (new FileInfo(path).Length < bytes)
            throw new ArgumentException($"The shared memory holds {new FileInfo(path).Length} bytes; shape [{string.Join(", ", shape)}] needs {bytes}.", nameof(shape));
        AliasSharedFile(tensor, path, (int)bytes);
        tensor._sharedMemoryPath = path;
        return tensor;
    }

    private static void AliasSharedFile(Tensor<T> tensor, string path, int bytes)
    {
        MmapTensorMemoryManager<T>? manager = null;
        try
        {
            manager = new MmapTensorMemoryManager<T>(path, 0, Math.Max(bytes, 1), tensor.Length, writable: true);
            tensor.AliasStorageFromMmap(Vector<T>.WrapMemory(manager.Memory), manager, writable: true);
        }
        catch
        {
            ((IDisposable?)manager)?.Dispose();
            throw;
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

    private static void DeleteAtExit(string path)
        => AppDomain.CurrentDomain.ProcessExit += (_, _) => { try { File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { } };
}
