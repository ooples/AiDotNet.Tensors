using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// #1715 mmap redesign, stage 2 — a streamed weight tensor's storage can VIEW a writable
/// memory-mapped slice directly (no resident GC copy), and mutations through that storage
/// persist to the file (MAP_SHARED). This is what kills the OOM: the param-IO round-trip
/// (GetParameters → mutate → SetParameters → read back) no longer materializes the whole
/// model as resident arrays — each weight aliases its mapped slice and the OS pages it.
/// Read-only aliases (inference) still gate writes (fail loud rather than fault).
/// </summary>
public class MmapWritableAliasTests
{
    private static string TempPath() =>
        Path.Combine(Path.GetTempPath(), "mmwa-test-" + Guid.NewGuid().ToString("N") + ".bin");

    private static void WriteFloats(string path, float[] values)
    {
        using var fs = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        fs.Write(bytes, 0, bytes.Length);
    }

    private static float[] ReadFloats(string path, int count)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        var bytes = new byte[count * sizeof(float)];
        int read = 0;
        while (read < bytes.Length)
        {
            int n = fs.Read(bytes, read, bytes.Length - read);
            if (n == 0) throw new EndOfStreamException($"Expected {bytes.Length} bytes, read {read}.");
            read += n;
        }
        var values = new float[count];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    [Fact]
    public void WritableManager_MutationPersistsToFile()
    {
        const int n = 1024;
        var path = TempPath();
        var initial = new float[n];
        for (int i = 0; i < n; i++) initial[i] = i * 0.5f;
        WriteFloats(path, initial);

        using (var mgr = new MmapTensorMemoryManager<float>(path, 0, n * sizeof(float), n, writable: true))
        {
            Assert.True(mgr.IsWritable);
            var span = mgr.GetSpan();
            for (int i = 0; i < n; i++) Assert.Equal(initial[i], span[i]);  // reads the file zero-copy
            for (int i = 0; i < n; i++) span[i] = span[i] + 1000f;          // mutate through the mapping
        } // Dispose flushes dirty pages to the file

        var roundTripped = ReadFloats(path, n);
        for (int i = 0; i < n; i++) Assert.Equal(initial[i] + 1000f, roundTripped[i]);
        File.Delete(path);
    }

    [Fact]
    public void ReadOnlyAlias_GatesWrites_WritableAlias_AllowsThem()
    {
        const int n = 64;
        var roPath = TempPath();
        var rwPath = TempPath();
        var vals = new float[n];
        for (int i = 0; i < n; i++) vals[i] = i;
        WriteFloats(roPath, vals);
        WriteFloats(rwPath, vals);

        var roMgr = new MmapTensorMemoryManager<float>(roPath, 0, n * sizeof(float), n, writable: false);
        var roStorage = new TensorStorage<float>(Vector<float>.WrapMemory(roMgr.Memory));
        roStorage.AttachMmapOwner(roMgr, writable: false);
        Assert.True(roStorage.IsReadOnlyMapped);
        Assert.Throws<InvalidOperationException>(() => roStorage.AsWritableSpan());
        roStorage.Release(); // disposes roMgr

        var rwMgr = new MmapTensorMemoryManager<float>(rwPath, 0, n * sizeof(float), n, writable: true);
        var rwStorage = new TensorStorage<float>(Vector<float>.WrapMemory(rwMgr.Memory));
        rwStorage.AttachMmapOwner(rwMgr, writable: true);
        Assert.False(rwStorage.IsReadOnlyMapped);
        var w = rwStorage.AsWritableSpan();   // must NOT throw
        w[0] = 42f;
        Assert.Equal(42f, rwStorage.AsSpan()[0]);
        rwStorage.Release(); // disposes rwMgr

        File.Delete(roPath);
        File.Delete(rwPath);
    }

    [Fact]
    public void TensorAliasedToWritableMmap_HasNoResidentArray_AndRoundTripsMutation()
    {
        const int n = 512;
        var path = TempPath();
        var initial = new float[n];
        for (int i = 0; i < n; i++) initial[i] = (i % 13) - 6f;
        WriteFloats(path, initial);

        var mgr = new MmapTensorMemoryManager<float>(path, 0, n * sizeof(float), n, writable: true);
        var aliased = Vector<float>.WrapMemory(mgr.Memory);
        var tensor = new Tensor<float>(new[] { n });
        tensor.AliasStorageFromMmap(aliased, mgr, writable: true);

        // The tensor now VIEWS the mapped slice — reading it returns the file's values with no
        // resident array materialized (an mmap-backed Vector is not array-backed).
        for (int i = 0; i < n; i++) Assert.Equal(initial[i], tensor[i]);

        // Mutate through the TENSOR API (not the raw alias) so this exercises the tensor-level write
        // gate (TensorBase.AsWritableSpan → TensorStorage, which allows writable mmaps and would throw
        // for a read-only one). The change lands in the mmap slice, shared with the file.
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < n; i++) span[i] = initial[i] * 2f + 1f;
        for (int i = 0; i < n; i++) Assert.Equal(initial[i] * 2f + 1f, tensor[i]); // tensor sees it (shared storage)

        tensor.Dispose(); // last ref → disposes mgr → flushes the mapping to the file

        var roundTripped = ReadFloats(path, n);
        for (int i = 0; i < n; i++) Assert.Equal(initial[i] * 2f + 1f, roundTripped[i]); // persisted
        File.Delete(path);
    }
}
