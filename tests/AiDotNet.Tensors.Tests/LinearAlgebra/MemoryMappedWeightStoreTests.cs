using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// #1715 mmap redesign — proves the memory-mapped slice store's core contract: slices
/// round-trip exactly, mutations through the writable span survive read-back WITH NO
/// write-back code (the property the bespoke streaming pool needed ReplaceEntryData +
/// write-back-before-drop for), the single file grows without corrupting earlier slices,
/// and Dispose deletes the backing file (no orphaned multi-GB temp files).
/// </summary>
public class MemoryMappedWeightStoreTests
{
    private static string TempPath() =>
        Path.Combine(Path.GetTempPath(), "mmws-test-" + Guid.NewGuid().ToString("N") + ".bin");

    private static byte[] MakeBytes(int len, int seed)
    {
        var r = new Random(seed);
        var b = new byte[len];
        r.NextBytes(b);
        return b;
    }

    [Fact]
    public void Allocate_GetSpan_RoundTripsExactly_AndDisposeDeletesFile()
    {
        var path = TempPath();
        using (var store = new MemoryMappedWeightStore(path, initialCapacity: 64 * 1024))
        {
            var a = MakeBytes(1000, seed: 1);
            var b = MakeBytes(2000, seed: 2);
            long oa = store.Allocate(a);
            long ob = store.Allocate(b);
            Assert.True(store.GetSpan(oa, a.Length).SequenceEqual(a));
            Assert.True(store.GetSpan(ob, b.Length).SequenceEqual(b));
            Assert.Equal(0, oa);              // first slice at offset 0
            Assert.True(ob >= 1000 && (ob % 64) == 0); // 64-byte aligned after the first slice
        }
        Assert.False(File.Exists(path), "Dispose must delete the backing file");
    }

    [Fact]
    public void Mutate_ThroughSpan_SurvivesReadBack_WithNoWriteBackCode()
    {
        // The exact round-trip the bespoke #707 pool needed manual write-back-before-drop for:
        // mutate a resident slice, read it back, observe the mutation. Free here — the span and
        // the read-back alias the same mapped pages.
        var path = TempPath();
        using var store = new MemoryMappedWeightStore(path, initialCapacity: 64 * 1024);
        var data = MakeBytes(4096, seed: 7);
        long off = store.Allocate(data);

        var span = store.GetSpan(off, data.Length);
        for (int i = 0; i < span.Length; i++) span[i] = (byte)(span[i] ^ 0xA5);

        var read = store.GetSpan(off, data.Length);
        for (int i = 0; i < data.Length; i++)
            Assert.Equal((byte)(data[i] ^ 0xA5), read[i]);
    }

    [Fact]
    public void Grow_PreservesEarlierSlices()
    {
        var path = TempPath();
        // Tiny initial capacity forces several doubling grows across the allocations.
        using var store = new MemoryMappedWeightStore(path, initialCapacity: 4096);
        const int n = 8, sz = 2000;
        var datas = new byte[n][];
        var offs = new long[n];
        for (int i = 0; i < n; i++)
        {
            datas[i] = MakeBytes(sz, seed: i + 10);
            offs[i] = store.Allocate(datas[i]);
        }
        Assert.True(store.Capacity >= store.UsedBytes);
        Assert.True(store.Capacity > 4096, "the allocations must have forced a grow");
        // Re-fetch each span AFTER all grows (per the span-lifetime contract) — all still exact.
        for (int i = 0; i < n; i++)
            Assert.True(store.GetSpan(offs[i], datas[i].Length).SequenceEqual(datas[i]),
                $"slice {i} corrupted by a grow");
    }

    [Fact]
    public void ManySlices_AcrossMmap_AllReadBackCorrectly()
    {
        // 256 × 1 MB = 256 MB of slices across the single mapped file; all correct. (Bounding the
        // RESIDENT set under host pressure is the OS page cache's job for a file-backed mapping —
        // inherent, not asserted here; this proves correctness at scale.)
        var path = TempPath();
        using var store = new MemoryMappedWeightStore(path, initialCapacity: 1024 * 1024);
        const int n = 256, sz = 1024 * 1024;
        var offs = new long[n];
        for (int i = 0; i < n; i++)
        {
            var d = new byte[sz];
            d[0] = (byte)i;
            d[sz - 1] = (byte)(i * 7 + 1);
            offs[i] = store.Allocate(d);
        }
        for (int i = 0; i < n; i++)
        {
            var s = store.GetSpan(offs[i], sz);
            Assert.Equal((byte)i, s[0]);
            Assert.Equal((byte)(i * 7 + 1), s[sz - 1]);
        }
    }
}
