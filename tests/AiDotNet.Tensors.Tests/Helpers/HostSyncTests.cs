// Copyright (c) AiDotNet. All rights reserved.

using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Host-sync state is per host ARRAY and shared by every storage over it (phase 3b: replaces the process-wide
/// materializer registry). A "download" here is a callback that fills the host array, standing in for a device copy.
/// </summary>
public sealed class HostSyncTests
{
    private static void Fill(float[] dst, float value)
    {
        for (int i = 0; i < dst.Length; i++) dst[i] = value;
    }

    [Fact]
    public void AliasCreatedBeforeRegistration_SeesPendingDownload()
    {
        var array = new float[4];
        var alias = Vector<float>.Wrap(array);          // exists before the device result is registered
        var owner = Vector<float>.Wrap(array);
        HostSync.Register(owner, _ => Fill(array, 7f));

        Assert.Equal(7f, alias.AsSpan()[0]);
        Assert.False(HostSync.IsPending(owner));
    }

    [Fact]
    public void SegmentView_OfFlatArray_DownloadsFlatOnce()
    {
        var flat = new float[10];
        var segment = Vector<float>.Wrap(flat, 4, 3);   // offset view (a parameter inside a flat buffer)
        int downloads = 0;
        HostSync.Register(flat, a => { Interlocked.Increment(ref downloads); Fill((float[])a, 3f); });

        Assert.Equal(3f, segment.AsSpan()[0]);          // span reads of an offset view used to skip the download
        Assert.Equal(3f, Vector<float>.Wrap(flat, 0, 4).AsSpan()[3]);
        Assert.Equal(1, downloads);
    }

    [Fact]
    public void LazyStorage_KeepsItsStateWhenItsArrayAppears()
    {
        var lazy = Vector<float>.CreateGpuResident(5);
        HostSync.Register(lazy, v => ((Vector<float>)v).MaterializeBacking(new[] { 1f, 2f, 3f, 4f, 5f }));
        Assert.True(HostSync.IsPending(lazy));

        var data = lazy.AsSpan().ToArray();             // installs a host array, then downloads into the storage
        Assert.Equal(5f, data[4]);
        Assert.False(HostSync.IsPending(lazy));
    }

    [Fact]
    public void TwoLazyStorages_KeepSeparateDownloads()
    {
        // Both have an empty backing (the shared Array.Empty<float>() instance) until their arrays appear.
        var first = Vector<float>.CreateGpuResident(2);
        var second = Vector<float>.CreateGpuResident(2);
        HostSync.Register(first, v => ((Vector<float>)v).MaterializeBacking(new[] { 1f, 1f }));
        HostSync.Register(second, v => ((Vector<float>)v).MaterializeBacking(new[] { 2f, 2f }));

        Assert.Equal(1f, first.AsSpan()[0]);
        Assert.Equal(2f, second.AsSpan()[0]);
    }

    [Fact]
    public void NewerRegistration_ReplacesPendingOlderOne()
    {
        var array = new float[2];
        var v = Vector<float>.Wrap(array);
        HostSync.Register(array, _ => Fill(array, 1f));
        HostSync.Register(v, _ => Fill(array, 2f));      // the other key of the same data: the newer result wins

        Assert.Equal(2f, v.AsSpan()[0]);
    }

    [Fact]
    public void ConcurrentReaders_WaitForOneDownload()
    {
        var array = new float[1 << 16];
        var v = Vector<float>.Wrap(array);
        int downloads = 0;
        using var started = new ManualResetEventSlim();
        HostSync.Register(v, _ =>
        {
            Interlocked.Increment(ref downloads);
            started.Set();
            Thread.Sleep(50);                            // readers arriving now must wait, not read a half-filled array
            Fill(array, 9f);
        });

        var readers = new Task<float>[8];
        readers[0] = Task.Run(() => v.AsSpan()[array.Length - 1]);
        started.Wait();
        for (int i = 1; i < readers.Length; i++) readers[i] = Task.Run(() => v.AsSpan()[array.Length - 1]);

        foreach (var r in readers) Assert.Equal(9f, r.Result);
        Assert.Equal(1, downloads);
    }

    [Fact]
    public void DownloadReadingItsOwnStorage_DoesNotRecurse()
    {
        var array = new float[3];
        var v = Vector<float>.Wrap(array);
        int downloads = 0;
        HostSync.Register(v, _ =>
        {
            downloads++;
            var span = v.AsWritableSpan();              // re-entrant read of the storage being filled
            for (int i = 0; i < span.Length; i++) span[i] = 4f;
        });

        Assert.Equal(4f, v.AsSpan()[2]);
        Assert.Equal(1, downloads);
    }

    [Fact]
    public void FailedDownload_StaysPending()
    {
        var array = new float[2];
        var v = Vector<float>.Wrap(array);
        bool fail = true;
        HostSync.Register(v, _ =>
        {
            if (fail) throw new System.InvalidOperationException("device busy");
            Fill(array, 6f);
        });

        Assert.Throws<System.InvalidOperationException>(() => v.AsSpan());
        Assert.True(HostSync.IsPending(v));
        fail = false;
        Assert.Equal(6f, v.AsSpan()[1]);
    }

    [Fact]
    public void ReleasedStorage_ThrowsOnHostRead_UntilRewritten()
    {
        var array = new float[2];
        var v = Vector<float>.Wrap(array);
        HostSync.Register(v, _ => Fill(array, 1f));
        Assert.True(HostSync.Release(array, "released intermediate"));

        var ex = Assert.Throws<System.InvalidOperationException>(() => v.AsSpan());
        Assert.Equal("released intermediate", ex.Message);
        HostSync.ClearReleased(v);
        Assert.Equal(0f, v.AsSpan()[0]);
    }
}
