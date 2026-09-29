using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

[CollectionDefinition("HostSyncRegistrySerial", DisableParallelization = true)]
public class HostSyncRegistrySerialCollection { }

/// <summary>
/// The registry contracts #1065 pinned on the process-wide DeferredArrayMaterializer, ported to HostSync, which
/// replaced it (state per host array in a weak table, no global registry or pending count). A property that only
/// existed as that count is checked directly: a collected result must really be collectable and never downloaded.
/// </summary>
[Collection("HostSyncRegistrySerial")]
public class HostSyncRegistryTests
{
    private static void CollectFully()
    {
        for (int i = 0; i < 3; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static List<WeakReference<float[]>> RegisterUnreferenced(int n)
    {
        var keys = new List<WeakReference<float[]>>(n);
        for (int i = 0; i < n; i++)
        {
            var key = new float[4];
            HostSync.Register(key, static _ => throw new InvalidOperationException("a collected result must never be downloaded"));
            keys.Add(new WeakReference<float[]>(key));
        }
        return keys;
    }

    [Fact]
    public void A_collected_pending_result_is_released_and_never_downloaded()
    {
        long downloadsBefore = HostSync.MaterializeCount;
        var keys = RegisterUnreferenced(1000);

        CollectFully();

        // The pending download must not root its host array (the old registry's strong keys leaked every result).
        Assert.DoesNotContain(keys, k => k.TryGetTarget(out _));
        Assert.Equal(downloadsBefore, HostSync.MaterializeCount);
    }

    [Fact]
    public void Concurrent_host_reads_download_each_result_exactly_once()
    {
        var keys = new List<float[]>();
        int downloads = 0;
        for (int i = 0; i < 5000; i++)
        {
            var key = new float[4];
            keys.Add(key);
            HostSync.Register(key, _ => Interlocked.Increment(ref downloads));
        }

        Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, _ =>
        {
            foreach (var key in keys) HostSync.TryMaterialize(key);
        });

        Assert.Equal(keys.Count, downloads);
        Assert.DoesNotContain(keys, k => HostSync.IsPending(k));
    }

    [Fact]
    public void A_second_registration_for_the_same_key_replaces_the_first_and_downloads_once()
    {
        // The old registry kept the FIRST registration. HostSync keeps the newest by design: a newer device result
        // for the same host array is the truth (HostSync remarks; HostSyncTests.NewerRegistration_ReplacesPendingOlderOne).
        // What carries over is that the pair still produces exactly one download and leaves nothing pending.
        var key = new float[4];
        int first = 0, second = 0;

        HostSync.Register(key, _ => first++);
        HostSync.Register(key, _ => second++);

        Assert.True(HostSync.TryMaterialize(key));
        Assert.False(HostSync.TryMaterialize(key));
        Assert.Equal((0, 1), (first, second));
        Assert.False(HostSync.IsPending(key));
        GC.KeepAlive(key);
    }

    [Fact]
    public void Removing_a_pending_result_skips_its_download()
    {
        var key = new float[4];
        int downloads = 0;

        HostSync.Register(key, _ => downloads++);
        HostSync.Remove(key);

        Assert.False(HostSync.IsPending(key));
        Assert.False(HostSync.TryMaterialize(key));
        Assert.Equal(0, downloads);
        GC.KeepAlive(key);
    }
}