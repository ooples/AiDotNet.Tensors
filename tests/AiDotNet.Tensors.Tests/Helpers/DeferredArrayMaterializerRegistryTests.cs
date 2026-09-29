using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

// The pending count is process-global, so these run alone: a parallel GPU test registering its own results would
// move the count under the exact assertions below.
[CollectionDefinition("DeferredMaterializerRegistrySerial", DisableParallelization = true)]
public class DeferredMaterializerRegistrySerialCollection { }

/// <summary>
/// Pins the registry that holds a GPU result's pending download. It is weakly keyed, so a result nobody
/// references takes its pending download (and its device buffer) with it, and it is lock-free.
/// </summary>
/// <remarks>
/// Measured failure these guard: with weak keys, a collected entry never decremented the pending count, so the
/// count only grew, TryMaterialize's "nothing pending" fast path never fired again, and every host read of every
/// tensor went through a global lock — parallel CPU loops in a training step spent a third of the step in
/// Monitor.Enter_Slowpath.
/// </remarks>
[Collection("DeferredMaterializerRegistrySerial")]
public class DeferredArrayMaterializerRegistryTests
{
    private static int PendingCount() =>
        (int)typeof(DeferredArrayMaterializer)
            .GetField("_pendingCount", BindingFlags.NonPublic | BindingFlags.Static)!
            .GetValue(null)!;

    private static void CollectFully()
    {
        for (int i = 0; i < 3; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RegisterUnreferenced(int n)
    {
        for (int i = 0; i < n; i++)
            DeferredArrayMaterializer.Register(new float[4], static _ => throw new InvalidOperationException("a collected result must never be downloaded"));
    }

    [Fact]
    public void A_collected_pending_result_gives_back_its_count()
    {
        CollectFully();
        int before = PendingCount();

        RegisterUnreferenced(1000);
        Assert.Equal(before + 1000, PendingCount());

        CollectFully();
        Assert.Equal(before, PendingCount());
    }

    [Fact]
    public void Concurrent_host_reads_download_each_result_exactly_once()
    {
        CollectFully();
        int before = PendingCount();
        var keys = new List<float[]>();
        int downloads = 0;
        for (int i = 0; i < 5000; i++)
        {
            var key = new float[4];
            keys.Add(key);
            DeferredArrayMaterializer.Register(key, _ => Interlocked.Increment(ref downloads));
        }

        Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, _ =>
        {
            foreach (var key in keys) DeferredArrayMaterializer.TryMaterialize(key);
        });

        Assert.Equal(keys.Count, downloads);
        Assert.Equal(before, PendingCount());
        GC.KeepAlive(keys);
    }

    [Fact]
    public void A_second_registration_for_the_same_key_is_ignored_and_not_counted()
    {
        CollectFully();
        int before = PendingCount();
        var key = new float[4];
        int first = 0, second = 0;

        DeferredArrayMaterializer.Register(key, _ => first++);
        DeferredArrayMaterializer.Register(key, _ => second++);
        Assert.Equal(before + 1, PendingCount());

        Assert.True(DeferredArrayMaterializer.TryMaterialize(key));
        Assert.False(DeferredArrayMaterializer.TryMaterialize(key));
        Assert.Equal((1, 0), (first, second));
        Assert.Equal(before, PendingCount());

        CollectFully(); // the losing duplicate's finalizer must not decrement a second time
        Assert.Equal(before, PendingCount());
        GC.KeepAlive(key);
    }

    [Fact]
    public void Removing_a_pending_result_skips_its_download_and_gives_back_its_count()
    {
        CollectFully();
        int before = PendingCount();
        var key = new float[4];
        int downloads = 0;

        DeferredArrayMaterializer.Register(key, _ => downloads++);
        DeferredArrayMaterializer.Remove(key);

        Assert.False(DeferredArrayMaterializer.TryMaterialize(key));
        Assert.Equal(0, downloads);
        Assert.Equal(before, PendingCount());
        CollectFully();
        Assert.Equal(before, PendingCount());
        GC.KeepAlive(key);
    }
}
