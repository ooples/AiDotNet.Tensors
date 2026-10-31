// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The weak registry a releasing GPU engine downloads its live results through: no device needed, so these run on
/// every lane.
/// </summary>
public class DeviceResultRegistryTests
{
    private sealed class FakeStorage : IDeviceResultStorage
    {
        internal readonly List<IDirectGpuBackend> EvacuatedFrom = new();
        internal bool Throw;

        public void EvacuateToHost(IDirectGpuBackend backend)
        {
            if (Throw) throw new InvalidOperationException("download failed");
            EvacuatedFrom.Add(backend);
        }
    }

    // Any backend object will do: the registry only hands it back.
    private static IDirectGpuBackend Backend() =>
        (IDirectGpuBackend)RuntimeHelpers.GetUninitializedObject(typeof(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend));

    [Fact]
    public void A_release_moves_every_live_result_from_the_backend_it_was_bound_on()
    {
        var registry = new DeviceResultRegistry();
        var first = Backend();
        var second = Backend();
        var a = new FakeStorage();
        var b = new FakeStorage();
        registry.Add(a, first);
        registry.Add(b, second);

        Assert.Equal(2, registry.EvacuateAll());
        Assert.Equal(new[] { first }, a.EvacuatedFrom);
        Assert.Equal(new[] { second }, b.EvacuatedFrom);
        Assert.Equal(0, registry.CountLive());
        GC.KeepAlive(a);
        GC.KeepAlive(b);
    }

    // Results registered on other threads (ops run in parallel) are moved too.
    [Fact]
    public void A_release_moves_results_registered_on_other_threads()
    {
        var registry = new DeviceResultRegistry();
        var backend = Backend();
        var storages = new FakeStorage[8];
        for (int i = 0; i < storages.Length; i++) storages[i] = new FakeStorage();
        Parallel.For(0, storages.Length, new ParallelOptions { MaxDegreeOfParallelism = 8 },
            i => registry.Add(storages[i], backend));

        Assert.Equal(storages.Length, registry.EvacuateAll());
        foreach (var storage in storages) Assert.Single(storage.EvacuatedFrom);
    }

    // A collected result costs only its slot until the next compaction: the registry stays proportional to the live
    // results however many were registered.
    [Fact]
    public void Collected_results_are_compacted_away()
    {
        var registry = new DeviceResultRegistry();
        var backend = Backend();
        var kept = new FakeStorage();
        registry.Add(kept, backend);
        AddDropped(registry, backend, 10_000);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        // 10,001 slots put the next compaction at 16,384; crossing it frees the 10,000 collected ones.
        AddDropped(registry, backend, 7_000);

        Assert.True(registry.CountSlots() <= 7_001, $"{registry.CountSlots()} slots for at most 7,001 live results");
        Assert.Equal(1, CountEvacuated(registry, kept));
        GC.KeepAlive(kept);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void AddDropped(DeviceResultRegistry registry, IDirectGpuBackend backend, int count)
    {
        for (int i = 0; i < count; i++) registry.Add(new FakeStorage(), backend);
    }

    private static int CountEvacuated(DeviceResultRegistry registry, FakeStorage kept)
    {
        registry.EvacuateAll();
        return kept.EvacuatedFrom.Count;
    }

    // One result's failed download leaves its error for the read that needs it; the others still move.
    [Fact]
    public void A_failed_download_does_not_stop_the_others()
    {
        var registry = new DeviceResultRegistry();
        var backend = Backend();
        var failing = new FakeStorage { Throw = true };
        var fine = new FakeStorage();
        registry.Add(failing, backend);
        registry.Add(fine, backend);

        Assert.Equal(1, registry.EvacuateAll());
        Assert.Single(fine.EvacuatedFrom);
        GC.KeepAlive(failing);
    }
}
