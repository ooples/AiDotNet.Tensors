using System;
using System.Reflection;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The module pins of a direct-PTX CUDA graph capture, driven without a device: only the bookkeeping that decides
/// which capture owns which pin. A capture's recording ends when it leaves the capture gate; its graph is counted and
/// instantiated after that, while another thread may already be recording the next capture on the same backend.
/// </summary>
public class CudaDirectPtxCapturePinOwnershipTests
{
    private const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;

    private sealed class Kernel : IDisposable
    {
        public void Dispose() { }
    }

    private static object? Call(CudaBackend backend, string method, params object[] arguments)
    {
        var info = typeof(CudaBackend).GetMethod(method, Private)
                   ?? throw new MissingMethodException(nameof(CudaBackend), method);
        try { return info.Invoke(backend, arguments); }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
    }

    private static object Begin(CudaBackend backend) => Call(backend, "BeginDirectPtxCapturePinTracking")!;

    // What leaving the capture gate does to the collector. A backend without the step leaves it installed, which is
    // the state the next capture then finds.
    private static void EndRecording(CudaBackend backend, object pins)
    {
        if (typeof(CudaBackend).GetMethod("DetachDirectPtxCapturePinTracking", Private) is not null)
            Call(backend, "DetachDirectPtxCapturePinTracking", pins);
    }

    private static object? ActiveCollector(CudaBackend backend)
        => typeof(CudaBackend).GetField("_activeDirectPtxCapturePins", Private)!.GetValue(backend);

    private static void Pin(object pins, DirectPtxKernelCache<string, Kernel> cache, string key)
    {
        var acquire = pins.GetType().GetMethod("Acquire", Private)!.MakeGenericMethod(typeof(string), typeof(Kernel));
        Assert.True((bool)acquire.Invoke(pins, new object[] { cache, key })!);
    }

    private static DirectPtxKernelCache<string, Kernel> Cache(params string[] keys)
    {
        var cache = new DirectPtxKernelCache<string, Kernel>(8);
        foreach (string key in keys) cache.AddOrGetExisting(key, new Kernel());
        return cache;
    }

    [Fact]
    public void A_graph_finalized_while_the_next_capture_records_keeps_its_own_pins()
    {
        // Declared first, so it outlives the backend: a backend still holding pins releases them into it on Dispose.
        using var cache = Cache("first", "second");
        using var backend = new CudaBackend();
        var graph = new IntPtr(0x1000);

        var first = Begin(backend);
        Pin(first, cache, "first");
        EndRecording(backend, first);

        var second = Begin(backend);   // another thread entered the gate
        Pin(second, cache, "second");
        Call(backend, "CompleteDirectPtxCapturePinTracking", first, graph);

        Assert.Same(second, ActiveCollector(backend));   // the recording capture still collects its own pins
        Assert.Equal(2, cache.PinnedCount);

        EndRecording(backend, second);
        Call(backend, "AbortDirectPtxCapturePinTracking", second);
        Assert.Null(ActiveCollector(backend));
        Assert.Equal(1, cache.PinnedCount);              // the first graph's module is still pinned by its graph

        Call(backend, "ReleaseDirectPtxGraphPins", graph);
        Assert.Equal(0, cache.PinnedCount);
    }

    [Fact]
    public void A_capture_that_fails_while_the_next_one_records_releases_only_its_own_pins()
    {
        // Declared first, so it outlives the backend: a backend still holding pins releases them into it on Dispose.
        using var cache = Cache("first", "second");
        using var backend = new CudaBackend();

        var first = Begin(backend);
        Pin(first, cache, "first");
        EndRecording(backend, first);

        var second = Begin(backend);
        Pin(second, cache, "second");
        Call(backend, "AbortDirectPtxCapturePinTracking", first);   // its graph did not instantiate

        Assert.Same(second, ActiveCollector(backend));
        Assert.Equal(1, cache.PinnedCount);

        var graph = new IntPtr(0x2000);
        EndRecording(backend, second);
        Call(backend, "CompleteDirectPtxCapturePinTracking", second, graph);
        Assert.Equal(1, cache.PinnedCount);
        Call(backend, "ReleaseDirectPtxGraphPins", graph);
        Assert.Equal(0, cache.PinnedCount);
    }

    [Fact]
    public void An_in_place_graph_update_swaps_the_graphs_pins_while_another_capture_records()
    {
        // Declared first, so it outlives the backend: a backend still holding pins releases them into it on Dispose.
        using var cache = Cache("old", "new", "other");
        using var backend = new CudaBackend();
        var graph = new IntPtr(0x3000);

        var original = Begin(backend);
        Pin(original, cache, "old");
        EndRecording(backend, original);
        Call(backend, "CompleteDirectPtxCapturePinTracking", original, graph);

        var update = Begin(backend);
        Pin(update, cache, "new");
        EndRecording(backend, update);

        var other = Begin(backend);
        Pin(other, cache, "other");
        Call(backend, "ReplaceDirectPtxGraphPins", graph, update);

        Assert.Same(other, ActiveCollector(backend));
        Assert.Equal(2, cache.PinnedCount);              // "new" for the graph and "other" still recording; "old" released

        EndRecording(backend, other);
        Call(backend, "AbortDirectPtxCapturePinTracking", other);
        Call(backend, "ReleaseDirectPtxGraphPins", graph);
        Assert.Equal(0, cache.PinnedCount);
    }
}
