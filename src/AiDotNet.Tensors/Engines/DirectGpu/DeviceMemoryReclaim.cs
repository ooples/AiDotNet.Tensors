using System;
using System.Collections.Concurrent;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Device frees of buffers whose owner was collected without being disposed. A finalizer only QUEUES the free: the
/// GPU driver is never called on the finalizer thread (that races op threads' driver calls). The owning backend runs
/// the queue on an op thread at its next allocation, and before an out-of-memory retry.
/// </summary>
internal sealed class DeviceFreeQueue
{
    private readonly ConcurrentQueue<Action> _frees = new();

    /// <summary>Queues <paramref name="free"/> (from a finalizer); dropped once the process is exiting.</summary>
    internal void Enqueue(Action free)
    {
        if (!DeviceMemoryReclaim.ProcessExiting) _frees.Enqueue(free);
    }

    internal bool IsEmpty => _frees.IsEmpty;

    /// <summary>Runs every queued free; returns how many ran. A failing free does not strand the rest.</summary>
    internal int Drain()
    {
        int ran = 0;
        while (_frees.TryDequeue(out var free))
        {
            try { free(); ran++; }
            catch { /* the device context may already be gone; the driver reclaims it with the context */ }
        }
        return ran;
    }
}

/// <summary>
/// A device allocation failed for lack of memory (<see cref="NativeCode"/> = the backend's error code). Derives from
/// <see cref="InvalidOperationException"/>, which these failures threw before, so existing handlers still catch it.
/// </summary>
public sealed class GpuOutOfMemoryException : InvalidOperationException
{
    public GpuOutOfMemoryException(string message, int nativeCode) : base(message) => NativeCode = nativeCode;

    /// <summary>The backend's native error code (e.g. CL_MEM_OBJECT_ALLOCATION_FAILURE, VK_ERROR_OUT_OF_DEVICE_MEMORY).</summary>
    public int NativeCode { get; }
}

/// <summary>
/// Out-of-memory recovery shared by every GPU backend: device memory held only by unreachable results is freed by
/// their finalizers (there is no cache evicting live data), so an allocation that runs out collects, lets the
/// finalizers queue their frees, runs them, drains the backend's buffer pool and retries once -- then fails with
/// the backend's own out-of-memory error. PyTorch's caching allocator frees its cached blocks and retries the same way.
/// </summary>
internal static class DeviceMemoryReclaim
{
    private static int s_processExiting;

    static DeviceMemoryReclaim()
    {
        AppDomain.CurrentDomain.ProcessExit += (_, _) => Volatile.Write(ref s_processExiting, 1);
    }

    internal static bool ProcessExiting => Volatile.Read(ref s_processExiting) != 0;

    /// <summary>Collects unreachable managed owners and waits for their finalizers to queue their device frees.</summary>
    internal static void CollectUnreachable()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    /// <summary>
    /// Runs <paramref name="allocate"/>; when it fails with out-of-memory (per <paramref name="isOutOfMemory"/>),
    /// collects unreachable owners, runs <paramref name="reclaim"/> (drain the free queue and the pool) and retries once.
    /// </summary>
    internal static T AllocateWithRetry<T>(Func<T> allocate, Func<Exception, bool> isOutOfMemory, Action reclaim)
    {
        try
        {
            return allocate();
        }
        catch (Exception ex) when (isOutOfMemory(ex))
        {
            CollectUnreachable();
            reclaim();
            return allocate();
        }
    }
}
