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
    /// <summary>
    /// Device bytes a backend may hand out (pool hits and driver allocations alike) before it asks for a
    /// young-generation collection. A dead result's buffer returns to the pool only when the GC finalizes it, and a GPU loop allocates so
    /// little managed memory that collections can be hundreds of steps apart: device memory grew by every step's dead
    /// results until one happened (measured on OpenCL: 20-330 MB for a ~10 MB working set). A gen-0 collection on a GPU
    /// loop's small managed heap costs well under a millisecond.
    /// </summary>
    internal const long CollectAfterDeviceBytes = 64L << 20;

    /// <summary>
    /// Driver allocation past which a backend collects on the spot rather than waiting for a step boundary: a
    /// collection in the middle of a step promotes the step's live results to an older generation, where young
    /// collections no longer reclaim them, so it is the backstop, not the rule.
    /// </summary>
    internal const long CollectNowAfterDeviceBytes = 4 * CollectAfterDeviceBytes;

    private static int s_collectionDue;

    /// <summary>Whether a backend has asked for a collection at the next step or inference boundary.</summary>
    internal static bool CollectionDue => Volatile.Read(ref s_collectionDue) != 0;

    /// <summary>Test hook: starts the byte and allocation counts from zero with no collection due.</summary>
    internal static void ResetCountsForTests()
    {
        ResetDeviceBytes();
        Interlocked.Exchange(ref s_allocationsSinceCollection, 0);
        Volatile.Write(ref s_collectionDue, 0);
    }
    // Device bytes handed out since the last collection, across every backend: one collection serves them all.
    private static long s_driverBytesSinceCollection;

    // Bumped by every reset of s_driverBytesSinceCollection. A charge belongs to the epoch it was counted in; once a
    // reset has dropped it, crediting it back would cancel bytes charged afterwards and hide an abandoned buffer.
    private static long s_collectionEpoch;

    /// <summary>Zeroes the byte count and starts a new epoch, so no charge counted before it is credited after it.</summary>
    private static void ResetDeviceBytes()
    {
        // Epoch first: a credit that read the old epoch and the pre-reset count then fails its exchange, re-reads, and
        // sees the new epoch.
        Interlocked.Increment(ref s_collectionEpoch);
        Interlocked.Exchange(ref s_driverBytesSinceCollection, 0);
    }

    /// <summary>
    /// Counts <paramref name="bytes"/> handed out since the last collection, pool hits included: a warm loop's
    /// allocations are almost all pool hits, and counting only driver allocations left collections to the managed
    /// heap's own pace (about every 100 steps once results stopped carrying host arrays), so a hundred steps of dead
    /// results had to be covered by device memory. Past
    /// <see cref="CollectAfterDeviceBytes"/> a gen-0 collection is due at the next step boundary
    /// (<see cref="CollectIfDueAtStepBoundary"/>); past <see cref="CollectNowAfterDeviceBytes"/> it runs now.
    /// </summary>
    /// <returns>The epoch the charge is counted in, for <see cref="OnDeviceRelease"/>. A charge that set off the
    /// collection here gets the epoch that collection ended, so its credit is dropped; one that raced another reset may
    /// too, which only brings a collection slightly earlier, never later.</returns>
    internal static long OnDeviceAllocation(long bytes)
    {
        long epoch = Volatile.Read(ref s_collectionEpoch);
        Interlocked.Increment(ref s_allocationsSinceCollection);
        long since = Interlocked.Add(ref s_driverBytesSinceCollection, bytes);
        if (since < CollectAfterDeviceBytes) return epoch;
        if (since < CollectNowAfterDeviceBytes)
        {
            Volatile.Write(ref s_collectionDue, 1);
            return epoch;
        }
        ResetDeviceBytes();
        Interlocked.Exchange(ref s_allocationsSinceCollection, 0);
        Volatile.Write(ref s_collectionDue, 0);
        GC.Collect(0, GCCollectionMode.Forced, blocking: true);
        return epoch;
    }

    /// <summary>
    /// Credits back a buffer its owner disposed explicitly (it is back in the pool, so no collection is needed to
    /// recover it). Without the credit, scratch that is always disposed (a Winograd convolution's transform planes, a
    /// split-K GEMM's partials) counted toward the collection threshold like an abandoned result: a ResNet-50 forward
    /// forced one blocking collection per call (about 1.5 ms) to recover nothing.
    /// </summary>
    /// <param name="bytes">What the allocation charged.</param>
    /// <param name="epoch">What <see cref="OnDeviceAllocation"/> returned for that charge: a reset since then already
    /// dropped it, and the credit is skipped.</param>
    internal static void OnDeviceRelease(long bytes, long epoch)
    {
        // Subtract and floor at zero in one atomic update: a negative counter would hide later allocations.
        long current = Volatile.Read(ref s_driverBytesSinceCollection);
        while (true)
        {
            if (Volatile.Read(ref s_collectionEpoch) != epoch) return;
            long next = Math.Max(0, current - bytes);
            long observed = Interlocked.CompareExchange(ref s_driverBytesSinceCollection, next, current);
            if (observed == current) return;
            current = observed;
        }
    }

    // Buffers handed out since the last collection this class ran, pool hits included.
    private static long s_allocationsSinceCollection;

    /// <summary>Buffers a backend hands out before a pool miss is worth a collection (see <see cref="CollectOnPoolMiss"/>).</summary>
    internal const int CollectOnMissAfterAllocations = 256;

    /// <summary>Whether a gradient tape is open anywhere in the process. Tests replace it.</summary>
    internal static Func<bool> TapeOpen = Autodiff.DifferentiableOps.AnyTapeActive;

    /// <summary>
    /// Called by a backend whose pool had nothing for an allocation; returns true when it collected, so the caller
    /// can look in its pool again. The byte thresholds above wait for a step boundary, and a loop that runs no
    /// gradient tape has none: its dropped results were first collected at <see cref="CollectNowAfterDeviceBytes"/>,
    /// some 2,000 small allocations in, every one of them a driver allocation. A miss is the moment a collection
    /// pays, so with no tape open it runs once <see cref="CollectOnMissAfterAllocations"/> buffers have gone out.
    /// It collects generation 1, not 0: a result still alive at one of these collections is promoted, and a young
    /// collection would never return it afterwards (measured on a forward holding 600 intermediates: 770 device
    /// buffers created with generation 1, 1,312 and still growing with generation 0, 1,913 without this).
    /// With a tape open the step boundary does the collecting, as before.
    /// </summary>
    internal static bool CollectOnPoolMiss()
    {
        if (Volatile.Read(ref s_allocationsSinceCollection) < CollectOnMissAfterAllocations) return false;
        if (TapeOpen()) return false;
        Interlocked.Exchange(ref s_allocationsSinceCollection, 0);
        ResetDeviceBytes();
        Volatile.Write(ref s_collectionDue, 0);
        GC.Collect(1, GCCollectionMode.Forced, blocking: true);
        // The collected buffers come back through their finalizers, which only queue a pool return.
        GC.WaitForPendingFinalizers();
        return true;
    }

    /// <summary>
    /// Runs the gen-0 collection a backend asked for, at a point where the finished step's intermediates are dead (an
    /// outermost gradient tape ending): they are collected young and their buffers return to the pool by the next
    /// allocation. Collecting inside an allocation promoted them instead (measured on OpenCL: device memory grew by
    /// 574 MB over 300 steps once results stopped carrying host arrays).
    /// </summary>
    /// <returns>Whether it collected.</returns>
    internal static bool CollectIfDueAtStepBoundary()
    {
        if (Interlocked.Exchange(ref s_collectionDue, 0) == 0) return false;
        // Owners may have disposed what was handed out since the request (a TensorScope ending an inference call, a
        // tape releasing its intermediates): those buffers are back in the pool and credited, so a collection would
        // recover nothing. Measured on ResNet-50 predicts under a TensorScope (OpenCL): every intermediate came back
        // that way and no buffer was ever finalized, yet a blocking collection ran on every call.
        if (Volatile.Read(ref s_driverBytesSinceCollection) < CollectAfterDeviceBytes) return false;
        ResetDeviceBytes();
        Interlocked.Exchange(ref s_allocationsSinceCollection, 0);
        GC.Collect(0, GCCollectionMode.Forced, blocking: true);
        // The collected buffers come back through their finalizers. Waiting for them (they only queue a pool return)
        // lets the next step's allocations find them in the pool instead of going to the driver.
        GC.WaitForPendingFinalizers();
        return true;
    }

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
