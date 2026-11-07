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
/// <summary>
/// Whether a backend's buffers charge <see cref="DeviceMemoryReclaim.OnDeviceAllocation"/> when handed out and credit it
/// when disposed. When they do, the engine adds no charge of its own for the results it defers onto them; when they do
/// not (Metal, WebGPU), the engine counts result bytes itself so abandoned results still prompt a collection.
/// </summary>
internal interface IChargesDeviceMemoryReclaim
{
    bool ChargesDeviceMemoryReclaim { get; }
}

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

    /// <summary>
    /// Cumulative unreleased device bytes across steps past which the step-boundary sweep upgrades from Gen 1 to Gen 2.
    /// Ephemeral managed-heap Gen-1 collections during an open <c>GradientTape</c> can promote live forward device-buffer
    /// wrappers into Gen 2; tracking cumulative unreleased bytes separately from the per-step Gen-1 counter prevents
    /// sub-256-MB steps from leaking Gen-2 device wrappers across hundreds of steps.
    /// </summary>
    internal const long CollectGen2AfterDeviceBytes = 4 * CollectAfterDeviceBytes;

    private static int s_collectionDue;

    /// <summary>Whether a backend has asked for a collection at the next step or inference boundary.</summary>
    internal static bool CollectionDue => Volatile.Read(ref s_collectionDue) != 0;

    /// <summary>Test hook: starts the byte and allocation counts from zero with no collection due.</summary>
    internal static void ResetCountsForTests()
    {
        ResetDeviceBytes(resetGen2: true);
        Interlocked.Exchange(ref s_allocationsSinceCollection, 0);
        Volatile.Write(ref s_collectionDue, 0);
    }
    // Device bytes handed out since the last collection, across every backend: one collection serves them all.
    private static long s_driverBytesSinceCollection;

    // Cumulative unreleased device bytes handed out since the last Gen-2 collection.
    private static long s_driverBytesSinceGen2Collection;

    // Bumped by every reset of s_driverBytesSinceCollection. A charge belongs to the epoch it was counted in; once a
    // reset has dropped it, crediting it back would cancel bytes charged afterwards and hide an abandoned buffer.
    private static long s_collectionEpoch;

    // The s_collectionEpoch value at the most recent Gen-2 reset; charges from >= this epoch still sit in
    // s_driverBytesSinceGen2Collection until explicitly disposed or swept by Gen 2.
    private static long s_gen2ResetCollectionEpoch;

    /// <summary>Zeroes the byte count and starts a new epoch, so no charge counted before it is credited after it.</summary>
    private static void ResetDeviceBytes(bool resetGen2 = false)
    {
        // Epoch first: a credit that read the old epoch and the pre-reset count then fails its exchange, re-reads, and
        // sees the new epoch.
        long newEpoch = Interlocked.Increment(ref s_collectionEpoch);
        Interlocked.Exchange(ref s_driverBytesSinceCollection, 0);
        if (resetGen2)
        {
            Volatile.Write(ref s_gen2ResetCollectionEpoch, newEpoch);
            Interlocked.Exchange(ref s_driverBytesSinceGen2Collection, 0);
        }
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
        long sinceGen2 = Interlocked.Add(ref s_driverBytesSinceGen2Collection, bytes);
        if (since < CollectAfterDeviceBytes && sinceGen2 < CollectGen2AfterDeviceBytes) return epoch;
        if ((since < CollectNowAfterDeviceBytes && sinceGen2 < CollectGen2AfterDeviceBytes) || TapeOpen())
        {
            Volatile.Write(ref s_collectionDue, 1);
            return epoch;
        }
        bool gen2 = sinceGen2 >= CollectGen2AfterDeviceBytes;
        ResetDeviceBytes(resetGen2: gen2);
        Interlocked.Exchange(ref s_allocationsSinceCollection, 0);
        Volatile.Write(ref s_collectionDue, 0);
        GC.Collect(gen2 ? 2 : 1, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
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
        if (epoch < 0) return;

        // Subtract and floor at zero in one atomic update: a negative counter would hide later allocations.
        long current = Volatile.Read(ref s_driverBytesSinceCollection);
        while (Volatile.Read(ref s_collectionEpoch) == epoch)
        {
            long next = Math.Max(0, current - bytes);
            long observed = Interlocked.CompareExchange(ref s_driverBytesSinceCollection, next, current);
            if (observed == current) break;
            current = observed;
        }

        CreditGen2(bytes, epoch);
    }

    /// <summary>
    /// Credits a buffer the GC reclaimed (its owner was collected and the finalizer queued it back to the pool or to
    /// the driver) against the cumulative Gen-2 count only. The young-generation collection that found it already
    /// reset the per-collection count, but the Gen-2 count tracks bytes still unreleased: leaving a finalized buffer in
    /// it forced a Gen-2 collection every <see cref="CollectGen2AfterDeviceBytes"/> even when young collections were
    /// recovering everything. Safe on the finalizer thread: it only updates counters.
    /// </summary>
    internal static void OnDeviceFinalized(long bytes, long epoch)
    {
        if (epoch < 0 || bytes <= 0) return;
        CreditGen2(bytes, epoch);
    }

    private static void CreditGen2(long bytes, long epoch)
    {
        long currentGen2 = Volatile.Read(ref s_driverBytesSinceGen2Collection);
        while (epoch >= Volatile.Read(ref s_gen2ResetCollectionEpoch))
        {
            long nextGen2 = Math.Max(0, currentGen2 - bytes);
            long observedGen2 = Interlocked.CompareExchange(ref s_driverBytesSinceGen2Collection, nextGen2, currentGen2);
            if (observedGen2 == currentGen2) break;
            currentGen2 = observedGen2;
        }
    }

    /// <summary>Test hook: the cumulative unreleased bytes the Gen-2 sweep is measured against.</summary>
    internal static long BytesSinceGen2CollectionForTests => Volatile.Read(ref s_driverBytesSinceGen2Collection);

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
        bool gen2 = Volatile.Read(ref s_driverBytesSinceGen2Collection) >= CollectGen2AfterDeviceBytes;
        Interlocked.Exchange(ref s_allocationsSinceCollection, 0);
        ResetDeviceBytes(resetGen2: gen2);
        Volatile.Write(ref s_collectionDue, 0);
        GC.Collect(gen2 ? 2 : 1, GCCollectionMode.Forced, blocking: true);
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
        long bytes = Volatile.Read(ref s_driverBytesSinceCollection);
        long gen2Bytes = Volatile.Read(ref s_driverBytesSinceGen2Collection);
        if (bytes < CollectAfterDeviceBytes && gen2Bytes < CollectGen2AfterDeviceBytes) return false;
        bool gen2 = gen2Bytes >= CollectGen2AfterDeviceBytes;
        ResetDeviceBytes(resetGen2: gen2);
        Interlocked.Exchange(ref s_allocationsSinceCollection, 0);
        // Normal managed-heap Gen-0/Gen-1 collections during a multi-layer training step promote early forward
        // activations into Gen 1 or Gen 2; collect through Gen 1 every 64 MB and Gen 2 every 256 MB of cumulative
        // unreleased device allocations once GradientTape._entries has been reset.
        GC.Collect(gen2 ? 2 : 1, GCCollectionMode.Forced, blocking: true);
        // The collected buffers come back through their finalizers. Waiting for them (they only queue a pool return)
        // lets the next step's allocations find them in the pool instead of going to the driver.
        GC.WaitForPendingFinalizers();
        return true;
    }

    internal static void CollectUnreachable()
    {
        ResetDeviceBytes(resetGen2: true);
        Interlocked.Exchange(ref s_allocationsSinceCollection, 0);
        Volatile.Write(ref s_collectionDue, 0);
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
