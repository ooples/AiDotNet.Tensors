using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Reuses native backing between completed, permanently uncaptured compiled steps. Logical action slots remain
/// private to each plan. A checked-out buffer is absent from this broker until its complete step returns it.
/// All calls are serialized by the engine's CompiledStepLock; nested steps therefore have separate checkouts too.
/// </summary>
internal sealed class CompiledTransientWorkspaceBroker
{
    private readonly StepScratchPool _owner;
    private readonly List<(long Key, IDirectGpuBackend Backend, int Thread, IGpuBuffer Buffer)> _free = new();
    private long _nextKey;
    internal long ReuseCount { get; private set; }
    internal long ReuseBytes { get; private set; }
    internal long ReturnedBytes { get; private set; }
    internal long RetainedBytes { get; private set; }
    internal long NativeReturnedForInferenceBytes { get; private set; }
    internal long NativeReturnedForInferenceCells { get; private set; }

    internal CompiledTransientWorkspaceBroker(DirectGpuTensorEngine engine) => _owner = engine.CreateScratchPool();

    internal IGpuBuffer? TryRent(IDirectGpuBackend backend, int length)
    {
        int pick = -1;
        for (int i = 0; i < _free.Count; i++)
        {
            var item = _free[i];
            if (!ReferenceEquals(item.Backend, backend) || item.Thread != Environment.CurrentManagedThreadId || item.Buffer.Handle == IntPtr.Zero
                || item.Buffer.Size < length || item.Buffer.Size > 2L * length) continue;
            if (pick < 0 || item.Buffer.Size < _free[pick].Buffer.Size) pick = i;
        }
        if (pick < 0) return null;
        var chosen = _free[pick];
        _free[pick] = _free[_free.Count - 1];
        _free.RemoveAt(_free.Count - 1);
        _owner.Slots.Remove(chosen.Key);
        RetainedBytes -= chosen.Buffer.SizeInBytes;
        ReuseCount++;
        ReuseBytes += chosen.Buffer.SizeInBytes;
        BufferAllocationDiagnostics.RecordCompiledWorkspaceLoan(length);
        // Keep the pooled marker until the receiving plan replaces its owner. No tensor may dispose this backing.
        return chosen.Buffer;
    }

    internal void Return(IDirectGpuBackend backend, IGpuBuffer buffer)
    {
        long key = ++_nextKey;
        _owner.Track(buffer);
        _owner.Slots.Add(key, buffer);
        // OpenCL's canonical compute queue is thread-local. Host locking alone does not order two different queues.
        // Keep independent native checkouts for execution threads rather than adding a synchronizing handoff.
        _free.Add((key, backend, Environment.CurrentManagedThreadId, buffer));
        ReturnedBytes += buffer.SizeInBytes;
        RetainedBytes += buffer.SizeInBytes;
    }

    internal void Release()
    {
        foreach (var item in _free)
        {
            _owner.Slots.Remove(item.Key);
            StepScratchPool.Untrack(item.Buffer);
            if (item.Backend is IQueuedBufferRelease queued) queued.ReleaseAfterQueuedWork(item.Buffer);
            else item.Buffer.Dispose();
        }
        _free.Clear();
        RetainedBytes = 0;
    }

    // The caller holds CompiledStepLock, as for ordinary checkouts. Only same-backend, same-thread idle cells are
    // considered: host serialization does not order OpenCL's independent thread-local native queues.
    internal long ReturnIncompatibleCellsToNativeAllocator(IDirectGpuBackend backend, long requestedBytes)
    {
        if (requestedBytes <= 0) return 0;
        int thread = Environment.CurrentManagedThreadId;
        long largest = 0;
        foreach (var item in _free)
            if (ReferenceEquals(item.Backend, backend) && item.Thread == thread && item.Buffer.Handle != IntPtr.Zero)
                largest = Math.Max(largest, item.Buffer.SizeInBytes);
        if (largest == 0 || requestedBytes <= largest) return 0;

        long returned = 0;
        int cells = 0;
        for (int index = _free.Count - 1; index >= 0; index--)
        {
            var item = _free[index];
            if (!ReferenceEquals(item.Backend, backend) || item.Thread != thread
                || item.Buffer is not IPoolableGpuBuffer native)
                continue;
            _free[index] = _free[_free.Count - 1];
            _free.RemoveAt(_free.Count - 1);
            _owner.Slots.Remove(item.Key);
            long bytes = item.Buffer.SizeInBytes;
            StepScratchPool.Untrack(item.Buffer);
            RetainedBytes -= bytes;
            // Release, not Dispose: Dispose would park these undersized cells in the host-side free cache.
            // This is the same native return contract that each backend's bounded buffer pool uses on eviction.
            // CUDA preserves capture pins and enqueues cuMemFreeAsync on its owning compute stream; other backends
            // retain their existing queued-work/native-release semantics. No global pool drain or trim is performed.
            native.Release();
            returned += bytes;
            cells++;
        }
        NativeReturnedForInferenceBytes += returned;
        NativeReturnedForInferenceCells += cells;
        if (cells > 0 && Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_DIAG") == "1")
            Console.Error.WriteLine($"[WORKSPACE-INFERENCE-RETURN] requested_bytes={requestedBytes} largest_cell_bytes={largest} cells={cells} native_bytes={returned} broker_remaining_bytes={RetainedBytes}");
        return returned;
    }
}
