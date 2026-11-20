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
}
