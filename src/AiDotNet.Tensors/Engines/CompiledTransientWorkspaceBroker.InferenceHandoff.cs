using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

internal sealed partial class CompiledTransientWorkspaceBroker
{
    internal long CacheReturnedForInferenceBytes { get; private set; }
    internal long CacheReturnedForInferenceCells { get; private set; }

    // The free list contains only fully detached backing from completed uncaptured steps. A public prediction
    // needs the engine allocator to reuse it too; keeping it in this training-only broker makes a second cache.
    internal long ReturnIdleCellsToBackendCache(IDirectGpuBackend backend)
    {
        int thread = Environment.CurrentManagedThreadId;
        long returned = 0;
        for (int index = _free.Count - 1; index >= 0; index--)
        {
            var item = _free[index];
            if (!ReferenceEquals(item.Backend, backend) || item.Thread != thread) continue;
            _free[index] = _free[_free.Count - 1];
            _free.RemoveAt(_free.Count - 1);
            _owner.Slots.Remove(item.Key);
            long bytes = item.Buffer.SizeInBytes;
            RetainedBytes -= bytes;
            StepScratchPool.Untrack(item.Buffer);
            if (backend is IQueuedBufferRelease queued) queued.ReleaseAfterQueuedWork(item.Buffer);
            else item.Buffer.Dispose();
            returned += bytes;
            CacheReturnedForInferenceBytes += bytes;
            CacheReturnedForInferenceCells++;
        }
        return returned;
    }
}
