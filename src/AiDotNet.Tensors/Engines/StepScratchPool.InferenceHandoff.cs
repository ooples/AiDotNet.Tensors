using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

internal sealed partial class StepScratchPool
{
    internal long CacheReturnedForInferenceBytes { get; private set; }

    // Called only for a completed, permanently uncaptured plan under CompiledStepLock. EndLeases classified
    // these buffers as unescaped; withdraw borrowed forward slots before giving idle backing to the allocator.
    internal long ReturnIdleLeasesToBackendCache(IDirectGpuBackend backend)
    {
        WithdrawLent();
        if (HasOutstandingWorkspaceLeases) return 0;
        var protectedBuffers = new HashSet<IGpuBuffer>(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
        foreach (var buffer in Slots.Values) protectedBuffers.Add(buffer);
        foreach (var buffer in Transient.Values) protectedBuffers.Add(buffer);
        var returnedBuffers = new HashSet<IGpuBuffer>(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
        long returned = 0;
        for (int index = _free.Count - 1; index >= 0; index--)
        {
            var buffer = _free[index];
            if (protectedBuffers.Contains(buffer)) continue;
            _free[index] = _free[_free.Count - 1];
            _free.RemoveAt(_free.Count - 1);
            if (!returnedBuffers.Add(buffer)) continue;
            Keys.Remove(buffer);
            _leasedThisRun.Remove(buffer);
            long bytes = buffer.SizeInBytes;
            Untrack(buffer);
            if (backend is IQueuedBufferRelease queued) queued.ReleaseAfterQueuedWork(buffer);
            else buffer.Dispose();
            returned += bytes;
        }
        CacheReturnedForInferenceBytes += returned;
        return returned;
    }
}
