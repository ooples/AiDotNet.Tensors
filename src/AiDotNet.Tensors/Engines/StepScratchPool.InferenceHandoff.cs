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
        long returned = ReturnFreeLeases(backend, unusedOnly: false);
        CacheReturnedForInferenceBytes += returned;
        return returned;
    }

    private long ReturnFreeLeases(IDirectGpuBackend backend, bool unusedOnly)
    {
        if (HasOutstandingWorkspaceLeases) return 0;
        HashSet<IGpuBuffer>? protectedBuffers = null;
        HashSet<IGpuBuffer>? returnedBuffers = null;
        long returned = 0;
        for (int index = _free.Count - 1; index >= 0; index--)
        {
            var buffer = _free[index];
            if (unusedOnly && _leasedThisRun.Contains(buffer)) continue;
            // A settled training plan reuses its working set. Build the ownership sets only
            // after finding an actual return candidate, never on its ordinary steady steps.
            if (protectedBuffers is null)
            {
                protectedBuffers = new HashSet<IGpuBuffer>(Slots.Values, Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
                protectedBuffers.UnionWith(Transient.Values);
                returnedBuffers = new HashSet<IGpuBuffer>(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
            }
            if (protectedBuffers.Contains(buffer)) continue;
            _free[index] = _free[_free.Count - 1];
            _free.RemoveAt(_free.Count - 1);
            if (!returnedBuffers!.Add(buffer)) continue;
            Keys.Remove(buffer);
            _leasedThisRun.Remove(buffer);
            long bytes = buffer.SizeInBytes;
            if (ReleaseOwnership(buffer)) continue;   // another plan still co-owns it (SharedStepScratch)
            if (backend is IQueuedBufferRelease queued) queued.ReleaseAfterQueuedWork(buffer);
            else buffer.Dispose();
            returned += bytes;
        }
        return returned;
    }
}
