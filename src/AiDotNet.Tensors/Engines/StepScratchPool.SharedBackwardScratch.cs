using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

internal sealed partial class StepScratchPool
{
    // Enabled only inside an explicitly uncaptured compiled body. The broker also enforces backend/thread affinity.
    internal bool ShareBackwardScratch { get; set; }
    internal long SharedBackwardScratchBytes { get; private set; }

    internal long ReturnIdleBackwardScratch(IDirectGpuBackend backend, CompiledTransientWorkspaceBroker broker)
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
            broker.Return(backend, buffer);
            returned += buffer.SizeInBytes;
        }
        SharedBackwardScratchBytes += returned;
        return returned;
    }
}
