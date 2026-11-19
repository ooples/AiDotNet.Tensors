using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

internal sealed partial class StepScratchPool
{
    internal bool ShareBackwardScratch { get; set; }
    internal long SharedBackwardScratchBytes { get; private set; }
    private readonly HashSet<IGpuBuffer> _backwardProtected = new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
    private readonly HashSet<IGpuBuffer> _backwardReturned = new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);

    internal long ReturnIdleBackwardScratch(IDirectGpuBackend backend, CompiledTransientWorkspaceBroker broker)
    {
        WithdrawLent();
        if (HasOutstandingWorkspaceLeases) return 0;
        _backwardProtected.Clear();
        _backwardReturned.Clear();
        foreach (var buffer in Slots.Values) _backwardProtected.Add(GpuBufferViewRoot.Of(buffer));
        foreach (var buffer in Transient.Values) _backwardProtected.Add(GpuBufferViewRoot.Of(buffer));
        long returned = 0;
        for (int index = _free.Count - 1; index >= 0; index--)
        {
            var buffer = _free[index];
            if (buffer.Handle == System.IntPtr.Zero || !ReferenceEquals(GpuBufferViewRoot.Of(buffer), buffer)
                || _backwardProtected.Contains(buffer)) continue;
            _free[index] = _free[_free.Count - 1];
            _free.RemoveAt(_free.Count - 1);
            if (!_backwardReturned.Add(buffer)) continue;
            Keys.Remove(buffer);
            _leasedThisRun.Remove(buffer);
            broker.Return(backend, buffer);
            returned += buffer.SizeInBytes;
        }
        SharedBackwardScratchBytes += returned;
        return returned;
    }
}
