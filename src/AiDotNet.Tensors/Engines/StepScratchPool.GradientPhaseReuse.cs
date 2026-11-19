using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

internal sealed partial class StepScratchPool
{
    // Only a completed action's free backing may change owner. A lent forward slot loses its slot too:
    // the next forward must rent again, rather than overwrite the public gradient's new storage.
    internal bool TryTakeGradientBacking(int length, out IGpuBuffer buffer, ISet<IGpuBuffer>? protectedRoots = null)
    {
        for (int i = 0; i < _free.Count; i++)
        {
            var candidate = _free[i];
            if (candidate.Handle == IntPtr.Zero || candidate.Size != length
                || protectedRoots?.Contains(candidate) == true
                || !ReferenceEquals(GpuBufferViewRoot.Of(candidate), candidate)) continue;
            bool lent = _lent.TryGetValue(candidate, out long forwardKey);
            bool otherOwner = false;
            foreach (var slot in Slots)
                if (ReferenceEquals(slot.Value, candidate) && (!lent || slot.Key != forwardKey)) otherOwner = true;
            foreach (var transient in Transient.Values)
                if (ReferenceEquals(transient, candidate)) otherOwner = true;
            if (otherOwner || _spanning.ContainsKey(candidate)) continue;
            if (lent)
            {
                _lent.Remove(candidate);
                Slots.Remove(forwardKey);
            }
            Keys.Remove(candidate);
            _leasedThisRun.Remove(candidate);
            _free[i] = _free[_free.Count - 1];
            _free.RemoveAt(_free.Count - 1);
            buffer = candidate;
            return true;
        }
        buffer = null!;
        return false;
    }
}
