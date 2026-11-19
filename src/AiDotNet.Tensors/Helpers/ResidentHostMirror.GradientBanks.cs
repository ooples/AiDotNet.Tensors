using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

internal static partial class ResidentHostMirror
{
    // The full bank is preserved, including padding and every tenant's currently visible prefix. Every vector
    // wrapper must appear in the mirror group; an unregistered wrapper, separate native copy or partial native
    // view declines the operation. Additional tensor views over a registered vector keep the same exact CPU data.
    internal static bool TryCollectGradientBankBindings<T>(T[] array, IGpuBuffer root, IDirectGpuBackend backend,
        ResultBufferOwner owner, List<Tensor<T>> bindings, HashSet<object> vectors)
    {
        bindings.Clear();
        vectors.Clear();
        if (owner.State.Escaped || root.Size != array.Length || !ReferenceEquals(GpuBufferViewRoot.Of(root), root)
            || !_groups.TryGetValue(array, out var value) || value is not Group<T> group) return false;
        lock (group.Members)
        {
            foreach (var weak in group.Members)
            {
                if (!weak.TryGetTarget(out var tensor) || tensor.IsDisposed
                    || !ReferenceEquals(tensor.GraphBackingArrayIdentity, array)) continue;
                vectors.Add(tensor.GraphVectorIdentity);
                if (tensor._gpuBuffer is not { } buffer) continue;
                if (!ReferenceEquals(buffer, root) || !ReferenceEquals(tensor._gpuBackend, backend)
                    || !tensor.IsContiguous || tensor._storageOffset != 0
                    || tensor._ownsGpuBuffer || tensor._storage.HasGpuBufferOwner || tensor.HasPendingCopyOnWritePeer
                    || !tensor._storage.TryGetBackingArrayIdentity(out var backing, out int offset)
                    || offset != 0 || !ReferenceEquals(backing, array)
                    || !ReferenceEquals(((Vector<T>)tensor.GraphVectorIdentity).ResultOwner, owner)) return false;
                bindings.Add(tensor);
            }
            return bindings.Count != 0 && vectors.Count == System.Threading.Volatile.Read(ref owner.State.Wrappers);
        }
    }
}
