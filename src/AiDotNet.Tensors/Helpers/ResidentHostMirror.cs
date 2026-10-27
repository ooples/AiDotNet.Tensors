using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Keeps the host copy of a tensor whose authoritative value lives in a device buffer honest: after a device-side
/// write, the next host read downloads the device data INTO the tensor's existing host slice. This is PyTorch's
/// <c>tensor.cpu()</c> contract for a parameter that an on-device optimizer keeps updating.
/// </summary>
/// <remarks>
/// <para>
/// Host reads are keyed by the storage's backing array, and AiDotNet parameters are commonly views into ONE shared
/// parameter buffer. A per-tensor registration therefore could not work: registration is first-write-wins per key,
/// so a second view either lost its download or displaced the first view's, and the old per-tensor callback copied
/// the view's device data to index 0 of the shared array — onto another parameter. Members are grouped per backing
/// array instead: ONE registration refreshes every resident view on that array, each into its own slice, and the
/// copy goes into the existing array (never a replacement), because other views and owners alias it.
/// </para>
/// </remarks>
internal static class ResidentHostMirror
{
    private sealed class Group<T>
    {
        public readonly List<WeakReference<Tensor<T>>> Members = new();
        public readonly Action<object> Callback;
        public Group() => Callback = Download;

        private void Download(object key)
        {
            var array = (T[])key;
            WeakReference<Tensor<T>>[] members;
            lock (Members) members = Members.ToArray();
            foreach (var weak in members)
            {
                if (!weak.TryGetTarget(out var t) || !HasContiguousResidentBuffer(t)) continue;
                if (!ReferenceEquals(t.GetBackingArrayForCacheLookupUnsafe(), array)) continue; // re-homed since
                // A member bound to a pooled scratch buffer that was since released (ClearActionScratchPool) has
                // nothing to contribute; its host slice is already the value. One such member must not abort the
                // host read for every other view on the array.
                if (t._gpuBuffer!.Handle == IntPtr.Zero) continue;
                float[] floats;
                try { floats = t._gpuBackend!.DownloadBuffer(t._gpuBuffer!); }
                catch (ObjectDisposedException) { continue; }
                var values = DirectGpuEngine.FromFloatArray<T>(floats);
                Array.Copy(values, 0, array, t._storageOffset, Math.Min(t.Length, values.Length));
            }
        }
    }

    private static readonly ConditionalWeakTable<object, object> _groups = new();

    private static bool HasContiguousResidentBuffer<T>(Tensor<T> t) =>
        t._gpuBuffer is not null && t._gpuBackend is not null && t.IsContiguous;

    /// <summary>
    /// Enrols a tensor whose device buffer holds (or is about to hold) its authoritative value, so a download of
    /// its backing array refreshes its slice. Returns the backing-array key, or null when the tensor cannot be
    /// mirrored (non-contiguous view, or no managed backing array) — callers keep their previous behaviour then.
    /// </summary>
    internal static object? Attach<T>(Tensor<T> tensor)
    {
        if (!HasContiguousResidentBuffer(tensor)) return null;
        var array = tensor.GetBackingArrayForCacheLookupUnsafe();
        if (array is null) return null;
        var group = (Group<T>)_groups.GetValue(array, static _ => new Group<T>());
        lock (group.Members)
        {
            group.Members.RemoveAll(w => !w.TryGetTarget(out var m) || ReferenceEquals(m, tensor));
            group.Members.Add(new WeakReference<Tensor<T>>(tensor));
        }
        tensor._gpuMaterializerKey = array;
        tensor._gpuMaterializerCallback = group.Callback;
        return array;
    }

    /// <summary>
    /// The resident member on <paramref name="array"/> whose device buffer holds storage range
    /// [<paramref name="offset"/>, <paramref name="offset"/> + <paramref name="length"/>), and where that range starts
    /// in its buffer. A slice of a resident step tensor shares this array but not the member's buffer field, so this
    /// is the only way to reach its device data without downloading it.
    /// </summary>
    internal static bool TryFindCovering<T>(T[] array, int offset, int length, object backend,
        out IGpuBuffer? buffer, out int bufferOffset)
    {
        buffer = null; bufferOffset = 0;
        if (!_groups.TryGetValue(array, out var g) || g is not Group<T> group) return false;
        WeakReference<Tensor<T>>[] members;
        lock (group.Members) members = group.Members.ToArray();
        foreach (var weak in members)
        {
            if (!weak.TryGetTarget(out var t) || !HasContiguousResidentBuffer(t)) continue;
            if (!ReferenceEquals(t._gpuBackend, backend) || !ReferenceEquals(t.GetBackingArrayForCacheLookupUnsafe(), array)) continue;
            var b = t._gpuBuffer!;
            if (b.Handle == IntPtr.Zero) continue;
            int start = offset - t._storageOffset;
            if (start < 0 || start + length > t.Length || b.Size < (long)start + length) continue;
            buffer = b; bufferOffset = start;
            return true;
        }
        return false;
    }

    /// <summary>
    /// Takes <paramref name="tensor"/> out of its backing array's group, so a download of that array no longer copies
    /// its device data back. For a tensor whose device value is being discarded: a retiring plan's intermediate
    /// gradients share pooled backing arrays with the parameter gradients it does land.
    /// </summary>
    internal static void Detach<T>(Tensor<T> tensor)
    {
        if (tensor.GetBackingArrayForCacheLookupUnsafe() is not { } array) return;
        if (!_groups.TryGetValue(array, out var g) || g is not Group<T> group) return;
        lock (group.Members)
            group.Members.RemoveAll(w => !w.TryGetTarget(out var m) || ReferenceEquals(m, tensor));
    }

    /// <summary>
    /// Arms the download for <paramref name="tensor"/>'s backing array after a device-side write: the next host read
    /// of ANY view on that array refreshes every resident view on it.
    /// </summary>
    internal static bool ArmDownload<T>(Tensor<T> tensor)
    {
        if (Attach(tensor) is not { } key) return false;
        HostSync.Register(key, tensor._gpuMaterializerCallback!);
        return true;
    }
}
