using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DevicePrimitives;
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
/// parameter buffer. A per-tensor registration would displace another view's pending download, and copying
/// the view's device data to index 0 would overwrite another parameter. Members are grouped per shared
/// storage instead: ONE registration refreshes every resident view, each into its own slice, and the
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
            var vector = key as VectorBase<T>;
            var array = vector is null ? (T[])key : vector.GetBackingArrayUnsafe() ?? new T[vector.Length];
            WeakReference<Tensor<T>>[] members;
            lock (Members) members = Members.ToArray();
            bool downloaded = false;
            foreach (var weak in members)
            {
                if (!weak.TryGetTarget(out var t) || !HasContiguousResidentBuffer(t)) continue;
                if (!ReferenceEquals(t.GetBackingArrayForCacheLookupUnsafe(), array)
                    && (vector is null || !ReferenceEquals(t.DataVector, vector))) continue; // re-homed since
                // A member bound to a pooled scratch buffer that was since released (ClearActionScratchPool) has
                // nothing to contribute; its host slice is already the value. One such member must not abort the
                // host read for every other view on the array.
                if (t._gpuBuffer is not { } gpuBuffer || gpuBuffer.Handle == IntPtr.Zero || t._gpuBackend is not { } gpuBackend) continue;
                if (array is float[] floatArray && t._storageOffset == 0 && floatArray.Length >= gpuBuffer.Size && t.Length >= gpuBuffer.Size)
                {
                    try { gpuBackend.DownloadBuffer(gpuBuffer, floatArray); }
                    catch (ObjectDisposedException) { continue; }
                    downloaded = true;
                    continue;
                }
                float[] floats;
                try { floats = gpuBackend.DownloadBuffer(gpuBuffer); }
                catch (ObjectDisposedException) { continue; }
                var values = DirectGpuEngine.FromFloatArray<T>(floats);
                Array.Copy(values, 0, array, t._storageOffset, Math.Min(t.Length, values.Length));
                downloaded = true;
            }
            // A pending device write has no proven current host copy. Do not turn
            // an empty weak-reader list into a successfully materialized zero array.
            if (!downloaded)
                throw new InvalidOperationException("The resident host mirror has no live reader for its pending device value.");
            if (vector is not null)
            {
                // Publish only after every slice downloaded successfully. Subsequent array-backed binds
                // must keep this same group, including siblings that have not been rebound yet.
                _groups.GetValue(array, _ => this);
                vector.MaterializeBacking(array);
            }
        }
    }

    private static readonly ConditionalWeakTable<object, object> _groups = new();

    private static bool HasContiguousResidentBuffer<T>(Tensor<T> t) =>
        t._gpuBuffer is not null && t._gpuBackend is not null && t.IsContiguous;

    /// <summary>
    /// Enrols a tensor whose device buffer holds (or is about to hold) its authoritative value, so a download of
    /// its shared storage refreshes its slice. Until a host array exists, the shared vector is the key.
    /// Returns null for a non-contiguous view or a tensor without a resident buffer.
    /// </summary>
    internal static object? Attach<T>(Tensor<T> tensor)
    {
        if (!HasContiguousResidentBuffer(tensor)) return null;
        var array = tensor.GetBackingArrayForCacheLookupUnsafe();
        object key = (object?)array ?? tensor.DataVector;
        var group = (Group<T>)_groups.GetValue(key, static _ => new Group<T>());
        lock (group.Members)
        {
            WeakReference<Tensor<T>>? existing = null;
            for (int i = group.Members.Count - 1; i >= 0; i--)
            {
                if (!group.Members[i].TryGetTarget(out var member))
                    group.Members.RemoveAt(i);
                else if (ReferenceEquals(member, tensor))
                {
                    existing ??= group.Members[i];
                    group.Members.RemoveAt(i);
                }
            }
            // Latest device writer wins overlapping ranges, without allocating on repeated binds.
            group.Members.Add(existing ?? new WeakReference<Tensor<T>>(tensor));
        }
        tensor._gpuMaterializerKey = key;
        tensor._gpuMaterializerCallback = group.Callback;
        if (array is not null) HostShadowMemory.Track(array);
        return key;
    }

    /// <summary>
    /// Keeps an existing mirror reader reachable through an equivalent metadata view.
    /// A partial view that merely inherited its parent's whole buffer is not enrolled.
    /// </summary>
    internal static void AttachShapeAlias<T>(Tensor<T> source, Tensor<T> view)
    {
        if (!source.IsContiguous || !view.IsContiguous || source._storageOffset != view._storageOffset
            || source.Length != view.Length || !ReferenceEquals(source.StorageIdentity, view.StorageIdentity)
            || !ReferenceEquals(source._gpuBuffer, view._gpuBuffer)
            || source._gpuMaterializerCallback?.Target is not Group<T> group)
            return;
        lock (group.Members)
        {
            int sourceIndex = -1;
            for (int i = 0; i < group.Members.Count; i++)
                if (group.Members[i].TryGetTarget(out var member))
                {
                    if (ReferenceEquals(member, view)) return;
                    if (ReferenceEquals(member, source)) sourceIndex = i;
                }
            if (sourceIndex < 0) return;
            // Alias creation is not a device write. Preserve the writer ordering:
            // a newer overlapping member must still override this source's range.
            group.Members.Insert(sourceIndex + 1, new WeakReference<Tensor<T>>(view));
        }
    }

    /// <summary>
    /// The resident member on <paramref name="array"/> whose device buffer holds storage range
    /// [<paramref name="offset"/>, <paramref name="offset"/> + <paramref name="length"/>), and where that range starts
    /// in its buffer. A slice of a resident step tensor shares this array but not the member's buffer field, so this
    /// is the only way to reach its device data without downloading it. Only a member over the SAME storage as
    /// <paramref name="view"/> counts: host arrays are pooled, so a member that merely shares the array may be a
    /// tensor of another plan or thread holding unrelated data, and copying from its buffer read that data (or a
    /// buffer its owner was releasing: a concurrent GPU Autoformer faulted with CUDA 700).
    /// </summary>
    internal static bool TryFindCovering<T>(Tensor<T> view, T[] array, int offset, int length, object backend,
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
            if (!t.SharesStorageWith(view)) continue;
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
        object key = (object?)tensor.GetBackingArrayForCacheLookupUnsafe() ?? tensor.DataVector;
        if (!_groups.TryGetValue(key, out var g) || g is not Group<T> group) return;
        lock (group.Members)
            group.Members.RemoveAll(w => !w.TryGetTarget(out var m) || ReferenceEquals(m, tensor));
    }

    /// <summary>
    /// Arms the download for <paramref name="tensor"/>'s backing array after a device-side write: the next host read
    /// of ANY view on that array refreshes every resident view on it.
    /// </summary>
    internal static bool ArmDownload<T>(Tensor<T> tensor)
    {
        if (Attach(tensor) is not { } key || tensor._gpuMaterializerCallback is not { } callback) return false;
        HostSync.Register(key, callback);
        return true;
    }
}
