// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Gpu;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Weights stored in the streaming pool's bf16 encoding stay bf16 on the device. Without this, every op read a bf16
/// streaming weight through its fp32 host view: materialized on the host, uploaded at 4 bytes per element and cached
/// keyed on an array the pool replaces when it sheds the weight, so a model in bf16 needed its full fp32 size on the
/// device (a 6.9B-parameter mixture-of-experts model: 27.6 GB against a 24 GB card) and re-uploaded on every re-materialization. Here the weight's
/// bf16 bytes are uploaded once (2 bytes per element) and each op that reads it gets an fp32 temporary widened on the
/// device. Widening is exact, so the op sees exactly the values the CPU engine sees for the same weight.
/// <para>
/// <see cref="Bf16WeightDeviceBudgetBytes"/> bounds the device bytes these copies hold. Unbounded, a model whose bf16
/// weights exceed the card cannot run at all (the same model: 13.85 GB of bf16 weights against a 12 GB card). Bounded,
/// the least recently used copies are released to admit a new one, and an evicted weight is uploaded again from the
/// host store on its next use. Values are unchanged; only the upload traffic grows.
/// </para>
/// </summary>
public partial class DirectGpuTensorEngine
{
    private sealed class Bf16DeviceWeight
    {
        internal IGpuBuffer Bits = null!;
        internal IDirectGpuBackend Backend = null!;
        internal long PoolHandle;
        internal int Length;
        internal long Bytes;
        internal bool Released;
        internal WeakReference<object> Key = null!;
        internal LinkedListNode<Bf16DeviceWeight>? Node;
    }

    // Replaced (not cleared: ConditionalWeakTable has no Clear on .NET Framework) when the engine is disposed.
    private ConditionalWeakTable<object, Bf16DeviceWeight> _bf16Weights = new();
    // Every device copy this engine made, most recently used first: the eviction order under a budget, and the list
    // Dispose frees. The list holds its copies strongly, so a copy whose weight has been collected is released by the
    // next upload (ReleaseCollectedBf16Copies), before any live copy is evicted, or by Dispose; until then its bytes
    // stay counted in Bf16WeightDeviceBytes.
    private readonly LinkedList<Bf16DeviceWeight> _bf16Lru = new();
    private readonly object _bf16WeightsLock = new();
    private long _bf16WeightBytesUploaded, _bf16WeightDeviceBytes, _bf16WeightEvictions, _bf16WeightDeviceBudget;

    /// <summary>The bf16 device copy this engine holds for <paramref name="weight"/>, or null (diagnostics, tests).</summary>
    internal IGpuBuffer? Bf16DeviceCopyOf(object weight)
    {
        lock (_bf16WeightsLock)
            return _bf16Weights.TryGetValue(weight, out var entry) ? entry.Bits : null;
    }

    /// <summary>
    /// Device bytes the bf16 copies of streamed weights may hold; 0 (the default) means unbounded. Over the budget, the
    /// least recently used copies are released before a new one is uploaded. A single weight larger than the budget is
    /// still uploaded, after every other copy is released. A lowered budget takes effect at the next upload.
    /// </summary>
    public long Bf16WeightDeviceBudgetBytes
    {
        get => Interlocked.Read(ref _bf16WeightDeviceBudget);
        set
        {
            if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), "The budget must be 0 (unbounded) or positive.");
            Interlocked.Exchange(ref _bf16WeightDeviceBudget, value);
        }
    }

    /// <summary>Device bytes currently held by bf16 copies of streamed weights.</summary>
    public long Bf16WeightDeviceBytes => Interlocked.Read(ref _bf16WeightDeviceBytes);

    /// <summary>bf16 weight copies released to stay within <see cref="Bf16WeightDeviceBudgetBytes"/>.</summary>
    public long Bf16WeightEvictions => Interlocked.Read(ref _bf16WeightEvictions);

    // Caller holds _bf16WeightsLock. Every widen from a copy is launched under the lock, so after this none can be
    // launched; one already queued may still read it, so the buffer is released in queue order.
    private void ReleaseBf16Copy(Bf16DeviceWeight entry, bool queued)
    {
        if (entry.Node is not null) { _bf16Lru.Remove(entry.Node); entry.Node = null; }
        if (entry.Key.TryGetTarget(out var key) && _bf16Weights.TryGetValue(key, out var current) && ReferenceEquals(current, entry))
            _bf16Weights.Remove(key);
        if (entry.Released) return;
        entry.Released = true;
        if (queued) DisposeBufferAfterQueuedUse(entry.Backend, entry.Bits);
        else entry.Bits.Dispose();
        Interlocked.Add(ref _bf16WeightDeviceBytes, -entry.Bytes);
    }

    // Caller holds _bf16WeightsLock. Releases the copies whose weight has been collected. Linear in the copies held; it
    // runs once per upload, and an upload copies the whole weight to the device.
    private void ReleaseCollectedBf16Copies()
    {
        for (var node = _bf16Lru.First; node is not null;)
        {
            var next = node.Next;
            if (!node.Value.Key.TryGetTarget(out _)) ReleaseBf16Copy(node.Value, queued: true);
            node = next;
        }
    }

    /// <summary>
    /// Frees every bf16 device copy. Engine teardown only: the engine is closed to ops, so no widen can read a copy
    /// after it is freed. Without this a copy outlived the engine until its weight was collected, holding device memory
    /// on a backend the engine shares with others.
    /// </summary>
    private void ReleaseBf16DeviceWeights()
    {
        lock (_bf16WeightsLock)
        {
            while (_bf16Lru.First is { } node) ReleaseBf16Copy(node.Value, queued: false);
            _bf16Weights = new ConditionalWeakTable<object, Bf16DeviceWeight>();
        }
    }

    // The weight's bf16 store uploaded as its raw bit pattern (the copy does not interpret the packed 32-bit words), or
    // null when the store is gone. An even element count streams straight from the pool's resident bytes through the
    // backend's pooled page-locked staging: no host array per upload. Each upload used to make two of the weight's size
    // (the store snapshot and a packed float[] of it), so under a device budget every eviction and re-upload left two
    // fresh large-object arrays behind, and host memory grew about two bytes per byte uploaded faster than the collector
    // reclaimed it. An odd count (a last half word), or a backend without async uploads, takes one packed array.
    private static IGpuBuffer? UploadBf16Store<T>(IDirectGpuBackend backend, Tensor<T> t, int words)
    {
        var bits = backend.AllocateBuffer(words);
        try
        {
            bool read;
            if (backend is IAsyncGpuBackend async && t.Length % 2 == 0)
            {
                read = WeightRegistry.TryReadBf16Store(t, bytes =>
                    async.UploadBufferAsync(MemoryMarshal.Cast<byte, float>(bytes), bits, async.DefaultStream));
            }
            else
            {
                var packed = new float[words];
                read = WeightRegistry.TryReadBf16Store(t, bytes => bytes.CopyTo(MemoryMarshal.AsBytes(packed.AsSpan())));
                if (read) backend.UploadBufferInPlace(packed, bits);
            }
            if (!read) { bits.Dispose(); return null; }
            return bits;
        }
        catch
        {
            bits.Dispose();
            throw;
        }
    }

    /// <summary>Bytes of bf16 weight data uploaded to the device by this engine (2 per element; diagnostics, tests).</summary>
    internal long Bf16WeightBytesUploaded => Interlocked.Read(ref _bf16WeightBytesUploaded);

    /// <summary>
    /// An fp32 temporary holding <paramref name="t"/>'s values, widened on the device from its bf16 device copy, when
    /// <paramref name="t"/> is a contiguous float weight whose streaming store is bf16; false for every other tensor.
    /// </summary>
    private bool TryBf16StreamingWeight<T>(IDirectGpuBackend backend, Tensor<T> t, out OwnedBuffer buffer)
    {
        buffer = default;
        if (typeof(T) != typeof(float) || t.Lifetime != WeightLifetime.Streaming || t.StreamingPoolHandle < 0)
            return false;
        if (t.StreamingStoreEncoding != StreamingEncoding.Bf16)
        {
            // The store left bf16 (promoted to a writable encoding for training): a copy made before is dead, and
            // nothing would read it again, so its device memory goes now rather than when the weight is collected.
            lock (_bf16WeightsLock)
                if (_bf16Weights.TryGetValue(t, out var stale)) ReleaseBf16Copy(stale, queued: true);
            return false;
        }
        if (!t.IsContiguous || t._storageOffset != 0) return false;
        if (backend is not IBf16WidenBackend widen || !widen.CanWidenBf16) return false;
        // The first use uploads (a synchronous copy), which a stream capture cannot record.
        if (backend is IGpuGraphCapture capture && capture.IsStreamCapturing()) return false;
        Bf16DeviceWeight? entry;
        lock (_bf16WeightsLock)
        {
            if (!_bf16Weights.TryGetValue(t, out entry) || entry.PoolHandle != t.StreamingPoolHandle
                || !ReferenceEquals(entry.Backend, backend) || entry.Bits.Handle == IntPtr.Zero)
            {
                if (!WeightRegistry.HasBf16Store(t)) return false;
                // The weight was re-registered (a new pool handle) or the backend changed: the old copy is dead.
                if (entry is not null) ReleaseBf16Copy(entry, queued: true);
                // Copies of weights that have been collected go first: neither the budget nor an unbounded engine should
                // keep a dropped model's weights on the device while it uploads another's.
                ReleaseCollectedBf16Copies();
                long byteLength = 2L * t.Length;
                int words = checked((int)((byteLength + 3) / 4));
                long size = 4L * words;
                long budget = Bf16WeightDeviceBudgetBytes;
                if (budget > 0)
                    while (_bf16Lru.Last is { } oldest && Bf16WeightDeviceBytes + size > budget)
                    {
                        ReleaseBf16Copy(oldest.Value, queued: true);
                        Interlocked.Increment(ref _bf16WeightEvictions);
                    }
                var bits = UploadBf16Store(backend, t, words);
                if (bits is null) return false;
                entry = new Bf16DeviceWeight
                {
                    Bits = bits,
                    Backend = backend,
                    PoolHandle = t.StreamingPoolHandle,
                    Length = t.Length,
                    Bytes = size,
                    Key = new WeakReference<object>(t),
                };
                entry.Node = _bf16Lru.AddFirst(entry);
                _bf16Weights.Add(t, entry);
                Interlocked.Add(ref _bf16WeightDeviceBytes, size);
                Interlocked.Add(ref _bf16WeightBytesUploaded, 2L * t.Length);
            }
            else if (entry.Node is not null && !ReferenceEquals(_bf16Lru.First, entry.Node))
            {
                _bf16Lru.Remove(entry.Node);
                _bf16Lru.AddFirst(entry.Node);
            }
            var fp32 = backend.AllocateBuffer(t.Length);
            widen.WidenBf16ToFp32(entry.Bits, fp32, t.Length);
            buffer = new OwnedBuffer(fp32, ownsBuffer: true);
        }
        return true;
    }
}
