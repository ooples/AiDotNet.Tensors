// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
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
/// </summary>
public partial class DirectGpuTensorEngine
{
    private sealed class Bf16DeviceWeight
    {
        internal IGpuBuffer Bits = null!;
        internal IDirectGpuBackend Backend = null!;
        internal long PoolHandle;
        internal int Length;
        internal bool Released;
    }

    // Replaced (not cleared: ConditionalWeakTable has no Clear on .NET Framework) when the engine is disposed.
    private ConditionalWeakTable<object, Bf16DeviceWeight> _bf16Weights = new();
    // Every device copy this engine made, so Dispose can free the ones whose weights are still alive. Weak: a copy
    // lives as long as its weight (the table's value), and a dead weight's copy is freed by the buffer's finalizer.
    private readonly List<WeakReference<Bf16DeviceWeight>> _bf16WeightCopies = new();
    private int _bf16WeightCopiesPruneAt = 64;
    private readonly object _bf16WeightsLock = new();
    private long _bf16WeightBytesUploaded;

    /// <summary>The bf16 device copy this engine holds for <paramref name="weight"/>, or null (diagnostics, tests).</summary>
    internal IGpuBuffer? Bf16DeviceCopyOf(object weight)
    {
        lock (_bf16WeightsLock)
            return _bf16Weights.TryGetValue(weight, out var entry) ? entry.Bits : null;
    }

    /// <summary>
    /// Frees every bf16 device copy whose weight is still alive. Engine teardown only: the engine is closed to ops, so
    /// no widen can read a copy after it is freed. Without this a copy outlived the engine until its weight was
    /// collected, holding device memory on a backend the engine shares with others.
    /// </summary>
    private void ReleaseBf16DeviceWeights()
    {
        lock (_bf16WeightsLock)
        {
            foreach (var weak in _bf16WeightCopies)
                if (weak.TryGetTarget(out var entry) && !entry.Released) { entry.Released = true; entry.Bits.Dispose(); }
            _bf16WeightCopies.Clear();
            _bf16Weights = new ConditionalWeakTable<object, Bf16DeviceWeight>();
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
        if (typeof(T) != typeof(float) || t.Lifetime != WeightLifetime.Streaming || t.StreamingPoolHandle < 0
            || t.StreamingStoreEncoding != StreamingEncoding.Bf16 || !t.IsContiguous || t._storageOffset != 0)
            return false;
        if (backend is not IBf16WidenBackend widen || !widen.CanWidenBf16) return false;
        // The first use uploads (a synchronous copy), which a stream capture cannot record.
        if (backend is IGpuGraphCapture capture && capture.IsStreamCapturing()) return false;
        Bf16DeviceWeight? entry;
        lock (_bf16WeightsLock)
        {
            if (!_bf16Weights.TryGetValue(t, out entry) || entry.PoolHandle != t.StreamingPoolHandle
                || !ReferenceEquals(entry.Backend, backend) || entry.Bits.Handle == IntPtr.Zero)
            {
                var bytes = WeightRegistry.TryReadBf16Store(t);
                if (bytes is null) return false;
                // Uploaded as the raw bit pattern: the copy does not interpret the packed 32-bit words.
                var packed = new float[(bytes.Length + 3) / 4];
                Buffer.BlockCopy(bytes, 0, packed, 0, bytes.Length);
                if (entry is not null)
                {
                    // The weight was re-registered (a new pool handle) or the backend changed: the old copy is dead.
                    // Disposed under the lock, where every widen from it is launched, so none can read it after.
                    _bf16Weights.Remove(t);
                    entry.Released = true;
                    entry.Bits.Dispose();
                }
                entry = new Bf16DeviceWeight
                {
                    Bits = backend.AllocateBuffer(packed),
                    Backend = backend,
                    PoolHandle = t.StreamingPoolHandle,
                    Length = t.Length,
                };
                _bf16Weights.Add(t, entry);
                _bf16WeightCopies.Add(new WeakReference<Bf16DeviceWeight>(entry));
                if (_bf16WeightCopies.Count >= _bf16WeightCopiesPruneAt)
                {
                    _bf16WeightCopies.RemoveAll(w => !w.TryGetTarget(out var e) || e.Released);
                    _bf16WeightCopiesPruneAt = Math.Max(64, 2 * _bf16WeightCopies.Count);
                }
                Interlocked.Add(ref _bf16WeightBytesUploaded, 2L * t.Length);
            }
            var fp32 = backend.AllocateBuffer(t.Length);
            widen.WidenBf16ToFp32(entry.Bits, fp32, t.Length);
            buffer = new OwnedBuffer(fp32, ownsBuffer: true);
        }
        return true;
    }
}
