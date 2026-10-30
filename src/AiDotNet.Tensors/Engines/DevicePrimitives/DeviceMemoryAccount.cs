// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DevicePrimitives;

/// <summary>
/// Device-memory counters for one GPU device, fed by every backend's owning buffer type. The two quantities match
/// PyTorch's caching allocator: <b>allocated</b> is memory in buffers handed to a caller and not yet given back,
/// <b>reserved</b> is device memory the library holds from the driver, which also counts buffers parked in a reuse
/// pool. Lock-free: each update is one or two interlocked adds on this device and on the process total.
/// </summary>
internal sealed class DeviceMemoryAccount
{
    private static readonly ConcurrentDictionary<string, DeviceMemoryAccount> s_devices = new(StringComparer.Ordinal);

    /// <summary>The sum over every device, with its own peaks (a sum of per-device peaks would overstate it).</summary>
    internal static readonly DeviceMemoryAccount Total = new("all", isTotal: true);

    private readonly bool _isTotal;
    private long _allocated, _peakAllocated, _reserved, _peakReserved, _allocations, _frees;

    // A driver-side cache the library's own counters cannot see: CUDA's stream-ordered pool keeps freed blocks
    // (release threshold at maximum), so its reserved memory is read from the pool itself.
    private volatile Func<(long Current, long Peak)>? _driverReserved;
    private volatile Action? _resetDriverPeak;

    private DeviceMemoryAccount(string device, bool isTotal)
    {
        Device = device;
        _isTotal = isTotal;
    }

    /// <summary>The device name, <c>backend:index</c> in lower case, such as <c>cuda:0</c>.</summary>
    internal string Device { get; }

    /// <summary>Returns the account for <paramref name="backend"/>'s device <paramref name="index"/>; every backend
    /// instance on one device shares it, as they share the device's memory.</summary>
    internal static DeviceMemoryAccount For(string backend, int index)
    {
        string key = backend.ToLowerInvariant() + ":" + index.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return s_devices.GetOrAdd(key, k => new DeviceMemoryAccount(k, isTotal: false));
    }

    internal static DeviceMemoryAccount[] Devices
    {
        get
        {
            var values = s_devices.Values;
            var devices = new DeviceMemoryAccount[values.Count];
            values.CopyTo(devices, 0);
            return devices;
        }
    }

    /// <summary>True once any memory was counted here: a backend probed on a machine without its device creates an
    /// account that never is.</summary>
    internal bool HasActivity =>
        Interlocked.Read(ref _peakAllocated) > 0 || Interlocked.Read(ref _peakReserved) > 0 || _driverReserved is not null;

    internal static bool TryGet(string device, out DeviceMemoryAccount account)
        => s_devices.TryGetValue(device.ToLowerInvariant(), out account!);

    /// <summary>Reads reserved memory from a driver pool the library allocates from asynchronously; that memory is
    /// then not counted by <see cref="Reserve"/>.</summary>
    internal void SetDriverReserved(Func<(long Current, long Peak)> read, Action resetPeak)
    {
        _driverReserved = read;
        _resetDriverPeak = resetPeak;
    }

    /// <summary>A fresh driver allocation of <paramref name="bytes"/>.</summary>
    internal void Reserve(long bytes)
    {
        Add(ref _reserved, ref _peakReserved, bytes);
        Interlocked.Increment(ref _allocations);
        if (!_isTotal) Total.Reserve(bytes);
    }

    /// <summary>The driver free of an allocation <see cref="Reserve"/> counted.</summary>
    internal void Unreserve(long bytes)
    {
        Interlocked.Add(ref _reserved, -bytes);
        Interlocked.Increment(ref _frees);
        if (!_isTotal) Total.Unreserve(bytes);
    }

    /// <summary>A buffer of <paramref name="bytes"/> handed to a caller.</summary>
    internal void Acquire(long bytes)
    {
        Add(ref _allocated, ref _peakAllocated, bytes);
        if (!_isTotal) Total.Acquire(bytes);
    }

    /// <summary>A buffer given back by its caller, to a pool or to the driver.</summary>
    internal void Relinquish(long bytes)
    {
        Interlocked.Add(ref _allocated, -bytes);
        if (!_isTotal) Total.Relinquish(bytes);
    }

    private static void Add(ref long current, ref long peak, long bytes)
    {
        long now = Interlocked.Add(ref current, bytes);
        long seen;
        while (now > (seen = Interlocked.Read(ref peak)))
            if (Interlocked.CompareExchange(ref peak, now, seen) == seen) break;
    }

    internal void ResetPeaks()
    {
        Interlocked.Exchange(ref _peakAllocated, Interlocked.Read(ref _allocated));
        Interlocked.Exchange(ref _peakReserved, Interlocked.Read(ref _reserved));
        if (_isTotal)
            foreach (var device in s_devices.Values) device._resetDriverPeak?.Invoke();
        else
            _resetDriverPeak?.Invoke();
    }

    internal GpuMemorySnapshot Snapshot()
    {
        long driverCurrent = 0, driverPeak = 0;
        if (_isTotal)
        {
            foreach (var device in s_devices.Values)
                if (device.ReadDriver() is { } d) { driverCurrent += d.Current; driverPeak += d.Peak; }
        }
        else if (ReadDriver() is { } d)
        {
            driverCurrent = d.Current;
            driverPeak = d.Peak;
        }

        return new GpuMemorySnapshot(
            Device,
            Interlocked.Read(ref _allocated),
            Interlocked.Read(ref _peakAllocated),
            Interlocked.Read(ref _reserved) + driverCurrent,
            Interlocked.Read(ref _peakReserved) + driverPeak,
            Interlocked.Read(ref _allocations),
            Interlocked.Read(ref _frees),
            HostShadowMemory.CurrentBytes,
            HostShadowMemory.PeakBytes);
    }

    private (long Current, long Peak)? ReadDriver()
    {
        var read = _driverReserved;
        if (read is null) return null;
        try { return read(); }
        catch { return null; }   // a statistics read never fails its caller
    }
}

/// <summary>
/// The size of one owning device buffer and what the account has counted for it. Embedded as a field in each
/// backend's buffer type; every transition is idempotent, so a double dispose or a finalizer after a dispose
/// counts once.
/// </summary>
internal struct DeviceMemoryHold
{
    private DeviceMemoryAccount? _account;
    private long _bytes;
    private int _reserved;
    private int _inUse;

    /// <summary>Counts a fresh allocation handed straight to its caller. <paramref name="countReserved"/> is false when
    /// the driver pool's own reserved counter covers this memory.</summary>
    internal void OnAllocated(DeviceMemoryAccount? account, long bytes, bool countReserved = true)
    {
        if (account is null || bytes <= 0) return;
        _account = account;
        _bytes = bytes;
        if (countReserved && Interlocked.Exchange(ref _reserved, 1) == 0) account.Reserve(bytes);
        if (Interlocked.Exchange(ref _inUse, 1) == 0) account.Acquire(bytes);
    }

    /// <summary>A pooled buffer handed to a caller again.</summary>
    internal void OnRented()
    {
        if (_account is { } account && Interlocked.Exchange(ref _inUse, 1) == 0) account.Acquire(_bytes);
    }

    /// <summary>The caller gave the buffer back; its memory may stay reserved in a pool.</summary>
    internal void OnReturned()
    {
        if (_account is { } account && Interlocked.Exchange(ref _inUse, 0) == 1) account.Relinquish(_bytes);
    }

    /// <summary>The memory went back to the driver (or to the context that is destroying it).</summary>
    internal void OnFreed()
    {
        OnReturned();
        if (_account is { } account && Interlocked.Exchange(ref _reserved, 0) == 1) account.Unreserve(_bytes);
    }

    /// <summary>Moves the reservation to the pool's new wrapper of the same memory, which starts not in use.</summary>
    internal DeviceMemoryHold TransferToLease()
    {
        OnReturned();
        var lease = new DeviceMemoryHold { _account = _account, _bytes = _bytes };
        lease._reserved = Interlocked.Exchange(ref _reserved, 0);
        return lease;
    }
}

/// <summary>
/// Host arrays kept beside GPU-resident tensor data: a downloaded copy of a resident tensor, or the array a host
/// tensor keeps after its data moved to the device. Process-wide (host memory belongs to no device).
/// </summary>
internal static class HostShadowMemory
{
    private static long s_current, s_peak;

    // One charge per array, released when the array is collected: the table keeps a value exactly as long as its key.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Array, Charge> s_arrays = new();

    private sealed class Charge
    {
        private readonly long _bytes;
        public Charge(long bytes) { _bytes = bytes; Add(bytes); }
        ~Charge() => Remove(_bytes);
    }

    /// <summary>Counts <paramref name="array"/> as a host shadow until it is collected. Counting one twice is a no-op.</summary>
    internal static void Track(Array array)
    {
        if (array is null || array.Length == 0) return;
        if (s_arrays.TryGetValue(array, out _)) return;
        long bytes = (long)array.Length * ElementBytes(array);
        s_arrays.GetValue(array, _ => new Charge(bytes));
    }

    // Managed element size: Buffer.ByteLength covers primitive arrays; anything else counts as a reference slot.
    private static long ElementBytes(Array array)
    {
        try { return Buffer.ByteLength(array) / array.Length; }
        catch (ArgumentException) { return IntPtr.Size; }
    }

    internal static long CurrentBytes => Interlocked.Read(ref s_current);
    internal static long PeakBytes => Interlocked.Read(ref s_peak);

    internal static void Add(long bytes)
    {
        long now = Interlocked.Add(ref s_current, bytes);
        long seen;
        while (now > (seen = Interlocked.Read(ref s_peak)))
            if (Interlocked.CompareExchange(ref s_peak, now, seen) == seen) break;
    }

    internal static void Remove(long bytes) => Interlocked.Add(ref s_current, -bytes);

    internal static void ResetPeak() => Interlocked.Exchange(ref s_peak, Interlocked.Read(ref s_current));
}
