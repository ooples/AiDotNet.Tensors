// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
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

    // The device's memory, as its backend reports it (0 until one does): what a fraction budget is a fraction of.
    private long _deviceTotalBytes;

    /// <summary>The device's total memory as its backend reported it, or 0 when none has.</summary>
    internal long DeviceTotalBytes
    {
        get => Interlocked.Read(ref _deviceTotalBytes);
        set => Interlocked.Exchange(ref _deviceTotalBytes, value);
    }

    /// <summary>
    /// The most this process may hold reserved on the device (<see cref="GpuMemoryBudget"/>), or 0 for no limit: an
    /// explicit byte limit, else a fraction of <see cref="DeviceTotalBytes"/>.
    /// </summary>
    internal long? LimitBytes => GpuMemoryBudget.LimitFor(Device, DeviceTotalBytes);

    /// <summary>Memory the library holds from the driver now: its own reservations and the driver pool's.</summary>
    internal long ReservedBytes => Interlocked.Read(ref _reserved) + (ReadDriver()?.Current ?? 0);

    // The driver pool's figures as last read. A query that fails keeps them: an unreadable pool is not an empty one,
    // and counting it as zero would admit allocations past the budget.
    private long _lastDriverCurrent, _lastDriverPeak;
    private int _driverWasRead;

    /// <summary>
    /// Admits a new driver allocation of <paramref name="bytes"/> under the device's budget. Over it, runs
    /// <paramref name="reclaim"/> (the backend's release of cached and unreachable memory, as for a driver
    /// out-of-memory) once and checks again; still over, throws the error <paramref name="outOfMemory"/> builds, the
    /// same type the backend raises when the driver itself is out of memory, so every caller's fallback applies alike.
    /// PyTorch's per-process memory fraction caps its caching allocator's reserved memory the same way.
    /// </summary>
    internal void Admit(long bytes, Action? reclaim, Func<string, Exception> outOfMemory)
    {
        if (!TryAdmit(bytes, reclaim, out string? refusal)) throw outOfMemory(refusal!);
    }

    /// <summary>As <see cref="Admit"/>, for an allocator that reports failure by its result: false (with the reason)
    /// where <see cref="Admit"/> would throw.</summary>
    // Bytes admitted under the budget whose allocation is still being made: admitted, not yet counted as reserved.
    // Every admission counts them, so threads allocating at the same moment cannot each be admitted against the
    // same headroom.
    private long _inFlight;

    // The admission this thread holds. One at a time: the next admission, or the reservation of the admitted
    // allocation, settles it.
    [ThreadStatic] private static DeviceMemoryAccount? t_admittedOn;
    [ThreadStatic] private static long t_admittedBytes;

    /// <summary>
    /// Whether an allocation of <paramref name="bytes"/> fits the device's budget, after running
    /// <paramref name="reclaim"/> once when it does not. An admitted request is held against the budget until this
    /// thread reserves it (<see cref="Reserve"/>), settles it (<see cref="SettleAdmission"/>, for an allocation the
    /// driver's own pool counts, or one that failed), or asks for its next admission.
    /// </summary>
    internal bool TryAdmit(long bytes, Action? reclaim, out string? refusal)
    {
        refusal = null;
        SettleAdmission();
        if (LimitBytes is not long limit || bytes <= 0) return true;
        if (Claim(bytes, limit)) return true;
        Interlocked.Increment(ref _budgetReclaims);
        reclaim?.Invoke();
        if (Claim(bytes, limit)) return true;
        Interlocked.Increment(ref _budgetRefusals);
        refusal = $"GPU memory budget of {limit} bytes on {Device} exceeded: {bytes} bytes requested with {ReservedBytes} "
                  + "reserved (GpuMemoryBudget).";
        return false;
    }

    private readonly object _admissionLock = new object();

    // Checked and counted under one lock (taken only on a device that has a budget): of the threads asking at the
    // same moment, exactly those that fit together are admitted -- none past the limit, and none refused for a
    // request that would have fit.
    private bool Claim(long bytes, long limit)
    {
        lock (_admissionLock)
        {
            long inFlight = Interlocked.Read(ref _inFlight);
            if (bytes > limit || ReservedBytes > limit - bytes - inFlight) return false;
            Interlocked.Add(ref _inFlight, bytes);
        }

        HoldAdmission(this, bytes);
        return true;
    }

    private static void HoldAdmission(DeviceMemoryAccount account, long bytes)
    {
        t_admittedOn = account;
        t_admittedBytes = bytes;
    }

    /// <summary>Releases the admission this thread holds, if any: its allocation is now counted as reserved, or was
    /// not made.</summary>
    internal static void SettleAdmission()
    {
        if (t_admittedOn is not { } account) return;
        Interlocked.Add(ref account._inFlight, -t_admittedBytes);
        t_admittedOn = null;
        t_admittedBytes = 0;
    }

    /// <summary>Bytes admitted and not yet reserved or settled.</summary>
    internal long InFlightBytes => Interlocked.Read(ref _inFlight);

    private long _budgetReclaims, _budgetRefusals;

    /// <summary>Allocations the budget sent through a reclaim, and those it refused (tests, diagnostics).</summary>
    internal long BudgetReclaims => Interlocked.Read(ref _budgetReclaims);

    internal long BudgetRefusals => Interlocked.Read(ref _budgetRefusals);

    /// <summary>Counts a refusal an allocator decided after allocating (it grew past the budget).</summary>
    internal void CountBudgetRefusal() => Interlocked.Increment(ref _budgetRefusals);

    /// <summary>A fresh driver allocation of <paramref name="bytes"/>.</summary>
    internal void Reserve(long bytes)
    {
        if (ReferenceEquals(t_admittedOn, this))
        {
            // The allocation this thread was admitted for is counted now. Counted and settled under the admission
            // lock, so an admission sees the bytes in exactly one of the two figures: never in neither (admitting
            // past the budget), never in both (refusing a request that fits).
            lock (_admissionLock)
            {
                Add(ref _reserved, ref _peakReserved, bytes);
                SettleAdmission();
            }
        }
        else
        {
            Add(ref _reserved, ref _peakReserved, bytes);
        }

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
            foreach (var d in s_devices.Values.Select(device => device.ReadDriver()).Where(d => d.HasValue))
            {
                driverCurrent += d!.Value.Current;
                driverPeak += d.Value.Peak;
            }
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
        try
        {
            var figures = read();
            Interlocked.Exchange(ref _lastDriverCurrent, figures.Current);
            Interlocked.Exchange(ref _lastDriverPeak, figures.Peak);
            Volatile.Write(ref _driverWasRead, 1);
            return figures;
        }
        // A statistics read never fails its caller. A query that fails reads as the last figures it returned (none
        // before the first successful one), not as an empty pool.
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            if (Volatile.Read(ref _driverWasRead) == 0) return null;
            return (Interlocked.Read(ref _lastDriverCurrent), Interlocked.Read(ref _lastDriverPeak));
        }
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
        s_arrays.GetValue(array, static key => new Charge((long)key.Length * ElementBytes(key)));
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
