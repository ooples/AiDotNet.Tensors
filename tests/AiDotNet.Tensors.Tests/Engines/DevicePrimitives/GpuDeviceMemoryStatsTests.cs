// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DevicePrimitives;

/// <summary>
/// The device counters of <see cref="GpuMemoryStats"/>: allocated and reserved bytes as PyTorch's caching allocator
/// reports them, fed by every backend's buffers, and the host-shadow bytes.
/// </summary>
[Collection(GpuMemoryStatsGlobalStateCollection.Name)]
public class GpuDeviceMemoryStatsTests
{
    [Fact]
    public void Hold_CountsAllocatedAndReservedThroughAPoolCycle()
    {
        var account = DeviceMemoryAccount.For("unittest", 1);
        var start = account.Snapshot();

        var hold = default(DeviceMemoryHold);
        hold.OnAllocated(account, 1000);
        AssertDelta(start, account.Snapshot(), allocated: 1000, reserved: 1000);

        // Returned to a pool under a new wrapper: no longer allocated, still reserved.
        var lease = hold.TransferToLease();
        AssertDelta(start, account.Snapshot(), allocated: 0, reserved: 1000);
        hold.OnFreed();   // the old wrapper no longer owns the memory
        AssertDelta(start, account.Snapshot(), allocated: 0, reserved: 1000);

        lease.OnRented();
        lease.OnRented();   // idempotent
        AssertDelta(start, account.Snapshot(), allocated: 1000, reserved: 1000);

        lease.OnFreed();
        lease.OnFreed();   // a finalizer after a dispose counts once
        var end = account.Snapshot();
        AssertDelta(start, end, allocated: 0, reserved: 0);
        Assert.True(end.PeakAllocatedBytes >= start.AllocatedBytes + 1000);
        Assert.Equal(1, end.DeviceAllocations - start.DeviceAllocations);
        Assert.Equal(1, end.DeviceFrees - start.DeviceFrees);
    }

    [Fact]
    public void Hold_DriverPooledMemoryIsNotCountedAsReserved()
    {
        var account = DeviceMemoryAccount.For("unittest", 2);
        var start = account.Snapshot();
        var hold = default(DeviceMemoryHold);
        hold.OnAllocated(account, 512, countReserved: false);
        AssertDelta(start, account.Snapshot(), allocated: 512, reserved: 0);
        hold.OnFreed();
        AssertDelta(start, account.Snapshot(), allocated: 0, reserved: 0);
    }

    [Fact]
    public void ResetPeak_DropsThePeakToTheCurrentValue()
    {
        var account = DeviceMemoryAccount.For("unittest", 3);
        var hold = default(DeviceMemoryHold);
        hold.OnAllocated(account, 4096);
        hold.OnFreed();
        Assert.True(account.Snapshot().PeakAllocatedBytes >= 4096);
        account.ResetPeaks();
        var after = account.Snapshot();
        Assert.Equal(after.AllocatedBytes, after.PeakAllocatedBytes);
        Assert.Equal(after.ReservedBytes, after.PeakReservedBytes);
    }

    [Fact]
    public void HostShadow_CountsAnArrayUntilItIsCollected()
    {
        Collect();
        long before = HostShadowMemory.CurrentBytes;
        TrackUnreachable(250_000);
        Assert.True(HostShadowMemory.CurrentBytes - before >= 1_000_000);
        Collect();
        Collect();
        Assert.True(HostShadowMemory.CurrentBytes - before < 1_000_000,
            $"host shadow still counts {HostShadowMemory.CurrentBytes - before} bytes after its array was collected");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void TrackUnreachable(int length)
    {
        var array = new float[length];
        HostShadowMemory.Track(array);
        HostShadowMemory.Track(array);   // once per array
        Assert.True(HostShadowMemory.CurrentBytes >= (long)length * sizeof(float));
    }

    public static TheoryData<string> Backends => new() { "cuda", "hip", "opencl", "vulkan", "metal", "webgpu" };

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void Backend_BufferIsAllocatedUntilDisposed_AndReservedNotBelowAllocated(string name)
    {
        using var owned = Create(name, out var backend);
        Skip.If(backend is null, $"{name} is not available on this machine");

        const int Elements = 262_147;   // an odd size, so no earlier test parked a buffer that fits it exactly
        Collect();
        var before = GpuMemoryStats.Snapshot(backend!);
        var buffer = backend!.AllocateBuffer(Elements);
        var during = GpuMemoryStats.Snapshot(backend);
        Assert.True(during.AllocatedBytes - before.AllocatedBytes >= (long)Elements * sizeof(float),
            $"allocating {Elements * sizeof(float)} B raised allocated by {during.AllocatedBytes - before.AllocatedBytes} B");
        Assert.True(during.ReservedBytes >= during.AllocatedBytes, $"reserved {during.ReservedBytes} < allocated {during.AllocatedBytes}");
        Assert.True(during.PeakAllocatedBytes >= during.AllocatedBytes);
        Assert.StartsWith(name + ":", during.Device, StringComparison.Ordinal);

        buffer.Dispose();
        var after = GpuMemoryStats.Snapshot(backend);
        Assert.Equal(before.AllocatedBytes, after.AllocatedBytes);
        Assert.True(after.ReservedBytes >= after.AllocatedBytes);
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void Backend_AnUndisposedBufferStopsCountingWhenCollected(string name)
    {
        using var owned = Create(name, out var backend);
        Skip.If(backend is null, $"{name} is not available on this machine");

        Collect();
        long before = GpuMemoryStats.Snapshot(backend!).AllocatedBytes;
        AllocateAndDrop(backend!, 131_101);
        Collect();
        Collect();
        Assert.Equal(before, GpuMemoryStats.Snapshot(backend!).AllocatedBytes);
    }

    [SkippableFact]
    public void Facade_ReportsTorchKeysAndTheDeviceName()
    {
        using var owned = Create("cuda", out var backend);
        Skip.If(backend is null, "CUDA is not available on this machine");
        using var buffer = backend!.AllocateBuffer(1024);
        var stats = GpuMemoryStats.Stats(backend);
        Assert.True(stats["allocated_bytes.all.current"] >= 4096);
        Assert.True(stats["reserved_bytes.all.current"] >= stats["allocated_bytes.all.current"]);
        Assert.Contains(GpuMemoryStats.Snapshot(backend).Device, GpuMemoryStats.DeviceNames);
        Assert.True(GpuMemoryStats.Stats()["allocated_bytes.all.current"] >= 4096);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void AllocateAndDrop(IDirectGpuBackend backend, int elements) => _ = backend.AllocateBuffer(elements);

    // The backend, or null when unavailable; the returned disposable disposes a backend the test created.
    private static IDisposable? Create(string name, out IDirectGpuBackend? backend)
    {
        backend = null;
        try
        {
            IDirectGpuBackend candidate = name switch
            {
                "cuda" => new CudaBackend(),
                "hip" => new HipBackend(),
                "opencl" => new OpenClBackend(),
                "vulkan" => VulkanBackend.Instance,   // the process-wide instance; never disposed here
                "metal" => new MetalBackend(),
#if NET7_0_OR_GREATER
                "webgpu" => new AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend(),
#endif
                _ => throw new PlatformNotSupportedException(name),
            };
            if (candidate is VulkanBackend vulkan && !vulkan.IsAvailable) vulkan.Initialize();
            if (!candidate.IsAvailable)
            {
                if (name != "vulkan") (candidate as IDisposable)?.Dispose();
                return null;
            }
            backend = candidate;
            return name == "vulkan" ? null : candidate as IDisposable;
        }
        catch (Exception ex) when (ex is not Xunit.Sdk.XunitException)
        {
            return null;   // no driver for this backend on this machine
        }
    }

    private static void AssertDelta(GpuMemorySnapshot start, GpuMemorySnapshot now, long allocated, long reserved)
    {
        Assert.Equal(allocated, now.AllocatedBytes - start.AllocatedBytes);
        Assert.Equal(reserved, now.ReservedBytes - start.ReservedBytes);
    }

    private static void Collect()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }
}
