// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
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
/// <see cref="GpuMemoryBudget"/>: a per-process cap on reserved device memory (PyTorch's per-process memory fraction)
/// that refuses an allocation past it with the backend's out-of-memory error after one reclaim.
/// </summary>
[Collection(GpuMemoryStatsGlobalStateCollection.Name)]
public class GpuMemoryBudgetTests
{
    [Fact]
    public void Admit_PassesUnderTheLimit_ReclaimsThenRefusesOverIt()
    {
        var account = DeviceMemoryAccount.For("budgettest", 1);
        GpuMemoryBudget.SetMemoryLimit(10_000, "budgettest:1");
        try
        {
            account.Reserve(6_000);
            int reclaims = 0;
            account.Admit(4_000, () => reclaims++, m => new InvalidOperationException(m));   // exactly at the limit
            Assert.Equal(0, reclaims);

            var ex = Assert.Throws<GpuOutOfMemoryException>(() =>
                account.Admit(4_001, () => reclaims++, m => new GpuOutOfMemoryException(m, 2)));
            Assert.Equal(1, reclaims);   // one release before refusing
            Assert.Contains("GpuMemoryBudget", ex.Message);
            Assert.Contains("budgettest:1", ex.Message);

            // A reclaim that frees enough admits the request.
            account.Admit(8_000, () => account.Unreserve(6_000), m => new InvalidOperationException(m));
            Assert.Equal(0, account.ReservedBytes);
        }
        finally
        {
            GpuMemoryBudget.ClearLimit("budgettest:1");
        }
    }

    [Fact]
    public void Fraction_ResolvesAgainstTheDeviceTotal_AndClearRemovesIt()
    {
        var account = DeviceMemoryAccount.For("budgettest", 2);
        GpuMemoryBudget.SetPerProcessMemoryFraction(0.25, "BudgetTest:2");   // names are case-insensitive
        try
        {
            Assert.Null(GpuMemoryBudget.GetLimitBytes("budgettest:2"));   // no total reported yet
            Assert.True(account.TryAdmit(long.MaxValue / 2, null, out _));

            account.DeviceTotalBytes = 1_000_000;
            Assert.Equal(250_000, GpuMemoryBudget.GetLimitBytes("budgettest:2"));
            Assert.True(account.TryAdmit(250_000, null, out _));
            Assert.False(account.TryAdmit(250_001, null, out string? refusal));
            Assert.Contains("250000", refusal);

            GpuMemoryBudget.SetMemoryLimit(500, "budgettest:2");   // a byte limit replaces the fraction
            Assert.Equal(500, GpuMemoryBudget.GetLimitBytes("budgettest:2"));
        }
        finally
        {
            GpuMemoryBudget.ClearLimit("budgettest:2");
        }
        Assert.Null(GpuMemoryBudget.GetLimitBytes("budgettest:2"));
        Assert.True(account.TryAdmit(long.MaxValue / 2, null, out _));
    }

    [Fact]
    public void InvalidArguments_AreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuMemoryBudget.SetPerProcessMemoryFraction(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuMemoryBudget.SetPerProcessMemoryFraction(1.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuMemoryBudget.SetPerProcessMemoryFraction(double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuMemoryBudget.SetMemoryLimit(0));
        Assert.Throws<ArgumentException>(() => GpuMemoryBudget.SetMemoryLimit(1, " "));
    }

    public static TheoryData<string> Backends => new() { "cuda", "hip", "opencl", "vulkan", "metal", "webgpu" };

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void Backend_AllocationPastTheBudgetIsRefused_AndSucceedsOnceCleared(string name)
    {
        using var owned = Create(name, out var backend);
        Skip.If(backend is null, $"{name} is not available on this machine");

        string device = GpuMemoryStats.Snapshot(backend!).Device;
        Assert.True(DeviceMemoryAccount.TryGet(device, out var account));
        const int Elements = 16 * 1024 * 1024 + 7;   // 64 MB, an odd size no pool holds
        long refusalsBefore = account.BudgetRefusals;
        GpuMemoryBudget.SetMemoryLimit(Math.Max(1, account.ReservedBytes) + 1024 * 1024, device);
        try
        {
            Assert.ThrowsAny<Exception>(() => backend!.AllocateBuffer(Elements).Dispose());
            Assert.True(account.BudgetRefusals > refusalsBefore, $"{device}: the budget did not refuse the allocation");
        }
        finally
        {
            GpuMemoryBudget.ClearLimit(device);
        }

        using var buffer = backend!.AllocateBuffer(Elements);   // no budget: the same request succeeds
        Assert.Equal(Elements, buffer.Size);
    }

    [SkippableFact]
    public void Cuda_RefusalIsTheBackendsOutOfMemoryError()
    {
        using var owned = Create("cuda", out var backend);
        Skip.If(backend is null, "CUDA is not available on this machine");
        string device = GpuMemoryStats.Snapshot(backend!).Device;
        Assert.True(DeviceMemoryAccount.TryGet(device, out var account));
        GpuMemoryBudget.SetMemoryLimit(Math.Max(1, account.ReservedBytes) + 1024 * 1024, device);
        try
        {
            // Larger than anything the other tests park in the pool: a reuse of pooled memory is no new reservation.
            var ex = Assert.Throws<GpuOutOfMemoryException>(() => backend!.AllocateBuffer(40 * 1024 * 1024 + 11).Dispose());
            Assert.Contains("GpuMemoryBudget", ex.Message);
        }
        finally
        {
            GpuMemoryBudget.ClearLimit(device);
        }
    }

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
            return null;
        }
    }
}
