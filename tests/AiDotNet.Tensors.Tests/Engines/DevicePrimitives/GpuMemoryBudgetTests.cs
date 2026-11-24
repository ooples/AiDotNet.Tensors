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
            DeviceMemoryAccount.SettleAdmission();
            GpuMemoryBudget.ClearLimit("budgettest:1");
        }
    }

    // What AIDOTNET_GPU_MEMORY_FRACTION leaves in force on a device of `total` bytes once its own limit is cleared.
    private static long? EnvironmentLimit(long total)
    {
        string? text = Environment.GetEnvironmentVariable(GpuMemoryBudget.FractionEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(text)) return null;
        double fraction = double.Parse(text, System.Globalization.CultureInfo.InvariantCulture);
        return (long)Math.Floor(total * fraction);
    }

    [Fact]
    public void AFractionSmallerThanOneByte_IsABudgetOfZero_NotTheAbsenceOfOne()
    {
        var account = DeviceMemoryAccount.For("budgettest", 3);
        account.DeviceTotalBytes = 24L * 1024 * 1024 * 1024;
        GpuMemoryBudget.SetPerProcessMemoryFraction(1e-12, "budgettest:3");
        try
        {
            Assert.Equal(0, GpuMemoryBudget.GetLimitBytes("budgettest:3"));
            int reclaims = 0;
            Assert.False(account.TryAdmit(64L * 1024 * 1024, () => reclaims++, out string? refusal));
            Assert.Equal(1, reclaims);
            Assert.Contains("budget of 0 bytes", refusal);
            Assert.False(account.TryAdmit(1, null, out _));
        }
        finally
        {
            DeviceMemoryAccount.SettleAdmission();
            GpuMemoryBudget.ClearLimit("budgettest:3");
        }
    }

    [Theory]
    [InlineData(600, 1)]    // two would pass 1000: exactly one of the eight is admitted
    [InlineData(125, 8)]    // all eight fit exactly
    [InlineData(300, 3)]    // three fit, a fourth would not
    public void ThreadsAdmittedAtTheSameMoment_NeverPassTheBudget_AndNoneThatFitsIsRefused(int bytes, int expectedAdmitted)
    {
        const int Threads = 8;
        var account = DeviceMemoryAccount.For("budgettest", 4 + bytes);
        string device = account.Device;
        GpuMemoryBudget.SetMemoryLimit(1_000, device);
        try
        {
            using var start = new System.Threading.Barrier(Threads);
            using var release = new System.Threading.ManualResetEventSlim(false);
            using var asked = new System.Threading.CountdownEvent(Threads);
            int admitted = 0;
            var threads = new System.Threading.Thread[Threads];
            for (int i = 0; i < Threads; i++)
            {
                threads[i] = new System.Threading.Thread(() =>
                {
                    start.SignalAndWait();
                    bool ok = account.TryAdmit(bytes, null, out _);
                    if (ok) System.Threading.Interlocked.Increment(ref admitted);
                    asked.Signal();
                    // Held, as an allocation in progress holds it, until every thread has asked.
                    release.Wait();
                    DeviceMemoryAccount.SettleAdmission();
                });
                threads[i].Start();
            }

            long heldTogether;
            try
            {
                asked.Wait();
                heldTogether = account.InFlightBytes;
            }
            finally
            {
                // Before any assertion: a failed one must not leave the eight threads waiting.
                release.Set();
                foreach (var thread in threads) thread?.Join();
            }

            Assert.Equal(expectedAdmitted, admitted);
            Assert.Equal((long)expectedAdmitted * bytes, heldTogether);
            Assert.Equal(0, account.InFlightBytes);
        }
        finally
        {
            GpuMemoryBudget.ClearLimit(device);
        }
    }

    [Fact]
    public void AnAdmission_IsHeldUntilItsBytesAreReserved_OrTheThreadAsksAgain()
    {
        var account = DeviceMemoryAccount.For("budgettest", 5);
        GpuMemoryBudget.SetMemoryLimit(1_000, "budgettest:5");
        try
        {
            Assert.True(account.TryAdmit(700, null, out _));
            Assert.Equal(700, account.InFlightBytes);
            account.Reserve(700);                       // the allocation is counted: no longer in flight
            Assert.Equal(0, account.InFlightBytes);
            Assert.Equal(700, account.ReservedBytes);
            Assert.False(account.TryAdmit(400, null, out _));
            Assert.True(account.TryAdmit(300, null, out _));
            Assert.True(account.TryAdmit(300, null, out _));   // the earlier 300 was never allocated: asking again replaces it
            Assert.Equal(300, account.InFlightBytes);
        }
        finally
        {
            DeviceMemoryAccount.SettleAdmission();
            account.Unreserve(700);
            GpuMemoryBudget.ClearLimit("budgettest:5");
        }
    }

    [Fact]
    public void ReservingAnAdmittedAllocation_NeverCountsItTwice_ForAThreadAdmittedAtTheSameMoment()
    {
        // 600 of 1,000 held by one thread that keeps admitting and reserving it; 400 always fits beside it. Counted in
        // both figures for a moment, the 600 would look like 1,200 and the 400 would be refused.
        var account = DeviceMemoryAccount.For("budgettest", 7);
        GpuMemoryBudget.SetMemoryLimit(1_000, "budgettest:7");
        try
        {
            using var stop = new System.Threading.ManualResetEventSlim(false);
            var holder = new System.Threading.Thread(() =>
            {
                while (!stop.IsSet)
                {
                    if (!account.TryAdmit(600, null, out _)) continue;
                    account.Reserve(600);
                    account.Unreserve(600);
                }

                DeviceMemoryAccount.SettleAdmission();
            });
            holder.Start();
            int refused = 0;
            try
            {
                for (int i = 0; i < 200_000; i++)
                {
                    if (!account.TryAdmit(400, null, out _)) refused++;
                    DeviceMemoryAccount.SettleAdmission();
                }
            }
            finally
            {
                stop.Set();
                holder.Join();
            }

            Assert.Equal(0, refused);
        }
        finally
        {
            DeviceMemoryAccount.SettleAdmission();
            GpuMemoryBudget.ClearLimit("budgettest:7");
        }
    }

    [Fact]
    public void ADriverPoolThatCannotBeRead_KeepsItsLastFigure_NotZero()
    {
        var account = DeviceMemoryAccount.For("budgettest", 8);
        GpuMemoryBudget.SetMemoryLimit(1_000, "budgettest:8");
        bool fail = false;
        account.SetDriverReserved(
            () => fail ? throw new InvalidOperationException("the pool query failed") : (900L, 900L),
            () => { });
        try
        {
            Assert.False(account.TryAdmit(200, null, out _));   // 900 in the pool
            fail = true;
            Assert.Equal(900, account.ReservedBytes);
            Assert.False(account.TryAdmit(200, null, out _));   // still 900: unreadable is not empty
            Assert.True(account.TryAdmit(100, null, out _));
        }
        finally
        {
            DeviceMemoryAccount.SettleAdmission();
            account.SetDriverReserved(() => (0L, 0L), () => { });
            GpuMemoryBudget.ClearLimit("budgettest:8");
        }
    }

    [Fact]
    public void BackendsSharingAContext_KeepItAccounted_UntilTheLastOneLeaves()
    {
        var context = new IntPtr(0x7000_2000);
        var first = DeviceMemoryAccount.For("budgettest", 9);
        var second = DeviceMemoryAccount.For("budgettest", 10);
        object firstToken = new object(), secondToken = new object();
        CudaAccountedMemory.Register(context, first, () => { }, firstToken);
        try
        {
            CudaAccountedMemory.Register(context, second, () => { }, secondToken);
            Assert.Same(second, CudaAccountedMemory.AccountOf(context));
            CudaAccountedMemory.Unregister(context, secondToken);
            // The first backend is still alive: its context must not fall back to uncounted allocation.
            Assert.Same(first, CudaAccountedMemory.AccountOf(context));
            Assert.Equal(1, CudaAccountedMemory.RegistrationCount(context));
            CudaAccountedMemory.Unregister(context, secondToken);   // a second dispose removes nothing more
            Assert.Same(first, CudaAccountedMemory.AccountOf(context));
        }
        finally
        {
            CudaAccountedMemory.Unregister(context, firstToken);
        }

        Assert.Null(CudaAccountedMemory.AccountOf(context));
        Assert.Equal(0, CudaAccountedMemory.RegistrationCount(context));
    }

    [Fact]
    public void TrackedScratchMemory_IsReservedOnItsDevice_UntilItIsFreed()
    {
        var account = DeviceMemoryAccount.For("budgettest", 6);
        var pointer = new IntPtr(0x6000_1000);
        long before = account.ReservedBytes;
        CudaAccountedMemory.Track(pointer, account, 4_096);
        try
        {
            Assert.Equal(before + 4_096, account.ReservedBytes);
            Assert.Equal(before + 4_096, GpuMemoryBudget.GetReservedBytes("budgettest:6"));
        }
        finally
        {
            CudaAccountedMemory.Untrack(pointer);
        }

        Assert.Equal(before, account.ReservedBytes);
        CudaAccountedMemory.Untrack(pointer);   // a pointer it does not hold returns nothing
        Assert.Equal(before, account.ReservedBytes);
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
        // Clearing a device's own limit leaves the environment's fraction in force, when one is set.
        long? fallback = EnvironmentLimit(1_000_000);
        Assert.Equal(fallback, GpuMemoryBudget.GetLimitBytes("budgettest:2"));
        Assert.Equal(fallback is null, account.TryAdmit(long.MaxValue / 2, null, out _));
        DeviceMemoryAccount.SettleAdmission();
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
        // Cached memory empty first: a request the pool's free memory covers is no new reservation.
        (backend as IDeviceMemoryPressure)?.ReleaseCachedBuffers();
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
        // Cached memory empty first: a request the pool's free memory covers is no new reservation.
        (backend as IDeviceMemoryPressure)?.ReleaseCachedBuffers();
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

    [SkippableFact]
    public void Cuda_ReuseOfTheStreamOrderedPoolsFreeMemoryIsAdmittedWithoutReclaim()
    {
        using var owned = Create("cuda", out var backend);
        Skip.If(backend is null, "CUDA is not available on this machine");
        var cuda = (CudaBackend)backend!;
        Skip.IfNot(cuda.UsesStreamOrderedAllocation,
            "This CUDA backend allocates synchronously (AIDOTNET_CUDA_ASYNC_ALLOC=0, or a driver without the stream-ordered pool).");
        string device = GpuMemoryStats.Snapshot(cuda).Device;
        Assert.True(DeviceMemoryAccount.TryGet(device, out var account));

        // 96 MB freed into the stream-ordered pool, which keeps it reserved.
        cuda.AllocateBuffer(24 * 1024 * 1024 + 13).Dispose();
        cuda.ReturnPooledBuffersToDriverPoolForTests();
        long reclaimsBefore = account.BudgetReclaims, refusalsBefore = account.BudgetRefusals;
        GpuMemoryBudget.SetMemoryLimit(account.ReservedBytes + 1024 * 1024, device);
        try
        {
            // 48 MB more than the budget leaves, but the pool's free memory covers it: no new reservation.
            using var reused = cuda.AllocateBuffer(12 * 1024 * 1024 + 17);
            Assert.Equal(reclaimsBefore, account.BudgetReclaims);
            Assert.Equal(refusalsBefore, account.BudgetRefusals);
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
        // Only what says the backend's runtime or hardware is not on this machine: any other failure to construct a
        // backend is a defect, and must fail the test rather than skip it.
        catch (Exception unavailable) when (unavailable is DllNotFoundException or EntryPointNotFoundException
                                                or PlatformNotSupportedException or BadImageFormatException)
        {
            return null;
        }
    }
}
