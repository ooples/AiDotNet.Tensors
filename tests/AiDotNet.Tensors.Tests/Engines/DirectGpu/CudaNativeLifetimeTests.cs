using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A CUDA backend dropped without Dispose has to release what it created - its modules, streams, events, cuBLAS /
/// cuBLASLt / cuDNN handles, pinned host blocks and its hold on the device's primary context - when it is collected.
/// The primary context is shared, so before the deferred release only the stream and the hold were dropped: while any
/// other backend lived (here, the anchor), every abandoned backend's modules and handles stayed allocated.
/// </summary>
[Collection("DirectGpuSerial")]
[Trait("Category", "GPU")]
public class CudaNativeLifetimeTests
{
    private const int Abandoned = 120;

    private readonly ITestOutputHelper _out;

    public CudaNativeLifetimeTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void AbandonedBackends_ReleaseTheirNativeResources()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        // A live backend keeps the primary context alive, so only the per-backend release can return memory.
        using var anchor = new CudaBackend(0);
        Skip.IfNot(anchor.IsAvailable, "The CUDA backend did not initialise.");
        IDeviceMemoryPressure pressure = anchor;
        Settle(anchor);
        int baselineHolders = CudaBackend.PrimaryContextHolderCount;
        Assert.True(pressure.TryGetDeviceMemory(out ulong freeBefore, out _), "device memory query failed");

        for (int i = 0; i < Abandoned; i++)
        {
            Assert.True(CreateUseAndAbandon(i, out string? error), $"backend {i}: {error}");
            Settle(anchor);
        }

        Assert.True(pressure.TryGetDeviceMemory(out ulong freeAfter, out _), "device memory query failed");
        int holders = CudaBackend.PrimaryContextHolderCount - baselineHolders;
        long lostMb = ((long)freeBefore - (long)freeAfter) >> 20;
        _out.WriteLine($"{Abandoned} backends abandoned: {holders} still hold the context; device memory lost {lostMb} MB");
        Assert.True(holders <= 1, $"{holders} abandoned backends still hold the primary context.");
        // A few MB of allocator slack is fine; a leaked backend's modules alone are about 4 MB each (480 MB here).
        Assert.True(lostMb < 64, $"{lostMb} MB of device memory was not returned by {Abandoned} abandoned backends.");
    }

    // Collect, then let the anchor's next operation run the deferred releases (as any op thread does).
    private static void Settle(CudaBackend anchor)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        using (var probe = anchor.AllocateBuffer(new float[16])) anchor.Synchronize();
        // torch.cuda.empty_cache: what the device's stream-ordered pool merely caches is not a leak.
        ((IDeviceMemoryPressure)anchor).ReleaseCachedBuffers();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool CreateUseAndAbandon(int seed, out string? error)
    {
        var backend = new CudaBackend(0);
        error = backend.InitializationError;
        if (!backend.IsAvailable) return false;
        var data = new float[1024];
        for (int i = 0; i < data.Length; i++) data[i] = i + seed;
        var a = backend.AllocateBuffer(data);
        var b = backend.AllocateBuffer(data);
        var c = backend.AllocateBuffer(new float[data.Length]);
        backend.Add(a, b, c, data.Length);
        var result = backend.DownloadBuffer(c);
        // a, b and c are abandoned with the backend, too.
        return result[3] == 2 * (3 + seed);
    }
}
