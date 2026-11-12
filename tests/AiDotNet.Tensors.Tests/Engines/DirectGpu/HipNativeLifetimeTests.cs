using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A HIP backend dropped without Dispose has to release its modules, hipBLAS handle, stream and pinned host blocks when
/// it is collected (HipBackend's finalizer hands its Dispose to the deferred-free queue). Before, there was no finalizer
/// at all and every abandoned backend kept them for the life of the process. Runs on an AMD (ROCm/HIP) host; skips
/// elsewhere.
/// </summary>
[Collection("DirectGpuSerial")]
[Trait("Category", "GPU")]
public class HipNativeLifetimeTests
{
    private const int Abandoned = 120;

    private readonly ITestOutputHelper _out;

    public HipNativeLifetimeTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void AbandonedBackends_ReleaseTheirNativeResources()
    {
        Skip.IfNot(HipBackend.IsHipAvailable, "HIP runtime is unavailable.");
        using var anchor = new HipBackend(0);
        Skip.IfNot(anchor.IsAvailable, $"The HIP backend did not initialise: {anchor.InitializationException?.Message}");
        IDeviceMemoryPressure pressure = anchor;
        Settle(anchor);
        Assert.True(pressure.TryGetDeviceMemory(out ulong freeBefore, out _), "device memory query failed");

        for (int i = 0; i < Abandoned; i++)
        {
            Assert.True(CreateUseAndAbandon(i, out string? error), $"backend {i}: {error}");
            Settle(anchor);
        }

        Assert.True(pressure.TryGetDeviceMemory(out ulong freeAfter, out _), "device memory query failed");
        long lostMb = ((long)freeBefore - (long)freeAfter) >> 20;
        _out.WriteLine($"{Abandoned} backends abandoned; device memory lost {lostMb} MB");
        Assert.True(lostMb < 64, $"{lostMb} MB of device memory was not returned by {Abandoned} abandoned backends.");
    }

    // Collect, then let the anchor's next allocation run the deferred releases; release what pools merely cache.
    private static void Settle(HipBackend anchor)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        using (var probe = anchor.AllocateBuffer(new float[16])) anchor.Synchronize();
        ((IDeviceMemoryPressure)anchor).ReleaseCachedBuffers();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool CreateUseAndAbandon(int seed, out string? error)
    {
        var backend = new HipBackend(0);
        error = backend.InitializationException?.Message;
        if (!backend.IsAvailable) return false;
        var data = new float[1024];
        for (int i = 0; i < data.Length; i++) data[i] = i + seed;
        var a = backend.AllocateBuffer(data);
        var b = backend.AllocateBuffer(data);
        var c = backend.AllocateBuffer(new float[data.Length]);
        backend.Add(a, b, c, data.Length);
        var result = backend.DownloadBuffer(c);
        return result[3] == 2 * (3 + seed);
    }
}
