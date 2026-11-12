using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// An OpenCL backend holds a context with every compiled program, roughly 110 MB of host memory. One that is never
/// disposed has to give that back when it is collected: before its native handles had a finalizer, about 56 abandoned
/// backends exhausted the driver's host budget and every later context creation failed with CL_OUT_OF_HOST_MEMORY.
/// </summary>
[Collection("DirectGpuSerial")]
public class OpenClNativeLifetimeTests
{
    // Comfortably past the measured ~56-context budget.
    private const int Abandoned = 120;

    private readonly ITestOutputHelper _out;

    public OpenClNativeLifetimeTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void AbandonedBackends_AreReclaimed_SoTheBudgetIsNeverExhausted()
    {
        Skip.IfNot(OpenClBackend.IsGpuAvailable, "Requires an OpenCL GPU.");
        int baseline = DirectOpenClContext.LiveContextCount;
        for (int i = 0; i < Abandoned; i++)
        {
            Assert.True(CreateUseAndAbandon(i, out string? error), $"backend {i} could not be created: {error}");
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
        int live = DirectOpenClContext.LiveContextCount - baseline;
        _out.WriteLine($"{Abandoned} backends abandoned; {live} of their contexts still alive; working set {Environment.WorkingSet >> 20} MB");
        Assert.True(live <= 2, $"{live} abandoned OpenCL contexts were never released.");
    }

    [SkippableFact]
    public void AbandonedTensorEngines_AreReclaimed_SoTheBudgetIsNeverExhausted()
    {
        Skip.IfNot(OpenClBackend.IsGpuAvailable, "Requires an OpenCL GPU.");
        for (int i = 0; i < Abandoned; i++)
        {
            Assert.True(CreateEngineUseAndAbandon(out string? error), $"engine {i} could not be created: {error}");
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
    }

    // Kept out of line so nothing it allocated is still on this frame's stack when the caller collects.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool CreateUseAndAbandon(int seed, out string? error)
    {
        var backend = new OpenClBackend();
        error = backend.InitializationException?.Message;
        if (!backend.IsAvailable) return false;
        // Allocate, compute and read back, so programs, kernels, buffers and queues all exist.
        var data = new float[1024];
        for (int i = 0; i < data.Length; i++) data[i] = i + seed;
        using var a = backend.AllocateBuffer(data);
        using var b = backend.AllocateBuffer(data);
        using var c = backend.AllocateBuffer(new float[data.Length]);
        backend.Add(a, b, c, data.Length);
        var result = backend.DownloadBuffer(c);
        return result[3] == 2 * (3 + seed);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool CreateEngineUseAndAbandon(out string? error)
    {
        error = null;
        var engine = new DirectGpuTensorEngine();
        if (!engine.IsGpuAvailable) { error = "the engine reported no GPU"; return false; }
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var y = engine.TensorAdd(x, x);
        return y[3] == 8f;
    }
}
