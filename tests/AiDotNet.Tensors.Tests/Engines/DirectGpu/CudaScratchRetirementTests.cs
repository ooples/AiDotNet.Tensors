// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A grow-only scratch buffer (convolution, BatchNorm split partials, the cuDNN channels-last output) is baked into any
/// graph captured while it is current. Growing it later -- one plan's eager warm-up after another plan captured, as a
/// GAN's two networks do -- used to free the old buffer to the pool, where another tensor could take the memory a
/// replay still writes. It is now retired (kept until the backend is disposed) when a capture began after it was
/// allocated, and freed as before otherwise.
/// </summary>
[Collection("DirectGpuSerial")]
public class CudaScratchRetirementTests
{
    [SkippableFact]
    public void A_scratch_buffer_a_capture_may_hold_is_retired_not_freed_when_it_grows()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend did not initialize.");
        var replace = typeof(CudaBackend).GetMethod("ReplaceScratch", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var generation = typeof(CudaBackend).GetField("_captureGeneration", BindingFlags.NonPublic | BindingFlags.Instance)!;

        var args = new object?[] { null, 16 };
        replace.Invoke(backend, args);
        var first = (IGpuBuffer)args[0]!;
        Assert.Equal(0, backend.RetiredScratchCount);

        // No capture since it was allocated: growing frees it.
        args[1] = 64;
        replace.Invoke(backend, args);
        var second = (IGpuBuffer)args[0]!;
        Assert.NotSame(first, second);
        Assert.Equal(0, backend.RetiredScratchCount);

        // A capture began (it may have baked `second` into a graph): growing retires it, still allocated.
        generation.SetValue(backend, (long)generation.GetValue(backend)! + 1);
        args[1] = 256;
        replace.Invoke(backend, args);
        Assert.Equal(1, backend.RetiredScratchCount);
        Assert.True(second.Size >= 64, "the retired buffer must stay allocated for the graph that may replay into it");

        // The replacement was allocated after that capture began: growing it again frees it.
        args[1] = 1024;
        replace.Invoke(backend, args);
        Assert.Equal(1, backend.RetiredScratchCount);
        ((IGpuBuffer)args[0]!).Dispose();
    }

    [SkippableFact]
    public void An_outgrown_cudnn_workspace_is_freed_unless_a_capture_began_since_it_was_allocated()
    {
        // The workspace grows mostly while the autotuner times candidates, before any step is captured: retiring every
        // outgrown size kept 102 MiB of a DCGAN step's 211 MiB of workspaces allocated for nothing.
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        Skip.IfNot(CuDnnConvolution.IsAvailable, "cuDNN is unavailable.");
        using var conv = new CuDnnConvolution();
        var ensure = typeof(CuDnnConvolution).GetMethod("EnsureWorkspaceUnlocked", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var retired = (System.Collections.ICollection)typeof(CuDnnConvolution)
            .GetField("_retiredWorkspaces", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(conv)!;
        var epoch = typeof(CudaBackend).GetField("s_captureEpoch", BindingFlags.NonPublic | BindingFlags.Static)!;

        ensure.Invoke(conv, new object[] { 1UL << 20, false });
        ensure.Invoke(conv, new object[] { 4UL << 20, false });
        Assert.Equal(0, retired.Count);      // no capture since the 1 MiB workspace was allocated: freed

        // A capture began (it may have recorded a convolution with the 4 MiB workspace): growing retires it.
        epoch.SetValue(null, (long)epoch.GetValue(null)! + 1);
        ensure.Invoke(conv, new object[] { 16UL << 20, false });
        Assert.Equal(1, retired.Count);

        // The 16 MiB workspace was allocated after that capture began: growing it frees it.
        ensure.Invoke(conv, new object[] { 64UL << 20, false });
        Assert.Equal(1, retired.Count);
    }
}
