// Copyright (c) AiDotNet. All rights reserved.
// #1630 / #91: CUDA Graph INFERENCE capture (counterpart to GraphedTrainingStep). The argument
// guards + state machine all fire BEFORE any GPU call, so they're host-agnostic and covered here
// via a DispatchProxy mock backend (gated to non-Framework — DispatchProxy is .NET Core 5+, same
// as the mock itself); the null-backend + options checks need no mock and run on every TFM. Real
// CUDA capture/replay requires an NVIDIA GPU and lives in the integration harness.

#nullable disable

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

public class GraphedInferenceStepTests
{
    // ─── Guards / options that need no backend — run on every TFM ──────────────────

    [Fact]
    public void Ctor_NullBackend_Throws()
        => Assert.Throws<ArgumentNullException>(() =>
            new GraphedInferenceStep(null, (IntPtr)0x1, () => { }));

    [Fact]
    public void Options_Default_HasReasonableValues()
    {
        var opts = GraphedInferenceStepOptions.Default;
        Assert.Equal(2, opts.WarmupIterations);
        Assert.True(opts.ThrowOnUnsupported);
    }

    [Fact]
    public void Options_CustomValues_RoundTrip()
    {
        var opts = new GraphedInferenceStepOptions { WarmupIterations = 4, ThrowOnUnsupported = false };
        Assert.Equal(4, opts.WarmupIterations);
        Assert.False(opts.ThrowOnUnsupported);
    }

    [Fact]
    public void Options_Default_IsSingleton()
        => Assert.Same(GraphedInferenceStepOptions.Default, GraphedInferenceStepOptions.Default);

#if !NETFRAMEWORK
    // ─── State machine + remaining guards — via DispatchProxy mock (.NET Core 5+) ──
    // All fire before any GPU call, so they're deterministic on a CUDA-less host.

    private static readonly IntPtr NonDefaultStream = (IntPtr)0x1;
    private static AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend Backend()
        => AiDotNet.Tensors.Tests.Engines.DirectGpu.MockDirectGpuBackend.Create(
               new AiDotNet.Tensors.Tests.Engines.DirectGpu.MockBackendState());

    [Fact]
    public void Ctor_DefaultStream_Throws()
        => Assert.Throws<ArgumentException>(() =>
            new GraphedInferenceStep(Backend(), IntPtr.Zero, () => { }));

    [Fact]
    public void Ctor_NullForward_Throws()
        => Assert.Throws<ArgumentNullException>(() =>
            new GraphedInferenceStep(Backend(), NonDefaultStream, null));

    [Fact]
    public void Capture_BeforePrepare_Throws()
    {
        using var step = new GraphedInferenceStep(Backend(), NonDefaultStream, () => { });
        var ex = Assert.Throws<InvalidOperationException>(() => step.Capture());
        Assert.Contains("Prepare", ex.Message);
    }

    [Fact]
    public void Prepare_RunsForward_WarmupIterations_Times()
    {
        int calls = 0;
        var opts = new GraphedInferenceStepOptions { WarmupIterations = 3 };
        using var step = new GraphedInferenceStep(Backend(), NonDefaultStream, () => calls++, opts);
        step.Prepare();
        Assert.Equal(3, calls); // warmup ran the forward exactly WarmupIterations times (no GPU)
    }

    [Fact]
    public void Prepare_ZeroWarmup_RunsForwardNever()
    {
        int calls = 0;
        var opts = new GraphedInferenceStepOptions { WarmupIterations = 0 };
        using var step = new GraphedInferenceStep(Backend(), NonDefaultStream, () => calls++, opts);
        step.Prepare();
        Assert.Equal(0, calls);
    }

    [Fact]
    public void Replay_BeforeCapture_Throws()
    {
        using var step = new GraphedInferenceStep(Backend(), NonDefaultStream, () => { });
        var ex = Assert.Throws<InvalidOperationException>(() => step.Replay());
        Assert.Contains("Capture", ex.Message);
    }

    [Fact]
    public void Replay_NullRebind_Throws()
    {
        using var step = new GraphedInferenceStep(Backend(), NonDefaultStream, () => { });
        Assert.Throws<ArgumentNullException>(() => step.Replay(null)); // null-check before capture-state
    }

    [Fact]
    public void HasGraph_FalseBeforeCapture_ReplayCountZero()
    {
        using var step = new GraphedInferenceStep(Backend(), NonDefaultStream, () => { });
        Assert.False(step.HasGraph);
        Assert.Equal(0, step.ReplayCount);
    }

    [Fact]
    public void Prepare_AfterDispose_Throws()
    {
        var step = new GraphedInferenceStep(Backend(), NonDefaultStream, () => { });
        step.Dispose();
        Assert.Throws<ObjectDisposedException>(() => step.Prepare());
    }
#endif
}
