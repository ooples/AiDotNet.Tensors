// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NETCOREAPP
using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class StaticShapeCudaGraphScopeTests
{
    [Theory]
    [InlineData(CudaGraphExecutionScopeKind.InferenceForward, true)]
    [InlineData(CudaGraphExecutionScopeKind.FullStepForwardBackward, true)]
    [InlineData(CudaGraphExecutionScopeKind.SegmentedLocalCreditBlock, false)]
    [InlineData(CudaGraphExecutionScopeKind.MultiTensorOptimizerStep, true)]
    public void AllFourScopes_WarmupCaptureReplay_AndInvalidateAutomaticallyOnPointerOrShapeOrScalarChange(
        CudaGraphExecutionScopeKind kind,
        bool synchronizeOnReplay)
    {
        var harness = FakeGraphCaptureHarness.Create();
        var options = new StaticShapeCudaGraphOptions
        {
            WarmupIterations = 2,
            SynchronizeOnReplay = synchronizeOnReplay,
            MaxCaptureFailuresBeforeFallback = 2
        };

        using var scope = new StaticShapeCudaGraphScope(harness.Backend, kind, options);
        Assert.Equal(kind, scope.ScopeKind);
        Assert.True(scope.IsGraphCaptureSupported);

        var bufIn = new MutableFakeGpuBuffer(new IntPtr(0x1000), 256);
        var bufWeight = new MutableFakeGpuBuffer(new IntPtr(0x2000), 512);
        var bufOut = new MutableFakeGpuBuffer(new IntPtr(0x3000), 256);
        var watched = new IGpuBuffer[] { bufIn, bufWeight, bufOut };
        long sig = StaticShapeCudaGraphScope.PackScalarSignature(1e-3f, 0.01f, intParam0: 1);

        int hostBodyRuns = 0;
        void StepBody() => hostBodyRuns++;

        // Warmup 1 & 2
        scope.Execute(StepBody, watched, sig);
        scope.Execute(StepBody, watched, sig);
        Assert.Equal(2, hostBodyRuns);
        Assert.Equal(2, scope.WarmupExecutionCount);
        Assert.False(scope.HasCapturedGraph);

        // Step 3: Capture + immediate replay launch
        scope.Execute(StepBody, watched, sig);
        Assert.Equal(3, hostBodyRuns);
        Assert.True(scope.HasCapturedGraph);
        Assert.Equal(1, scope.CaptureCount);
        Assert.Equal(1, scope.GraphReplayCount);

        // Steps 4 & 5: Pure CUDA Graph replay — host StepBody does NOT re-execute!
        scope.Execute(StepBody, watched, sig);
        scope.Execute(StepBody, watched, sig);
        Assert.Equal(3, hostBodyRuns);
        Assert.Equal(3, scope.GraphReplayCount);
        if (synchronizeOnReplay)
        {
            Assert.Equal(3, harness.SyncLaunches);
            Assert.Equal(0, harness.AsyncEnqueues);
        }
        else
        {
            Assert.Equal(0, harness.SyncLaunches);
            Assert.Equal(3, harness.AsyncEnqueues);
        }

        // Trigger 1: Pointer change on bufIn invalidates old graph, destroys it, and restarts warmup -> capture
        bufIn.Handle = new IntPtr(0x9000);
        scope.Execute(StepBody, watched, sig);
        Assert.Equal(1, scope.InvalidationCount);
        Assert.Equal(1, harness.DestroyedGraphs);
        Assert.False(scope.HasCapturedGraph);
        Assert.Contains("pointer", scope.LastInvalidationReason ?? string.Empty, StringComparison.OrdinalIgnoreCase);

        // Finish warmup 2 + recapture
        scope.Execute(StepBody, watched, sig);
        scope.Execute(StepBody, watched, sig);
        Assert.True(scope.HasCapturedGraph);
        Assert.Equal(2, scope.CaptureCount);

        // Trigger 2: Logical buffer size change invalidates automatically
        bufOut.Size = 192;
        scope.Execute(StepBody, watched, sig);
        Assert.Equal(2, scope.InvalidationCount);
        Assert.Equal(2, harness.DestroyedGraphs);
        Assert.False(scope.HasCapturedGraph);

        // Trigger 3: Scalar hyperparameter signature change after recapture
        scope.Execute(StepBody, watched, sig);
        scope.Execute(StepBody, watched, sig);
        Assert.True(scope.HasCapturedGraph);
        long nextSig = StaticShapeCudaGraphScope.PackScalarSignature(5e-4f, 0.01f, intParam0: 2);
        scope.Execute(StepBody, watched, nextSig);
        Assert.Equal(3, scope.InvalidationCount);
        Assert.Equal(3, harness.DestroyedGraphs);
    }

    [Fact]
    public void CaptureFailure_ExecutesStepEagerlyWithoutDroppingWork_AndStopsRetryingAfterBudget()
    {
        var harness = FakeGraphCaptureHarness.Create();
        harness.FailCaptures = true;

        using var scope = new StaticShapeCudaGraphScope(
            harness.Backend,
            CudaGraphExecutionScopeKind.FullStepForwardBackward,
            new StaticShapeCudaGraphOptions { WarmupIterations = 1, MaxCaptureFailuresBeforeFallback = 2 });

        var watched = new IGpuBuffer[] { new MutableFakeGpuBuffer(new IntPtr(0x100), 64) };
        int workCompleted = 0;

        // Call 1: warmup
        scope.Execute(() => workCompleted++, watched, 7L);
        Assert.Equal(1, workCompleted);

        // Call 2: capture attempt 1 fails -> runs eagerly so workCompleted becomes 2
        scope.Execute(() => workCompleted++, watched, 7L);
        Assert.Equal(2, workCompleted);
        Assert.Equal(1, harness.CaptureAttempts);
        Assert.False(scope.HasCapturedGraph);

        // Call 3: capture attempt 2 fails -> hits MaxCaptureFailuresBeforeFallback (2), still runs eagerly
        scope.Execute(() => workCompleted++, watched, 7L);
        Assert.Equal(3, workCompleted);
        Assert.Equal(2, harness.CaptureAttempts);

        // Call 4: stays on eager fallback without attempting CaptureGraph again for this signature
        scope.Execute(() => workCompleted++, watched, 7L);
        Assert.Equal(4, workCompleted);
        Assert.Equal(2, harness.CaptureAttempts);
        Assert.Equal(3, scope.EagerFallbackCount);
    }

    [Fact]
    public void GpuCompiledPlan_CapturesReplaysAndInvalidatesOnResidentTensors()
    {
        var harness = FakeGraphCaptureHarness.Create();
        var engine = new CpuEngine();

        var inputBuf = new MutableFakeGpuBuffer(new IntPtr(0xA000), 16);
        var outputBuf = new MutableFakeGpuBuffer(new IntPtr(0xB000), 16);
        var input = Tensor<float>.FromGpuBuffer(harness.Backend, inputBuf, new[] { 4, 4 }, AiDotNet.Tensors.Engines.Gpu.GpuTensorRole.Activation, ownsBuffer: false);
        var output = Tensor<float>.FromGpuBuffer(harness.Backend, outputBuf, new[] { 4, 4 }, AiDotNet.Tensors.Engines.Gpu.GpuTensorRole.Activation, ownsBuffer: false);

        int eagerStepExecutions = 0;
        var step = new CompiledStep<float>(
            "FusedSwiGlu",
            (_, _) => eagerStepExecutions++,
            output,
            new[] { input },
            null,
            null);

        using var plan = new GpuCompiledPlan<float>(new[] { step }, output, engine, harness.Backend);
        Assert.False(plan.IsCaptured);

        // Call 1: Warmup (1 iteration)
        plan.Execute();
        Assert.Equal(1, eagerStepExecutions);
        Assert.False(plan.IsCaptured);

        // Call 2: Captures + launches graph
        plan.Execute();
        Assert.Equal(2, eagerStepExecutions);
        Assert.True(plan.IsCaptured);
        Assert.Equal(1, harness.SyncLaunches);

        // Call 3: Replays CUDA graph without running host step closure
        plan.Execute();
        Assert.Equal(2, eagerStepExecutions);
        Assert.Equal(2, harness.SyncLaunches);

        // Mutate device pointer -> invalidates and falls back to warmup
        inputBuf.Handle = new IntPtr(0xC000);
        plan.Execute();
        Assert.False(plan.IsCaptured);
        Assert.Equal(1, harness.DestroyedGraphs);
    }

    [Fact]
    public void Gpu1Ready_CudaGraphAndFusedKernelsEndToEndVerification()
    {
        // Guarded strictly off by default while GPU 1 is running production workloads.
        // Set AIDOTNET_RUN_GPU1_VERIFICATION=1 once GPU 1 is free to run end-to-end hardware verification.
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_GPU1_VERIFICATION") != "1")
        {
            return;
        }

        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend();
        if (!backend.IsAvailable)
        {
            return;
        }

        using var direct = new DirectGpuEngine(backend);
        using var gpuEngine = new DirectGpuTensorEngine(direct);
        using var scope = new StaticShapeCudaGraphScope(backend, CudaGraphExecutionScopeKind.InferenceForward);

        using var b0 = backend.AllocateBuffer(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f });
        using var b1 = backend.AllocateBuffer(new float[] { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f });
        using var bGamma = backend.AllocateBuffer(new float[] { 1f, 1f, 1f, 1f });
        using var bOut = backend.AllocateBuffer(8);
        using var bRms = backend.AllocateBuffer(2);

        var watched = new IGpuBuffer[] { b0, b1, bGamma, bOut, bRms };
        for (int iter = 0; iter < 5; iter++)
        {
            scope.Execute(
                () => Assert.True(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.TryDispatchRmsNormResidual(
                    backend, b0, b1, bGamma, bOut, bRms, 2, 4, 1e-5f)),
                watched);
        }

        Assert.True(scope.HasCapturedGraph);
        Assert.Equal(1, scope.CaptureCount);
        Assert.True(scope.GraphReplayCount >= 3);
    }

    private sealed class MutableFakeGpuBuffer : IGpuBuffer
    {
        public MutableFakeGpuBuffer(IntPtr handle, int size)
        {
            Handle = handle;
            Size = size;
        }

        public GpuElementType ElementType => GpuElementType.Float32;
        public int Size { get; set; }
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle { get; set; }
        public void Dispose() => Handle = IntPtr.Zero;
    }

    private class FakeGraphCaptureProxy : DispatchProxy
    {
        internal FakeGraphCaptureHarness? Harness { get; set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            if (targetMethod is null || Harness is null)
                return null;

            switch (targetMethod.Name)
            {
                case "get_SupportsGraphCapture":
                    return Harness.SupportsGraphCapture;
                case "get_ContextIsLive":
                    return Harness.ContextIsLive;
                case "get_CaptureActiveOnThisThread":
                    return false;
                case "IsStreamCapturing":
                    return false;
                case "StreamCaptureStatusRaw":
                    return 0;
                case "get_LastCaptureFailure":
                    return Harness.LastFailure;
                case "get_LastCaptureNodeCounts":
                    return new GraphNodeCounts(4, 0, 0);
                case "CaptureGraph":
                {
                    Harness.CaptureAttempts++;
                    if (Harness.FailCaptures)
                    {
                        Harness.LastFailure = "Simulated non-capturable operation.";
                        return IntPtr.Zero;
                    }
                    if (args is { Length: > 0 } && args[0] is Action action)
                    {
                        action();
                    }
                    Harness.LastFailure = null;
                    return new IntPtr(0x5000 + Harness.CaptureAttempts);
                }
                case "LaunchCapturedGraph":
                    Harness.SyncLaunches++;
                    return null;
                case "EnqueueCapturedGraph":
                    Harness.AsyncEnqueues++;
                    return null;
                case "DestroyCapturedGraph":
                    Harness.DestroyedGraphs++;
                    return null;
                case "get_IsAvailable":
                    return true;
                case "get_BackendName":
                    return "FakeGraphCapture";
                case "get_DeviceName":
                    return "SyntheticCPU";
                default:
                    return targetMethod.ReturnType.IsValueType
                        ? Activator.CreateInstance(targetMethod.ReturnType)
                        : null;
            }
        }
    }

    private sealed class FakeGraphCaptureHarness
    {
        public IGpuGraphCapture Backend { get; private set; }
        public bool SupportsGraphCapture { get; set; } = true;
        public bool ContextIsLive { get; set; } = true;
        public bool FailCaptures { get; set; }
        public string? LastFailure { get; set; }
        public int CaptureAttempts { get; set; }
        public int SyncLaunches { get; set; }
        public int AsyncEnqueues { get; set; }
        public int DestroyedGraphs { get; set; }

        private FakeGraphCaptureHarness(IGpuGraphCapture backend)
        {
            Backend = backend;
        }

        public static FakeGraphCaptureHarness Create()
        {
            var proxy = DispatchProxy.Create<IGpuGraphCapture, FakeGraphCaptureProxy>();
            var harness = new FakeGraphCaptureHarness(proxy);
            if ((object)proxy is FakeGraphCaptureProxy impl)
            {
                impl.Harness = harness;
            }
            return harness;
        }
    }
}
#endif
