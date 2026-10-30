// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// cuDNN binds its stream to the handle (cudnnSetStream), and the backend's handle is shared by every thread. A launch
/// is bind-then-launch, so without a lock spanning both, another thread could rebind the handle in between: one
/// thread's convolution then ran on the other's stream - inside that thread's step capture. Measured on the CUDA lane:
/// "cuDNN ConvolutionForward failed: CUDNN_STATUS_EXECUTION_FAILED_CUDART", with cuDNN's log reporting CUDA 900
/// "operation not permitted when stream is capturing" on one thread and 901 on the capturer, in a different conv test
/// each run. Here one thread trains captured convolution plans while another runs eager convolutions on the same engine.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class CudnnStreamBindingRaceTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public CudnnStreamBindingRaceTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Make(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < data.Length; i++) data[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<float>(data, shape);
    }

    [SkippableFact]
    public void Eager_convolutions_on_another_thread_do_not_break_a_captured_step()
    {
        Skip.IfNot(_fixture.IsAvailable, "No DirectGpu backend available.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "cuDNN is CUDA-only.");
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0", "Graph capture is disabled.");

        // The reference candidate of the cuDNN vendor slot is cuDNN itself; tuning could route a shape elsewhere.
        var previousMode = TunedKernelPolicy.ModeOverride;
        TunedKernelPolicy.ModeOverride = TunedKernelMode.Off;
        try { Run(gpu); }
        finally { TunedKernelPolicy.ModeOverride = previousMode; }
    }

    private static void Run(DirectGpuTensorEngine gpu)
    {
        var ex = Make(new[] { 4, 8, 16, 16 }, 21, 1f);
        var eb = Make(new[] { 16 }, 23, 0.1f);
        var ew = Make(new[] { 16, 8, 3, 3 }, 22, 0.2f);
        var expected = gpu.FusedConv2D(ex, ew, eb, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();

        Exception? eagerError = null;
        int eagerRuns = 0;
        using var done = new ManualResetEventSlim(false);
        var eager = new Thread(() =>
        {
            try
            {
                AiDotNetEngine.Current = gpu;
                while (!done.IsSet)
                {
                    var y = gpu.FusedConv2D(ex, ew, eb, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();
                    for (int i = 0; i < y.Length; i++)
                        if (Math.Abs(y[i] - expected[i]) > 1e-4f * Math.Max(1f, Math.Abs(expected[i])))
                            throw new InvalidOperationException($"eager conv element {i}: {y[i]} != {expected[i]}");
                    eagerRuns++;
                }
            }
            catch (Exception e) { eagerError = e; }
        }) { IsBackground = true };

        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        eager.Start();
        try
        {
            // Each plan captures its step on the second Step(); fresh plans capture again.
            for (int round = 0; round < 40 && eagerError is null; round++)
            {
                var x = Make(new[] { 2, 4, 16, 16 }, 1 + round, 1f).Gpu();
                var w0 = Make(new[] { 16, 4, 3, 3 }, 2, 0.3f).Gpu();
                var w1 = Make(new[] { 8, 16, 3, 3 }, 3, 0.1f).Gpu();
                ICompiledTrainingPlan<float> plan;
                using (var scope = GraphMode.Enable())
                {
                    var h = gpu.ReLU(gpu.Conv2D(x, w0, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }));
                    var o = gpu.Conv2D(h, w1, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 });
                    gpu.ReduceSum(gpu.TensorMultiply(o, o), new[] { 0, 1, 2, 3 }, keepDims: false);
                    plan = scope.CompileTraining(new[] { w0, w1 });
                }
                using (plan)
                    for (int st = 0; st < 4; st++)
                    {
                        plan.Step();
                        foreach (var g in plan.Gradients[1].ToArray())
                            Assert.False(float.IsNaN(g) || float.IsInfinity(g), $"round {round} step {st}: non-finite gradient");
                    }
            }
        }
        finally
        {
            done.Set();
            eager.Join();
            AiDotNetEngine.Current = previous;
        }

        Assert.True(eagerError is null, $"the eager thread failed: {eagerError}");
        Assert.True(eagerRuns > 0, "the eager thread never ran a convolution");
    }
}
