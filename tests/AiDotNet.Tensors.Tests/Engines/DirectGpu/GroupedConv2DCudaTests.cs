using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class GroupedConv2DCudaTests
{
    private static (float[] Y, float[] Dx, float[] Dw, float[] X, float[] W, float[] Dy) Reference(GroupedConv2DGeometry g)
    {
        var cpu = new CpuEngine();
        using var input = GroupedConv2DSemanticTests.Values<float>(g.InputShape);
        using var kernel = GroupedConv2DSemanticTests.Values<float>(g.KernelShape, .71);
        using var dy = GroupedConv2DSemanticTests.Values<float>(g.OutputShape, .37);
        using var tape = new GradientTape<float>(); tape.BindEngineIfUnset(cpu);
        var y = GroupedConvolution.SplitForward(cpu, input, kernel, g);
        var loss = cpu.ReduceSum(cpu.TensorMultiply(y, dy), null);
        var gradients = tape.ComputeGradients(loss, new[] { input, kernel });
        return (y.ToArray(), gradients[input].ToArray(), gradients[kernel].ToArray(), input.ToArray(), kernel.ToArray(), dy.ToArray());
    }

    private static void NativeCase(CudaBackend backend, GroupedConv2DGeometry g, bool vendor)
    {
        var reference = Reference(g); const float sentinel = 123.75f;
        var native = Assert.IsAssignableFrom<IGroupedConv2DBackend>(backend);
        using var input = backend.AllocateBuffer(reference.X); using var kernel = backend.AllocateBuffer(reference.W);
        using var dy = backend.AllocateBuffer(reference.Dy);
        using var y = backend.AllocateBuffer(Enumerable.Repeat(sentinel, g.OutputCount + 7).ToArray());
        using var dx = backend.AllocateBuffer(Enumerable.Repeat(sentinel, g.InputCount + 7).ToArray());
        using var dw = backend.AllocateBuffer(Enumerable.Repeat(sentinel, g.KernelCount + 7).ToArray());
        long fwd = backend.CudnnForwardExecutions, data = backend.CudnnBackwardDataExecutions,
            filter = backend.CudnnBackwardFilterExecutions, launches = GpuLaunchProbe.Count, readbacks = GpuLaunchProbe.ReadbackBytes;
        Assert.True(native.TryGroupedConv2D(input, kernel, y, g));
        Assert.True(native.TryGroupedConv2DBackwardInput(dy, kernel, dx, g, false));
        Assert.True(native.TryGroupedConv2DBackwardKernel(input, dy, dw, g, false));
        Assert.Equal(readbacks, GpuLaunchProbe.ReadbackBytes);
        void Check(IGpuBuffer value, float[] expected, int logical)
        {
            var actual = backend.DownloadBuffer(value);
            GroupedConv2DSemanticTests.Close(expected, actual.Take(logical), 2e-4);
            Assert.All(actual.Skip(logical), item => Assert.Equal(sentinel, item));
        }
        Check(y, reference.Y, g.OutputCount); Check(dx, reference.Dx, g.InputCount); Check(dw, reference.Dw, g.KernelCount);
        Assert.True(native.TryGroupedConv2DBackwardInput(dy, kernel, dx, g, true));
        Assert.True(native.TryGroupedConv2DBackwardKernel(input, dy, dw, g, true));
        Check(dx, reference.Dx.Select(value => value * 2).ToArray(), g.InputCount);
        Check(dw, reference.Dw.Select(value => value * 2).ToArray(), g.KernelCount);
        Assert.Equal(reference.X, backend.DownloadBuffer(input)); Assert.Equal(reference.W, backend.DownloadBuffer(kernel));
        Assert.Equal(reference.Dy, backend.DownloadBuffer(dy));
        if (vendor)
        {
            Assert.Equal(fwd + 1, backend.CudnnForwardExecutions);
            Assert.Equal(data + 2, backend.CudnnBackwardDataExecutions);
            Assert.Equal(filter + 2, backend.CudnnBackwardFilterExecutions);
        }
        else
        {
            Assert.Equal(fwd, backend.CudnnForwardExecutions); Assert.Equal(data, backend.CudnnBackwardDataExecutions);
            Assert.Equal(filter, backend.CudnnBackwardFilterExecutions); Assert.Equal(launches + 5, GpuLaunchProbe.Count);
        }
        long rejected = GpuLaunchProbe.Count;
        Assert.False(native.TryGroupedConv2D(input, kernel, input, g));
        Assert.Equal(rejected, GpuLaunchProbe.Count);
    }

    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Portable_and_cudnn_fwd_dx_dw_overwrite_add_and_readonly_inputs(bool vendor)
    {
        using var backend = new CudaBackend { CudnnExecutionDiagnosticsEnabled = true };
        CudaTestDevice.Require(backend);
        Skip.If(vendor && !CuDnnConvolution.IsAvailable, "cuDNN unavailable.");
        bool previous = TensorCodecOptions.Current.UseCudnn;
        try
        {
            TensorCodecOptions.Current.UseCudnn = vendor;
            NativeCase(backend, new(2, 4, 5, 7, 4, 3, 3, 1, 2, 1, 1, 1, 1, 2), vendor);
            NativeCase(backend, new(1, 128, 1, 65, 256, 1, 41, 1, 2, 0, 20, 1, 1, 16), vendor);
        }
        finally { TensorCodecOptions.Current.UseCudnn = previous; }
    }

    [SkippableFact]
    public void Cudnn_cache_distinguishes_same_full_shape_with_different_group_counts()
    {
        using var backend = new CudaBackend { CudnnExecutionDiagnosticsEnabled = true };
        CudaTestDevice.Require(backend);
        Skip.IfNot(CuDnnConvolution.IsAvailable, "cuDNN unavailable.");
        bool previous = TensorCodecOptions.Current.UseCudnn, previousHeuristic = CuDnnConvolution.HeuristicForwardEnabled;
        try
        {
            TensorCodecOptions.Current.UseCudnn = true; CuDnnConvolution.HeuristicForwardEnabled = true;
            NativeCase(backend, new(2, 4, 5, 7, 4, 3, 3, 1, 2, 1, 1, 1, 1, 1), true);
            NativeCase(backend, new(2, 4, 5, 7, 4, 3, 3, 1, 2, 1, 1, 1, 1, 2), true);
            NativeCase(backend, new(2, 4, 5, 7, 4, 3, 3, 1, 2, 1, 1, 1, 1, 1), true);
            var convolution = typeof(CudaBackend).GetField("_cudnnConv", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(backend)
                as CuDnnConvolution ?? throw new InvalidOperationException("No actual grouped cuDNN context.");
            var cache = typeof(CuDnnConvolution).GetField("_heuristicForward", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(convolution)
                as IDictionary ?? throw new InvalidOperationException("No group-aware native heuristic cache.");
            Assert.Equal(2, cache.Count);
        }
        finally { TensorCodecOptions.Current.UseCudnn = previous; CuDnnConvolution.HeuristicForwardEnabled = previousHeuristic; }
    }

    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Eager_and_compiled_cuda_gradients_and_updates_match_split_control(bool compile)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable && engine.GetBackend() is CudaBackend, "CUDA unavailable.");
        var g = new GroupedConv2DGeometry(2, 8, 1, 17, 8, 1, 5, 1, 2, 0, 2, 1, 1, 4);
        var expected = GroupedConv2DSemanticTests.Trajectory<float>(engine, g, false, compile);
        var actual = GroupedConv2DSemanticTests.Trajectory<float>(engine, g, true, compile);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i) GroupedConv2DSemanticTests.Close(expected[i], actual[i], 2e-4);
    }

    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Recorded_grouped_node_preserves_public_reads_and_private_retirement_across_training(bool inputGradients)
    {
#if INFERENCE_WORKSPACE_DEMAND
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable && engine.GetBackend() is CudaBackend, "CUDA unavailable.");
        var previousEngine = AiDotNetEngine.Current;
        bool? previousInputGradients = AiDotNet.Tensors.Engines.Compilation.CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        string previous = Environment.GetEnvironmentVariable("AIDOTNET_INFERENCE_PRIVATE_GRADIENT_RETURN");
        try
        {
            AiDotNetEngine.Current = engine;
            AiDotNet.Tensors.Engines.Compilation.CompiledTrainingPlan<float>.CompiledInputGradientsForTests = inputGradients;
            List<float[]> Run(bool release, out long released)
            {
                Environment.SetEnvironmentVariable("AIDOTNET_INFERENCE_PRIVATE_GRADIENT_RETURN", release ? null : "0");
                var g = new GroupedConv2DGeometry(1, 4, 1, 9, 4, 1, 3, 1, 1, 0, 1, 1, 1, 2);
                using var input = GroupedConv2DSemanticTests.Values<float>(g.InputShape);
                using var kernel = GroupedConv2DSemanticTests.Values<float>(g.KernelShape, .71);
                AiDotNet.Tensors.Engines.Compilation.CompiledTrainingPlan<float> plan;
                using (var scope = AiDotNet.Tensors.Engines.Compilation.GraphMode.EnableTraining(new[] { kernel }))
                {
                    var value = GroupedConvolution.Forward(engine, input, kernel, g.Stride, g.Padding, g.Dilation, g.Groups);
                    for (int layer = 0; layer < 4; ++layer) value = engine.Tanh(engine.TensorMultiplyScalar(value, .9f));
                    var loss = engine.ReduceMean(engine.TensorMultiply(value, value), new[] { 0, 1, 2, 3 }, false);
                    plan = Assert.IsType<AiDotNet.Tensors.Engines.Compilation.CompiledTrainingPlan<float>>(scope.CompileTraining(new[] { kernel }, loss));
                }
                using (plan)
                {
                    plan.DisableGraphStep(); plan.ConfigureOptimizer(AiDotNet.Tensors.Engines.Compilation.OptimizerType.Adam, learningRate: .001);
                    var samples = new List<float[]>();
                    for (int step = 0; step < 4; ++step)
                    {
                        input[0] = .1f + .01f * step;
                        var loss = plan.Step(); var gradient = plan.Gradients[0];
                        using var publicView = gradient.Reshape(gradient.Length);
                        var publicBuffer = gradient._gpuBuffer;
                        using (new NoGradScope<float>())
                        using (InferenceWorkspaceDemandScope.Enter(engine))
                        using (var prediction = engine.TensorAdd(new Tensor<float>(new[] { 4096 }), new Tensor<float>(new[] { 4096 })))
                            Assert.Equal(4096, prediction.Length);
                        GC.Collect(); GC.WaitForPendingFinalizers();
                        Assert.Same(publicBuffer, gradient._gpuBuffer);
                        if (publicBuffer is not null) Assert.NotEqual(IntPtr.Zero, publicBuffer.Handle);
                        samples.Add(loss.ToArray()); samples.Add(publicView.ToArray()); samples.Add(kernel.ToArray());
                        if (inputGradients) samples.Add(input.Grad?.ToArray() ?? throw new InvalidOperationException("Input gradient lost."));
                    }
                    released = plan.PrivateGradientNativeReleaseRequestedBytes;
                    return samples;
                }
            }
            var baseline = Run(false, out long baselineReleased); var actual = Run(true, out long released);
            Assert.Equal(0, baselineReleased);
            Assert.True(released > 0, "A recorded grouped node must preserve real private-gradient retirement eligibility.");
            Assert.Equal(baseline.Count, actual.Count);
            for (int i = 0; i < baseline.Count; ++i)
                Assert.Equal(baseline[i].Select(BitConverter.SingleToInt32Bits), actual[i].Select(BitConverter.SingleToInt32Bits));
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_INFERENCE_PRIVATE_GRADIENT_RETURN", previous);
            AiDotNet.Tensors.Engines.Compilation.CompiledTrainingPlan<float>.CompiledInputGradientsForTests = previousInputGradients;
            AiDotNetEngine.Current = previousEngine;
        }
#else
        Skip.If(true, "The grouped convolution and inference workspace integration test requires both implementations.");
#endif
    }

    [SkippableFact]
    public void Aliased_primal_targets_keep_original_values_for_both_native_gradients()
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable && engine.GetBackend() is CudaBackend, "CUDA unavailable.");
        var g = new GroupedConv2DGeometry(2, 4, 1, 9, 4, 1, 1, 1, 1, 0, 0, 1, 1, 2);
        var reference = Reference(g);
        using var input = new Tensor<float>(reference.X, g.InputShape);
        using var kernel = new Tensor<float>(reference.W, g.KernelShape);
        using var dy = new Tensor<float>(reference.Dy, g.OutputShape);
        GroupedConvolution.BackwardInto(engine, dy, input, kernel, input, kernel, g, false, false);
        GroupedConv2DSemanticTests.Close(reference.Dx, input.ToArray(), 2e-4);
        GroupedConv2DSemanticTests.Close(reference.Dw, kernel.ToArray(), 2e-4);
        Assert.Equal(reference.Dy, dy.ToArray());
    }

    [SkippableFact]
    public void Unread_native_output_view_survives_producer_disposal_GC_and_unrelated_pool_activity()
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable && engine.GetBackend() is CudaBackend, "CUDA unavailable.");
        var g = new GroupedConv2DGeometry(2, 4, 1, 9, 4, 1, 3, 1, 1, 0, 1, 1, 1, 2);
        var reference = Reference(g);
        using var input = new Tensor<float>(reference.X, g.InputShape);
        using var kernel = new Tensor<float>(reference.W, g.KernelShape);
        var result = GroupedConvolution.Forward(engine, input, kernel, g.Stride, g.Padding, g.Dilation, g.Groups);
        using var view = result.Reshape(result.Length);
        result.Dispose(); GC.Collect(); GC.WaitForPendingFinalizers();
        for (int i = 0; i < 4; ++i)
        {
            using var other = engine.TensorAdd(new Tensor<float>(g.OutputShape), new Tensor<float>(g.OutputShape));
            Assert.Equal(g.OutputCount, other.Length);
        }
        GroupedConv2DSemanticTests.Close(reference.Y, view.ToArray(), 2e-4);
        Assert.Equal(reference.X, input.ToArray()); Assert.Equal(reference.W, kernel.ToArray());
    }
}
