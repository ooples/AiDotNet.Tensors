// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A compiled training plan on the GPU ran its device-resident body only inside CUDA-graph capture; every other
/// step (warmup, capture off, capture failed) took the host-centric StepEager, which zeroes the gradient
/// accumulators on the host and so downloads every device gradient it accumulates (~1 GB/step on an LM). Those
/// steps now run the resident body uncaptured.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class ResidentCompiledStepTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public ResidentCompiledStepTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    [SkippableFact]
    public void CompiledMlpStep_ReadsBackOnlyTheLoss()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var x = Filled(new[] { 64, 32 }, 1, 1f);
            // Parameters on the device, as AiDotNet's fused path places them: the optimizer then updates them in
            // place. (Host parameters are updated by the host optimizer, which needs their gradients downloaded.)
            var w1 = Filled(new[] { 32, 64 }, 2, 0.2f).Gpu();
            var w2 = Filled(new[] { 64, 16 }, 3, 0.2f).Gpu();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var h = gpu.TensorTanh(gpu.TensorMatMul(x, w1));
                var y = gpu.TensorMatMul(h, w2);
                gpu.ReduceSum(gpu.TensorMultiply(y, y), null);
                plan = scope.CompileTraining(new[] { w1, w2 });
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-3f);
                var losses = new List<float> { plan.Step().ToArray()[0] };
                long readback;
                string sites;
                bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
                try
                {
                    GpuLaunchProbe.CaptureReadbackSites = true;
                    GpuLaunchProbe.Reset();
                    losses.Add(plan.Step().ToArray()[0]);    // a warmup step: uncaptured on every path
                    readback = GpuLaunchProbe.ReadbackBytes;
                    sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
                }
                finally
                {
                    GpuLaunchProbe.CaptureReadbackSites = savedCapture;
                }
                Assert.True(readback <= 64, $"an uncaptured compiled step read back {readback} bytes: {sites}");
                Assert.True(losses[1] < losses[0], $"the step must still train: {losses[0]} -> {losses[1]}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    /// <summary>
    /// Tensor.Gpu() uploaded through the process-wide DirectGpu backend, a different instance (and CUDA stream)
    /// from the current engine's. AiDotNet's fused path moves every parameter that way, so the fused optimizer
    /// updated them on one stream while the engine's kernels ran on the other.
    /// </summary>
    [SkippableFact]
    public void Gpu_UsesTheCurrentEnginesBackend()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var t = Filled(new[] { 16 }, 4, 1f).Gpu();
            Assert.True(ReferenceEquals(t._gpuBackend, gpu.GetBackend()), "Gpu() bound the tensor to a backend other than the current engine's");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    /// <summary>
    /// With parameters moved by Gpu() (as AiDotNet's fused path does), the global-norm clip scaled the gradients on
    /// the engine's stream while the optimizer read them on the parameters' backend stream -- the update used the
    /// unclipped gradients (measured on an LM: update norm 2.80 with a max of 1.0).
    /// </summary>
    [SkippableFact]
    public void GlobalNormClip_WithGpuParameters_LimitsTheUpdate()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var ws = new[] { Filled(new[] { 1_000_000 }, 5, 1f).Gpu(), Filled(new[] { 4096 }, 6, 1f).Gpu() };
            var cs = new[] { Filled(new[] { 1_000_000 }, 7, 1f), Filled(new[] { 4096 }, 8, 1f) };
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var loss = gpu.TensorAdd(gpu.ReduceSum(gpu.TensorMultiply(ws[0], cs[0]), null),
                                         gpu.ReduceSum(gpu.TensorMultiply(ws[1], cs[1]), null));
                plan = scope.CompileTraining(ws);
            }
            using (plan)
            {
                plan.SetMaxGradNorm(1.0);
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1.0f);
                var before = ws.Select(w => w.ToArray()).ToArray();
                plan.Step();
                double sq = 0;
                for (int p = 0; p < ws.Length; p++)
                {
                    var after = ws[p].ToArray();
                    for (int i = 0; i < after.Length; i++) { double d = after[i] - before[p][i]; sq += d * d; }
                }
                Assert.True(Math.Abs(Math.Sqrt(sq) - 1.0) < 1e-3, $"a clipped SGD step (lr 1, max norm 1) moved the parameters by {Math.Sqrt(sq):G6}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
