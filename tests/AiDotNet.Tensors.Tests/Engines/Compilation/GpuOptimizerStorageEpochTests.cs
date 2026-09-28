#nullable disable
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The one configuration where host weight-derived caches (here the fused W·W2 product, keyed on the storage epoch)
/// coexist with an on-device optimizer update: a GPU plan compiled with host specializations and cross-layer fusion
/// (AIDOTNET_GPU_PREFER_GENERIC=0, AIDOTNET_CROSS_LAYER_FUSION=1) over GPU-resident parameters. The update runs on the
/// device (the CPU updater refuses a GPU-device tensor), and training must still match the same plan on the CPU engine.
/// </summary>
/// <remarks>
/// Advancing the storage epoch after the device write, the obvious way to invalidate those caches, is wrong: the same
/// epoch gates the GPU persistent weight cache, which then re-uploads the stale host copy over the updated device
/// weights (FP16-resident training went flat at 7.7022). This pins that the current bookkeeping already trains correctly.
/// </remarks>
[Collection(MixedPrecisionTestCollection.Name)]
public class GpuOptimizerStorageEpochTests
{
    private const int TrainingSteps = 6;
    private const float LearningRate = 0.01f;
    // Relative loss agreement between the two plans; the GPU and CPU reductions differ only by float reordering.
    private const float RelativeLossTolerance = 1e-3f;
    private static float[] Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() - 0.5);
        return d;
    }

    private static float[] Train(IEngine engine, bool residentParams, OptimizerType optimizer, int steps)
    {
        var input = new Tensor<float>(Rand(8 * 16, 1), new[] { 8, 16 });
        var w = new Tensor<float>(Rand(16 * 16, 2), new[] { 16, 16 });
        var w2 = new Tensor<float>(Rand(16 * 8, 3), new[] { 16, 8 });
        if (residentParams) { w.Gpu(); w2.Gpu(); }

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            // x·W·W2: with cross-layer fusion the plan caches the W·W2 product, keyed on the weights' storage epoch.
            var y = engine.TensorMatMul(engine.TensorMatMul(input, w), w2);
            engine.ReduceSum(engine.TensorMultiply(y, y), null);
            plan = scope.CompileTraining(new[] { w, w2 });
        }
        GraphMode.SetCurrent(null);

        var losses = new float[steps];
        using (plan)
        {
            plan.ConfigureOptimizer(optimizer, learningRate: LearningRate);
            for (int s = 0; s < steps; s++) losses[s] = plan.Step().GetFlat(0);
        }
        return losses;
    }

    [SkippableTheory]
    [InlineData(OptimizerType.Adam)]
    [InlineData(OptimizerType.SGD)]
    public void HostSpecializedGpuPlan_WithResidentParams_TrainsLikeTheCpuPlan(OptimizerType optimizer)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");

        string prior = Environment.GetEnvironmentVariable("AIDOTNET_GPU_PREFER_GENERIC");
        string priorFusion = Environment.GetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION");
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_PREFER_GENERIC", "0");
        Environment.SetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION", "1");
        try
        {
            var reference = Train(new CpuEngine(), residentParams: false, optimizer, TrainingSteps);
            var resident = Train(gpu, residentParams: true, optimizer, TrainingSteps);

            Assert.True(reference[TrainingSteps - 1] < reference[0], $"{optimizer}: the CPU reference did not train.");
            for (int s = 0; s < reference.Length; s++)
                Assert.True(Math.Abs(resident[s] - reference[s]) <= RelativeLossTolerance * Math.Max(1f, Math.Abs(reference[s])),
                    $"{optimizer} step {s}: host-specialized GPU plan loss {resident[s]} vs CPU plan {reference[s]}");
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_GPU_PREFER_GENERIC", prior);
            Environment.SetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION", priorFusion);
        }
    }
}
