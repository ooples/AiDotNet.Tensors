using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A fused BatchNorm + ReLU node whose channels are larger than the register kernel takes (batch x spatial above 8192)
/// normalizes and then applies the ReLU. A compiled step keeps one buffer per output an action allocates, so a ReLU
/// written to a second buffer held the normalized values all step as well: DenseNet-121's first two blocks, 0.8 GB.
/// The ReLU runs in place, and nothing is rented for the register kernel before it declines the shape, so the fused
/// node's action keeps one activation-sized buffer; the plan trains like BatchNorm then ReLU unfused.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class FusedBatchNormReluFallbackScratchTests : IClassFixture<DirectGpuTensorEngineTestFixture>, IDisposable
{
    // 4 x 48 x 48 = 9216 values per channel: past the register kernel's 8192.
    private const int Batch = 4, Channels = 4, Size = 48, Steps = 3;
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public FusedBatchNormReluFallbackScratchTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static (double[] Losses, int FusedActionActivations, string[] Steps) Train(IEngine engine)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([Batch, Channels, Size, Size], 1, 1f);
        var y = Rand([Batch, Channels], 2, 1f);
        var w = Rand([Channels, Channels, 3, 3], 3, 0.3f);
        var gamma = Rand([Channels], 4, 1f);
        var beta = Rand([Channels], 5, 0.5f);
        var parameters = new[] { w, gamma, beta };
        foreach (var p in parameters) p.Gpu();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.FusedConv2D(x, w, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
            h = engine.ReLU(engine.BatchNorm(h, gamma, beta, 1e-5, out _, out _));
            engine.TensorMSELoss(engine.ReduceMean(h, new[] { 2, 3 }, keepDims: false), y);
            plan = scope.CompileTraining(parameters);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05f);
            var losses = new double[Steps];
            for (int s = 0; s < Steps; s++) losses[s] = plan.Step().ToArray()[0];
            var concrete = plan as CompiledTrainingPlan<float>;
            Assert.NotNull(concrete);
            var steps = concrete.ForwardStepsForSerialization?.Select(st => st.OpName).ToArray() ?? Array.Empty<string>();
            return (losses, ActivationSlots(concrete, Array.IndexOf(steps, "BatchNormReLU")), steps);
        }
    }

    // The activation-sized buffers the plan's scratch pool keeps for forward action <paramref name="action"/> (one per
    // buffer the action rents; the key's high half is the action).
    private static int ActivationSlots(CompiledTrainingPlan<float> plan, int action)
    {
        if (action < 0) return -1;
        var pool = typeof(CompiledTrainingPlan<float>).GetField("_scratchPool", BindingFlags.NonPublic | BindingFlags.Instance)
            ?.GetValue(plan);
        var slots = pool?.GetType().GetProperty("Slots", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(pool)
            as System.Collections.Generic.Dictionary<long, IGpuBuffer>;
        if (slots is null) return -1;
        long activationBytes = (long)Batch * Channels * Size * Size * sizeof(float);
        return slots.Count(kv => (kv.Key >> 32) == action && kv.Value.SizeInBytes >= activationBytes);
    }

    [SkippableFact]
    public void FusedFallback_KeepsOneActivationBuffer_AndTrainsLikeUnfused()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");

        var fused = Train(gpu);
        Skip.IfNot(fused.Steps.Contains("BatchNormReLU"), "This engine leaves BatchNorm + ReLU unfused.");
        Skip.If(fused.FusedActionActivations <= 0, "This engine's compiled step draws no scratch from the plan's pool.");

        var pattern = FusionPatternRegistry.Patterns.First(p => p.Name == "BatchNormReLU");
        Assert.True(FusionPatternRegistry.Unregister(pattern.Name));
        (double[] Losses, int FusedActionActivations, string[] Steps) unfused;
        try { unfused = Train(gpu); }
        finally { FusionPatternRegistry.Register(pattern); }
        Assert.DoesNotContain("BatchNormReLU", unfused.Steps);

        for (int s = 0; s < Steps; s++)
            Assert.True(Math.Abs(unfused.Losses[s] - fused.Losses[s]) <= 1e-4 * Math.Max(1, Math.Abs(unfused.Losses[s])),
                $"step {s}: fused loss {fused.Losses[s]:G7} != unfused {unfused.Losses[s]:G7}");

        Assert.True(fused.FusedActionActivations == 1,
            $"the fused BatchNorm + ReLU action keeps {fused.FusedActionActivations} activation-sized buffers, not 1");
    }
}
