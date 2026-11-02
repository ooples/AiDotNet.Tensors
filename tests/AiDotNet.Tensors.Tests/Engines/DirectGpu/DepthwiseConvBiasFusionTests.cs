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
/// A depthwise convolution followed by its channel bias add compiles to one <c>DepthwiseConv2DBias</c> node on a GPU
/// engine. The two nodes kept the raw convolution output beside the biased one for the whole compiled step, though
/// neither backward reads it (MobileNetV2: 17 of them). The fused node writes the convolution into its output and adds
/// the bias there, so its action keeps one activation-sized buffer, and it trains bit for bit like the two nodes.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class DepthwiseConvBiasFusionTests : IClassFixture<DirectGpuTensorEngineTestFixture>, IDisposable
{
    private const int Batch = 4, Channels = 8, Size = 16, Steps = 3;
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public DepthwiseConvBiasFusionTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static (double[] Losses, float[][] Parameters, int FusedActionActivations, string[] Steps) Train(IEngine engine)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([Batch, Channels, Size, Size], 1, 1f);
        var y = Rand([Batch, Channels], 2, 1f);
        var kernel = Rand([Channels, 1, 3, 3], 3, 0.5f);
        var bias = Rand([Channels], 4, 0.5f);
        var parameters = new[] { kernel, bias };
        foreach (var p in parameters) p.Gpu();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.DepthwiseConv2D(x, kernel, new[] { 1, 1 }, new[] { 1, 1 });
            h = engine.TensorChannelBiasAdd(h, bias);
            engine.TensorMSELoss(engine.ReduceMean(engine.ReLU(h), new[] { 2, 3 }, keepDims: false), y);
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
            return (losses, parameters.Select(p => p.ToArray()).ToArray(),
                ActivationSlots(concrete, Array.IndexOf(steps, "DepthwiseConv2DBias")), steps);
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
    public void DepthwiseConvBias_KeepsOneActivationBuffer_AndTrainsLikeUnfused()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");

        var fused = Train(gpu);
        Assert.Contains("DepthwiseConv2DBias", fused.Steps);
        Assert.DoesNotContain("TensorChannelBiasAdd", fused.Steps);

        var pattern = FusionPatternRegistry.Patterns.First(p => p.Name == "DepthwiseConv2DBias");
        Assert.True(FusionPatternRegistry.Unregister(pattern.Name));
        (double[] Losses, float[][] Parameters, int FusedActionActivations, string[] Steps) unfused;
        try { unfused = Train(gpu); }
        finally { FusionPatternRegistry.Register(pattern); }
        Assert.Contains("TensorChannelBiasAdd", unfused.Steps);

        Assert.Equal(unfused.Losses, fused.Losses);
        for (int p = 0; p < unfused.Parameters.Length; p++)
            Assert.Equal(unfused.Parameters[p], fused.Parameters[p]);

        Skip.If(fused.FusedActionActivations < 0, "This engine's compiled step draws no scratch from the plan's pool.");
        Assert.True(fused.FusedActionActivations == 1,
            $"the fused depthwise convolution + bias action keeps {fused.FusedActionActivations} activation-sized buffers, not 1");
    }
}
