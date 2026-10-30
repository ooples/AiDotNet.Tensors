// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Two device paths that the GPU training work routes through on every backend, not only CUDA: the linear backward's
/// transposed-flag GEMMs (no materialized Wᵀ or Xᵀ), and the compiled plan's reshape gradient aliasing on a GPU
/// engine, where a reshape's input gradient becomes a view bound to the output gradient's device buffer. Each is
/// checked against the CPU engine, so a wrong transpose flag or a view bound to the wrong buffer shows as a value.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GpuReshapeAliasAndLinearBackwardTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public GpuReshapeAliasAndLinearBackwardTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[i])),
                $"{what}[{i}] cpu {expected[i]} gpu {actual[i]}");
    }

    /// <summary>Runs the linear backward on <paramref name="engine"/> and returns the gradients it produced, keyed
    /// by input index; <paramref name="relevant"/> limits which inputs need one, as a backward toward chosen sources
    /// does.</summary>
    private static Dictionary<int, float[]> LinearBackward(IEngine engine, bool[] relevant)
    {
        var x = Filled(new[] { 8, 6 }, 21, 1f);
        var w = Filled(new[] { 6, 5 }, 22, 0.5f);
        var b = Filled(new[] { 5 }, 23, 0.5f);
        var gradOutput = Filled(new[] { 8, 5 }, 24, 1f);
        var inputs = new[] { x, w, b };
        var output = new Tensor<float>(new[] { 8, 5 });
        var grads = new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        var wanted = new HashSet<Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        for (int i = 0; i < inputs.Length; i++) if (relevant[i]) wanted.Add(inputs[i]);
        using (DifferentiableOps.PushGradientRelevance(wanted))
            BackwardFunctions<float>.FusedLinearBackward(gradOutput, inputs, output, Array.Empty<object>(), engine, grads);
        var result = new Dictionary<int, float[]>();
        for (int i = 0; i < inputs.Length; i++)
            if (grads.TryGetValue(inputs[i], out var g)) result[i] = g.ToArray();
        return result;
    }

    [SkippableFact]
    public void LinearBackward_OnTheGpuEngine_MatchesCpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var all = new[] { true, true, true };
        var cpu = LinearBackward(new CpuEngine(), all);
        var gpu = LinearBackward(_fixture.Engine!, all);
        Assert.Equal(new[] { 0, 1, 2 }, gpu.Keys.OrderBy(k => k).ToArray());
        AssertClose(cpu[0], gpu[0], "dX");
        AssertClose(cpu[1], gpu[1], "dW");
        AssertClose(cpu[2], gpu[2], "db");
    }

    /// <summary>A network's first layer: nothing reads its input gradient, so the GPU computes only the weight's.</summary>
    [SkippableFact]
    public void LinearBackward_WeightOnly_OnTheGpuEngine_MatchesCpuAndSkipsTheInputGradient()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var weightAndBias = new[] { false, true, true };
        var cpu = LinearBackward(new CpuEngine(), weightAndBias);
        var gpu = LinearBackward(_fixture.Engine!, weightAndBias);
        Assert.False(gpu.ContainsKey(0), "the input gradient was computed although nothing needs it");
        AssertClose(cpu[1], gpu[1], "dW");
        AssertClose(cpu[2], gpu[2], "db");
    }

    /// <summary>
    /// A chain of reshapes between two layers: on a GPU engine each reshape's input gradient is aliased onto the last
    /// one's device buffer instead of copied. Three AdamW steps must land the parameters where the CPU plan does.
    /// </summary>
    [SkippableFact]
    public void CompiledAdamWSteps_ThroughAReshapeChain_MatchTheCpuPlan()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        float[] Run(IEngine e, out float[] losses)
        {
            var previous = AiDotNetEngine.Current;
            AiDotNetEngine.Current = e;
            try
            {
                var x = Filled(new[] { 8, 12 }, 31, 1f);
                var w1 = Filled(new[] { 12, 12 }, 32, 0.3f);
                var w2 = Filled(new[] { 12, 4 }, 33, 0.3f);
                if (e is DirectGpuTensorEngine) { w1 = w1.Gpu(); w2 = w2.Gpu(); }
                ICompiledTrainingPlan<float> plan;
                using (var scope = GraphMode.Enable())
                {
                    var h = e.TensorMatMul(x, w1);
                    var r = e.Reshape(e.Reshape(e.Reshape(h, new[] { 8, 3, 4 }), new[] { 2, 4, 12 }), new[] { 8, 12 });
                    var y = e.TensorMatMul(e.TensorTanh(r), w2);
                    e.ReduceSum(e.TensorMultiply(y, y), null);
                    plan = scope.CompileTraining(new[] { w1, w2 });
                }
                using (plan)
                {
                    plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 1e-2, weightDecay: 0.01);
                    losses = new float[3];
                    for (int step = 0; step < losses.Length; step++) losses[step] = plan.Step().ToArray()[0];
                    return w1.ToArray().Concat(w2.ToArray()).ToArray();
                }
            }
            finally
            {
                AiDotNetEngine.Current = previous;
            }
        }

        var cpu = Run(new CpuEngine(), out var cpuLosses);
        var gpu = Run(_fixture.Engine!, out var gpuLosses);
        Assert.True(cpuLosses[2] < cpuLosses[0], $"the CPU reference must train: {cpuLosses[0]} -> {cpuLosses[2]}");
        AssertClose(cpuLosses, gpuLosses, "loss");
        AssertClose(cpu, gpu, "param");
    }
}