using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A host compiled training plan turns a reshape of an activation into a storage alias: the reshape emits no forward
/// action, and when it is its input's only consumer the input gradient shares the output gradient's storage and the
/// reshape emits no backward action either. Gradients must match the eager tape and stay bit-identical across
/// steps at fixed weights (an alias that dropped a contribution, or a buffer that missed its zeroing, would show).
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class CompiledReshapeAliasTests
{
    private static Tensor<float> Rnd(int[] shape, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    /// <summary>x·W → reshape → reshape → tanh → Σ s²; with <paramref name="shared"/> the matmul output also feeds a
    /// second consumer, so the first reshape's input gradient must keep accumulating through its backward.</summary>
    private static Tensor<float> Forward(IEngine engine, Tensor<float> x, Tensor<float> w, bool shared)
    {
        var h = engine.TensorMatMul(x, w);                       // [4, 8]
        var r = engine.Reshape(engine.Reshape(h, new[] { 2, 16 }), new[] { 32 });
        var s = engine.Tanh(r);
        var loss = engine.ReduceSum(engine.TensorMultiply(s, s), null);
        if (shared)
            loss = engine.TensorAdd(loss, engine.ReduceSum(engine.TensorMultiply(h, h), null));
        return loss;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReshapeAlias_MatchesTape_AndIsStableAcrossSteps(bool shared)
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var engine = new CpuEngine();
            var x = Rnd(new[] { 4, 6 }, 1);
            var w = Rnd(new[] { 6, 8 }, 2, 0.5f);

            float[] tape;
            using (var t = new GradientTape<float>())
            {
                var loss = Forward(engine, x, w, shared);
                tape = (float[])t.ComputeGradients(loss, new[] { w })[w].GetFlattenedData().Clone();
            }

            var wPlan = Rnd(new[] { 6, 8 }, 2, 0.5f);
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(engine, x, wPlan, shared);
                plan = scope.CompileTraining(new[] { wPlan });
            }
            try
            {
                var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
                var steps = concrete.ForwardStepsForSerialization;
                Assert.NotNull(steps);
                var reshapes = steps!.Where(s => s.OpName == "Reshape").ToArray();
                Assert.Equal(2, reshapes.Length);
                foreach (var r in reshapes)
                    Assert.True(r.OutputBuffer.SharesStorageWith(r.Inputs[0]), "a reshape of an activation was not aliased");
                // Both reshapes emit no forward action (other fusions may only lower the count further).
                Assert.True(concrete.ForwardActions.Length <= steps.Length - 2,
                    $"{concrete.ForwardActions.Length} forward actions for {steps.Length} steps; the aliased reshapes still emit actions");

                float[]? first = null;
                for (int step = 0; step < 4; step++)
                {
                    plan.Step();
                    var now = plan.Gradients[0].AsSpan().ToArray();
                    for (int i = 0; i < tape.Length; i++)
                    {
                        Assert.True(Math.Abs(now[i] - tape[i]) <= 1e-4f * (1f + Math.Abs(tape[i])),
                            $"shared={shared} step {step} element {i}: tape {tape[i]:R} plan {now[i]:R}");
                        if (first is not null)
                            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(first[i]) == TestHelpers.MathCompat.SingleToInt32Bits(now[i]),
                                $"shared={shared} step {step} element {i} drifted from step 0: {first[i]:R} -> {now[i]:R}");
                    }
                    first ??= now;
                }
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    [Fact]
    public void ReshapeAlias_DisabledByEnvironment_KeepsTheCopy()
    {
        // The alias is on by default; this pins the documented kill switch so a regression can be bisected.
        var prior = Environment.GetEnvironmentVariable("AIDOTNET_RESHAPE_ALIAS");
        var priorEngine = AiDotNetEngine.Current;
        Environment.SetEnvironmentVariable("AIDOTNET_RESHAPE_ALIAS", "0");
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var engine = new CpuEngine();
            var x = Rnd(new[] { 4, 6 }, 3);
            var w = Rnd(new[] { 6, 8 }, 4, 0.5f);
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(engine, x, w, shared: false);
                plan = scope.CompileTraining(new[] { w });
            }
            using (plan)
            {
                var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
                var steps = concrete.ForwardStepsForSerialization;
                Assert.NotNull(steps);
                foreach (var r in steps!.Where(s => s.OpName == "Reshape"))
                    Assert.False(r.OutputBuffer.SharesStorageWith(r.Inputs[0]));
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_RESHAPE_ALIAS", prior);
            AiDotNetEngine.Current = priorEngine;
        }
    }
}
