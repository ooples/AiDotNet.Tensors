using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The CPU compiled plan's float AvgPool2D backward spreads the gradient straight into the plan's input-gradient buffer
/// (the eager op rented a fresh input-sized result every step to copy or add in). It runs when the pool is its input
/// gradient's sole writer; an input another op also reads takes the generic, accumulating backward. Against the eager
/// tape both must give every parameter's gradient, on an overlapping pooling (stride below the window) and a plain 2x2
/// one; later steps at learning rate zero must repeat step 0 bit for bit; and the sole-writer plan must have run the
/// direct path.
/// </summary>
[Collection("CompilationGlobalState")]
public class CompiledAvgPoolBackwardTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    [Theory]
    [InlineData(2, 2, false)]
    [InlineData(2, 2, true)]
    [InlineData(3, 2, false)]
    [InlineData(3, 2, true)]
    public void MatchesEager_AndRepeats(int pool, int stride, bool sharedInput)
    {
        var engine = new CpuEngine();
        var input = Filled(new[] { 2, 3, 9, 9 }, 1, 2f);
        Tensor<float>[] MakeParams() => new[] { Filled(new[] { 4, 3, 3, 3 }, 2, 0.6f), Filled(new[] { 4, 4, 3, 3 }, 3, 0.6f) };
        Tensor<float> Forward(Tensor<float>[] p)
        {
            int[] one = { 1, 1 }, pad = { 1, 1 };
            var t = engine.Tanh(engine.Conv2D(input, p[0], one, pad, one));
            var pooled = engine.AvgPool2D(t, new[] { pool, pool }, new[] { stride, stride });
            var loss = engine.ReduceSum(engine.TensorMultiply(pooled, pooled), null);
            if (!sharedInput) return loss;
            // t is read again: its gradient accumulates the pool's contribution and this one's.
            var other = engine.Tanh(engine.Conv2D(t, p[1], one, pad, one));
            return engine.TensorAdd(loss, engine.ReduceSum(engine.TensorMultiply(other, other), null));
        }

        var eagerParams = MakeParams();
        var eager = new float[eagerParams.Length][];
        using (var tape = new GradientTape<float>())
        {
            var grads = tape.ComputeGradients(Forward(eagerParams), sources: eagerParams);
            for (int i = 0; i < eagerParams.Length; i++)
                eager[i] = grads.TryGetValue(eagerParams[i], out var g) ? g.ToArray() : new float[eagerParams[i].Length];
        }

        var ps = MakeParams();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Forward(ps);
            plan = scope.CompileTraining(ps);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            int directBefore = CompiledTrainingPlan<float>.DirectAvgPoolBackwardsOnThisThread;
            plan.Step();
            int directRuns = CompiledTrainingPlan<float>.DirectAvgPoolBackwardsOnThisThread - directBefore;
            if (sharedInput) Assert.Equal(0, directRuns);   // the generic backward sums the two contributions
            else Assert.True(directRuns > 0, "the AvgPool2D backward did not run straight into the gradient buffer");
            var step0 = new float[ps.Length][];
            for (int i = 0; i < ps.Length; i++)
            {
                step0[i] = ps[i].Grad?.ToArray() ?? new float[ps[i].Length];
                for (int k = 0; k < step0[i].Length; k++)
                {
                    float e = eager[i][k], a = step0[i][k];
                    Assert.True(Math.Abs(e - a) <= 1e-4f * (1f + Math.Abs(e)), $"param {i}[{k}]: eager {e}, compiled {a}");
                }
            }
            for (int step = 1; step < 3; step++)
            {
                plan.Step();
                for (int i = 0; i < ps.Length; i++)
                    Assert.Equal(step0[i], ps[i].Grad?.ToArray() ?? new float[ps[i].Length]);
            }
        }
    }
}
