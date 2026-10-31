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
/// tape both must give every parameter's gradient (nonzero where connected), on an overlapping pooling (stride below
/// the window), a plain 2x2 one, and padded ones that do and do not count padded cells in the divisor; later steps at learning rate zero must repeat step 0 bit for bit; and the sole-writer plan must have run the
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
    [InlineData(2, 2, 0, false, false)]
    [InlineData(2, 2, 0, false, true)]
    [InlineData(3, 2, 0, false, false)]
    [InlineData(3, 2, 0, false, true)]
    [InlineData(3, 2, 1, false, false)]   // padded: edge windows divide by the cells they cover
    [InlineData(3, 2, 1, true, false)]    // padded, padded cells counted in the divisor
    public void MatchesEager_AndRepeats(int pool, int stride, int padding, bool countIncludePad, bool sharedInput)
    {
        var engine = new CpuEngine();
        var input = Filled(new[] { 2, 3, 9, 9 }, 1, 2f);
        Tensor<float>[] MakeParams() => new[] { Filled(new[] { 4, 3, 3, 3 }, 2, 0.6f), Filled(new[] { 4, 4, 3, 3 }, 3, 0.6f) };
        Tensor<float> Forward(Tensor<float>[] p)
        {
            int[] one = { 1, 1 }, pad = { 1, 1 };
            var t = engine.Tanh(engine.Conv2D(input, p[0], one, pad, one));
            // The array overload records pool and stride only; the scalar one also padding and the divisor setting.
            var pooled = padding == 0 && !countIncludePad
                ? engine.AvgPool2D(t, new[] { pool, pool }, new[] { stride, stride })
                : engine.AvgPool2D(t, pool, stride, padding, countIncludePad);
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
            // Both paths share the pooling kernel, so agreement alone could hide a kernel that wrote nothing.
            int connected = sharedInput ? eagerParams.Length : 1;
            for (int i = 0; i < connected; i++)
                Assert.True(Array.Exists(eager[i], v => v != 0f), $"parameter {i}: the eager gradient is all zero");
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
                if (i < (sharedInput ? ps.Length : 1)) Assert.True(ps[i].Grad is not null, $"parameter {i}: no compiled gradient");
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
