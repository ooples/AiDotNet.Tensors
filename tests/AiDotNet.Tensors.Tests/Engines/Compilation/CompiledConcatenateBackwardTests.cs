using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The CPU compiled plan's concatenation backward copies (or adds) each input's slice of the output gradient straight
/// into that input's gradient buffer. Against the eager tape it must give every parameter's gradient for a concatenation
/// on a middle, the leading and the last axis; with an input repeated in the same concatenation (its two slices must
/// sum); with a constant input that has no gradient; and with an input another op also reads (its contributions sum).
/// Later steps at learning rate zero must repeat step 0 bit for bit.
/// </summary>
[Collection("CompilationGlobalState")]
public class CompiledConcatenateBackwardTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    [InlineData(3)]
    public void MatchesEager_WithRepeatedConstantAndSharedInputs(int axis)
    {
        var engine = new CpuEngine();
        var input = Filled(new[] { 2, 3, 4, 5 }, 1, 2f);
        var constant = Filled(new[] { 2, 3, 4, 5 }, 2, 1f);
        Tensor<float>[] MakeParams() => new[] { Filled(new[] { 2, 3, 4, 5 }, 3, 1f), Filled(new[] { 2, 3, 4, 5 }, 4, 1f) };
        string[] names = { "a", "b" };
        Tensor<float> Forward(Tensor<float>[] p)
        {
            var x = engine.TensorMultiply(input, p[0]);
            var y = engine.TensorMultiply(input, p[1]);
            var z = engine.TensorConcatenate(new[] { x, y, x, constant }, axis);   // x twice, a constant
            var shared = engine.TensorMultiply(y, y);                               // y also read here
            return engine.TensorAdd(engine.ReduceSum(engine.TensorMultiply(z, z), null), engine.ReduceSum(shared, null));
        }

        var eagerParams = MakeParams();
        var eager = new Tensor<float>[eagerParams.Length];
        using (var tape = new GradientTape<float>())
        {
            var grads = tape.ComputeGradients(Forward(eagerParams), sources: eagerParams);
            for (int i = 0; i < eagerParams.Length; i++) eager[i] = grads[eagerParams[i]];
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
            var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            Assert.Contains("specialized:Concatenate", concrete.BackwardStepKinds);
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
            var step0 = new float[ps.Length][];
            for (int i = 0; i < ps.Length; i++)
            {
                var grad = ps[i].Grad;
                Assert.True(grad is not null, $"{names[i]}: no gradient");
                if (grad is null) return;
                step0[i] = grad.ToArray();
                for (int k = 0; k < step0[i].Length; k++)
                {
                    float e = eager[i][k], a = step0[i][k];
                    Assert.True(System.Math.Abs(e - a) <= 1e-4f * (1f + System.Math.Abs(e)), $"{names[i]}[{k}]: eager {e}, compiled {a}");
                }
            }
            for (int step = 1; step < 4; step++)
            {
                plan.Step();
                for (int i = 0; i < ps.Length; i++)
                {
                    var grad = ps[i].Grad;
                    Assert.True(grad is not null, $"{names[i]} step {step}: no gradient");
                    if (grad is null) return;
                    Assert.Equal(step0[i], grad.ToArray());
                }
            }
        }
    }
}
