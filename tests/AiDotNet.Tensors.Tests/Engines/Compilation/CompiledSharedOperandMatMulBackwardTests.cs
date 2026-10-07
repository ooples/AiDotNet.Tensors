using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A MatMul whose operand has several consumers -- a weight reused across RNN time steps, one activation feeding
/// several projections -- takes the specialized compiled backward (one GEMM per needed gradient, accumulated into
/// the shared buffer) instead of the generic dictionary path. Gradients must match the eager tape, every
/// consumer's contribution must be summed, and repeated steps at fixed weights must reproduce step 0 bit for bit.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class CompiledSharedOperandMatMulBackwardTests
{
    private static Tensor<float> Rnd(int[] shape, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    /// <summary>A two-step recurrence h1 = tanh(x0·W + h0·U), h2 = tanh(x1·W + h1·U): W and U are each read by two
    /// MatMuls, and h1 by the second MatMul and nothing else. Then a projection fan-out: q = h2·Wq and k = h2·Wk
    /// share h2.</summary>
    private static Tensor<float> Forward(IEngine e, Tensor<float> x0, Tensor<float> x1, Tensor<float> h0,
        Tensor<float> w, Tensor<float> u, Tensor<float> wq, Tensor<float> wk)
    {
        var h1 = e.Tanh(e.TensorAdd(e.TensorMatMul(x0, w), e.TensorMatMul(h0, u)));
        var h2 = e.Tanh(e.TensorAdd(e.TensorMatMul(x1, w), e.TensorMatMul(h1, u)));
        var q = e.TensorMatMul(h2, wq);
        var k = e.TensorMatMul(h2, wk);
        return e.ReduceSum(e.TensorMultiply(q, k), null);
    }

    [Fact]
    public void SharedOperandMatMul_IsSpecialized_MatchesTape_AndIsStable()
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var engine = new CpuEngine();
            var x0 = Rnd(new[] { 5, 7 }, 1);
            var x1 = Rnd(new[] { 5, 7 }, 2);
            var h0 = Rnd(new[] { 5, 6 }, 3);
            Tensor<float>[] MakeParams() => new[]
            {
                Rnd(new[] { 7, 6 }, 4, 0.4f), Rnd(new[] { 6, 6 }, 5, 0.4f),
                Rnd(new[] { 6, 3 }, 6, 0.4f), Rnd(new[] { 6, 3 }, 7, 0.4f),
            };

            var tapeParams = MakeParams();
            float[][] tape;
            using (var t = new GradientTape<float>())
            {
                var loss = Forward(engine, x0, x1, h0, tapeParams[0], tapeParams[1], tapeParams[2], tapeParams[3]);
                var g = t.ComputeGradients(loss, tapeParams);
                tape = tapeParams.Select(p => (float[])g[p].GetFlattenedData().Clone()).ToArray();
            }

            var planParams = MakeParams();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(engine, x0, x1, h0, planParams[0], planParams[1], planParams[2], planParams[3]);
                plan = scope.CompileTraining(planParams);
            }
            try
            {
                var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
                Assert.DoesNotContain("generic:TensorMatMul", concrete.BackwardActionNames);
                Assert.Contains("specialized:TensorMatMul", concrete.BackwardActionNames);

                float[][]? first = null;
                for (int step = 0; step < 4; step++)
                {
                    plan.Step();
                    var now = plan.Gradients.Select(g => g.AsSpan().ToArray()).ToArray();
                    for (int p = 0; p < planParams.Length; p++)
                        for (int i = 0; i < tape[p].Length; i++)
                        {
                            Assert.True(Math.Abs(now[p][i] - tape[p][i]) <= 1e-4f * (1f + Math.Abs(tape[p][i])),
                                $"step {step} parameter {p} element {i}: tape {tape[p][i]:R} plan {now[p][i]:R}");
                            if (first is not null)
                                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(first[p][i]) == TestHelpers.MathCompat.SingleToInt32Bits(now[p][i]),
                                    $"step {step} parameter {p} element {i} drifted from step 0: {first[p][i]:R} -> {now[p][i]:R}");
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
}
