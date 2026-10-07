using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A batched MatMul ([b.., M, K] x [b.., K, N], equal batch dims -- attention's score and context products) takes the
/// specialized compiled forward and backward (one GEMM per batch slice straight into the plan buffers) instead of the
/// generic engine path. Gradients must match the eager tape, an operand read by two batched MatMuls must receive both
/// contributions, and repeated steps at fixed weights must reproduce step 0 bit for bit.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class CompiledBatchedMatMulTests
{
    private static Tensor<float> Rnd(int[] shape, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    /// <summary>h = tanh(x·W) [2,4,6] feeds two batched MatMuls (h·P and h·Q, so h is a shared operand); tanh(h·P) is
    /// the sole-consumer A operand of a third batched MatMul against the rank-3 parameter R.</summary>
    private static Tensor<float> Forward(IEngine e, Tensor<float> x, Tensor<float> w, Tensor<float> p,
        Tensor<float> q, Tensor<float> r)
    {
        var h = e.Tanh(e.TensorMatMul(x, w));
        var s1 = e.TensorMatMul(h, p);
        var s2 = e.TensorMatMul(h, q);
        var v = e.TensorMatMul(e.Tanh(s1), r);
        return e.TensorAdd(e.ReduceSum(e.TensorMultiply(v, v), null), e.ReduceSum(e.TensorMultiply(s2, s2), null));
    }

    [Fact]
    public void BatchedMatMul_IsSpecialized_MatchesTape_AndIsStable()
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var engine = new CpuEngine();
            var x = Rnd(new[] { 2, 4, 5 }, 1);
            Tensor<float>[] MakeParams() => new[]
            {
                Rnd(new[] { 5, 6 }, 2, 0.5f), Rnd(new[] { 2, 6, 3 }, 3, 0.5f),
                Rnd(new[] { 2, 6, 3 }, 4, 0.5f), Rnd(new[] { 2, 3, 5 }, 5, 0.5f),
            };

            var tapeParams = MakeParams();
            float[][] tape;
            using (var t = new GradientTape<float>())
            {
                var loss = Forward(engine, x, tapeParams[0], tapeParams[1], tapeParams[2], tapeParams[3]);
                var g = t.ComputeGradients(loss, tapeParams);
                tape = tapeParams.Select(p => (float[])g[p].GetFlattenedData().Clone()).ToArray();
            }

            var planParams = MakeParams();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(engine, x, planParams[0], planParams[1], planParams[2], planParams[3]);
                plan = scope.CompileTraining(planParams);
            }
            try
            {
                var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
                Assert.DoesNotContain("generic:TensorMatMul", concrete.ForwardActionNames);
                Assert.DoesNotContain("generic:TensorMatMul", concrete.BackwardActionNames);

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