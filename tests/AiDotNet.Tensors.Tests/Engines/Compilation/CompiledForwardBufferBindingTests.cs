using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Every forward output buffer must expose its live backing once the plan is compiled.
/// </summary>
/// <remarks>
/// <para>
/// Forward outputs come from <c>TensorAllocator.RentUninitialized</c>, and a pooled tensor whose
/// storage is larger than its logical length does not settle which array IS the tensor until
/// something asks for writable storage. Until then a writer reaching for the data array can be
/// handed a pool-padded COPY while the plan reads the live backing, so what was written is not
/// what is read back.
/// </para>
/// <para>
/// It presented as a first-step loss that was intermittently enormous (1e10..1e35) or NaN on
/// roughly 60% of freshly built plans, and it vanished whenever anything touched those buffers
/// first — including the diagnostics used to look at it, which is what made it read as a
/// heisenbug rather than a defect. Binding at compile time settles the backing before any kernel
/// can bind it.
/// </para>
/// </remarks>
public class CompiledForwardBufferBindingTests
{
    private static Tensor<float> Filled(int[] shape, float scale, int salt)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = scale * (float)Math.Sin((i + salt) * 0.37);
        return t;
    }

    /// <summary>
    /// Compiling the same graph repeatedly must give the same loss every time. The defect this
    /// pins was intermittent, so a single comparison could pass on a lucky pool block.
    /// </summary>
    [Fact]
    public void RepeatedlyCompiledPlans_AgreeWithEachOtherAndWithEager()
    {
        const int S = 8, D = 8, C = 9;
        var engine = new CpuEngine();

        static (Tensor<float> output, Tensor<float>[] parameters) Build(CpuEngine e)
        {
            var hs = Filled(new[] { 1, S, D }, 0.5f, 1);
            var he = Filled(new[] { 1, S, D }, 0.5f, 7);
            var bias = Filled(new[] { C }, 0.01f, 5);
            var heT = e.TensorPermute(he, new[] { 0, 2, 1 });
            var scores = e.TensorBatchMatMul<float>(hs, heT);                       // [1, S, S]
            var grid = e.TensorBroadcastTo(e.Reshape(scores, new[] { 1, S, S, 1 }), new[] { 1, S, S, C });
            var biasGrid = e.TensorBroadcastTo(e.Reshape(bias, new[] { 1, 1, 1, C }), new[] { 1, S, S, C });
            return (e.TensorAdd(grid, biasGrid), new[] { hs, he, bias });
        }

        var (eagerOutput, _) = Build(engine);
        double eager = engine.ReduceSum(eagerOutput, null)[0];

        for (int attempt = 0; attempt < 12; attempt++)
        {
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var (output, parameters) = Build(engine);
                engine.ReduceSum(output, null);
                plan = scope.CompileTraining(parameters);
            }

            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                var loss = new Tensor<float>(new[] { 1 });
                plan.StepInto(loss);

                Assert.False(float.IsNaN(loss[0]) || float.IsInfinity(loss[0]),
                    $"attempt {attempt}: compiled replay produced {loss[0]}; eager was {eager:G8}.");
                Assert.True(Math.Abs(loss[0] - eager) <= 1e-3 * Math.Max(1.0, Math.Abs(eager)),
                    $"attempt {attempt}: compiled {loss[0]:G8} != eager {eager:G8}.");
            }
        }
    }
}
