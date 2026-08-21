using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Rebuilding a plan's forward actions must not change what the forward computes.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="ICompiledPlan{T}.EnableFrozenWeightOptimizations"/> rebuilds the forward action list.
/// It used to re-derive each step's verdict from scratch while knowing only three of the seven
/// cases the compile-time walk weighs, so a slice step already consumed by a fused MatMul was
/// re-executed as a standalone action and analytic MatMul-into-loss forwards were dropped
/// entirely. Nothing in either repository calls the method, so no test covered the drift.
/// </para>
/// <para>
/// These compare the loss across the rebuild rather than against a constant: whatever the plan
/// computed before must be exactly what it computes after.
/// </para>
/// </remarks>
public class CompiledForwardRebuildTests
{
    private static Tensor<float> Filled(int[] shape, float scale, int salt)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = scale * (float)Math.Sin((i + salt) * 0.37);
        return t;
    }

    private static void AssertRebuildPreservesForward(
        string what, Func<CpuEngine, (Tensor<float> output, Tensor<float>[] parameters)> build)
    {
        var engine = new CpuEngine();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var (output, parameters) = build(engine);
            engine.ReduceSum(output, null);
            plan = scope.CompileTraining(parameters);
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);

            var before = new Tensor<float>(new[] { 1 });
            plan.StepInto(before);

            plan.EnableFrozenWeightOptimizations();

            var after = new Tensor<float>(new[] { 1 });
            plan.StepInto(after);

            Assert.False(float.IsNaN(after[0]) || float.IsInfinity(after[0]),
                $"{what}: forward returned {after[0]} after the rebuild (was {before[0]:G8}).");
            Assert.True(Math.Abs(before[0] - after[0]) <= 1e-4f * Math.Max(1f, Math.Abs(before[0])),
                $"{what}: rebuild changed the forward — {before[0]:G8} became {after[0]:G8}.");
        }
    }

    [Fact]
    public void Rebuild_PreservesForward_ForAPointwiseChain()
    {
        AssertRebuildPreservesForward("pointwise chain", e =>
        {
            var a = Filled(new[] { 8, 8 }, 0.3f, 1);
            var b = Filled(new[] { 8, 8 }, 0.2f, 5);
            var scaled = e.TensorMultiply(a, b);
            return (e.TensorAdd(scaled, a), new[] { a, b });
        });
    }

    [Fact]
    public void Rebuild_PreservesForward_ForAMatMulIntoLoss()
    {
        AssertRebuildPreservesForward("matmul into loss", e =>
        {
            var x = Filled(new[] { 1, 8, 8 }, 0.3f, 1);
            var w = Filled(new[] { 1, 8, 8 }, 0.1f, 5);
            return (e.TensorBatchMatMul<float>(x, w), new[] { x, w });
        });
    }

    /// <summary>
    /// A MatMul whose output is sliced to a prefix is fused: the MatMul step gets a specialized
    /// action and the Slice step is CONSUMED, emitting nothing. The rebuild knew nothing about
    /// that pairing, so it re-emitted the consumed Slice as a standalone action.
    /// </summary>
    [Fact]
    public void Rebuild_PreservesForward_WhenASliceIsConsumedByAFusedMatMul()
    {
        AssertRebuildPreservesForward("matmul + prefix slice", e =>
        {
            var a = Filled(new[] { 8, 16 }, 0.3f, 1);
            var w = Filled(new[] { 16, 32 }, 0.1f, 5);
            var product = e.TensorMatMul<float>(a, w);                       // [8, 32]
            var prefix = e.TensorSlice(product, new[] { 0, 0 }, new[] { 8, 16 });
            return (prefix, new[] { a, w });
        });
    }

    [Fact]
    public void Rebuild_PreservesForward_ForABroadcastExpansion()
    {
        AssertRebuildPreservesForward("broadcast expansion", e =>
        {
            var rowVector = Filled(new[] { 8, 1 }, 0.3f, 2);
            var full = Filled(new[] { 8, 9 }, 0.1f, 7);
            return (e.TensorAdd(e.TensorBroadcastTo(rowVector, new[] { 8, 9 }), full),
                    new[] { rowVector, full });
        });
    }
}
