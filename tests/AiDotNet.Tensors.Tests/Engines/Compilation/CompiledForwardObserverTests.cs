using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The forward observer must report each replayed step with the op that produced it.
/// </summary>
/// <remarks>
/// <para>
/// A compiled plan otherwise reports one number — the loss — which says a replay went wrong but not
/// where, and the intermediate values cannot be recovered from outside: the tensors handed back
/// while tracing are placeholders, and reading one after a Step shows whatever the buffer holds
/// now rather than what that step produced. Both look like measurements and neither is one, which
/// is how a wrong conclusion gets built on them.
/// </para>
/// <para>
/// The step label matters as much as the values. Actions and steps are not 1:1 — a skipped step
/// emits none and a fused group emits one for several — so indexing the step list by action
/// position mislabels every op after the first skip, which is what the old probe did.
/// </para>
/// </remarks>
/// <remarks>
/// Serialized against the other global-state suites: <c>ForwardStepObserver</c> is static, matching
/// the existing <c>StepProbe</c>, so any plan replayed on another thread reports into whatever
/// observer is installed. Run in parallel these assertions see a neighbouring test's steps.
/// </remarks>
[Collection("EngineCurrentGlobalState")]
public class CompiledForwardObserverTests
{
    private static Tensor<float> Filled(int[] shape, float scale, int salt)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = scale * (float)Math.Sin((i + salt) * 0.37);
        return t;
    }

    [Fact]
    public void Observer_ReportsEveryReplayedStep_WithItsOpNameAndValues()
    {
        var engine = new CpuEngine();
        var a = Filled(new[] { 8, 8 }, 0.3f, 1);
        var b = Filled(new[] { 8, 8 }, 0.2f, 5);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var scaled = engine.TensorMultiply(a, b);
            engine.ReduceSum(engine.TensorAdd(scaled, a), null);
            plan = scope.CompileTraining(new[] { a, b });
        }

        var seen = new List<(int Index, string Op, int Length, bool AllFinite)>();
        try
        {
            CompiledTrainingPlan<float>.ForwardStepObserver = (index, op, buffer) =>
            {
                bool finite = true;
                for (int i = 0; i < buffer.Length; i++)
                    if (float.IsNaN(buffer[i]) || float.IsInfinity(buffer[i])) { finite = false; break; }
                seen.Add((index, op, buffer.Length, finite));
            };

            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                plan.StepInto(new Tensor<float>(new[] { 1 }));
            }
        }
        finally { CompiledTrainingPlan<float>.ForwardStepObserver = null; }

        Assert.NotEmpty(seen);

        // Indices are dense and ascending: one report per emitted action, none skipped or repeated.
        for (int i = 0; i < seen.Count; i++)
            Assert.Equal(i, seen[i].Index);

        // Every step is named and carries real values.
        foreach (var (index, op, length, allFinite) in seen)
        {
            Assert.False(string.IsNullOrWhiteSpace(op), $"step {index} was reported without an op name.");
            Assert.True(length > 0, $"step {index} ({op}) reported an empty buffer.");
            Assert.True(allFinite, $"step {index} ({op}) reported a non-finite value.");
        }

        // The ops actually traced must be the ops reported.
        Assert.Contains(seen, s => s.Op.Contains("Multiply", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(seen, s => s.Op.Contains("Add", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void Observer_IsOffByDefault_AndReplayIsUnaffected()
    {
        var engine = new CpuEngine();

        static (Tensor<float> a, Tensor<float> b) Operands()
            => (Filled(new[] { 8, 8 }, 0.3f, 1), Filled(new[] { 8, 8 }, 0.2f, 5));

        float Run(bool observe)
        {
            var (a, b) = Operands();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                engine.ReduceSum(engine.TensorAdd(engine.TensorMultiply(a, b), a), null);
                plan = scope.CompileTraining(new[] { a, b });
            }

            int reports = 0;
            if (observe) CompiledTrainingPlan<float>.ForwardStepObserver = (_, _, _) => reports++;
            try
            {
                using (plan)
                {
                    plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                    var loss = new Tensor<float>(new[] { 1 });
                    plan.StepInto(loss);
                    if (observe) Assert.True(reports > 0, "observer was set but never called.");
                    else Assert.Equal(0, reports);
                    return loss[0];
                }
            }
            finally { CompiledTrainingPlan<float>.ForwardStepObserver = null; }
        }

        float unobserved = Run(observe: false);
        float observed = Run(observe: true);

        Assert.Equal(unobserved, observed, 4);
    }
}
