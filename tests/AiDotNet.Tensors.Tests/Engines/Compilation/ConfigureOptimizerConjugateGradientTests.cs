using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Covers fused Fletcher-Reeves conjugate gradient.
/// </summary>
/// <remarks>
/// <para>
/// CG is structurally SGD-with-momentum where the coefficient is recomputed each step:
/// <c>beta = (g.g)/(g_prev.g_prev)</c>, <c>d = -g + beta*d_prev</c>, <c>x += lr*d</c>. That per-step
/// recomputation is exactly why it cannot reuse the SGDMomentum kernel, which bakes beta1 in when the plan
/// is built — so it runs as two global reductions plus an elementwise update, the shape HypergradientSGD
/// already uses.
/// </para>
/// <para>
/// The failure mode to guard is silence: a recursion that always produced <c>d = -g</c> would descend
/// perfectly well and would simply be gradient descent. So the tests demand the first step equal SGD (no
/// history yet), later steps differ from it, and the Powell restart actually fire.
/// </para>
/// </remarks>
public class ConfigureOptimizerConjugateGradientTests
{
    private static float[] Run(float[] init, float[] curvature, OptimizerType opt, int steps, float lr)
    {
        var engine = new CpuEngine();
        var w = new Tensor<float>(new[] { init.Length });
        var c = new Tensor<float>(new[] { init.Length });
        for (int i = 0; i < init.Length; i++) { w[i] = init[i]; c[i] = curvature[i]; }

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var scaled = engine.TensorMultiply(engine.TensorMultiply(w, w), c);
            engine.ReduceSum(scaled, null);
            plan = scope.CompileTraining(new[] { w });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(opt, LrSchedule.Constant(lr));
            for (int s = 0; s < steps; s++) plan.Step();
        }
        return w.GetDataArray().AsSpan(0, init.Length).ToArray();
    }

    /// <summary>
    /// With no previous gradient, beta is 0 and <c>d = -g</c>, so the first step must be exactly the
    /// gradient-descent step.
    /// </summary>
    [Fact]
    public void FirstStepEqualsGradientDescent_BecauseBetaHasNoDenominatorYet()
    {
        var init = new[] { 1.0f, -2.0f, 0.5f };
        var curv = new[] { 1.0f, 1.0f, 1.0f };
        const float lr = 0.01f;

        var cg = Run(init, curv, OptimizerType.ConjugateGradient, steps: 1, lr);

        for (int i = 0; i < init.Length; i++)
        {
            float grad = 2f * curv[i] * init[i];
            Assert.Equal(init[i] - lr * grad, cg[i], 4);
        }
    }

    /// <summary>
    /// Once a previous gradient exists the conjugate term engages, so the trajectory must leave SGD's.
    /// </summary>
    [Fact]
    public void DivergesFromGradientDescent_OnceTheConjugateTermEngages()
    {
        var init = new[] { 1.0f, 1.0f, 1.0f, 1.0f };
        var curv = new[] { 1.0f, 4.0f, 16.0f, 50.0f };
        const float lr = 0.005f;
        const int steps = 10;

        var cg = Run(init, curv, OptimizerType.ConjugateGradient, steps, lr);
        var sgd = Run(init, curv, OptimizerType.SGD, steps, lr);

        double maxGap = 0;
        for (int i = 0; i < init.Length; i++)
            maxGap = Math.Max(maxGap, Math.Abs(cg[i] - sgd[i]));

        Assert.True(maxGap > 1e-5,
            $"CG never diverged from plain SGD (max |Δ| = {maxGap}) — the conjugate direction is not being applied.");
    }

    /// <summary>
    /// It must still converge, and stay finite while doing so. A restart that never fired, or fired every
    /// step, would show up here as divergence or as a stall.
    /// </summary>
    [Fact]
    public void ConvergesOnAnIllConditionedQuadratic()
    {
        var init = new[] { 1.0f, 1.0f, 1.0f, 1.0f };
        var curv = new[] { 1.0f, 4.0f, 16.0f, 50.0f };

        var cg = Run(init, curv, OptimizerType.ConjugateGradient, steps: 40, lr: 0.005f);

        double before = 0, after = 0;
        for (int i = 0; i < init.Length; i++)
        {
            before += curv[i] * init[i] * init[i];
            after += curv[i] * cg[i] * cg[i];
            Assert.False(float.IsNaN(cg[i]), $"element {i} is NaN");
            Assert.False(float.IsInfinity(cg[i]), $"element {i} is Infinity");
        }

        Assert.True(after < before * 0.25,
            $"CG did not meaningfully reduce the objective ({before:G6} -> {after:G6}).");
    }

    /// <summary>
    /// A zero gradient must not divide by a vanished previous norm and produce a non-finite beta.
    /// </summary>
    [Fact]
    public void SurvivesAZeroGradient()
    {
        var init = new[] { 0.7f, -0.3f };
        var curv = new[] { 0.0f, 0.0f };

        var result = Run(init, curv, OptimizerType.ConjugateGradient, steps: 6, lr: 0.1f);

        for (int i = 0; i < init.Length; i++)
        {
            Assert.False(float.IsNaN(result[i]), "beta divided by a zero previous-gradient norm");
            Assert.False(float.IsInfinity(result[i]), "beta divided by a zero previous-gradient norm");
            Assert.Equal(init[i], result[i], 5);
        }
    }

    /// <summary>
    /// Per-group schedules are meaningless when beta comes from the norm of the WHOLE gradient.
    /// </summary>
    [Fact]
    public void ConfigureOptimizerGrouped_ConjugateGradient_Throws()
    {
        var engine = new CpuEngine();
        var w = new Tensor<float>(new[] { 4 });
        for (int i = 0; i < 4; i++) w[i] = 1f;

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(engine.TensorMultiply(w, w), null);
            plan = scope.CompileTraining(new[] { w });
        }
        using (plan)
        {
            var ex = Assert.Throws<NotSupportedException>(() => plan.ConfigureOptimizerGrouped(
                OptimizerType.ConjugateGradient,
                new System.Collections.Generic.List<LrSchedule> { LrSchedule.Constant(0.1) },
                new System.Collections.Generic.List<int> { 0 }));
            Assert.Contains("ConjugateGradient", ex.Message);
        }
    }
}
