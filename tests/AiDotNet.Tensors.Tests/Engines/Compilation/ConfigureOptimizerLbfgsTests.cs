using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Covers fused L-BFGS: the two-loop recursion (Nocedal 1980; Nocedal &amp; Wright, Algorithm 7.4) running in
/// the compiled plan as a sequence of global reductions.
/// </summary>
/// <remarks>
/// <para>
/// L-BFGS is not a per-element kernel and never can be: each of the 2m+2 inner products couples every
/// parameter. It is dispatchable anyway because the plan does not actually require an elementwise kernel —
/// HypergradientSGD and DAdaptationSGD already run as global reductions, and this is the same shape repeated
/// 2m times.
/// </para>
/// <para>
/// The step takes a fixed <c>lr</c> rather than running a line search, matching <c>torch.optim.LBFGS</c>'s
/// default (<c>line_search_fn=None</c>). A line search would need extra loss evaluations inside the step,
/// which the plan does not offer.
/// </para>
/// <para>
/// The tests target the parts that fail silently. A broken two-loop recursion still descends — it just
/// descends like gradient descent — so "loss went down" proves nothing. Instead: the first step must equal
/// SGD exactly (no history yet), later steps must differ, curvature pairs violating <c>s.y &gt; 0</c> must be
/// rejected, and on an ill-conditioned quadratic it must beat SGD outright.
/// </para>
/// </remarks>
public class ConfigureOptimizerLbfgsTests
{
    private static ICompiledTrainingPlan<float> CompileQuadratic(
        CpuEngine engine, Tensor<float> weight, Tensor<float> curvature)
    {
        using var scope = GraphMode.Enable();
        // loss = sum(c_i * w_i^2) -> grad_i = 2*c_i*w_i, an ill-conditioned quadratic when c varies.
        var scaled = engine.TensorMultiply(engine.TensorMultiply(weight, weight), curvature);
        engine.ReduceSum(scaled, null);
        return scope.CompileTraining(new[] { weight });
    }

    private static float[] Run(float[] init, float[] curvature, OptimizerType opt, int steps, float lr,
        FusedOptimizerExtras? extras = null)
    {
        var engine = new CpuEngine();
        var w = new Tensor<float>(new[] { init.Length });
        var c = new Tensor<float>(new[] { init.Length });
        for (int i = 0; i < init.Length; i++) { w[i] = init[i]; c[i] = curvature[i]; }

        var plan = CompileQuadratic(engine, w, c);
        using (plan)
        {
            plan.ConfigureOptimizer(opt, LrSchedule.Constant(lr), 0.9f, 0.999f, 1e-8f, 0f, extras);
            for (int s = 0; s < steps; s++) plan.Step();
        }
        return w.GetDataArray().AsSpan(0, init.Length).ToArray();
    }

    private static double Objective(float[] c, float[] x)
    {
        double sum = 0;
        for (int i = 0; i < c.Length; i++) sum += (double)c[i] * x[i] * x[i];
        return sum;
    }

    /// <summary>
    /// L-BFGS must configure without throwing. It was previously rejected outright at configure time.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_Lbfgs_IsAccepted()
    {
        var engine = new CpuEngine();
        var w = new Tensor<float>(new[] { 4 });
        var c = new Tensor<float>(new[] { 4 });
        for (int i = 0; i < 4; i++) { w[i] = 1f; c[i] = 1f; }

        var plan = CompileQuadratic(engine, w, c);
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.LBFGS, LrSchedule.Constant(0.1f));
            plan.Step();
        }

        foreach (var v in w.GetDataArray().AsSpan(0, 4).ToArray())
        {
            Assert.False(float.IsNaN(v));
            Assert.False(float.IsInfinity(v));
        }
    }

    /// <summary>
    /// With no curvature pairs yet, the recursion reduces to <c>r = g</c> and the step must be exactly the
    /// gradient-descent step. That is correct L-BFGS behaviour and it is what makes the first update of an
    /// existing run unchanged.
    /// </summary>
    [Fact]
    public void FirstStepEqualsGradientDescent_BecauseNoCurvaturePairsExistYet()
    {
        var init = new[] { 1.0f, -2.0f, 0.5f };
        var curv = new[] { 1.0f, 1.0f, 1.0f };
        const float lr = 0.01f;

        var lbfgs = Run(init, curv, OptimizerType.LBFGS, steps: 1, lr);

        for (int i = 0; i < init.Length; i++)
        {
            // grad_i = 2*c_i*w_i
            float expected = init[i] - lr * (2f * curv[i] * init[i]);
            Assert.Equal(expected, lbfgs[i], 4);
        }
    }

    /// <summary>
    /// Once curvature pairs exist the direction must stop being the raw gradient, so the trajectory has to
    /// diverge from SGD. A recursion that silently produced <c>r = g</c> every step would pass every
    /// convergence check while not being L-BFGS.
    /// </summary>
    [Fact]
    public void DivergesFromGradientDescent_OnceCurvatureIsAccumulated()
    {
        var init = new[] { 1.0f, 1.0f, 1.0f, 1.0f };
        var curv = new[] { 1.0f, 4.0f, 16.0f, 50.0f };
        const float lr = 0.01f;
        const int steps = 12;

        var lbfgs = Run(init, curv, OptimizerType.LBFGS, steps, lr);
        var sgd = Run(init, curv, OptimizerType.SGD, steps, lr);

        double maxGap = 0;
        for (int i = 0; i < init.Length; i++)
            maxGap = Math.Max(maxGap, Math.Abs(lbfgs[i] - sgd[i]));

        Assert.True(maxGap > 1e-4,
            $"L-BFGS never diverged from plain SGD (max |Δ| = {maxGap}) — the two-loop recursion is not being applied.");
    }

    /// <summary>
    /// The payoff: on an ill-conditioned quadratic, curvature information must beat plain gradient descent.
    /// </summary>
    /// <remarks>
    /// This is the end-to-end check that the recursion is assembled the right way round. A sign error or a
    /// swapped loop order still produces a finite, plausible trajectory that diverges from SGD — satisfying
    /// the test above — but loses to it here.
    /// </remarks>
    [Fact]
    public void ConvergesRapidlyOnAnIllConditionedQuadratic()
    {
        var init = new[] { 1.0f, 1.0f, 1.0f, 1.0f };
        var curv = new[] { 1.0f, 4.0f, 16.0f, 50.0f };
        const int steps = 20;

        // lr = 1 is the natural quasi-Newton step: the two-loop recursion already returns H*g, so the
        // direction carries its own scale. Comparing L-BFGS and SGD at a SHARED small lr would measure
        // nothing about curvature — it would just show both crawling.
        var lbfgs = Run(init, curv, OptimizerType.LBFGS, steps, lr: 1.0f);

        double initial = Objective(curv, init);
        double final = Objective(curv, lbfgs);

        // Gradient descent cannot do this at any single fixed step size. Its best-case rate on a quadratic
        // with condition number k is ((k-1)/(k+1))^n; at k = 50 over 20 steps that is ~0.45 of the initial
        // objective. Requiring a 1e-3 reduction is therefore a claim only curvature information can satisfy.
        Assert.True(final < initial * 1e-3,
            $"L-BFGS reduced the objective only from {initial:G6} to {final:G6} in {steps} steps at lr=1 " +
            "— the curvature estimate is not being used (gradient descent's best case here is ~0.45x).");
    }

    /// <summary>
    /// The history size must actually bound memory and behaviour: m=1 and m=10 must produce different
    /// trajectories, or the ring buffer is not being consulted.
    /// </summary>
    [Fact]
    public void HistorySizeChangesTheTrajectory()
    {
        var init = new[] { 1.0f, 1.0f, 1.0f, 1.0f };
        var curv = new[] { 1.0f, 4.0f, 16.0f, 50.0f };
        const float lr = 0.01f;
        const int steps = 20;

        var small = Run(init, curv, OptimizerType.LBFGS, steps, lr, new FusedOptimizerExtras { LbfgsMemorySize = 1 });
        var large = Run(init, curv, OptimizerType.LBFGS, steps, lr, new FusedOptimizerExtras { LbfgsMemorySize = 10 });

        double maxGap = 0;
        for (int i = 0; i < init.Length; i++)
            maxGap = Math.Max(maxGap, Math.Abs(small[i] - large[i]));

        Assert.True(maxGap > 1e-5,
            $"m=1 and m=10 produced the same trajectory (max |Δ| = {maxGap}) — the history ring is not being used.");
    }

    /// <summary>
    /// A stationary point produces s = 0 and y = 0, so <c>s.y</c> is zero and the pair must be rejected rather
    /// than yielding <c>rho = 1/0</c> and poisoning the recursion with infinities.
    /// </summary>
    [Fact]
    public void RejectsDegenerateCurvaturePairs_WithoutProducingNonFiniteParameters()
    {
        // Zero curvature => zero gradient everywhere => s and y are both zero on every step.
        var init = new[] { 0.7f, -0.3f, 1.5f };
        var curv = new[] { 0.0f, 0.0f, 0.0f };

        var result = Run(init, curv, OptimizerType.LBFGS, steps: 8, lr: 0.1f);

        for (int i = 0; i < init.Length; i++)
        {
            Assert.False(float.IsNaN(result[i]), $"element {i} is NaN — a degenerate (s.y = 0) pair was accepted and rho divided by zero.");
            Assert.False(float.IsInfinity(result[i]), $"element {i} is Infinity — a degenerate pair was accepted.");
            // Zero gradient means nothing should move.
            Assert.Equal(init[i], result[i], 5);
        }
    }

    /// <summary>
    /// Per-group schedules are meaningless for L-BFGS: its curvature history spans every parameter, so it
    /// must be refused there rather than silently applying one group's schedule to a global recursion.
    /// </summary>
    [Fact]
    public void ConfigureOptimizerGrouped_Lbfgs_Throws()
    {
        var engine = new CpuEngine();
        var w = new Tensor<float>(new[] { 4 });
        var c = new Tensor<float>(new[] { 4 });
        for (int i = 0; i < 4; i++) { w[i] = 1f; c[i] = 1f; }

        var plan = CompileQuadratic(engine, w, c);
        using (plan)
        {
            var ex = Assert.Throws<NotSupportedException>(() => plan.ConfigureOptimizerGrouped(
                OptimizerType.LBFGS,
                new System.Collections.Generic.List<LrSchedule> { LrSchedule.Constant(0.1) },
                new System.Collections.Generic.List<int> { 0 }));
            Assert.Contains("LBFGS", ex.Message);
        }
    }
}
