using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Covers the fused trust-region Cauchy-point step.
/// </summary>
/// <remarks>
/// <para>
/// The step is <c>alpha = min(radius/||g||, lr); x -= alpha*g</c> — Nocedal &amp; Wright Algorithm 4.1 with
/// B = I. It is not a per-element kernel because <c>||g||</c> couples every parameter, but the plan does not
/// need one: this is a global reduction followed by an elementwise step, the same shape HypergradientSGD
/// uses.
/// </para>
/// <para>
/// What makes it trust-region rather than SGD is the CAP: once the gradient is large enough that
/// <c>radius/||g|| &lt; lr</c>, the step length stops growing with the gradient and saturates at the radius.
/// That is the property these tests target, because an implementation that silently dropped the cap would
/// still descend and would still look correct on a small-gradient fixture.
/// </para>
/// </remarks>
public class ConfigureOptimizerTrustRegionTests
{
    private static float[] Run(float[] init, float[] curvature, int steps, float lr, float radius)
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
            plan.ConfigureOptimizer(
                OptimizerType.TrustRegion, LrSchedule.Constant(lr), 0f, 0f, 0f, 0f,
                new FusedOptimizerExtras { TrustRegionRadius = radius });
            for (int s = 0; s < steps; s++) plan.Step();
        }
        return w.GetDataArray().AsSpan(0, init.Length).ToArray();
    }

    /// <summary>
    /// The defining property: a large gradient must NOT produce a proportionally large step. Once
    /// <c>radius/||g|| &lt; lr</c> the step saturates, so the movement is bounded by the radius rather than
    /// scaling with the gradient.
    /// </summary>
    /// <remarks>
    /// A plain SGD step would move 100x further for a 100x larger gradient. This is the assertion that
    /// separates the two, and it is why the fixture uses a deliberately huge gradient.
    /// </remarks>
    [Fact]
    public void StepLengthSaturatesAtTheRadius_RatherThanScalingWithTheGradient()
    {
        const float radius = 0.1f, lr = 1.0f;

        // grad = 2*c*w. Small: 2*1*1 = 2. Large: 2*100*1 = 200.
        var small = Run(new[] { 1.0f }, new[] { 1.0f }, steps: 1, lr, radius);
        var large = Run(new[] { 1.0f }, new[] { 100.0f }, steps: 1, lr, radius);

        double smallMove = Math.Abs(1.0 - small[0]);
        double largeMove = Math.Abs(1.0 - large[0]);

        // Both gradients exceed radius/lr, so alpha = radius/||g|| in both cases and the MOVE is exactly
        // the radius regardless of gradient magnitude.
        Assert.Equal(radius, smallMove, 4);
        Assert.Equal(radius, largeMove, 4);

        // A 100x gradient under plain SGD would move 100x further. It does not.
        Assert.True(Math.Abs(largeMove - smallMove) < 1e-4,
            $"step grew with the gradient ({smallMove:G6} -> {largeMove:G6}) — the trust-region cap is not applied.");
    }

    /// <summary>
    /// Below the cap the step is the ordinary gradient step, so a tiny gradient must behave like SGD.
    /// </summary>
    [Fact]
    public void FallsBackToTheGradientStep_WhenTheRadiusIsNotBinding()
    {
        const float lr = 0.01f, radius = 1000f;   // radius/||g|| >> lr, so alpha = lr
        var init = new[] { 1.0f, -2.0f };
        var curv = new[] { 1.0f, 1.0f };

        var result = Run(init, curv, steps: 1, lr, radius);

        for (int i = 0; i < init.Length; i++)
        {
            float grad = 2f * curv[i] * init[i];
            Assert.Equal(init[i] - lr * grad, result[i], 4);
        }
    }

    /// <summary>
    /// A larger radius must permit a larger step, or the radius is being ignored.
    /// </summary>
    [Fact]
    public void RadiusControlsTheStepLength()
    {
        var init = new[] { 1.0f };
        var curv = new[] { 100.0f };   // large gradient so the radius binds

        var tight = Run(init, curv, steps: 1, lr: 1.0f, radius: 0.01f);
        var loose = Run(init, curv, steps: 1, lr: 1.0f, radius: 0.20f);

        double tightMove = Math.Abs(1.0 - tight[0]);
        double looseMove = Math.Abs(1.0 - loose[0]);

        Assert.True(looseMove > tightMove * 10,
            $"a 20x larger radius did not produce a proportionally larger step ({tightMove:G6} vs {looseMove:G6}).");
    }

    /// <summary>
    /// A vanishing gradient must not divide radius by ~0 and launch the parameters to infinity.
    /// </summary>
    [Fact]
    public void ZeroGradientTakesNoStep_RatherThanAnUnboundedOne()
    {
        var result = Run(new[] { 0.7f, -0.3f }, new[] { 0.0f, 0.0f }, steps: 5, lr: 1.0f, radius: 1.0f);

        Assert.Equal(0.7f, result[0], 5);
        Assert.Equal(-0.3f, result[1], 5);
        foreach (var v in result)
        {
            Assert.False(float.IsNaN(v));
            Assert.False(float.IsInfinity(v));
        }
    }

    /// <summary>
    /// It still converges: the cap bounds the step, it does not stop progress.
    /// </summary>
    [Fact]
    public void ConvergesOnAQuadratic()
    {
        var init = new[] { 1.0f, 1.0f, 1.0f };
        var curv = new[] { 1.0f, 4.0f, 16.0f };

        var result = Run(init, curv, steps: 60, lr: 1.0f, radius: 0.05f);

        double before = 0, after = 0;
        for (int i = 0; i < init.Length; i++)
        {
            before += curv[i] * init[i] * init[i];
            after += curv[i] * result[i] * result[i];
        }
        Assert.True(after < before * 0.5,
            $"trust-region did not reduce the objective ({before:G6} -> {after:G6}).");
    }

    /// <summary>
    /// Per-group schedules are meaningless when the step length depends on the norm of the WHOLE gradient.
    /// </summary>
    [Fact]
    public void ConfigureOptimizerGrouped_TrustRegion_Throws()
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
                OptimizerType.TrustRegion,
                new System.Collections.Generic.List<LrSchedule> { LrSchedule.Constant(0.1) },
                new System.Collections.Generic.List<int> { 0 }));
            Assert.Contains("TrustRegion", ex.Message);
        }
    }
}
