using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// #76: wire the remaining kernel-backed optimizers (Nadam, RAdam, LAMB, RMSprop,
/// Adagrad, Lion, SGDMomentum, AdaMax) into CompiledTrainingPlan's float CPU
/// fused-update dispatch. Each previously threw NotSupportedException at
/// ConfigureOptimizer (kernel existed in FusedOptimizer but no plan dispatch
/// branch), forcing the eager tape.
///
/// These tests exercise the new DISPATCH wiring end-to-end through the plan
/// (buffer-slot selection + hyperparameter mapping + per-step replay): each wired
/// optimizer must configure without throwing, update parameters in place, keep
/// them finite, and move them meaningfully over several steps. The numerical
/// correctness of each kernel's math is covered by the kernels' own tests; this
/// validates that the plan routes the right buffers/params to the right kernel.
/// (AMSGrad additionally has full kernel-direct parity in
/// ConfigureOptimizerAMSGradTests.)
/// </summary>
public class ConfigureOptimizerFusedDispatchTests
{
    private const float Lr = 0.03f, B1 = 0.9f, B2 = 0.99f, Eps = 1e-8f;

    public static TheoryData<OptimizerType> WiredFloatOptimizers => new()
    {
        OptimizerType.SGD,
        OptimizerType.Adam,
        OptimizerType.AdamW,
        OptimizerType.AMSGrad,
        OptimizerType.Nadam,
        OptimizerType.RAdam,
        OptimizerType.LAMB,
        OptimizerType.RMSprop,
        OptimizerType.Adagrad,
        OptimizerType.Lion,
        OptimizerType.SGDMomentum,
        OptimizerType.AdaMax,
    };

    /// <summary>
    /// Every wired float optimizer must dispatch through the fused plan (no
    /// NotSupportedException), update the parameter in place, stay finite, and
    /// move it meaningfully over several steps. Catches a missing buffer
    /// (crash/NaN), a missing gate entry (throw), or a grossly wrong mapping
    /// (no movement / blow-up).
    /// </summary>
    [Theory]
    [MemberData(nameof(WiredFloatOptimizers))]
    public void ConfigureOptimizer_WiredFloatOptimizer_DispatchesAndUpdatesInPlace(OptimizerType opt)
    {
        var engine = new CpuEngine();
        const int n = 16;
        var rng = new Random(101);
        var weight = new Tensor<float>(new[] { n });
        var init = new float[n];
        for (int i = 0; i < n; i++) { init[i] = (float)(rng.NextDouble() * 2.0 - 1.0); weight[i] = init[i]; }

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var sq = engine.TensorMultiply(weight, weight);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            // Must NOT throw — this is the regression for the per-optimizer
            // NotSupportedException that forced eager fallback.
            plan.ConfigureOptimizer(opt, Lr, B1, B2, Eps);
            for (int s = 0; s < 5; s++) plan.Step();
        }

        var post = weight.GetDataArray();
        double maxAbs = 0;
        for (int i = 0; i < n; i++)
        {
            Assert.True(!float.IsNaN(post[i]) && !float.IsInfinity(post[i]),
                $"{opt}: fused step produced a non-finite parameter at {i} ({post[i]}).");
            maxAbs = Math.Max(maxAbs, Math.Abs(post[i] - init[i]));
        }
        Assert.True(maxAbs > 1e-6,
            $"{opt}: fused dispatch did not move the parameter over 5 steps (max |Δ| = {maxAbs}).");
    }

    /// <summary>
    /// The newly-wired optimizers have float-only kernels. A double plan must
    /// reject them at ConfigureOptimizer (the dtype-aware gate) rather than
    /// configure-then-throw at step time, so callers cleanly stay on eager.
    /// </summary>
    [Theory]
    [InlineData(OptimizerType.Nadam)]
    [InlineData(OptimizerType.RMSprop)]
    [InlineData(OptimizerType.Lion)]
    [InlineData(OptimizerType.AdaMax)]
    public void ConfigureOptimizer_FloatOnlyOptimizer_OnDoublePlan_Throws(OptimizerType opt)
    {
        var engine = new CpuEngine();
        var weight = new Tensor<double>(new[] { 8 });
        for (int i = 0; i < 8; i++) weight[i] = 0.1 * (i + 1);

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var sq = engine.TensorMultiply(weight, weight);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            Assert.Throws<NotSupportedException>(() => plan.ConfigureOptimizer(opt, Lr, B1, B2, Eps));
        }
    }

    /// <summary>
    /// Optimizers whose kernels need hyperparameters the ConfigureOptimizer API
    /// doesn't carry (AdaDelta rho+2 accumulators, FTRL l1/l2/lr_power, LARS
    /// trust coeff, ASGD lambd/alpha/mu) are intentionally NOT wired and must
    /// still be rejected at configure time — so models using them fall back to
    /// the eager tape cleanly instead of configuring then throwing mid-step.
    /// </summary>
    [Theory]
    [InlineData(OptimizerType.AdaDelta)]
    [InlineData(OptimizerType.FTRL)]
    [InlineData(OptimizerType.LARS)]
    [InlineData(OptimizerType.ASGD)]
    public void ConfigureOptimizer_UnwiredOptimizer_Throws(OptimizerType opt)
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(new[] { 8 });
        for (int i = 0; i < 8; i++) weight[i] = 0.1f * (i + 1);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var sq = engine.TensorMultiply(weight, weight);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            Assert.Throws<NotSupportedException>(() => plan.ConfigureOptimizer(opt, Lr, B1, B2, Eps));
        }
    }
}
