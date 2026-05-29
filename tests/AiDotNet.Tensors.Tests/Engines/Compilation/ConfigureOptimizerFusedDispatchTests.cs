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
        // #76 part 2: optimizers wired via the FusedOptimizerExtras API extension.
        OptimizerType.AdaDelta,
        OptimizerType.LARS,
        OptimizerType.FTRL,
        OptimizerType.ASGD,
        OptimizerType.Rprop,
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
    /// Optimizers that need an execution model the per-parameter fused
    /// optimizer-step loop can't provide are intentionally NOT wired and must be
    /// rejected at configure time, so models using them fall back to the eager
    /// tape cleanly: SparseAdam (sparse-gradient index lists), LBFGS (closure
    /// line-search over the whole loss), HypergradientSGD / DAdaptationSGD (need
    /// a GLOBAL cross-parameter reduction, not per-tensor), ScheduleFreeSGD
    /// (needs a y-buffer written before the forward pass).
    /// </summary>
    [Theory]
    [InlineData(OptimizerType.SparseAdam)]
    [InlineData(OptimizerType.LBFGS)]
    [InlineData(OptimizerType.HypergradientSGD)]
    [InlineData(OptimizerType.ScheduleFreeSGD)]
    [InlineData(OptimizerType.DAdaptationSGD)]
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

    private static float[] SeedFloat(int n, int seed)
    {
        var rng = new Random(seed);
        var w = new float[n];
        for (int i = 0; i < n; i++) w[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return w;
    }

    private static float L1Norm(float[] w)
    {
        float s = 0;
        foreach (var x in w) s += Math.Abs(x);
        return s;
    }

    private static float[] RunFtrl(float[] init, float l1)
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(new[] { init.Length });
        for (int i = 0; i < init.Length; i++) weight[i] = init[i];

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var sq = engine.TensorMultiply(weight, weight);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            // The 6th positional arg is weightDecay; the 7th is the extras object.
            plan.ConfigureOptimizer(OptimizerType.FTRL, 0.1f, B1, B2, Eps, 0f,
                new FusedOptimizerExtras { L1 = l1, L2 = 0f, LrPower = -0.5f });
            for (int s = 0; s < 6; s++) plan.Step();
        }
        return weight.GetDataArray().AsSpan(0, init.Length).ToArray();
    }

    /// <summary>
    /// Proves the FusedOptimizerExtras values actually flow into the kernel (not
    /// ignored): FTRL with a strong L1 penalty must drive parameters closer to
    /// zero (sparsity via the L1 proximal soft-threshold) than FTRL with L1=0 on
    /// the same trajectory. If extras were dropped, both runs would be identical.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_FtrlExtras_L1_DrivesSparsity()
    {
        var init = SeedFloat(16, seed: 55);
        var noL1 = RunFtrl(init, l1: 0f);
        var strongL1 = RunFtrl(init, l1: 5f);

        foreach (var x in strongL1)
            Assert.True(!float.IsNaN(x) && !float.IsInfinity(x), "FTRL produced a non-finite parameter");

        Assert.True(L1Norm(strongL1) < L1Norm(noL1),
            $"FTRL L1 extras did not increase sparsity: ||w||₁(L1=5)={L1Norm(strongL1)} should be < " +
            $"||w||₁(L1=0)={L1Norm(noL1)} — extras may not be flowing into the kernel.");
    }
}
