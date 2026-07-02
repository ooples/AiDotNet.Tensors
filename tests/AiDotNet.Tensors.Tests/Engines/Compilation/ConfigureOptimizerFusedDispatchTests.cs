using System;
using System.Reflection;
using System.Runtime.ExceptionServices;
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

    public static TheoryData<OptimizerType> GpuPlanSupportedOptimizers => new()
    {
        OptimizerType.SGD,
        OptimizerType.Adam,
        OptimizerType.AdamW,
        OptimizerType.AMSGrad,
        OptimizerType.Nadam,
        OptimizerType.RMSprop,
        OptimizerType.Adagrad,
        OptimizerType.Lion,
        OptimizerType.SGDMomentum,
        OptimizerType.AdaMax,
    };

    [Theory]
    [MemberData(nameof(GpuPlanSupportedOptimizers))]
    public void ConfigureOptimizer_GpuPlanSupportedOptimizer_PassesEarlyGate(OptimizerType opt)
    {
        InvokeValidatePlanOptimizerSupport(opt, isFloat: true, hasGpuParams: true);
    }

    [Theory]
    [InlineData(OptimizerType.RAdam)]
    [InlineData(OptimizerType.LAMB)]
    [InlineData(OptimizerType.AdaDelta)]
    [InlineData(OptimizerType.LARS)]
    [InlineData(OptimizerType.FTRL)]
    [InlineData(OptimizerType.ASGD)]
    [InlineData(OptimizerType.Rprop)]
    [InlineData(OptimizerType.HypergradientSGD)]
    [InlineData(OptimizerType.DAdaptationSGD)]
    [InlineData(OptimizerType.ScheduleFreeSGD)]
    [InlineData(OptimizerType.SparseAdam)]
    [InlineData(OptimizerType.LBFGS)]
    public void ConfigureOptimizer_GpuPlanSemanticGap_ThrowsAtEarlyGate(OptimizerType opt)
    {
        Assert.Throws<NotSupportedException>(
            () => InvokeValidatePlanOptimizerSupport(opt, isFloat: true, hasGpuParams: true));
    }

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
    /// Optimizers that need an execution model the fused plan can't provide are
    /// intentionally NOT wired and must be rejected at configure time, so models
    /// using them fall back to the eager tape cleanly:
    ///  • SparseAdam — sparse-gradient index lists (the plan operates on dense grads),
    ///  • LBFGS — closure line-search (multiple loss evaluations per step).
    /// (HypergradientSGD and DAdaptationSGD are wired via the two-phase
    /// global-reduction path; ScheduleFreeSGD is wired via the pre-forward
    /// parameter-transform hook — see the dedicated tests below.)
    /// </summary>
    [Theory]
    [InlineData(OptimizerType.SparseAdam)]
    [InlineData(OptimizerType.LBFGS)]
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

    // ---- global-state optimizers (Hypergradient / D-Adaptation) -----------

    private static float[] RunGlobal(float[] init, OptimizerType opt, int steps, FusedOptimizerExtras? extras)
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
            plan.ConfigureOptimizer(opt, Lr, B1, B2, Eps, 0f, extras);
            for (int s = 0; s < steps; s++) plan.Step();
        }
        return weight.GetDataArray().AsSpan(0, init.Length).ToArray();
    }

    /// <summary>
    /// Hypergradient SGD adapts ONE global learning rate from the inner product of
    /// successive gradients. Step 1 (no prior gradient) equals plain SGD; over a
    /// consistent-direction trajectory the adapted LR grows, so it must diverge from
    /// plain SGD — proving the global reduction is wired (not per-tensor / a no-op).
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_HypergradientSGD_AdaptsGlobalLrAndDivergesFromSGD()
    {
        var init = SeedFloat(16, seed: 71);
        const int steps = 12;
        var hg = RunGlobal(init, OptimizerType.HypergradientSGD, steps, new FusedOptimizerExtras { HyperLr = 1e-3f });
        var sgd = RunGlobal(init, OptimizerType.SGD, steps, null);

        double maxAbs = 0;
        for (int i = 0; i < init.Length; i++)
        {
            Assert.True(!float.IsNaN(hg[i]) && !float.IsInfinity(hg[i]), "Hypergradient produced a non-finite parameter");
            maxAbs = Math.Max(maxAbs, Math.Abs(hg[i] - sgd[i]));
        }
        Assert.True(maxAbs > 1e-5, $"Hypergradient never diverged from SGD — global LR adaptation appears unwired (max |Δ| = {maxAbs}).");
    }

    /// <summary>
    /// D-Adaptation maintains a single global distance estimate d that should GROW
    /// from a small d0, so the parameters move substantially more than d0·lr·g would
    /// in one step. Asserts finite, moves meaningfully, and (vs a tiny fixed d0 SGD-like
    /// step) that the global d estimate has taken effect.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_DAdaptationSGD_GrowsGlobalDistanceEstimate()
    {
        var init = SeedFloat(16, seed: 73);
        const int steps = 20;
        var d = RunGlobal(init, OptimizerType.DAdaptationSGD, steps, new FusedOptimizerExtras { D0 = 1e-6f });

        double maxMove = 0;
        for (int i = 0; i < init.Length; i++)
        {
            Assert.True(!float.IsNaN(d[i]) && !float.IsInfinity(d[i]), "D-Adaptation produced a non-finite parameter");
            maxMove = Math.Max(maxMove, Math.Abs(d[i] - init[i]));
        }
        // With d frozen at d0=1e-6 the total move over 20 steps would be ~O(1e-6·lr·20·|g|) ≈ 1e-6.
        // A meaningfully larger move proves d grew well above d0 (the adaptation is active).
        Assert.True(maxMove > 1e-3, $"D-Adaptation distance estimate did not grow (max move = {maxMove}).");
    }

    /// <summary>
    /// Schedule-Free SGD (Defazio et al. 2024) evaluates gradients at the
    /// interpolation y = (1-β)z + βx (written into the live backing by the
    /// plan's pre-forward hook) and returns the running-average eval copy x as
    /// the parameter. On the convex loss Σ wᵢ² the eval weights must shrink
    /// toward 0 (training works) AND differ from plain SGD's trajectory (the
    /// averaging is active, not a silent SGD passthrough) while staying finite.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_ScheduleFreeSGD_AveragesAndDivergesFromSGD()
    {
        var init = SeedFloat(16, seed: 79);
        const int steps = 30;
        var sf = RunGlobal(init, OptimizerType.ScheduleFreeSGD, steps, new FusedOptimizerExtras { SfBeta = 0.9f });
        var sgd = RunGlobal(init, OptimizerType.SGD, steps, null);

        double initNorm = L1Norm(init), sfNorm = L1Norm(sf), maxAbs = 0;
        for (int i = 0; i < init.Length; i++)
        {
            Assert.True(!float.IsNaN(sf[i]) && !float.IsInfinity(sf[i]), "Schedule-Free produced a non-finite parameter");
            maxAbs = Math.Max(maxAbs, Math.Abs(sf[i] - sgd[i]));
        }
        Assert.True(sfNorm < initNorm,
            $"Schedule-Free eval weights did not shrink on Σwᵢ²: ‖w‖₁ {sfNorm} should be < initial {initNorm}.");
        Assert.True(maxAbs > 1e-5,
            $"Schedule-Free never diverged from plain SGD — the averaging/pre-forward hook appears unwired (max |Δ| = {maxAbs}).");
    }

    /// <summary>Global-state optimizers are rejected with per-group schedules (configure-time).</summary>
    [Theory]
    [InlineData(OptimizerType.HypergradientSGD)]
    [InlineData(OptimizerType.DAdaptationSGD)]
    [InlineData(OptimizerType.ScheduleFreeSGD)]
    public void ConfigureOptimizerGrouped_GlobalStateOptimizer_Throws(OptimizerType opt)
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
            var schedules = new System.Collections.Generic.List<LrSchedule> { LrSchedule.Constant(0.01) };
            var map = new System.Collections.Generic.List<int> { 0 };
            Assert.Throws<NotSupportedException>(() => plan.ConfigureOptimizerGrouped(opt, schedules, map, B1, B2, Eps));
        }
    }

    private static void InvokeValidatePlanOptimizerSupport(OptimizerType opt, bool isFloat, bool hasGpuParams)
    {
        var method = typeof(CompiledTrainingPlan<float>).GetMethod(
            "ValidatePlanOptimizerSupport",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        try
        {
            method!.Invoke(null, [opt, isFloat, hasGpuParams]);
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
        }
    }
}
