using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class CompiledOptimizerEnrollmentTests
{
    private static Tensor<float> Scalar(float value) => new(new[] { value }, new[] { 1 });

    private static CompiledTrainingPlan<float> Plan(CpuEngine engine, Tensor<float> p, Tensor<float> q, bool freezeP)
    {
        using var scope = GraphMode.EnableTraining(new[] { p, q });
        Tensor<float> twiceP;
        if (freezeP)
        {
            using (new NoGradScope<float>()) twiceP = engine.TensorMultiplyScalar(p, 2f);
        }
        else twiceP = engine.TensorMultiplyScalar(p, 2f);
        return Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(new[] { p, q },
            engine.ReduceSum(engine.TensorMultiply(twiceP, q), null, false)));
    }

    // Independent scalar recurrence: an absent gradient does not enter this function at all.
    private static void AdamW(ref double value, double gradient, ref double m, ref double v, ref int age, double lr)
    {
        age++;
        m = .9 * m + .1 * gradient;
        v = .999 * v + .001 * gradient * gradient;
        value = value * (1 - lr * .2) - lr * (m / (1 - Math.Pow(.9, age)))
            / (Math.Sqrt(v / (1 - Math.Pow(.999, age))) + 1e-8);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AdamW_handover_keeps_each_parameters_accepted_update_age(bool scheduled)
    {
        var engine = new CpuEngine();
        var p = Scalar(2); var q = Scalar(3);
        using var a = Plan(engine, p, q, true);
        using var b = Plan(engine, p, q, false);
        var schedule = scheduled ? LrSchedule.Exponential(.1, .8) : LrSchedule.Constant(.1);
        a.ConfigureOptimizer(OptimizerType.AdamW, schedule, weightDecay: .2);
        b.ConfigureOptimizer(OptimizerType.AdamW, schedule, weightDecay: .2);
        double expectedP = 2, expectedQ = 3, pm = 0, pv = 0, qm = 0, qv = 0;
        int pa = 0, qa = 0;
        var sequence = new[] { a, a, b, b, a, b, a };
        CompiledTrainingPlan<float>? previous = null;
        int accepted = 0;
        foreach (var current in sequence)
        {
            if (previous is not null && !ReferenceEquals(previous, current)) current.ContinueOptimizerFrom(previous);
            double gp = 2 * expectedQ, gq = 2 * expectedP;
            double lr = scheduled ? .1 * Math.Pow(.8, accepted) : .1;
            accepted++;
            if (ReferenceEquals(current, b)) AdamW(ref expectedP, gp, ref pm, ref pv, ref pa, lr);
            AdamW(ref expectedQ, gq, ref qm, ref qv, ref qa, lr);
            current.Step();
            Assert.Equal(expectedP, p.ToArray()[0], 5);
            Assert.Equal(expectedQ, q.ToArray()[0], 5);
            Assert.Equal(new long[] { pa, qa }, current.ParameterAcceptedUpdates);
            previous = current;
        }
    }

    [Fact]
    public void Connected_zero_gradient_still_decays_and_advances_age()
    {
        var engine = new CpuEngine(); var p = Scalar(2); var q = Scalar(3);
        using var scope = GraphMode.EnableTraining(new[] { p, q });
        using var plan = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(new[] { p, q },
            engine.ReduceSum(engine.TensorAdd(engine.TensorMultiplyScalar(p, 0f), q), null, false)));
        plan.ConfigureOptimizer(OptimizerType.AdamW, .1, weightDecay: .2);
        plan.Step();
        Assert.Equal(1.96f, p.ToArray()[0], 6);
        Assert.Equal(new long[] { 1, 1 }, plan.ParameterAcceptedUpdates);
    }

    [Fact]
    public void External_membership_temporarily_activates_a_frozen_parameter()
    {
        var engine = new CpuEngine(); var p = Scalar(2); var q = Scalar(3);
        using var plan = Plan(engine, p, q, true);
        plan.ConfigureOptimizer(OptimizerType.SGD, .1, weightDecay: 0);
        Assert.True(plan.TryApplyGradients(new Dictionary<Tensor<float>, Tensor<float>> { [p] = Scalar(5) }));
        Assert.Equal(1.5f, p.ToArray()[0]); Assert.Equal(3f, q.ToArray()[0]);
        plan.Step();
        Assert.Equal(1.5f, p.ToArray()[0]); Assert.Equal(2.7f, q.ToArray()[0], 6);
        Assert.Equal(new long[] { 1, 1 }, plan.ParameterAcceptedUpdates);
    }

    [Fact]
    public void External_shape_preflight_does_not_partially_update_parameters()
    {
        var engine = new CpuEngine(); var p = Scalar(2); var q = Scalar(3);
        using var plan = Plan(engine, p, q, false);
        plan.ConfigureOptimizer(OptimizerType.AdamW, .1, weightDecay: .2);
        Assert.Throws<ArgumentException>(() => plan.TryApplyGradients(new Dictionary<Tensor<float>, Tensor<float>>
        {
            [p] = Scalar(5), [q] = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 })
        }));
        Assert.Equal(2f, p.ToArray()[0]); Assert.Equal(3f, q.ToArray()[0]);
        Assert.Equal(new long[] { 0, 0 }, plan.ParameterAcceptedUpdates);
    }

    [Fact]
    public void Where_condition_is_absent_but_unselected_branch_has_a_connected_zero_gradient()
    {
        var engine = new CpuEngine(); var condition = Scalar(1); var x = Scalar(2); var y = Scalar(3);
        using var scope = GraphMode.EnableTraining(new[] { condition, x, y });
        using var plan = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(new[] { condition, x, y },
            engine.ReduceSum(engine.TensorWhere(condition, x, y), null, false)));
        plan.ConfigureOptimizer(OptimizerType.AdamW, .1, weightDecay: .2);
        plan.Step();
        Assert.Equal(1f, condition.ToArray()[0]);
        Assert.Equal(1.86f, x.ToArray()[0], 5);
        Assert.Equal(2.94f, y.ToArray()[0], 5);
        Assert.Equal(new long[] { 0, 1, 1 }, plan.ParameterAcceptedUpdates);
    }

    [Fact]
    public void Nonfinite_external_gradient_does_not_advance_any_parameter_age()
    {
        var engine = new CpuEngine(); var p = Scalar(2); var q = Scalar(3);
        using var plan = Plan(engine, p, q, false);
        plan.ConfigureOptimizer(OptimizerType.AdamW, .1, weightDecay: .2);
        plan.SetNonFiniteGradientPolicy(AiDotNet.Tensors.Engines.Gpu.NonFiniteGradientPolicy.LeaveUnscaled);
        Assert.True(plan.TryApplyGradients(new Dictionary<Tensor<float>, Tensor<float>> { [p] = Scalar(float.NaN) }));
        Assert.Equal(2f, p.ToArray()[0]); Assert.Equal(3f, q.ToArray()[0]);
        Assert.Equal(new long[] { 0, 0 }, plan.ParameterAcceptedUpdates);
        plan.Step();
        Assert.Equal(new long[] { 1, 1 }, plan.ParameterAcceptedUpdates);
        Assert.Equal(1.86f, p.ToArray()[0], 5); Assert.Equal(2.84f, q.ToArray()[0], 5);
    }

    [Theory]
    [InlineData(OptimizerType.SGD)]
    [InlineData(OptimizerType.Adam)]
    [InlineData(OptimizerType.AdamW)]
    [InlineData(OptimizerType.AMSGrad)]
    public void Double_optimizer_skips_absent_parameters(OptimizerType optimizer)
    {
        var engine = new CpuEngine();
        var p = new Tensor<double>(new[] { 2d }, new[] { 1 });
        var q = new Tensor<double>(new[] { 3d }, new[] { 1 });
        using var scope = GraphMode.EnableTraining(new[] { p, q });
        Tensor<double> data;
        using (new NoGradScope<double>()) data = engine.TensorMultiplyScalar(p, 2d);
        using var plan = Assert.IsType<CompiledTrainingPlan<double>>(scope.CompileTraining(new[] { p, q },
            engine.ReduceSum(engine.TensorMultiply(data, q), null, false)));
        plan.ConfigureOptimizer(optimizer, .1, weightDecay: .2);
        plan.Step(); plan.Step();
        Assert.Equal(2d, p.ToArray()[0]);
        Assert.True(q.ToArray()[0] < 3d);
        Assert.Equal(new long[] { 0, 2 }, plan.ParameterAcceptedUpdates);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Grouped_AdamW_preserves_membership_and_individual_ages(bool doubles)
    {
        if (doubles)
        {
            var engine = new CpuEngine();
            var p = new Tensor<double>(new[] { 2d }, new[] { 1 });
            var q = new Tensor<double>(new[] { 3d }, new[] { 1 });
            using var scope = GraphMode.EnableTraining(new[] { p, q });
            Tensor<double> data;
            using (new NoGradScope<double>()) data = engine.TensorMultiplyScalar(p, 2d);
            using var plan = Assert.IsType<CompiledTrainingPlan<double>>(scope.CompileTraining(new[] { p, q },
                engine.ReduceSum(engine.TensorMultiply(data, q), null, false)));
            plan.ConfigureOptimizerGrouped(OptimizerType.AdamW,
                new[] { LrSchedule.Constant(.1), LrSchedule.Constant(.01) }, new[] { 0, 1 }, weightDecay: .2);
            plan.RequestInt8MomentStorage(true);
            Assert.Throws<NotSupportedException>(() => plan.ConfigureOptimizerGrouped(OptimizerType.AdamW,
                new[] { OptimizerType.AdamW, OptimizerType.SGDMomentum },
                new[] { LrSchedule.Constant(.1), LrSchedule.Constant(.01) }, new[] { 0, 1 }, weightDecay: .2));
            plan.RequestInt8MomentStorage(false);
            plan.Step();
            Assert.Equal(2d, p.ToArray()[0]); Assert.Equal(2.984d, q.ToArray()[0], 7);
            Assert.True(plan.TryApplyGradients(new Dictionary<Tensor<double>, Tensor<double>>
                { [p] = new(new[] { 5d }, new[] { 1 }) }));
            Assert.Equal(1.86d, p.ToArray()[0], 7); Assert.Equal(2.984d, q.ToArray()[0], 7);
            Assert.Equal(new long[] { 1, 1 }, plan.ParameterAcceptedUpdates);
        }
        else
        {
            var engine = new CpuEngine(); var p = Scalar(2); var q = Scalar(3);
            using var plan = Plan(engine, p, q, true);
            plan.ConfigureOptimizerGrouped(OptimizerType.AdamW,
                new[] { LrSchedule.Constant(.1), LrSchedule.Constant(.01) }, new[] { 0, 1 }, weightDecay: .2);
            plan.RequestInt8MomentStorage(true);
            Assert.Throws<NotSupportedException>(() => plan.ConfigureOptimizerGrouped(OptimizerType.AdamW,
                new[] { OptimizerType.AdamW, OptimizerType.SGDMomentum },
                new[] { LrSchedule.Constant(.1), LrSchedule.Constant(.01) }, new[] { 0, 1 }, weightDecay: .2));
            plan.RequestInt8MomentStorage(false);
            plan.Step();
            Assert.Equal(2f, p.ToArray()[0]); Assert.Equal(2.984f, q.ToArray()[0], 5);
            Assert.True(plan.TryApplyGradients(new Dictionary<Tensor<float>, Tensor<float>> { [p] = Scalar(5) }));
            Assert.Equal(1.86f, p.ToArray()[0], 5); Assert.Equal(2.984f, q.ToArray()[0], 5);
            Assert.Equal(new long[] { 1, 1 }, plan.ParameterAcceptedUpdates);
        }
    }

    [Theory]
    [InlineData(-1, 1)]
    [InlineData(1, 1)]
    [InlineData(1, 0)]
    [InlineData(0, 2)]
    public void Partial_edge_contract_rejects_invalid_operand_indices(int first, int second)
    {
        var p = Scalar(2); var q = Scalar(3);
        using var scope = GraphMode.EnableTraining(new[] { p, q });
        Assert.Throws<ArgumentException>(() => scope.RecordVariadic<float>(LazyNodeType.Custom, "TensorMultiply",
            new[] { p, q }, new[] { 1 }, (_, _) => { }, differentiableInputIndices: new[] { first, second }));
    }

    [Fact]
    public void Partial_edge_contract_distinguishes_repeated_operand_occurrences()
    {
        var engine = new CpuEngine(); var p = Scalar(2);
        using var scope = GraphMode.EnableTraining(new[] { p });
        var declared = new[] { 1 };
        var product = scope.RecordVariadic(LazyNodeType.Custom, "TensorMultiply", new[] { p, p }, new[] { 1 },
            (e, output) => DirectGpuTensorEngine.CopyResultInto(e, e.TensorMultiply(p, p), output),
            backwardFn: (gradient, inputs, _, _, e, gradients) =>
                DifferentiableOps.AccumulateGrad(gradients, inputs[1], e.TensorMultiply(gradient, inputs[0]), e),
            differentiableInputIndices: declared);
        declared[0] = 0; // The trace owns an immutable copy of its declared operand contract.
        using var plan = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(new[] { p },
            engine.ReduceSum(product, null, false)));
        plan.ConfigureOptimizer(OptimizerType.SGD, .1);
        plan.Step();
        Assert.Equal(1.8f, p.ToArray()[0], 6); // One occurrence contributes p, rather than 2*p.
        Assert.Equal(new long[] { 1 }, plan.ParameterAcceptedUpdates);
    }

    [Theory]
    [InlineData(OptimizerType.SGD)]
    [InlineData(OptimizerType.Adam)]
    [InlineData(OptimizerType.AdamW)]
    [InlineData(OptimizerType.AMSGrad)]
    [InlineData(OptimizerType.Nadam)]
    [InlineData(OptimizerType.RAdam)]
    [InlineData(OptimizerType.LAMB)]
    [InlineData(OptimizerType.RMSprop)]
    [InlineData(OptimizerType.Adagrad)]
    [InlineData(OptimizerType.Lion)]
    [InlineData(OptimizerType.SGDMomentum)]
    [InlineData(OptimizerType.AdaMax)]
    [InlineData(OptimizerType.AdaDelta)]
    [InlineData(OptimizerType.LARS)]
    [InlineData(OptimizerType.FTRL)]
    [InlineData(OptimizerType.ASGD)]
    [InlineData(OptimizerType.Rprop)]
    [InlineData(OptimizerType.ProximalL1)]
    [InlineData(OptimizerType.ADMM)]
    public void Per_parameter_optimizer_handover_matches_independent_parameter_timelines(OptimizerType optimizer)
    {
        var engine = new CpuEngine(); var p = Scalar(2); var q = Scalar(3);
        var referenceP = Scalar(2); var referenceQ = Scalar(3);
        CompiledTrainingPlan<float> Trace(bool freeze)
        {
            using var scope = GraphMode.EnableTraining(new[] { p, q });
            Tensor<float> first;
            if (freeze) { using (new NoGradScope<float>()) first = engine.TensorMultiplyScalar(p, 2f); }
            else first = engine.TensorMultiplyScalar(p, 2f);
            return Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(new[] { p, q },
                engine.ReduceSum(engine.TensorAdd(first, engine.TensorMultiplyScalar(q, 3f)), null, false)));
        }
        CompiledTrainingPlan<float> Reference(Tensor<float> parameter, float gradient)
        {
            using var scope = GraphMode.EnableTraining(new[] { parameter });
            return Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(new[] { parameter },
                engine.ReduceSum(engine.TensorMultiplyScalar(parameter, gradient), null, false)));
        }
        using var a = Trace(true); using var b = Trace(false);
        using var pOnly = Reference(referenceP, 2); using var qOnly = Reference(referenceQ, 3);
        foreach (var plan in new[] { a, b, pOnly, qOnly }) plan.ConfigureOptimizer(optimizer, .001);
        CompiledTrainingPlan<float>? previous = null;
        int pAge = 0, qAge = 0;
        foreach (var current in new[] { a, a, b, b, a, b, a, b })
        {
            if (previous is not null && !ReferenceEquals(current, previous)) current.ContinueOptimizerFrom(previous);
            if (ReferenceEquals(current, b)) { pOnly.Step(); pAge++; }
            qOnly.Step(); qAge++;
            current.Step();
            Assert.Equal(referenceP.ToArray()[0], p.ToArray()[0], 6);
            Assert.Equal(referenceQ.ToArray()[0], q.ToArray()[0], 6);
            Assert.Equal(new long[] { pAge, qAge }, current.ParameterAcceptedUpdates);
            var actual = Assert.IsType<FusedOptimizerCheckpoint>(current.CaptureFusedOptimizerCheckpoint());
            var pState = Assert.IsType<FusedOptimizerCheckpoint>(pOnly.CaptureFusedOptimizerCheckpoint()).Parameters[0];
            var qState = Assert.IsType<FusedOptimizerCheckpoint>(qOnly.CaptureFusedOptimizerCheckpoint()).Parameters[0];
            foreach (var pair in new[] { (Actual: actual.Parameters[0], Expected: pState),
                         (Actual: actual.Parameters[1], Expected: qState) })
            {
                Assert.Equal(pair.Expected.MFloat, pair.Actual.MFloat);
                Assert.Equal(pair.Expected.VFloat, pair.Actual.VFloat);
                Assert.Equal(pair.Expected.VMaxFloat, pair.Actual.VMaxFloat);
            }
            previous = current;
        }
    }

    [Theory]
    [InlineData(OptimizerType.LBFGS)]
    [InlineData(OptimizerType.TrustRegion)]
    [InlineData(OptimizerType.ConjugateGradient)]
    [InlineData(OptimizerType.HypergradientSGD)]
    [InlineData(OptimizerType.DAdaptationSGD)]
    [InlineData(OptimizerType.ScheduleFreeSGD)]
    public void Coupled_optimizer_matches_active_subspace_and_refuses_changed_domain(OptimizerType optimizer)
    {
        var engine = new CpuEngine(); var p = Scalar(2); var q = Scalar(3); var referenceQ = Scalar(3);
        using var plan = Plan(engine, p, q, true);
        CompiledTrainingPlan<float> reference;
        using (var scope = GraphMode.EnableTraining(new[] { referenceQ }))
            reference = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(new[] { referenceQ },
                engine.ReduceSum(engine.TensorMultiplyScalar(referenceQ, 4f), null, false)));
        using (reference)
        {
            plan.ConfigureOptimizer(optimizer, .01);
            reference.ConfigureOptimizer(optimizer, .01);
            for (int step = 0; step < 4; step++)
            {
                plan.Step(); reference.Step();
                Assert.Equal(2f, p.ToArray()[0]);
                Assert.Equal(referenceQ.ToArray()[0], q.ToArray()[0], 6);
                Assert.Equal(new long[] { 0, step + 1 }, plan.ParameterAcceptedUpdates);
            }
            float before = q.ToArray()[0];
            plan.RequestInt8MomentStorage(true);
            Assert.Throws<NotSupportedException>(() => plan.ConfigureOptimizer(OptimizerType.SGDMomentum, .01));
            plan.RequestInt8MomentStorage(false);
            var changed = new Dictionary<Tensor<float>, Tensor<float>> { [p] = Scalar(1), [q] = Scalar(4) };
            if (optimizer == OptimizerType.ScheduleFreeSGD)
                Assert.Throws<NotSupportedException>(() => plan.TryApplyGradients(changed));
            else Assert.Throws<InvalidOperationException>(() => plan.TryApplyGradients(changed));
            Assert.Equal(2f, p.ToArray()[0]); Assert.Equal(before, q.ToArray()[0]);
            Assert.Equal(new long[] { 0, 4 }, plan.ParameterAcceptedUpdates);
            plan.Step(); reference.Step();
            Assert.Equal(2f, p.ToArray()[0]);
            Assert.Equal(referenceQ.ToArray()[0], q.ToArray()[0], 6);
            Assert.Equal(new long[] { 0, 5 }, plan.ParameterAcceptedUpdates);
        }
    }
}

public sealed class CompiledOptimizerHandoverIsolationTests
{
    [Fact]
    public void ContinueOptimizerFrom_clones_parameter_accepted_update_ages_from_source_plan()
    {
        var engine = new CpuEngine();
        var p = new Tensor<float>(new[] { 2f }, new[] { 1 });
        var q = new Tensor<float>(new[] { 3f }, new[] { 1 });
        using var scopeA = GraphMode.EnableTraining(new[] { p, q });
        Tensor<float> detachedP;
        using (new NoGradScope<float>()) detachedP = engine.TensorMultiplyScalar(p, 2f);
        using var planA = Assert.IsType<CompiledTrainingPlan<float>>(scopeA.CompileTraining(
            new[] { p, q }, engine.ReduceSum(engine.TensorMultiply(detachedP, q), null, false)));

        using var scopeB = GraphMode.EnableTraining(new[] { p, q });
        using var planB = Assert.IsType<CompiledTrainingPlan<float>>(scopeB.CompileTraining(
            new[] { p, q }, engine.ReduceSum(engine.TensorMultiply(engine.TensorMultiplyScalar(p, 2f), q), null, false)));

        planA.ConfigureOptimizer(OptimizerType.AdamW, .05, weightDecay: .1);
        planB.ConfigureOptimizer(OptimizerType.AdamW, .05, weightDecay: .1);

        planA.Step();
        Assert.Equal(new long[] { 0, 1 }, planA.ParameterAcceptedUpdates);

        planB.ContinueOptimizerFrom(planA);
        planB.Step();

        // Stepping planB without re-handing back to planA must not mutate planA's own accepted-update array
        // beyond planA.OptimizerStep, so planA's checkpoint remains valid and self-consistent.
        Assert.Equal(new long[] { 1, 2 }, planB.ParameterAcceptedUpdates);
        Assert.Equal(new long[] { 0, 1 }, planA.ParameterAcceptedUpdates);
        var bytes = Assert.IsType<byte[]>(planA.ExportOptimizerState());
        Assert.NotEmpty(bytes);
    }

    [Fact]
    public void Float_master_weights_feeding_differentiable_fp16_cast_bridges_enroll_in_optimizer()
    {
        var engine = new CpuEngine();
        var activeMaster = new Tensor<float>(new[] { 2.0f, -1.5f }, new[] { 2 });
        var detachedMaster = new Tensor<float>(new[] { 3.0f, 4.0f }, new[] { 2 });
        var parameters = new[] { activeMaster, detachedMaster };

        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var activeHalf = scope.RecordCrossTypeWithBackward<float, Half>(
                LazyNodeType.Custom,
                "CastToFp16",
                activeMaster,
                new[] { 2 },
                (e, o) => MixedPrecisionCast.CastToFp16(activeMaster).AsSpan().CopyTo(o.AsWritableSpan()),
                (gradOut, inp, o, s, e) => MixedPrecisionCast.CastToFp16Backward(gradOut));

            // Cast bridge recorded inside NoGradScope must not enroll detachedMaster.
            Tensor<Half> detachedHalf;
            using (new NoGradScope<float>())
            {
                detachedHalf = scope.RecordCrossTypeWithBackward<float, Half>(
                    LazyNodeType.Custom,
                    "CastToFp16Detached",
                    detachedMaster,
                    new[] { 2 },
                    (e, o) => MixedPrecisionCast.CastToFp16(detachedMaster).AsSpan().CopyTo(o.AsWritableSpan()),
                    (gradOut, inp, o, s, e) => MixedPrecisionCast.CastToFp16Backward(gradOut));
            }

            var sumHalf = engine.TensorAdd(engine.TensorMultiply(activeHalf, activeHalf), detachedHalf);
            var lossHalf = engine.ReduceSum(sumHalf, null, false);
            var lossFloat = scope.RecordCrossTypeWithBackward<Half, float>(
                LazyNodeType.Custom,
                "CastToFp32",
                lossHalf,
                lossHalf.Shape.ToArray(),
                (e, o) => MixedPrecisionCast.CastToFp32(lossHalf).AsSpan().CopyTo(o.AsWritableSpan()),
                (gradOut, inp, o, s, e) => MixedPrecisionCast.CastToFp32Backward(gradOut));

            plan = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(parameters, lossFloat));
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0.1, weightDecay: 0.05);
            float initial0 = activeMaster[0];
            float initial1 = activeMaster[1];
            plan.Step();

            Assert.Equal(new long[] { 1, 0 }, plan.ParameterAcceptedUpdates);
            // Independent scalar AdamW recurrence: d(sum(master^2))/d(master) = [4, -3].
            static double Expected(double value, double gradient)
            {
                double firstMoment = 0.1 * gradient;
                double secondMoment = 0.001 * gradient * gradient;
                double correctedFirst = firstMoment / (1 - 0.9);
                double correctedSecond = secondMoment / (1 - 0.999);
                return value * (1 - 0.1 * 0.05)
                    - 0.1 * correctedFirst / (Math.Sqrt(correctedSecond) + 1e-8);
            }
            Assert.InRange(Math.Abs(activeMaster[0] - Expected(initial0, 4)), 0, 2e-6);
            Assert.InRange(Math.Abs(activeMaster[1] - Expected(initial1, -3)), 0, 2e-6);
            Assert.Equal(new[] { 3.0f, 4.0f }, detachedMaster.ToArray());
        }
    }

    [Fact]
    public void Explicit_leaf_loss_without_a_recorded_producer_is_rejected()
    {
        var engine = new CpuEngine();
        var parameter = new Tensor<float>(new[] { 2f }, new[] { 1 });
        using var scope = GraphMode.EnableTraining(new[] { parameter });
        _ = engine.TensorMultiply(parameter, parameter);
        var error = Assert.Throws<ArgumentException>(() => scope.CompileTraining(new[] { parameter }, parameter));
        Assert.Contains("produced by an operation recorded", error.Message);
    }

    [Fact]
    public void Foreign_type_no_grad_operation_does_not_enroll_its_float_master()
    {
        var engine = new CpuEngine();
        var master = new Tensor<float>(new[] { 2f, -1.5f }, new[] { 2 });
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { master }))
        {
            var half = scope.RecordCrossTypeWithBackward<float, Half>(
                LazyNodeType.Custom, "MasterToHalf", master, new[] { 2 },
                (e, o) => MixedPrecisionCast.CastToFp16(master).AsSpan().CopyTo(o.AsWritableSpan()),
                (gradient, input, output, state, e) => MixedPrecisionCast.CastToFp16Backward(gradient));
            Tensor<Half> detached;
            using (new NoGradScope<Half>())
                detached = engine.TensorMultiply(half, half);
            var floating = scope.RecordCrossTypeWithBackward<Half, float>(
                LazyNodeType.Custom, "DetachedToFloat", detached, new[] { 2 },
                (e, o) => MixedPrecisionCast.CastToFp32(detached).AsSpan().CopyTo(o.AsWritableSpan()),
                (gradient, input, output, state, e) => MixedPrecisionCast.CastToFp32Backward(gradient));
            var loss = engine.ReduceSum(floating, null, false);
            plan = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(new[] { master }, loss));
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0.1, weightDecay: 0.2);
            plan.Step();
            Assert.Equal(new[] { 2f, -1.5f }, master.ToArray());
            Assert.Equal(new long[] { 0 }, plan.ParameterAcceptedUpdates);
        }
    }

    [Fact]
    public void Explicit_intermediate_loss_survives_linear_fusion()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 3f }, new[] { 1, 1 });
        var weight = new Tensor<float>(new[] { 2f }, new[] { 1, 1 });
        var bias = new Tensor<float>(new[] { 10f }, new[] { 1 });
        var parameters = new[] { weight, bias };
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var selectedLoss = engine.TensorMatMul(input, weight);
            _ = engine.TensorAdd(selectedLoss, bias);
            plan = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(parameters, selectedLoss));
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1);
            plan.Step();
            Assert.InRange(Math.Abs(weight[0] - 1.7f), 0, 1e-6f);
            Assert.Equal(10f, bias[0]);
            Assert.Equal(new long[] { 1, 0 }, plan.ParameterAcceptedUpdates);
        }
    }

    [Fact]
    public void Foreign_type_no_grad_in_place_write_preserves_prior_writer_gradient_only()
    {
        var engine = new CpuEngine();
        var master = new Tensor<float>(new[] { 2f, -1.5f }, new[] { 2 });
        var detached = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var parameters = new[] { master, detached };
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            Tensor<Half> CastMaster(Tensor<float> value, string name) =>
                scope.RecordCrossTypeWithBackward<float, Half>(
                    LazyNodeType.Custom, name, value, new[] { 2 },
                    (e, o) => MixedPrecisionCast.CastToFp16(value).AsSpan().CopyTo(o.AsWritableSpan()),
                    (gradient, input, output, state, e) => MixedPrecisionCast.CastToFp16Backward(gradient));
            var halfMaster = CastMaster(master, "LiveMasterToHalf");
            var halfDetached = CastMaster(detached, "DetachedOperandToHalf");
            var live = engine.TensorMultiplyScalar(halfMaster, (Half)3);
            using (new NoGradScope<Half>())
                engine.TensorAddInPlace(live, halfDetached);
            var halfLoss = engine.ReduceSum(live, null, false);
            var loss = scope.RecordCrossTypeWithBackward<Half, float>(
                LazyNodeType.Custom, "PriorWriterLossToFloat", halfLoss, halfLoss.Shape.ToArray(),
                (e, o) => MixedPrecisionCast.CastToFp32(halfLoss).AsSpan().CopyTo(o.AsWritableSpan()),
                (gradient, input, output, state, e) => MixedPrecisionCast.CastToFp32Backward(gradient));
            plan = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(parameters, loss));
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1);
            // Forward sum: (3 * 2 + 1) + (3 * -1.5 + 2) = 4.5.
            // NoGrad addition passes the prior writer's identity gradient, not its added operand's.
            Assert.Equal(4.5f, plan.Step().ToArray()[0]);
            Assert.Equal(new[] { 3f, 3f }, plan.Gradients[0].ToArray());
            Assert.Equal(new[] { 0f, 0f }, plan.Gradients[1].ToArray());
            Assert.InRange(Math.Abs(master[0] - (2f - 0.1f * 3f)), 0, 1e-6f);
            Assert.InRange(Math.Abs(master[1] - (-1.5f - 0.1f * 3f)), 0, 1e-6f);
            Assert.Equal(new[] { 1f, 2f }, detached.ToArray());
            Assert.Equal(new long[] { 1, 0 }, plan.ParameterAcceptedUpdates);
        }
    }
}
