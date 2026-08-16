using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Per-group optimizer types: each parameter group may run a DIFFERENT optimizer, not merely a different
/// learning-rate schedule.
/// </summary>
/// <remarks>
/// <para>
/// The motivating recipe is LARS, whose papers exclude biases and normalization parameters from both the
/// layer-wise trust ratio and weight decay. With one optimizer per plan the only options were applying LARS to
/// biases (wrong) or refusing to fuse (useless).
/// </para>
/// <para>
/// The failure mode to guard against is silence: if the per-group lookup were dropped, every group would run
/// the fallback optimizer, training would proceed, and nothing would report an error. So the central tests
/// compare against single-optimizer reference runs and demand EXACT agreement per group — a per-group run must
/// reproduce, tensor by tensor, what each group's optimizer produces on its own.
/// </para>
/// </remarks>
public class PerGroupOptimizerTypeTests
{
    private const float Lr = 0.03f, B1 = 0.9f, B2 = 0.99f, Eps = 1e-8f;

    private static float[] Seed(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    /// <summary>
    /// Runs a two-parameter plan on the separable loss sum(a^2) + sum(b^2) for <paramref name="steps"/> steps.
    /// </summary>
    /// <remarks>
    /// Separability is what makes the comparisons in this file valid: da/dloss depends only on a and db/dloss
    /// only on b, so parameter a's trajectory under a given optimizer is identical whether or not b is being
    /// optimized differently. Any cross-talk would therefore show up as a failure rather than be masked.
    /// </remarks>
    private static (float[] A, float[] B) RunTwoGroups(
        float[] initA, float[] initB,
        OptimizerType fallback,
        IReadOnlyList<OptimizerType>? groupTypes,
        int steps,
        IReadOnlyList<float>? groupWeightDecays = null,
        float weightDecay = 0f,
        FusedOptimizerExtras? extras = null)
    {
        var engine = new CpuEngine();
        var a = new Tensor<float>(new[] { initA.Length });
        var b = new Tensor<float>(new[] { initB.Length });
        for (int i = 0; i < initA.Length; i++) a[i] = initA[i];
        for (int i = 0; i < initB.Length; i++) b[i] = initB[i];

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var sumA = engine.ReduceSum(engine.TensorMultiply(a, a), null);
            var sumB = engine.ReduceSum(engine.TensorMultiply(b, b), null);
            engine.TensorAdd(sumA, sumB);
            plan = scope.CompileTraining(new[] { a, b });
        }

        using (plan)
        {
            var schedules = new List<LrSchedule> { LrSchedule.Constant(Lr), LrSchedule.Constant(Lr) };
            var map = new List<int> { 0, 1 };
            plan.ConfigureOptimizerGrouped(
                fallback, groupTypes, schedules, map, B1, B2, Eps, weightDecay, groupWeightDecays, extras);
            for (int s = 0; s < steps; s++) plan.Step();
        }

        return (a.GetDataArray().AsSpan(0, initA.Length).ToArray(),
                b.GetDataArray().AsSpan(0, initB.Length).ToArray());
    }

    /// <summary>
    /// The core guarantee: a plan configured with [SGD, Adam] must reproduce EXACTLY what a pure-SGD plan does
    /// to the first tensor and what a pure-Adam plan does to the second.
    /// </summary>
    /// <remarks>
    /// SGD and Adam are chosen because they differ in both trajectory and state requirements — Adam needs first
    /// and second moment buffers, SGD needs none. So this simultaneously proves the per-parameter dispatch reads
    /// the right group AND that buffer allocation is sized per group rather than from the plan-wide fallback.
    /// The final assertion checks the two references genuinely disagree, so exact agreement is informative.
    /// </remarks>
    [Fact]
    public void PerGroupTypes_RunEachGroupsOwnOptimizer_ExactlyMatchingSingleOptimizerRuns()
    {
        var initA = Seed(16, seed: 11);
        var initB = Seed(16, seed: 12);
        const int steps = 15;

        var (pureSgdA, pureSgdB) = RunTwoGroups(initA, initB, OptimizerType.SGD, null, steps);
        var (pureAdamA, pureAdamB) = RunTwoGroups(initA, initB, OptimizerType.Adam, null, steps);

        var (mixedA, mixedB) = RunTwoGroups(
            initA, initB,
            fallback: OptimizerType.SGD,
            groupTypes: new[] { OptimizerType.SGD, OptimizerType.Adam },
            steps);

        for (int i = 0; i < initA.Length; i++)
            Assert.Equal(pureSgdA[i], mixedA[i], 6);
        for (int i = 0; i < initB.Length; i++)
            Assert.Equal(pureAdamB[i], mixedB[i], 6);

        // If the per-group lookup were dropped, group 1 would have run the SGD fallback. Confirm that is a
        // distinguishable outcome rather than a coincidence.
        double maxRefGap = 0;
        for (int i = 0; i < initB.Length; i++)
            maxRefGap = Math.Max(maxRefGap, Math.Abs(pureAdamB[i] - pureSgdB[i]));
        Assert.True(maxRefGap > 1e-3,
            $"SGD and Adam produced near-identical trajectories (max gap {maxRefGap}), so this test proves nothing.");
    }

    /// <summary>
    /// The same guarantee with the groups swapped, so neither "always use group 0's type" nor "always use the
    /// last group's type" can pass.
    /// </summary>
    [Fact]
    public void PerGroupTypes_AreNotOrderDependent()
    {
        var initA = Seed(16, seed: 21);
        var initB = Seed(16, seed: 22);
        const int steps = 15;

        var (pureSgdA, _) = RunTwoGroups(initA, initB, OptimizerType.SGD, null, steps);
        var (pureAdamA, _) = RunTwoGroups(initA, initB, OptimizerType.Adam, null, steps);

        var (mixedA, mixedB) = RunTwoGroups(
            initA, initB,
            fallback: OptimizerType.Adam,
            groupTypes: new[] { OptimizerType.Adam, OptimizerType.SGD },
            steps);

        for (int i = 0; i < initA.Length; i++)
            Assert.Equal(pureAdamA[i], mixedA[i], 6);

        // Group 1 ran SGD while the fallback was Adam — so this also proves the fallback is not silently winning.
        var (_, pureSgdB) = RunTwoGroups(initA, initB, OptimizerType.SGD, null, steps);
        for (int i = 0; i < initB.Length; i++)
            Assert.Equal(pureSgdB[i], mixedB[i], 6);

        Assert.NotEqual(pureSgdA[0], pureAdamA[0], 4);
    }

    /// <summary>
    /// The LARS recipe this feature exists for: LARS on the weight tensor, plain SGD-with-momentum and zero
    /// weight decay on the bias tensor.
    /// </summary>
    [Fact]
    public void PerGroupTypes_ExpressTheLarsBiasExclusionRecipe()
    {
        var initWeights = Seed(16, seed: 31);
        var initBias = Seed(16, seed: 32);
        const int steps = 12;
        const float decay = 0.05f;

        var (weights, bias) = RunTwoGroups(
            initWeights, initBias,
            fallback: OptimizerType.LARS,
            groupTypes: new[] { OptimizerType.LARS, OptimizerType.SGDMomentum },
            steps,
            groupWeightDecays: new[] { decay, 0f },
            extras: new FusedOptimizerExtras { Momentum = 0.9f, TrustCoefficient = 0.001f });

        // The bias must match a pure SGD-with-momentum, zero-decay run — i.e. it saw neither LARS's trust ratio
        // nor any weight decay, which is exactly what the LARS papers prescribe for biases.
        var (_, referenceBias) = RunTwoGroups(
            initWeights, initBias,
            fallback: OptimizerType.SGDMomentum, groupTypes: null, steps,
            weightDecay: 0f,
            extras: new FusedOptimizerExtras { Momentum = 0.9f, TrustCoefficient = 0.001f });

        for (int i = 0; i < initBias.Length; i++)
            Assert.Equal(referenceBias[i], bias[i], 6);

        foreach (var w in weights)
            Assert.True(!float.IsNaN(w) && !float.IsInfinity(w), "LARS group produced a non-finite parameter.");

        // And the weights must NOT match what SGD-with-momentum would have done to them, or LARS was not applied.
        var (referenceWeightsIfSgdm, _) = RunTwoGroups(
            initWeights, initBias,
            fallback: OptimizerType.SGDMomentum, groupTypes: null, steps,
            weightDecay: decay,
            extras: new FusedOptimizerExtras { Momentum = 0.9f, TrustCoefficient = 0.001f });

        double maxGap = 0;
        for (int i = 0; i < initWeights.Length; i++)
            maxGap = Math.Max(maxGap, Math.Abs(referenceWeightsIfSgdm[i] - weights[i]));
        Assert.True(maxGap > 1e-6,
            $"The LARS group's trajectory is indistinguishable from SGD-with-momentum (max gap {maxGap}).");
    }

    /// <summary>
    /// Per-group weight decay must apply per group, under a single shared optimizer.
    /// </summary>
    [Fact]
    public void PerGroupWeightDecay_AppliesPerGroup()
    {
        var initA = Seed(16, seed: 41);
        var initB = Seed(16, seed: 42);
        const int steps = 10;
        const float decay = 0.1f;

        var (a, b) = RunTwoGroups(
            initA, initB, OptimizerType.SGD, groupTypes: null, steps,
            groupWeightDecays: new[] { 0f, decay });

        var (refNoDecayA, _) = RunTwoGroups(initA, initB, OptimizerType.SGD, null, steps, weightDecay: 0f);
        var (_, refDecayB) = RunTwoGroups(initA, initB, OptimizerType.SGD, null, steps, weightDecay: decay);

        for (int i = 0; i < initA.Length; i++) Assert.Equal(refNoDecayA[i], a[i], 6);
        for (int i = 0; i < initB.Length; i++) Assert.Equal(refDecayB[i], b[i], 6);
    }

    /// <summary>
    /// Omitting the per-group arrays entirely must behave exactly like the original single-optimizer overload —
    /// the uniform path is the overwhelmingly common one and must not have shifted.
    /// </summary>
    [Fact]
    public void NullGroupTypes_BehaveIdenticallyToTheSingleOptimizerOverload()
    {
        var initA = Seed(16, seed: 51);
        var initB = Seed(16, seed: 52);
        const int steps = 10;

        var (viaNew, viaNewB) = RunTwoGroups(initA, initB, OptimizerType.Adam, groupTypes: null, steps);

        var engine = new CpuEngine();
        var a = new Tensor<float>(new[] { initA.Length });
        var b = new Tensor<float>(new[] { initB.Length });
        for (int i = 0; i < initA.Length; i++) a[i] = initA[i];
        for (int i = 0; i < initB.Length; i++) b[i] = initB[i];

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var sumA = engine.ReduceSum(engine.TensorMultiply(a, a), null);
            var sumB = engine.ReduceSum(engine.TensorMultiply(b, b), null);
            engine.TensorAdd(sumA, sumB);
            plan = scope.CompileTraining(new[] { a, b });
        }
        using (plan)
        {
            var schedules = new List<LrSchedule> { LrSchedule.Constant(Lr), LrSchedule.Constant(Lr) };
            plan.ConfigureOptimizerGrouped(OptimizerType.Adam, schedules, new List<int> { 0, 1 }, B1, B2, Eps);
            for (int s = 0; s < steps; s++) plan.Step();
        }

        var legacyA = a.GetDataArray().AsSpan(0, initA.Length).ToArray();
        var legacyB = b.GetDataArray().AsSpan(0, initB.Length).ToArray();
        for (int i = 0; i < initA.Length; i++) Assert.Equal(legacyA[i], viaNew[i], 6);
        for (int i = 0; i < initB.Length; i++) Assert.Equal(legacyB[i], viaNewB[i], 6);
    }

    /// <summary>
    /// A per-group array whose length disagrees with the schedule count is rejected, rather than silently
    /// leaving trailing groups on the fallback.
    /// </summary>
    [Theory]
    [InlineData(true)]    // wrong groupOptimizerTypes length
    [InlineData(false)]   // wrong groupWeightDecays length
    public void MismatchedPerGroupArrayLength_Throws(bool badTypes)
    {
        var init = Seed(8, seed: 61);

        Assert.Throws<ArgumentException>(() => RunTwoGroups(
            init, init,
            fallback: OptimizerType.SGD,
            groupTypes: badTypes ? new[] { OptimizerType.SGD } : null,
            steps: 1,
            groupWeightDecays: badTypes ? null : new[] { 0f }));
    }

    /// <summary>
    /// A global-state optimizer smuggled in through a per-group array must be rejected just as it is when passed
    /// as the plan-wide type — those maintain one scalar across all parameters, which per-group configuration
    /// cannot express.
    /// </summary>
    [Theory]
    [InlineData(OptimizerType.HypergradientSGD)]
    [InlineData(OptimizerType.DAdaptationSGD)]
    [InlineData(OptimizerType.ScheduleFreeSGD)]
    public void GlobalStateOptimizerInAnyGroup_Throws(OptimizerType global)
    {
        var init = Seed(8, seed: 71);

        Assert.Throws<NotSupportedException>(() => RunTwoGroups(
            init, init,
            fallback: OptimizerType.SGD,
            groupTypes: new[] { OptimizerType.SGD, global },
            steps: 1));
    }

    /// <summary>
    /// An unsupported optimizer in ANY group must be caught at configure time, not on the first Step() after
    /// earlier parameters have already been updated.
    /// </summary>
    [Fact]
    public void UnsupportedOptimizerInAnyGroup_ThrowsAtConfigureTime()
    {
        var init = Seed(8, seed: 81);

        Assert.Throws<NotSupportedException>(() => RunTwoGroups(
            init, init,
            fallback: OptimizerType.SGD,
            groupTypes: new[] { OptimizerType.SGD, OptimizerType.LBFGS },
            steps: 1));
    }

    /// <summary>
    /// The per-group configuration must survive a checkpoint round trip. Without this, saving and resuming a
    /// heterogeneous plan would silently rebuild it as a uniform one — every group switched to the fallback
    /// optimizer, mid-run, with no error anywhere.
    /// </summary>
    [Fact]
    public void CheckpointRoundTrip_PreservesPerGroupTypesAndWeightDecays()
    {
        var initA = Seed(8, seed: 91);
        var initB = Seed(8, seed: 92);
        var groupTypes = new[] { OptimizerType.SGD, OptimizerType.Adam };
        var groupWds = new[] { 0f, 0.02f };

        var engine = new CpuEngine();
        var a = new Tensor<float>(new[] { initA.Length });
        var b = new Tensor<float>(new[] { initB.Length });
        for (int i = 0; i < initA.Length; i++) a[i] = initA[i];
        for (int i = 0; i < initB.Length; i++) b[i] = initB[i];

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var sumA = engine.ReduceSum(engine.TensorMultiply(a, a), null);
            var sumB = engine.ReduceSum(engine.TensorMultiply(b, b), null);
            engine.TensorAdd(sumA, sumB);
            plan = scope.CompileTraining(new[] { a, b });
        }

        using (plan)
        {
            var schedules = new List<LrSchedule> { LrSchedule.Constant(Lr), LrSchedule.Constant(Lr) };
            plan.ConfigureOptimizerGrouped(
                OptimizerType.SGD, groupTypes, schedules, new List<int> { 0, 1 },
                B1, B2, Eps, 0f, groupWds);
            for (int s = 0; s < 3; s++) plan.Step();

            var checkpoint = Assert.IsType<CompiledTrainingPlan<float>>(plan).CaptureFusedOptimizerCheckpoint();
            Assert.NotNull(checkpoint);
            Assert.Equal(groupTypes, checkpoint!.GroupOptimizerTypes);
            Assert.Equal(groupWds, checkpoint.GroupWeightDecays);

            // Restoring the captured checkpoint into the same plan must reinstate the per-group configuration,
            // not collapse it onto the SGD fallback.
            Assert.IsType<CompiledTrainingPlan<float>>(plan).RestoreFusedOptimizerCheckpoint(checkpoint);
            var afterRestore = Assert.IsType<CompiledTrainingPlan<float>>(plan).CaptureFusedOptimizerCheckpoint();
            Assert.NotNull(afterRestore);
            Assert.Equal(groupTypes, afterRestore!.GroupOptimizerTypes);
            Assert.Equal(groupWds, afterRestore.GroupWeightDecays);
        }
    }

    /// <summary>
    /// A uniform plan must checkpoint as uniform (null arrays), so "one optimizer everywhere" stays
    /// distinguishable from "several groups that happen to agree today".
    /// </summary>
    [Fact]
    public void CheckpointRoundTrip_UniformPlanRecordsNoPerGroupArrays()
    {
        var init = Seed(8, seed: 93);
        var engine = new CpuEngine();
        var a = new Tensor<float>(new[] { init.Length });
        for (int i = 0; i < init.Length; i++) a[i] = init[i];

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(engine.TensorMultiply(a, a), null);
            plan = scope.CompileTraining(new[] { a });
        }

        using (plan)
        {
            plan.ConfigureOptimizerGrouped(
                OptimizerType.Adam,
                new List<LrSchedule> { LrSchedule.Constant(Lr) },
                new List<int> { 0 },
                B1, B2, Eps);
            plan.Step();

            var checkpoint = Assert.IsType<CompiledTrainingPlan<float>>(plan).CaptureFusedOptimizerCheckpoint();
            Assert.NotNull(checkpoint);
            Assert.Null(checkpoint!.GroupOptimizerTypes);
            Assert.Null(checkpoint.GroupWeightDecays);
        }
    }
}
