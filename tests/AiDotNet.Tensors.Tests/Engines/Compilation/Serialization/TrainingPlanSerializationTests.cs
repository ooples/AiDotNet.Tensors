using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Serialization;

/// <summary>
/// Acceptance tests for issue #166 — training plan serialization.
/// Validates SaveAsync → LoadTrainingAsync round-trip produces plans whose
/// Step() returns matching loss + gradients.
/// </summary>
public class TrainingPlanSerializationTests
{
    public static TheoryData<OptimizerType> SupportedFloatOptimizerCheckpointCases => new()
    {
        OptimizerType.SGD,
        OptimizerType.SGDMomentum,
        OptimizerType.Adam,
        OptimizerType.AdamW,
        OptimizerType.AMSGrad,
        OptimizerType.Nadam,
        OptimizerType.RAdam,
        OptimizerType.LAMB,
        OptimizerType.RMSprop,
        OptimizerType.Adagrad,
        OptimizerType.Lion,
        OptimizerType.AdaMax,
        OptimizerType.AdaDelta,
        OptimizerType.LARS,
        OptimizerType.FTRL,
        OptimizerType.ASGD,
        OptimizerType.Rprop,
        OptimizerType.HypergradientSGD,
        OptimizerType.DAdaptationSGD,
        OptimizerType.ScheduleFreeSGD,
    };

    public static TheoryData<OptimizerType> SupportedDoubleOptimizerCheckpointCases => new()
    {
        OptimizerType.SGD,
        OptimizerType.Adam,
        OptimizerType.AdamW,
        OptimizerType.AMSGrad,
    };

    // ── Training plan round-trip: forward + backward, gradients match ────────
    [Fact]
    public async Task SaveLoad_TrainingPlan_StepProducesSameLossAndGradients()
    {
        var engine = new CpuEngine();

        // Use a weight-only loss so the forward result is independent of any
        // leaf-input tensor (which the loader doesn't populate). The loss is
        // simply ReduceSum(weight) — the gradient w.r.t. weight is all-ones.
        var weight = Tensor<float>.CreateRandom([3, 2]);

        // Compile training plan
        ICompiledTrainingPlan<float> original;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(weight, null);
            original = scope.CompileTraining(new[] { weight });
        }

        // Run one step to populate loss + gradients
        var origLoss = original.Step();
        float origLossVal = origLoss[0];
        var origGrad = original.Gradients[0].AsSpan().ToArray();

        // Serialize
        using var ms = new MemoryStream();
        await original.SaveAsync(ms);
        ms.Position = 0;

        // Deserialize with the SAME weight tensor
        var loaded = await CompiledPlanLoader.LoadTrainingAsync<float>(
            ms, engine, new[] { weight });
        Assert.NotNull(loaded);

        // Run one step on the loaded plan — should produce a valid loss.
        // The loaded plan re-compiles backward from the forward graph
        // structure; the backward closures may differ in implementation
        // detail from the original's specialized closures, so we check
        // approximate equivalence rather than bitwise equality.
        var loadedLoss = loaded!.Step();
        float loadedLossVal = loadedLoss[0];
        Assert.False(float.IsNaN(loadedLossVal), "Loaded plan produced NaN loss");
        Assert.False(float.IsInfinity(loadedLossVal), "Loaded plan produced Infinity loss");

        // Forward-pass loss (Step returns the loss from the pre-backward
        // forward sweep) must match to within tight float precision. Only the
        // gradient-build path is re-specialized during load, so forward
        // numerics are bitwise-identical modulo reduction order — reduction-
        // order variance is bounded well below 1e-4 relative for ReduceSum.
        Assert.True(
            Math.Abs(origLossVal - loadedLossVal) <= Math.Max(Math.Abs(origLossVal) * 1e-4f, 1e-5f),
            $"Forward loss diverged: original={origLossVal}, loaded={loadedLossVal}");

        // Gradients should exist and be non-trivial.
        Assert.NotNull(loaded.Gradients);
        Assert.True(loaded.Gradients.Length > 0, "No gradients produced");
        var loadedGrad = loaded.Gradients[0].AsSpan().ToArray();
        Assert.Equal(origGrad.Length, loadedGrad.Length);

        // At least some gradient values should be non-zero.
        bool anyNonZero = false;
        for (int i = 0; i < loadedGrad.Length; i++)
            if (Math.Abs(loadedGrad[i]) > 1e-8f) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "All loaded gradients are zero — backward likely broken");

        original.Dispose();
        loaded.Dispose();
    }

    [Theory]
    [MemberData(nameof(SupportedFloatOptimizerCheckpointCases))]
    public async Task SaveLoad_TrainingPlanWithEverySupportedFloatOptimizer_RestoresContinuation(OptimizerType optimizer)
    {
        var engine = new CpuEngine();
        var weight = CreateTensor(new[] { 3, 2 }, seed: 71);

        var original = CompileLinearPlan(engine, weight);
        ConfigureCheckpointOptimizer(original, optimizer);

        for (int i = 0; i < 5; i++) original.Step();
        var savedCheckpoint = CaptureCheckpoint(original);
        Assert.Equal(optimizer, savedCheckpoint.OptimizerType);
        Assert.Equal(5, savedCheckpoint.OptimizerStep);
        AssertFloatOptimizerStatePresent(savedCheckpoint);

        using var ms = new MemoryStream();
        await original.SaveAsync(ms);
        var savedWeight = weight.AsSpan().ToArray();

        var expectedLoss = original.Step()[0];
        var expectedGrad = original.Gradients[0].AsSpan().ToArray();
        var expectedWeight = weight.AsSpan().ToArray();

        var loadedWeight = CreateTensor(new[] { 3, 2 }, seed: 0);
        savedWeight.AsSpan().CopyTo(loadedWeight.AsWritableSpan());
        ms.Position = 0;
        var loaded = await CompiledPlanLoader.LoadTrainingAsync<float>(
            ms, engine, new[] { loadedWeight });
        Assert.NotNull(loaded);

        var loadedCheckpoint = CaptureCheckpoint(loaded!);
        AssertOptimizerCheckpointEqual(savedCheckpoint, loadedCheckpoint, optimizer.ToString());

        var actualLoss = loaded!.Step()[0];
        var actualGrad = loaded.Gradients[0].AsSpan().ToArray();
        var actualWeight = loadedWeight.AsSpan().ToArray();

        AssertClose(expectedLoss, actualLoss, 1e-4f, $"loss ({optimizer})");
        AssertEqual(expectedGrad, actualGrad, 1e-5f, $"gradients ({optimizer})");
        AssertEqual(expectedWeight, actualWeight, 5e-4f, $"weights ({optimizer})");

        original.Dispose();
        loaded.Dispose();
    }

    [Theory]
    [MemberData(nameof(SupportedDoubleOptimizerCheckpointCases))]
    public async Task SaveLoad_TrainingPlanWithSupportedDoubleOptimizer_RestoresContinuation(OptimizerType optimizer)
    {
        var engine = new CpuEngine();
        var weight = CreateTensorDouble(new[] { 3, 2 }, seed: 91);

        var original = CompileLinearPlan(engine, weight);
        original.ConfigureOptimizer(
            optimizer,
            learningRate: 0.01f,
            beta1: 0.8f,
            beta2: 0.9f,
            eps: 1e-8f,
            weightDecay: optimizer == OptimizerType.AdamW ? 0.001f : 0.0005f);

        for (int i = 0; i < 5; i++) original.Step();
        var savedCheckpoint = CaptureCheckpoint(original);
        Assert.Equal(optimizer, savedCheckpoint.OptimizerType);
        Assert.Equal(5, savedCheckpoint.OptimizerStep);
        AssertDoubleOptimizerStatePresent(savedCheckpoint);

        using var ms = new MemoryStream();
        await original.SaveAsync(ms);
        var savedWeight = weight.AsSpan().ToArray();

        var expectedLoss = original.Step()[0];
        var expectedGrad = original.Gradients[0].AsSpan().ToArray();
        var expectedWeight = weight.AsSpan().ToArray();

        var loadedWeight = CreateTensorDouble(new[] { 3, 2 }, seed: 0);
        savedWeight.AsSpan().CopyTo(loadedWeight.AsWritableSpan());
        ms.Position = 0;
        var loaded = await CompiledPlanLoader.LoadTrainingAsync<double>(
            ms, engine, new[] { loadedWeight });
        Assert.NotNull(loaded);

        var loadedCheckpoint = CaptureCheckpoint(loaded!);
        AssertOptimizerCheckpointEqual(savedCheckpoint, loadedCheckpoint, $"double {optimizer}");

        var actualLoss = loaded!.Step()[0];
        var actualGrad = loaded.Gradients[0].AsSpan().ToArray();
        var actualWeight = loadedWeight.AsSpan().ToArray();

        AssertClose(expectedLoss, actualLoss, 1e-10, $"loss (double {optimizer})");
        AssertEqual(expectedGrad, actualGrad, 1e-12, $"gradients (double {optimizer})");
        AssertEqual(expectedWeight, actualWeight, 1e-10, $"weights (double {optimizer})");

        original.Dispose();
        loaded.Dispose();
    }

    [Fact]
    public async Task SaveLoad_TrainingPlanWithGroupedOptimizer_RestoresSchedulesAndState()
    {
        var engine = new CpuEngine();
        var backbone = CreateTensor(new[] { 2 }, seed: 111);
        var head = CreateTensor(new[] { 2 }, seed: 112);

        var original = CompileTwoParameterLinearPlan(engine, backbone, head);
        original.ConfigureOptimizerGrouped(
            OptimizerType.Adam,
            new LrSchedule[] { LrSchedule.Step(0.01, stepSize: 2, gamma: 0.5), LrSchedule.Cosine(0.02, totalSteps: 8, lrMin: 0.002) },
            new[] { 0, 1 },
            beta1: 0.8f,
            beta2: 0.9f,
            eps: 1e-6f,
            weightDecay: 0.0005f);

        for (int i = 0; i < 5; i++) original.Step();
        var savedCheckpoint = CaptureCheckpoint(original);
        Assert.True(savedCheckpoint.IsGrouped);
        Assert.Equal(new[] { 0, 1 }, savedCheckpoint.ParamToGroup);
        Assert.Equal(2, savedCheckpoint.Schedules.Length);

        using var ms = new MemoryStream();
        await original.SaveAsync(ms);
        var savedBackbone = backbone.AsSpan().ToArray();
        var savedHead = head.AsSpan().ToArray();

        var expectedLoss = original.Step()[0];
        var expectedBackbone = backbone.AsSpan().ToArray();
        var expectedHead = head.AsSpan().ToArray();

        var loadedBackbone = CreateTensor(new[] { 2 }, seed: 0);
        var loadedHead = CreateTensor(new[] { 2 }, seed: 1);
        savedBackbone.AsSpan().CopyTo(loadedBackbone.AsWritableSpan());
        savedHead.AsSpan().CopyTo(loadedHead.AsWritableSpan());
        ms.Position = 0;
        var loaded = await CompiledPlanLoader.LoadTrainingAsync<float>(
            ms, engine, new[] { loadedBackbone, loadedHead });
        Assert.NotNull(loaded);

        var loadedCheckpoint = CaptureCheckpoint(loaded!);
        AssertOptimizerCheckpointEqual(savedCheckpoint, loadedCheckpoint, "grouped Adam");

        var actualLoss = loaded!.Step()[0];
        AssertClose(expectedLoss, actualLoss, 1e-4f, "grouped loss");
        AssertEqual(expectedBackbone, loadedBackbone.AsSpan().ToArray(), 2e-4f, "grouped backbone");
        AssertEqual(expectedHead, loadedHead.AsSpan().ToArray(), 2e-4f, "grouped head");

        original.Dispose();
        loaded.Dispose();
    }

    // ── Training plan with optimizer state ───────────────────────────────────
    [Theory]
    [InlineData(FusedMomentStorageMode.Float32)]
    [InlineData(FusedMomentStorageMode.BFloat16)]
    [InlineData(FusedMomentStorageMode.Int8BlockQuantized)]
    public async Task SaveLoad_TrainingPlanWithFusedAdam_RestoresOptimizerState(FusedMomentStorageMode momentMode)
    {
        var engine = new CpuEngine();
        var weight = CreateTensor(new[] { 3, 2 }, seed: 42);

        var original = CompileLinearPlan(engine, weight);
        ConfigureAdam(original, momentMode);

        // Take 5 steps to accumulate optimizer state
        for (int i = 0; i < 5; i++) original.Step();
        var savedCheckpoint = CaptureCheckpoint(original);
        Assert.Equal(5, savedCheckpoint.OptimizerStep);
        AssertParameterStatePresent(savedCheckpoint.Parameters[0], momentMode);

        // Serialize after 5 steps
        using var ms = new MemoryStream();
        await original.SaveAsync(ms);

        var savedWeight = weight.AsSpan().ToArray();

        // Continue the uninterrupted plan for one more step. The loss returned
        // by Step() is the pre-update loss at the saved weights, and the
        // post-step weights include Adam's restored m/v and step counter.
        var expectedLoss = original.Step()[0];
        var expectedWeight = weight.AsSpan().ToArray();

        var loadedWeight = CreateTensor(new[] { 3, 2 }, seed: 0);
        savedWeight.AsSpan().CopyTo(loadedWeight.AsWritableSpan());
        ms.Position = 0;
        var loaded = await CompiledPlanLoader.LoadTrainingAsync<float>(
            ms, engine, new[] { loadedWeight });
        Assert.NotNull(loaded);
        AssertEqual(savedWeight, loadedWeight.AsSpan().ToArray(), 0f, $"loaded initial weights ({momentMode})");

        var loadedCheckpoint = CaptureCheckpoint(loaded!);
        Assert.Equal(savedCheckpoint.OptimizerStep, loadedCheckpoint.OptimizerStep);
        AssertParameterStateEqual(savedCheckpoint.Parameters[0], loadedCheckpoint.Parameters[0], momentMode);

        var actualLoss = loaded!.Step()[0];
        var actualWeight = loadedWeight.AsSpan().ToArray();

        AssertClose(expectedLoss, actualLoss, 1e-4f, $"loss ({momentMode})");
        AssertEqual(
            original.Gradients[0].AsSpan().ToArray(),
            loaded.Gradients[0].AsSpan().ToArray(),
            1e-5f,
            $"gradients ({momentMode})");
        AssertEqual(expectedWeight, actualWeight, 2e-4f, $"weights ({momentMode})");

        original.Dispose();
        loaded.Dispose();
    }

    private static Tensor<float> CreateTensor(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<float>(shape);
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)(rng.NextDouble() * 0.8 - 0.4);
        return tensor;
    }

    private static Tensor<double> CreateTensorDouble(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<double>(shape);
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = rng.NextDouble() * 0.8 - 0.4;
        return tensor;
    }

    private static ICompiledTrainingPlan<float> CompileLinearPlan(CpuEngine engine, Tensor<float> weight)
    {
        using var scope = GraphMode.Enable();
        engine.ReduceSum(weight, null);
        return scope.CompileTraining(new[] { weight });
    }

    private static ICompiledTrainingPlan<double> CompileLinearPlan(CpuEngine engine, Tensor<double> weight)
    {
        using var scope = GraphMode.Enable();
        engine.ReduceSum(weight, null);
        return scope.CompileTraining(new[] { weight });
    }

    private static ICompiledTrainingPlan<float> CompileTwoParameterLinearPlan(
        CpuEngine engine,
        Tensor<float> first,
        Tensor<float> second)
    {
        using var scope = GraphMode.Enable();
        var firstLoss = engine.ReduceSum(first, null);
        var secondLoss = engine.ReduceSum(second, null);
        engine.TensorAdd(firstLoss, secondLoss);
        return scope.CompileTraining(new[] { first, second });
    }

    private static void ConfigureAdam(ICompiledTrainingPlan<float> plan, FusedMomentStorageMode momentMode)
    {
        if (momentMode == FusedMomentStorageMode.BFloat16)
            plan.RequestBf16MomentStorage(true);
        else if (momentMode == FusedMomentStorageMode.Int8BlockQuantized)
            plan.RequestInt8MomentStorage(true, blockSize: 4);

        plan.ConfigureOptimizer(
            OptimizerType.Adam,
            learningRate: 0.01f,
            beta1: 0.8f,
            beta2: 0.9f,
            eps: 1e-6f);
    }

    private static void ConfigureCheckpointOptimizer(ICompiledTrainingPlan<float> plan, OptimizerType optimizer)
    {
        float weightDecay = optimizer is OptimizerType.HypergradientSGD
            or OptimizerType.DAdaptationSGD
            or OptimizerType.ScheduleFreeSGD
            or OptimizerType.FTRL
            or OptimizerType.Rprop
            ? 0f
            : 0.0005f;

        plan.ConfigureOptimizer(
            optimizer,
            learningRate: optimizer == OptimizerType.Rprop ? 0.001f : 0.01f,
            beta1: 0.8f,
            beta2: 0.9f,
            eps: 1e-6f,
            weightDecay: weightDecay,
            extras: CreateCheckpointExtras());
    }

    private static FusedOptimizerExtras CreateCheckpointExtras() => new()
    {
        Momentum = 0.75f,
        TrustCoefficient = 0.005f,
        L1 = 0.0001f,
        L2 = 0.001f,
        LrPower = -0.5f,
        Lambd = 0.0002f,
        Alpha = 0.75f,
        T0 = 2f,
        RpropEtaPlus = 1.2f,
        RpropEtaMinus = 0.5f,
        RpropStepMin = 1e-6f,
        RpropStepMax = 1f,
        RpropInitialStep = 0.02f,
        HyperLr = 1e-4f,
        D0 = 1e-4f,
        DGrowthRate = 2f,
        SfBeta = 0.75f,
    };

    private static FusedOptimizerCheckpoint CaptureCheckpoint<T>(ICompiledTrainingPlan<T> plan)
        => Assert.IsType<CompiledTrainingPlan<T>>(plan).CaptureFusedOptimizerCheckpoint()
            ?? throw new Xunit.Sdk.XunitException("Expected fused optimizer checkpoint to be present.");

    private static void AssertFloatOptimizerStatePresent(FusedOptimizerCheckpoint checkpoint)
    {
        var state = checkpoint.Parameters[0];
        if (NeedsFirstFloatState(checkpoint.OptimizerType))
            Assert.NotNull(state.MFloat);
        if (NeedsSecondFloatState(checkpoint.OptimizerType))
            Assert.NotNull(state.VFloat);
        if (NeedsThirdFloatState(checkpoint.OptimizerType))
            Assert.NotNull(state.VMaxFloat);

        if (checkpoint.OptimizerType == OptimizerType.HypergradientSGD)
            Assert.NotEqual(0f, checkpoint.Scalars.HypergradientAdjustment);
        if (checkpoint.OptimizerType == OptimizerType.DAdaptationSGD)
            Assert.True(checkpoint.Scalars.DAdaptationRAccum > 0f, "D-Adaptation scalar accumulator was not restored.");
        if (checkpoint.OptimizerType == OptimizerType.ScheduleFreeSGD)
            Assert.True(checkpoint.Scalars.ScheduleFreeWeightSum > 0f, "Schedule-Free scalar weight sum was not restored.");
    }

    // Double-path mirror of AssertFloatOptimizerStatePresent — asserts the fp64 moment state is
    // actually captured before save (the double test previously only checked type + step).
    private static void AssertDoubleOptimizerStatePresent(FusedOptimizerCheckpoint checkpoint)
    {
        var state = checkpoint.Parameters[0];
        if (NeedsFirstFloatState(checkpoint.OptimizerType))
            Assert.NotNull(state.MDouble);
        if (NeedsSecondFloatState(checkpoint.OptimizerType))
            Assert.NotNull(state.VDouble);
        if (NeedsThirdFloatState(checkpoint.OptimizerType))
            Assert.NotNull(state.VMaxDouble);

        if (checkpoint.OptimizerType == OptimizerType.HypergradientSGD)
            Assert.NotEqual(0f, checkpoint.Scalars.HypergradientAdjustment);
        if (checkpoint.OptimizerType == OptimizerType.DAdaptationSGD)
            Assert.True(checkpoint.Scalars.DAdaptationRAccum > 0f, "D-Adaptation scalar accumulator was not restored.");
        if (checkpoint.OptimizerType == OptimizerType.ScheduleFreeSGD)
            Assert.True(checkpoint.Scalars.ScheduleFreeWeightSum > 0f, "Schedule-Free scalar weight sum was not restored.");
    }

    private static bool NeedsFirstFloatState(OptimizerType optimizer) => optimizer is
        OptimizerType.Adam or OptimizerType.AdamW or OptimizerType.AMSGrad or OptimizerType.Nadam or
        OptimizerType.RAdam or OptimizerType.LAMB or OptimizerType.Lion or OptimizerType.SGDMomentum or
        OptimizerType.AdaMax or OptimizerType.LARS or OptimizerType.ASGD or OptimizerType.Rprop or
        OptimizerType.HypergradientSGD or OptimizerType.DAdaptationSGD or OptimizerType.ScheduleFreeSGD;

    private static bool NeedsSecondFloatState(OptimizerType optimizer) => optimizer is
        OptimizerType.Adam or OptimizerType.AdamW or OptimizerType.AMSGrad or OptimizerType.Nadam or
        OptimizerType.RAdam or OptimizerType.LAMB or OptimizerType.RMSprop or OptimizerType.Adagrad or
        OptimizerType.AdaMax or OptimizerType.AdaDelta or OptimizerType.FTRL or OptimizerType.Rprop or
        OptimizerType.ScheduleFreeSGD;

    private static bool NeedsThirdFloatState(OptimizerType optimizer) => optimizer is
        OptimizerType.AMSGrad or OptimizerType.AdaDelta or OptimizerType.FTRL;

    private static void AssertOptimizerCheckpointEqual(
        FusedOptimizerCheckpoint expected,
        FusedOptimizerCheckpoint actual,
        string label)
    {
        Assert.Equal(expected.OptimizerType, actual.OptimizerType);
        Assert.Equal(expected.IsGrouped, actual.IsGrouped);
        Assert.Equal(expected.OptimizerStep, actual.OptimizerStep);
        Assert.Equal(expected.Beta1, actual.Beta1);
        Assert.Equal(expected.Beta2, actual.Beta2);
        Assert.Equal(expected.Epsilon, actual.Epsilon);
        Assert.Equal(expected.WeightDecay, actual.WeightDecay);
        Assert.Equal(expected.MomentStorageMode, actual.MomentStorageMode);
        Assert.Equal(expected.Int8MomentBlockSize, actual.Int8MomentBlockSize);
        Assert.Equal(expected.MaxGradNorm, actual.MaxGradNorm);
        AssertExtrasEqual(expected.Extras, actual.Extras);
        AssertScalarCheckpointEqual(expected.Scalars, actual.Scalars);
        AssertNullableArrayEqual(expected.ParamToGroup, actual.ParamToGroup, $"{label} paramToGroup");
        Assert.Equal(expected.Schedules.Length, actual.Schedules.Length);
        for (int i = 0; i < expected.Schedules.Length; i++)
        {
            Assert.Equal(expected.Schedules[i].Kind, actual.Schedules[i].Kind);
            Assert.Equal(expected.Schedules[i].Doubles, actual.Schedules[i].Doubles);
            Assert.Equal(expected.Schedules[i].Ints, actual.Schedules[i].Ints);
        }

        Assert.Equal(expected.Parameters.Length, actual.Parameters.Length);
        for (int i = 0; i < expected.Parameters.Length; i++)
            AssertParameterCheckpointEqual(expected.Parameters[i], actual.Parameters[i], $"{label} parameter {i}");
    }

    private static void AssertExtrasEqual(FusedOptimizerExtras expected, FusedOptimizerExtras actual)
    {
        Assert.Equal(expected.Momentum, actual.Momentum);
        Assert.Equal(expected.TrustCoefficient, actual.TrustCoefficient);
        Assert.Equal(expected.L1, actual.L1);
        Assert.Equal(expected.L2, actual.L2);
        Assert.Equal(expected.LrPower, actual.LrPower);
        Assert.Equal(expected.Lambd, actual.Lambd);
        Assert.Equal(expected.Alpha, actual.Alpha);
        Assert.Equal(expected.T0, actual.T0);
        Assert.Equal(expected.RpropEtaPlus, actual.RpropEtaPlus);
        Assert.Equal(expected.RpropEtaMinus, actual.RpropEtaMinus);
        Assert.Equal(expected.RpropStepMin, actual.RpropStepMin);
        Assert.Equal(expected.RpropStepMax, actual.RpropStepMax);
        Assert.Equal(expected.RpropInitialStep, actual.RpropInitialStep);
        Assert.Equal(expected.HyperLr, actual.HyperLr);
        Assert.Equal(expected.D0, actual.D0);
        Assert.Equal(expected.DGrowthRate, actual.DGrowthRate);
        Assert.Equal(expected.SfBeta, actual.SfBeta);
    }

    private static void AssertScalarCheckpointEqual(
        FusedOptimizerScalarCheckpoint expected,
        FusedOptimizerScalarCheckpoint actual)
    {
        Assert.Equal(expected.HypergradientAdjustment, actual.HypergradientAdjustment);
        Assert.Equal(expected.DAdaptationEstimate, actual.DAdaptationEstimate);
        Assert.Equal(expected.DAdaptationRAccum, actual.DAdaptationRAccum);
        Assert.Equal(expected.ScheduleFreeWeightSum, actual.ScheduleFreeWeightSum);
    }

    private static void AssertParameterCheckpointEqual(
        FusedOptimizerParameterCheckpoint expected,
        FusedOptimizerParameterCheckpoint actual,
        string label)
    {
        AssertNullableArrayEqual(expected.MFloat, actual.MFloat, $"{label} m");
        AssertNullableArrayEqual(expected.VFloat, actual.VFloat, $"{label} v");
        AssertNullableArrayEqual(expected.VMaxFloat, actual.VMaxFloat, $"{label} vMax");
        AssertNullableArrayEqual(expected.MDouble, actual.MDouble, $"{label} mDouble");
        AssertNullableArrayEqual(expected.VDouble, actual.VDouble, $"{label} vDouble");
        AssertNullableArrayEqual(expected.VMaxDouble, actual.VMaxDouble, $"{label} vMaxDouble");
        AssertNullableArrayEqual(expected.MBFloat16, actual.MBFloat16, $"{label} mBf16");
        AssertNullableArrayEqual(expected.VBFloat16, actual.VBFloat16, $"{label} vBf16");
        AssertNullableArrayEqual(expected.MQuantized, actual.MQuantized, $"{label} mQuantized");
        AssertNullableArrayEqual(expected.VQuantized, actual.VQuantized, $"{label} vQuantized");
        AssertNullableArrayEqual(expected.MScales, actual.MScales, $"{label} mScales");
        AssertNullableArrayEqual(expected.VScales, actual.VScales, $"{label} vScales");
    }

    private static void AssertNullableArrayEqual<T>(T[]? expected, T[]? actual, string label)
    {
        Assert.True(expected is null == actual is null, $"{label} nullability diverged.");
        if (expected is null || actual is null) return;
        Assert.Equal(expected, actual);
    }

    private static void AssertParameterStatePresent(
        FusedOptimizerParameterCheckpoint state,
        FusedMomentStorageMode momentMode)
    {
        switch (momentMode)
        {
            case FusedMomentStorageMode.Float32:
                Assert.NotNull(state.MFloat);
                Assert.NotNull(state.VFloat);
                break;
            case FusedMomentStorageMode.BFloat16:
                Assert.NotNull(state.MBFloat16);
                Assert.NotNull(state.VBFloat16);
                break;
            case FusedMomentStorageMode.Int8BlockQuantized:
                Assert.NotNull(state.MQuantized);
                Assert.NotNull(state.VQuantized);
                Assert.NotNull(state.MScales);
                Assert.NotNull(state.VScales);
                break;
            default:
                throw new Xunit.Sdk.XunitException($"Unhandled moment mode {momentMode}.");
        }
    }

    private static void AssertParameterStateEqual(
        FusedOptimizerParameterCheckpoint expected,
        FusedOptimizerParameterCheckpoint actual,
        FusedMomentStorageMode momentMode)
    {
        switch (momentMode)
        {
            case FusedMomentStorageMode.Float32:
                AssertEqual(expected.MFloat!, actual.MFloat!, 0f, "m");
                AssertEqual(expected.VFloat!, actual.VFloat!, 0f, "v");
                break;
            case FusedMomentStorageMode.BFloat16:
                Assert.Equal(expected.MBFloat16, actual.MBFloat16);
                Assert.Equal(expected.VBFloat16, actual.VBFloat16);
                break;
            case FusedMomentStorageMode.Int8BlockQuantized:
                Assert.Equal(expected.MQuantized, actual.MQuantized);
                Assert.Equal(expected.VQuantized, actual.VQuantized);
                Assert.Equal(expected.MScales, actual.MScales);
                Assert.Equal(expected.VScales, actual.VScales);
                break;
            default:
                throw new Xunit.Sdk.XunitException($"Unhandled moment mode {momentMode}.");
        }
    }

    private static void AssertEqual(float[] expected, float[] actual, float tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            AssertClose(expected[i], actual[i], tolerance, $"{label}[{i}]");
    }

    private static void AssertClose(float expected, float actual, float tolerance, string label)
    {
        Assert.False(float.IsNaN(actual), $"{label} is NaN");
        Assert.False(float.IsInfinity(actual), $"{label} is infinity");
        float diff = Math.Abs(expected - actual);
        Assert.True(
            diff <= tolerance,
            $"{label} diverged: expected={expected:R}, actual={actual:R}, diff={diff:R}, tolerance={tolerance:R}");
    }

    private static void AssertEqual(double[] expected, double[] actual, double tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            AssertClose(expected[i], actual[i], tolerance, $"{label}[{i}]");
    }

    private static void AssertClose(double expected, double actual, double tolerance, string label)
    {
        Assert.False(double.IsNaN(actual), $"{label} is NaN");
        Assert.False(double.IsInfinity(actual), $"{label} is infinity");
        double diff = Math.Abs(expected - actual);
        Assert.True(
            diff <= tolerance,
            $"{label} diverged: expected={expected:R}, actual={actual:R}, diff={diff:R}, tolerance={tolerance:R}");
    }
}
