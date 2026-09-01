using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The fused optimizer must DISCARD a step whose gradients are not finite, leaving every parameter
/// exactly as it was.
/// </summary>
/// <remarks>
/// <para>
/// The fused kernels read a gradient and write its parameter in one pass, so a NaN or infinite
/// gradient is not just a wasted step — it is written into the weights, and every subsequent step
/// reads it back. The loss reports NaN from then on with nothing to indicate which step poisoned it,
/// and because the eager autograd tape has no such fusion, the same model can show perfectly finite
/// gradients from a direct gradient call while training destroys it.
/// </para>
/// <para>
/// <c>torch.amp.GradScaler</c> is the reference behaviour: check the gradients each iteration, skip
/// the optimizer step when any are inf or NaN, and leave the parameters untouched so the run
/// continues from the last good state. These pin that contract.
/// </para>
/// </remarks>
public class FusedOptimizerNonFiniteGradientTests
{
    /// <summary>
    /// Builds <c>loss = sum(param * x)</c>, whose gradient with respect to <c>param</c> is exactly
    /// <c>x</c>. Feeding a non-finite <c>x</c> therefore produces a non-finite gradient by
    /// construction, with no reliance on a model diverging.
    /// </summary>
    private static (ICompiledTrainingPlan<float> plan, Tensor<float> param, Tensor<float> input) BuildPlan(float xValue)
    {
        var param = new Tensor<float>(new[] { 16 });
        var x = new Tensor<float>(new[] { 16 });
        for (int i = 0; i < param.Length; i++)
        {
            param[i] = 1.0f + (0.1f * i);
            x[i] = xValue;
        }

        var engine = new CpuEngine();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var product = engine.TensorMultiply(param, x);
            engine.ReduceSum(product, null);
            plan = scope.CompileTraining(new[] { param });
        }

        return (plan, param, x);
    }

    private static (ICompiledTrainingPlan<double> plan, Tensor<double> param) BuildDoublePlan(double xValue)
    {
        var param = new Tensor<double>(new[] { 16 });
        var x = new Tensor<double>(new[] { 16 });
        for (int i = 0; i < param.Length; i++)
        {
            param[i] = 1.0 + (0.1 * i);
            x[i] = xValue;
        }

        var engine = new CpuEngine();
        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var product = engine.TensorMultiply(param, x);
            engine.ReduceSum(product, null);
            plan = scope.CompileTraining(new[] { param });
        }

        return (plan, param);
    }

    private static (ICompiledTrainingPlan<float> plan, Tensor<float>[] parameters) BuildTwoParameterPlan()
    {
        var first = new Tensor<float>(new[] { 16 });
        var second = new Tensor<float>(new[] { 16 });
        var input = new Tensor<float>(new[] { 16 });
        for (int i = 0; i < input.Length; i++)
        {
            first[i] = 1f;
            second[i] = 2f;
            input[i] = 1f;
        }

        var engine = new CpuEngine();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var firstProduct = engine.TensorMultiply(first, input);
            var secondProduct = engine.TensorMultiply(second, input);
            var combined = engine.TensorAdd(firstProduct, secondProduct);
            engine.ReduceSum(combined, null);
            plan = scope.CompileTraining(new[] { first, second });
        }

        return (plan, new[] { first, second });
    }

    private static void AttachGpuBuffer(
        Tensor<float> tensor,
        AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend backend,
        float value)
    {
        var data = new float[tensor.Length];
        for (int i = 0; i < data.Length; i++) data[i] = value;
        tensor._gpuBuffer = new MockGpuBuffer(data);
        tensor._gpuBackend = backend;
        tensor._gpuBufferVersion = tensor.Version;
    }

    private static Tensor<float>[] GetPlanGradients(ICompiledTrainingPlan<float> plan)
    {
        var field = plan.GetType().GetField("_gradients", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("Compiled plan gradient field was not found.");
        return (Tensor<float>[])field.GetValue(plan)!;
    }

    private static void InvokeOptimizerUpdate(ICompiledTrainingPlan<float> plan)
    {
        var field = plan.GetType().GetField("_optimizerUpdate", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("Compiled plan optimizer closure was not found.");
        var update = (Action?)field.GetValue(plan)
            ?? throw new InvalidOperationException("Compiled plan optimizer was not configured.");
        update();
    }

    private static int GetOptimizerStep(ICompiledTrainingPlan<float> plan)
    {
        var field = plan.GetType().GetField("_optimizerStep", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("Compiled plan optimizer step field was not found.");
        return (int)field.GetValue(plan)!;
    }

    private static float[] Snapshot(Tensor<float> t)
    {
        var copy = new float[t.Length];
        for (int i = 0; i < t.Length; i++) copy[i] = t[i];
        return copy;
    }

    private static double[] Snapshot(Tensor<double> t)
    {
        var copy = new double[t.Length];
        for (int i = 0; i < t.Length; i++) copy[i] = t[i];
        return copy;
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void NonFiniteGradient_LeavesEveryParameterUntouched(float badGradient)
    {
        var (plan, param, _) = BuildPlan(badGradient);
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1f);
            var before = Snapshot(param);

            plan.Step();

            Assert.True(plan.LastStepSkippedNonFiniteGradients,
                "a step with non-finite gradients must report that it was skipped");
            Assert.Equal(1, plan.NonFiniteStepsSkipped);

            for (int i = 0; i < param.Length; i++)
            {
                Assert.True(before[i].Equals(param[i]),
                    $"parameter[{i}] changed on a discarded step: {before[i]:G9} -> {param[i]:G9}");
            }
        }
    }

    [Fact]
    public void FiniteGradient_StillUpdates_AndReportsNoSkip()
    {
        // The gate must not become a blanket refusal: an ordinary step still has to train.
        var (plan, param, _) = BuildPlan(2.0f);
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1f);
            var before = Snapshot(param);

            plan.Step();

            Assert.False(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(0, plan.NonFiniteStepsSkipped);

            // d(sum(param * x))/d(param) = x = 2, so SGD at lr 0.1 moves every entry by -0.2.
            for (int i = 0; i < param.Length; i++)
            {
                Assert.True(System.Math.Abs((before[i] - 0.2f) - param[i]) < 1e-5f,
                    $"parameter[{i}] did not take the expected step: {before[i]:G9} -> {param[i]:G9}");
            }
        }
    }

    [Fact]
    public void TrainingRecovers_AfterASkippedStep()
    {
        // The point of skipping rather than clamping: the model is still trainable afterwards.
        // A single poisoned step used to make every later step NaN forever.
        var (badPlan, param, _) = BuildPlan(float.NaN);
        using (badPlan)
        {
            badPlan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1f);
            badPlan.Step();
            Assert.True(badPlan.LastStepSkippedNonFiniteGradients);
        }

        for (int i = 0; i < param.Length; i++)
        {
            Assert.False(float.IsNaN(param[i]), $"parameter[{i}] is NaN after a step that was skipped");
            Assert.False(float.IsInfinity(param[i]), $"parameter[{i}] is Infinity after a skipped step");
        }
    }

    [Fact]
    public void GroupedSchedule_NonFiniteGradient_SkipsTheWholeStep()
    {
        var (plan, param, _) = BuildPlan(float.NaN);
        using (plan)
        {
            plan.ConfigureOptimizerGrouped(
                OptimizerType.SGD,
                new[] { LrSchedule.Constant(0.1) },
                new[] { 0 });
            var before = Snapshot(param);

            plan.Step();

            Assert.True(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(1, plan.NonFiniteStepsSkipped);
            for (int i = 0; i < param.Length; i++)
                Assert.True(before[i].Equals(param[i]), $"grouped parameter[{i}] changed");
        }
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void GpuResidentGradient_NonFiniteValue_SkipsBeforeAnyGpuUpdate(float badGradient)
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var (plan, parameters) = BuildTwoParameterPlan();
        using (plan)
        {
            foreach (var parameter in parameters) AttachGpuBuffer(parameter, backend, 1f);
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1f);

            var gradients = GetPlanGradients(plan);
            AttachGpuBuffer(gradients[0], backend, 1f);
            AttachGpuBuffer(gradients[1], backend, badGradient);

            InvokeOptimizerUpdate(plan);

            Assert.True(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(1, plan.NonFiniteStepsSkipped);
            Assert.Equal(2, state.ClassifyFloatCalls);
            Assert.Equal(1, state.FillCalls);
            Assert.Equal(2, state.MultiplyCalls);
            Assert.Equal(1, state.ScalarMinCalls);
            Assert.Empty(state.OptimizerCalls);
            Assert.Equal(0, state.DownloadBufferCalls);
            Assert.Equal(new[] { 16, 16 }, state.AllocationSizes);
            Assert.Equal(0, GetOptimizerStep(plan));

            foreach (var gradient in gradients) AttachGpuBuffer(gradient, backend, 1f);
            InvokeOptimizerUpdate(plan);

            Assert.False(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(1, plan.NonFiniteStepsSkipped);
            Assert.Equal(1, GetOptimizerStep(plan));
            Assert.Equal(new[] { "SgdUpdate", "SgdUpdate" }, state.OptimizerCalls);
        }
    }

    [Fact]
    public void GpuResidentFiniteGradients_RunEveryGpuUpdate()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var (plan, parameters) = BuildTwoParameterPlan();
        using (plan)
        {
            foreach (var parameter in parameters) AttachGpuBuffer(parameter, backend, 1f);
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1f);

            foreach (var gradient in GetPlanGradients(plan))
                AttachGpuBuffer(gradient, backend, 1f);

            InvokeOptimizerUpdate(plan);

            Assert.False(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(0, plan.NonFiniteStepsSkipped);
            Assert.Equal(2, state.ClassifyFloatCalls);
            Assert.Equal(1, state.FillCalls);
            Assert.Equal(2, state.MultiplyCalls);
            Assert.Equal(1, state.ScalarMinCalls);
            Assert.Equal(new[] { "SgdUpdate", "SgdUpdate" }, state.OptimizerCalls);
            Assert.Equal(0, state.DownloadBufferCalls);
            Assert.Equal(new[] { 16, 16 }, state.AllocationSizes);
        }
    }

    [Fact]
    public void MultiTensorGpuRoute_NonFiniteGradient_SkipsBeforeBatchedUpdate()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.CreateMultiTensor(state);
        var (plan, parameters) = BuildTwoParameterPlan();
        using (plan)
        {
            foreach (var parameter in parameters) AttachGpuBuffer(parameter, backend, 1f);
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.001f);

            var gradients = GetPlanGradients(plan);
            AttachGpuBuffer(gradients[0], backend, 1f);
            AttachGpuBuffer(gradients[1], backend, float.NaN);

            InvokeOptimizerUpdate(plan);

            Assert.True(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(1, plan.NonFiniteStepsSkipped);
            Assert.Empty(state.OptimizerCalls);
            Assert.Equal(2, state.ClassifyFloatCalls);
            Assert.Equal(1, state.FillCalls);
            Assert.Equal(2, state.MultiplyCalls);
            Assert.Equal(1, state.ScalarMinCalls);
            Assert.Equal(0, state.DownloadBufferCalls);
        }
    }

    [Fact]
    public void MultiTensorGpuRoute_FiniteGradients_UsesOneBatchedUpdate()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.CreateMultiTensor(state);
        var (plan, parameters) = BuildTwoParameterPlan();
        using (plan)
        {
            foreach (var parameter in parameters) AttachGpuBuffer(parameter, backend, 1f);
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.001f);

            foreach (var gradient in GetPlanGradients(plan))
                AttachGpuBuffer(gradient, backend, 1f);

            InvokeOptimizerUpdate(plan);

            Assert.False(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(0, plan.NonFiniteStepsSkipped);
            Assert.Equal(new[] { "AdamMultiTensorUpdate" }, state.OptimizerCalls);
            Assert.Equal(2, state.ClassifyFloatCalls);
            Assert.Equal(1, state.FillCalls);
            Assert.Equal(2, state.MultiplyCalls);
            Assert.Equal(1, state.ScalarMinCalls);
            Assert.Equal(0, state.DownloadBufferCalls);
        }
    }

    [Fact]
    public void GroupedGpuPlan_StaleResidentGradientChecksAuthoritativeHostValues()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var (plan, parameters) = BuildTwoParameterPlan();
        using (plan)
        {
            foreach (var parameter in parameters) AttachGpuBuffer(parameter, backend, 1f);
            plan.ConfigureOptimizerGrouped(
                OptimizerType.SGD,
                new[] { LrSchedule.Constant(0.1) },
                new[] { 0, 0 });

            var gradients = GetPlanGradients(plan);
            AttachGpuBuffer(gradients[0], backend, 1f);
            AttachGpuBuffer(gradients[1], backend, 1f);
            for (int i = 0; i < gradients[1].Length; i++) gradients[1][i] = float.NaN;

            InvokeOptimizerUpdate(plan);

            Assert.True(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(1, plan.NonFiniteStepsSkipped);
            Assert.Empty(state.OptimizerCalls);
            Assert.Equal(0, state.FillCalls);
            Assert.Equal(0, state.ClassifyFloatCalls);
            Assert.Equal(0, state.MultiplyCalls);
            Assert.Equal(0, state.ScalarMinCalls);
            Assert.Equal(0, state.DownloadBufferCalls);
        }
    }

#if !NETFRAMEWORK
    [SkippableFact]
    public void ActiveGpuBackend_ClassifyAndReduceDetectsEveryNonFiniteKind()
    {
        AiDotNet.Tensors.Engines.DirectGpuTensorEngine? engine = null;
        try
        {
            engine = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine();
        }
        catch
        {
            Skip.If(true, "No direct GPU backend is available on this host.");
        }

        using (engine)
        {
            Skip.If(engine is null || !engine.IsGpuAvailable, "No direct GPU backend is available on this host.");
            var backend = engine!.GetBackend();
            Skip.If(backend is null, "No direct GPU backend is available on this host.");

            foreach (int length in new[] { 1, 4, 31, 32, 33, 255, 256, 257, 1025 })
            {
                foreach (float badValue in new[]
                         { 0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity })
                {
                    var values = new float[length];
                    for (int i = 0; i < length; i++) values[i] = 1f;
                    bool allFinite = badValue == 0f;
                    if (!allFinite) values[length - 1] = badValue;

                    using var gradient = backend!.AllocateBuffer(values);
                    using var scratch = backend.AllocateBuffer(length);
                    using var aggregate = backend.AllocateBuffer(length);
                    using var finitePrefix = backend.AllocateBuffer(new float[] { 1f });
                    backend.Fill(aggregate, 1f, length);
                    backend.ClassifyFloat(finitePrefix, scratch, mode: 2, size: 1);
                    backend.Multiply(scratch, aggregate, aggregate, size: 1);
                    backend.ClassifyFloat(gradient, scratch, mode: 2, size: length);
                    backend.Multiply(scratch, aggregate, aggregate, size: length);
                    // Reported with the case. A bare Equal here said only "expected 0, actual 1",
                    // which does not say WHICH length or which non-finite kind slipped through —
                    // and the bad value is deliberately placed at values[length - 1], so the length
                    // is the whole diagnosis when a kernel drops its tail.
                    float observed = backend.Min(aggregate, length);
                    Assert.True(
                        observed == (allFinite ? 1f : 0f),
                        $"ClassifyFloat(mode 2) over {length} values with {badValue} at index "
                            + $"{length - 1}: expected min {(allFinite ? 1f : 0f)}, got {observed}. "
                            + "A miss here means the non-finite element was never classified — check "
                            + "whether the kernel covers a size that is not a multiple of its "
                            + "workgroup.");
                }
            }
        }
    }
#endif

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void DoublePlan_NonFiniteGradient_SkipsTheWholeStep(double badGradient)
    {
        var (plan, param) = BuildDoublePlan(badGradient);
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1f);
            var before = Snapshot(param);

            plan.Step();

            Assert.True(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(1, plan.NonFiniteStepsSkipped);
            for (int i = 0; i < param.Length; i++)
                Assert.True(before[i].Equals(param[i]), $"double parameter[{i}] changed");
        }
    }

    [Fact]
    public void DoubleGroupedSchedule_NonFiniteGradient_SkipsTheWholeStep()
    {
        var (plan, param) = BuildDoublePlan(double.NaN);
        using (plan)
        {
            plan.ConfigureOptimizerGrouped(
                OptimizerType.SGD,
                new[] { LrSchedule.Constant(0.1) },
                new[] { 0 });
            var before = Snapshot(param);

            plan.Step();

            Assert.True(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(1, plan.NonFiniteStepsSkipped);
            for (int i = 0; i < param.Length; i++)
                Assert.True(before[i].Equals(param[i]), $"double grouped parameter[{i}] changed");
        }
    }

    [Theory]
    [InlineData(OptimizerType.LBFGS)]
    [InlineData(OptimizerType.TrustRegion)]
    [InlineData(OptimizerType.ConjugateGradient)]
    [InlineData(OptimizerType.HypergradientSGD)]
    [InlineData(OptimizerType.DAdaptationSGD)]
    [InlineData(OptimizerType.ScheduleFreeSGD)]
    public void GlobalStateOptimizer_NonFiniteGradient_SkipsWithoutChangingParameters(
        OptimizerType optimizerType)
    {
        var (plan, param, input) = BuildPlan(2.0f);
        using (plan)
        {
            plan.ConfigureOptimizer(optimizerType, learningRate: 0.1f);

            // Establish non-trivial optimizer state first. In particular, Schedule-Free's z and x
            // copies now differ, so the bad step's pre-forward y write must be rolled back to x.
            plan.Step();
            for (int i = 0; i < input.Length; i++) input[i] = float.NaN;
            var before = Snapshot(param);
            int skippedBefore = plan.NonFiniteStepsSkipped;

            plan.Step();

            Assert.True(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(skippedBefore + 1, plan.NonFiniteStepsSkipped);
            for (int i = 0; i < param.Length; i++)
            {
                Assert.True(before[i].Equals(param[i]),
                    $"{optimizerType} changed parameter[{i}] on a discarded step: " +
                    $"{before[i]:G9} -> {param[i]:G9}");
            }
        }
    }

    [Fact]
    public void AllFiniteScan_AgreesWithElementwiseChecks_AcrossVectorBoundaries()
    {
        // The scan is vectorized with a scalar tail, so a bad value has to be caught wherever it
        // falls relative to the vector width -- including the very last element of the tail.
        for (int length = 1; length <= 40; length++)
        {
            for (int bad = 0; bad < length; bad++)
            {
                var values = new float[length];
                for (int i = 0; i < length; i++) values[i] = 1.0f;
                values[bad] = float.NaN;

                bool expected = false;
                bool actual;
                unsafe
                {
                    fixed (float* p = values)
                    {
                        actual = AiDotNet.Tensors.Engines.Compilation.FusedOptimizer.AllFiniteSimd(p, length);
                    }
                }

                Assert.True(expected == actual,
                    $"length={length} bad-index={bad}: scan returned {actual}");
            }
        }
    }

    [Fact]
    public void DoubleAllFiniteScan_AgreesWithElementwiseChecks_AcrossVectorBoundaries()
    {
        for (int length = 1; length <= 24; length++)
        {
            for (int bad = 0; bad < length; bad++)
            {
                var values = new double[length];
                for (int i = 0; i < length; i++) values[i] = 1.0;
                values[bad] = double.PositiveInfinity;

                bool actual;
                unsafe
                {
                    fixed (double* p = values)
                    {
                        actual = AiDotNet.Tensors.Engines.Compilation.FusedOptimizer.AllFiniteSimd(p, length);
                    }
                }

                Assert.False(actual, $"length={length} bad-index={bad}: scan missed infinity");
            }
        }
    }
}
