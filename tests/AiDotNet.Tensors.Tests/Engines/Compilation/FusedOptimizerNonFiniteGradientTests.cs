using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
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
