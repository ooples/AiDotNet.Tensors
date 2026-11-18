using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("DirectGpuSerial")]
public sealed class NanToNumReplayTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    [Fact]
    public void Half_defaults_use_finite_half_endpoints_and_preserve_finite_bits()
    {
        var engine = new CpuEngine();
        Half[] values = [Half.NaN, Half.PositiveInfinity, Half.NegativeInfinity,
            (Half)(-0.0), Half.Epsilon, Half.MaxValue, Half.MinValue, (Half)1.5];
        Half[] expected = [(Half)0, Half.MaxValue, Half.MinValue,
            (Half)(-0.0), Half.Epsilon, Half.MaxValue, Half.MinValue, (Half)1.5];
        var result = engine.TensorNanToNum(new Tensor<Half>(values, [2, 4]));
        Assert.Equal(new[] { 2, 4 }, result.Shape.ToArray());
        Assert.Equal(expected.Select(BitConverter.HalfToInt16Bits), result.ToArray().Select(BitConverter.HalfToInt16Bits));
    }

    [Fact]
    public void Float_defaults_use_finite_float_endpoints_and_preserve_finite_bits()
    {
        var engine = new CpuEngine();
        float[] values = [float.NaN, float.PositiveInfinity, float.NegativeInfinity,
            -0.0f, float.Epsilon, float.MaxValue, float.MinValue, 1.5f];
        float[] expected = [0, float.MaxValue, float.MinValue,
            -0.0f, float.Epsilon, float.MaxValue, float.MinValue, 1.5f];
        var result = engine.TensorNanToNum(new Tensor<float>(values, [2, 4]));
        Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), result.ToArray().Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void Double_defaults_use_full_finite_double_range_and_preserve_finite_bits()
    {
        var engine = new CpuEngine();
        double[] values = [double.NaN, double.PositiveInfinity, double.NegativeInfinity,
            -0.0, double.Epsilon, double.MaxValue, double.MinValue, 1e200];
        double[] expected = [0, double.MaxValue, double.MinValue,
            -0.0, double.Epsilon, double.MaxValue, double.MinValue, 1e200];
        var result = engine.TensorNanToNum(new Tensor<double>(values, [2, 4]));
        Assert.Equal(expected.Select(BitConverter.DoubleToInt64Bits), result.ToArray().Select(BitConverter.DoubleToInt64Bits));
    }

    [Theory]
    [InlineData("Half")]
    [InlineData("float")]
    [InlineData("double")]
    public void Explicit_replacements_are_independent_and_finite_values_pass_through(string dtype)
        => Dispatch(dtype, new ExplicitReplacementCheck());

    [Fact]
    public void Integral_and_decimal_finite_values_remain_exact()
    {
        var engine = new CpuEngine();
        long[] integers = [long.MinValue, -9007199254740993L, 0, 9007199254740993L, long.MaxValue];
        decimal[] decimals = [decimal.MinValue, -0.1234567890123456789012345678m, 0m, decimal.MaxValue];
        Assert.Equal(integers, engine.TensorNanToNum(new Tensor<long>(integers, [5])).ToArray());
        Assert.Equal(decimals, engine.TensorNanToNum(new Tensor<decimal>(decimals, [4])).ToArray());
    }

    [Fact]
    public void Finite_integral_bool_and_decimal_inputs_ignore_invalid_unused_replacements()
    {
        var engine = new CpuEngine();
        long[] integers = [long.MinValue, 0, long.MaxValue];
        bool[] booleans = [false, true];
        decimal[] decimals = [decimal.MinValue, 0m, decimal.MaxValue];
        Assert.Equal(integers, engine.TensorNanToNum(new Tensor<long>(integers, [3]),
            nan: double.NaN, posinf: double.PositiveInfinity, neginf: double.MinValue).ToArray());
        Assert.Equal(booleans, engine.TensorNanToNum(new Tensor<bool>(booleans, [2]),
            nan: double.NaN, posinf: double.PositiveInfinity, neginf: double.MinValue).ToArray());
        Assert.Equal(decimals, engine.TensorNanToNum(new Tensor<decimal>(decimals, [3]),
            nan: double.NaN, posinf: double.PositiveInfinity, neginf: double.MinValue).ToArray());
    }

    [Theory]
    [InlineData("Half")]
    [InlineData("float")]
    [InlineData("double")]
    public void Eager_noncontiguous_view_preserves_layout_and_masked_backward(string dtype)
        => Dispatch(dtype, new EagerViewCheck());

    [Theory]
    [InlineData("Half")]
    [InlineData("float")]
    [InlineData("double")]
    public void Lazy_replay_refreshes_nonfinite_mask_and_rebinds_fresh_weights(string dtype)
        => Dispatch(dtype, new ReplayCheck());

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Original_half_layout_coordinate_pipeline_rounds_ties_to_even_and_clamps(bool compiled)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var source = new Tensor<Half>(new[] { Half.NaN, (Half)2.5, (Half)3.5, (Half)(-1), (Half)20 }, [5]);
        Half[][] expected = [[(Half)0, (Half)2, (Half)4, (Half)0, (Half)7],
            [(Half)7, (Half)0, (Half)4, (Half)6, (Half)0]];
        Tensor<Half> Pipeline() => engine.TensorClamp(engine.TensorRound(engine.TensorNanToNum(source, 0, null, null)), (Half)0, (Half)7);
        Tensor<Half>? observed = null;
        CompiledTrainingPlan<Half>? plan = null;
        if (compiled)
        {
            using var scope = GraphMode.EnableTraining(new[] { source });
            observed = Pipeline();
            Assert.False(Assert.IsType<LazyNode<Half>>(observed.LazySource).IsRealized);
            Assert.Null(scope.HostReadOp);
            GradientRecording.RetainOutput(observed);
            plan = scope.CompileTraining(new[] { source }, engine.ReduceSum(observed, null));
        }
        using (plan)
        {
            for (int step = 0; step < 2; step++)
            {
                if (step == 1)
                {
                    Half[] next = [Half.PositiveInfinity, Half.NegativeInfinity, (Half)4.5, (Half)5.5, Half.NaN];
                    for (int i = 0; i < next.Length; i++) source.SetFlat(i, next[i]);
                    source.IncrementVersion();
                }
                if (plan is null) observed = Pipeline();
                else
                {
                    plan.Step();
                    Assert.All(plan.Gradients[0].ToArray(), value => Assert.Equal((Half)0, value));
                }
                Assert.Equal(expected[step], observed!.ToArray());
            }
        }
    }

    private interface ICheck { void Run<T>(); }
    private static void Dispatch(string dtype, ICheck check)
    {
        switch (dtype)
        {
            case "Half": check.Run<Half>(); break;
            case "float": check.Run<float>(); break;
            case "double": check.Run<double>(); break;
            default: throw new ArgumentOutOfRangeException(nameof(dtype));
        }
    }

    private static Tensor<T> Make<T>(double[] values, int[] shape)
        => new(values.Select(MathHelper.GetNumericOperations<T>().FromDouble).ToArray(), shape);

    private sealed class ExplicitReplacementCheck : ICheck
    {
        public void Run<T>()
        {
            var engine = new CpuEngine();
            var source = Make<T>([double.NaN, double.PositiveInfinity, double.NegativeInfinity, -1.5, 0, 3.25], [2, 3]);
            var result = engine.TensorNanToNum(source, nan: 2.5, posinf: 7.5, neginf: -9.25);
            Assert.Equal(Make<T>([2.5, 7.5, -9.25, -1.5, 0, 3.25], [2, 3]).ToArray(), result.ToArray());
            // Explicit nonfinite replacements remain permitted; only defaults use finite endpoints.
            var nonfinite = engine.TensorNanToNum(source, nan: double.NaN,
                posinf: double.PositiveInfinity, neginf: double.NegativeInfinity).ToArray();
            var ops = MathHelper.GetNumericOperations<T>();
            Assert.True(double.IsNaN(ops.ToDouble(nonfinite[0])));
            Assert.Equal(double.PositiveInfinity, ops.ToDouble(nonfinite[1]));
            Assert.Equal(double.NegativeInfinity, ops.ToDouble(nonfinite[2]));
            var replaced = engine.TensorNanToNum(Make<T>([double.PositiveInfinity], [1]), posinf: double.NaN);
            Assert.True(double.IsNaN(ops.ToDouble(replaced.ToArray()[0])));
        }
    }

    private sealed class EagerViewCheck : ICheck
    {
        public void Run<T>()
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var source = Make<T>([double.NaN, 1, double.PositiveInfinity, -2, double.NegativeInfinity, 3], [2, 3]);
            using var tape = new GradientTape<T>();
            tape.BindEngineIfUnset(engine);
            var view = engine.TensorPermute(source, [1, 0]);
            Assert.False(view.IsContiguous);
            var output = engine.TensorNanToNum(view, nan: 2, posinf: 5, neginf: -4);
            Assert.Equal(new[] { 3, 2 }, output.Shape.ToArray());
            Assert.Equal(Make<T>([2, -2, 1, -4, 5, 3], [3, 2]).ToArray(), output.ToArray());
            var loss = engine.ReduceSum(engine.TensorMultiply(output, Make<T>([1, 2, 3, 4, 5, 6], [3, 2])), null);
            var gradients = tape.ComputeGradients(loss, new[] { source });
            Assert.Equal(Make<T>([0, 3, 0, 2, 0, 6], [2, 3]).ToArray(), gradients[source]!.ToArray());
        }
    }

    private sealed class ReplayCheck : ICheck
    {
        public void Run<T>()
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var ops = MathHelper.GetNumericOperations<T>();
            double[][] points = [[double.NaN, 1, double.PositiveInfinity, -2, double.NegativeInfinity, 3],
                [4, double.NegativeInfinity, double.NaN, double.PositiveInfinity, 2, -1],
                [double.NegativeInfinity, double.PositiveInfinity, 4, double.NaN, -3, 2]];
            double[][] outputs = [[2, -2, 1, -4, 5, 3], [4, 5, -4, 2, 2, -1], [-4, 2, 5, -3, 4, 2]];
            double[][] gradients = [[0, 3, 0, 2, 0, 6], [2, 0, 0, 0, 5, 7], [0, 0, 7, 0, 6, 8]];
            double[] losses = [28, 22, 47];
            var source = Make<T>(points[0], [2, 3]);
            Tensor<T>[] parameters = [source];
            var weights = Make<T>([1, 2, 3, 4, 5, 6], [3, 2]);
            Tensor<T>? observed = null;
            Tensor<T> Objective()
            {
                observed = engine.TensorNanToNum(engine.TensorPermute(source, [1, 0]), nan: 2, posinf: 5, neginf: -4);
                GradientRecording.RetainOutput(observed);
                return engine.ReduceSum(engine.TensorMultiply(observed, weights), null);
            }
            using var recording = ClosureObjectiveGuard<T>.BeginRecording();
            CompiledTrainingPlan<T> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var loss = Objective();
                Assert.False(Assert.IsType<LazyNode<T>>(observed!.LazySource).IsRealized);
                Assert.Null(scope.HostReadOp);
                plan = scope.CompileTraining(parameters, loss);
            }
            var tracedOutput = observed!;
            using (plan)
            {
                var guard = ClosureObjectiveGuard<T>.TakeRecorded(out var reason);
                Assert.True(guard is not null, reason);
                guard!.AttachTo(plan);
                for (int step = 0; step < 3; step++)
                {
                    for (int i = 0; i < 6; i++) source.SetFlat(i, ops.FromDouble(points[step][i]));
                    source.IncrementVersion();
                    weights = Make<T>(Enumerable.Range(1 + step, 6).Select(i => (double)i).ToArray(), [3, 2]);
                    Assert.True(guard.TryRebind(Objective, parameters, out _) == ClosureGuardResult.Matched, guard.LastReason);
                    Assert.Equal(ops.FromDouble(losses[step]), plan.Step().ToArray()[0]);
                    Assert.Equal(Make<T>(outputs[step], [3, 2]).ToArray(), tracedOutput.ToArray());
                    Assert.Equal(Make<T>(gradients[step], [2, 3]).ToArray(), plan.Gradients[0].ToArray());
                }
            }
        }
    }
}
