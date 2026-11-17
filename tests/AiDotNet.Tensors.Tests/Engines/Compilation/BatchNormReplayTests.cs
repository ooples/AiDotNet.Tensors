using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class BatchNormReplayTests
{
    private const double Epsilon = 1e-5;

    private static int[] Shape(int rank) => rank switch
    {
        1 => new[] { 3 },
        2 => new[] { 4, 3 },
        3 => new[] { 3, 2, 4 },
        4 => new[] { 2, 3, 2, 2 },
        _ => throw new ArgumentOutOfRangeException(nameof(rank)),
    };

    private static int Channel(int index, int[] shape)
        => shape.Length switch
        {
            1 => index,
            2 => index % shape[1],
            3 => index / (shape[1] * shape[2]),
            4 => (index / (shape[2] * shape[3])) % shape[1],
            _ => throw new ArgumentException(nameof(shape)),
        };

    private static (double[] Output, double[] Mean, double[] Variance) Reference(
        double[] input, int[] shape, double[] gamma, double[] beta)
    {
        var means = new double[gamma.Length];
        var variances = new double[gamma.Length];
        int count = input.Length / gamma.Length;
        for (int i = 0; i < input.Length; i++) means[Channel(i, shape)] += input[i] / count;
        for (int i = 0; i < input.Length; i++)
        {
            int c = Channel(i, shape);
            double centered = input[i] - means[c];
            variances[c] += centered * centered / count;
        }
        var output = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            int c = Channel(i, shape);
            output[i] = (input[i] - means[c]) / Math.Sqrt(variances[c] + Epsilon) * gamma[c] + beta[c];
        }
        return (output, means, variances);
    }

    private static void Close(double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"element {i}: expected {expected[i]:R}, actual {actual[i]:R}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Double_derived_input_and_statistics_stay_lazy_and_preserve_layout(int rank)
    {
        var engine = new CpuEngine();
        int[] shape = Shape(rank);
        var source = new Tensor<double>(shape);
        for (int i = 0; i < source.Length; i++) source.SetFlat(i, Math.Sin(i * 0.7) + i * 0.2);
        var gamma = new Tensor<double>(new[] { 0.7, 1.1, -0.6 }, new[] { 3 });
        var beta = new Tensor<double>(new[] { 0.2, -0.3, 0.4 }, new[] { 3 });
        double[] scaled = source.ToArray();
        for (int i = 0; i < scaled.Length; i++) scaled[i] *= 1.3;
        var expected = Reference(scaled, shape, gamma.ToArray(), beta.ToArray());

        using var scope = GraphMode.EnableTraining(new[] { source, gamma, beta });
        var derived = engine.Reshape(engine.TensorMultiplyScalar(source, 1.3), shape);
        var output = engine.BatchNorm(derived, gamma, beta, Epsilon, out var mean, out var variance);
        Assert.False(derived.LazySource?.IsRealized ?? true);
        Assert.False(output.LazySource?.IsRealized ?? true);
        Assert.False(mean.LazySource?.IsRealized ?? true);
        Assert.False(variance.LazySource?.IsRealized ?? true);
        // Reading statistics first must realize their producer, not expose uninitialized saved buffers.
        Close(expected.Mean, mean.ToArray(), 1e-12);
        Close(expected.Variance, variance.ToArray(), 1e-12);
        Close(expected.Output, output.ToArray(), 1e-12);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Double_closure_rebind_refreshes_output_statistics_and_saved_backward(int rank)
    {
        var engine = new CpuEngine();
        int[] shape = Shape(rank);
        int length = 1;
        foreach (int extent in shape) length *= extent;
        var source = new Tensor<double>(new[] { length });
        var gamma = new Tensor<double>(new[] { 3 });
        gamma.Fill(1.0);
        var beta = new Tensor<double>(new[] { 3 });
        var parameters = new[] { source, gamma, beta };
        var weights = new Tensor<double>(shape);
        weights.Fill(1.0);
        var observedMean = new Tensor<double>(new[] { 3 });
        var observedVariance = new Tensor<double>(new[] { 3 });
        Tensor<double> Objective()
        {
            var derived = engine.Reshape(engine.TensorMultiplyScalar(source, 1.3), shape);
            var output = engine.BatchNorm(derived, gamma, beta, Epsilon, out var mean, out var variance);
            using (new NoGradScope<double>())
            {
                engine.TensorMultiplyScalarInPlace(observedMean, 0.0);
                engine.TensorAddInPlace(observedMean, mean);
                engine.TensorMultiplyScalarInPlace(observedVariance, 0.0);
                engine.TensorAddInPlace(observedVariance, variance);
            }
            return engine.ReduceSum(engine.TensorMultiply(output, weights), null);
        }

        using var recording = ClosureObjectiveGuard<double>.BeginRecording();
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            guard!.AttachTo(plan);
            for (int step = 0; step < 3; step++)
            {
                for (int i = 0; i < length; i++) source.SetFlat(i, Math.Sin(i + step * 0.4) * (step + 1) + 0.2 * i * step);
                for (int c = 0; c < 3; c++)
                {
                    gamma.SetFlat(c, 0.8 + c * 0.1 + step * 0.2);
                    beta.SetFlat(c, c * 0.2 - step * 0.15);
                }
                foreach (var parameter in parameters) parameter.IncrementVersion();
                var weightData = new double[length];
                for (int i = 0; i < length; i++) weightData[i] = Math.Cos(i * 0.6 - step);
                weights = new Tensor<double>(weightData, shape);
                Assert.True(guard.TryRebind(Objective, parameters, out _) == ClosureGuardResult.Matched, guard.LastReason);

                double Loss(double[] x, double[] g, double[] b)
                {
                    var scaled = (double[])x.Clone();
                    for (int i = 0; i < scaled.Length; i++) scaled[i] *= 1.3;
                    var result = Reference(scaled, shape, g, b);
                    double total = 0;
                    for (int i = 0; i < length; i++) total += result.Output[i] * weightData[i];
                    return total;
                }
                var values = new[] { source.ToArray(), gamma.ToArray(), beta.ToArray() };
                double expectedLoss = Loss(values[0], values[1], values[2]);
                Assert.Equal(expectedLoss, plan.Step().ToArray()[0], 10);
                var scaledInput = (double[])values[0].Clone();
                for (int i = 0; i < length; i++) scaledInput[i] *= 1.3;
                var expected = Reference(scaledInput, shape, values[1], values[2]);
                Close(expected.Mean, observedMean.ToArray(), 1e-11);
                Close(expected.Variance, observedVariance.ToArray(), 1e-11);
                for (int parameter = 0; parameter < parameters.Length; parameter++)
                {
                    var gradient = plan.Gradients[parameter].ToArray();
                    for (int index = 0; index < gradient.Length; index++)
                    {
                        const double delta = 1e-5;
                        double saved = values[parameter][index];
                        values[parameter][index] = saved + delta;
                        double plus = Loss(values[0], values[1], values[2]);
                        values[parameter][index] = saved - delta;
                        double minus = Loss(values[0], values[1], values[2]);
                        values[parameter][index] = saved;
                        Assert.True(Math.Abs(gradient[index] - (plus - minus) / (2 * delta)) < 2e-6,
                            $"rank={rank}, step={step}, parameter={parameter}, index={index}");
                    }
                }
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Rank1_eager_and_compiled_gradients_match_the_single_observation_oracle(bool useHalf)
    {
        if (useHalf) CheckSingleObservation<Half>();
        else CheckSingleObservation<double>();
    }

    private static void CheckSingleObservation<T>()
    {
        var engine = new CpuEngine();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        T[] ConvertValues(params double[] values)
        {
            var converted = new T[values.Length];
            for (int i = 0; i < values.Length; i++) converted[i] = ops.FromDouble(values[i]);
            return converted;
        }
        var input = new Tensor<T>(ConvertValues(-1.0, 2.0, 4.0), new[] { 3 });
        var gamma = new Tensor<T>(ConvertValues(0.5, 2.0, -1.0), new[] { 3 });
        var beta = new Tensor<T>(ConvertValues(0.25, -0.5, 1.0), new[] { 3 });
        var weights = new Tensor<T>(ConvertValues(1.0, -2.0, 0.5), new[] { 3 });
        var output = engine.BatchNorm(input, gamma, beta, Epsilon, out var mean, out var variance);
        Assert.Equal(beta.ToArray(), output.ToArray());
        var gradInput = engine.BatchNormBackward(weights, input, gamma, mean, variance, Epsilon, out var gradGamma, out var gradBeta);
        Assert.Equal(new[] { 3 }, gradInput.Shape.ToArray());
        Assert.All(gradInput.ToArray(), value => Assert.Equal(ops.Zero, value));
        Assert.All(gradGamma.ToArray(), value => Assert.Equal(ops.Zero, value));
        Assert.Equal(weights.ToArray(), gradBeta.ToArray());
        Assert.Throws<ArgumentException>(() => engine.BatchNormBackward(
            engine.Reshape(weights, new[] { 1, 3 }), input, gamma, mean, variance, Epsilon, out _, out _));

        var parameters = new[] { input, gamma, beta };
        CompiledTrainingPlan<T> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var derived = engine.Reshape(engine.TensorMultiplyScalar(input, ops.One), new[] { 3 });
            var normalized = engine.BatchNorm(derived, gamma, beta, Epsilon, out _, out _);
            plan = scope.CompileTraining(parameters, engine.ReduceSum(engine.TensorMultiply(normalized, weights), null));
        }
        using (plan)
        {
            Assert.Equal(ops.FromDouble(1.75), plan.Step().ToArray()[0]);
            Assert.All(plan.Gradients[0].ToArray(), value => Assert.Equal(ops.Zero, value));
            Assert.All(plan.Gradients[1].ToArray(), value => Assert.Equal(ops.Zero, value));
            Assert.Equal(weights.ToArray(), plan.Gradients[2].ToArray());
        }
    }

    [Fact]
    public void Half_rank3_replay_preserves_float32_statistics_for_backward()
    {
        var engine = new CpuEngine();
        var input = new Tensor<Half>(new[] { (Half)0, (Half)0.0001, (Half)0.0002, (Half)0.0003 }, new[] { 1, 2, 2 });
        var gamma = new Tensor<Half>(new[] { (Half)1 }, new[] { 1 });
        var beta = new Tensor<Half>(new[] { (Half)0 }, new[] { 1 });
        var weights = new Tensor<Half>(new[] { (Half)1, (Half)(-2), (Half)0.5, (Half)3 }, input.Shape.ToArray());
        var parameters = new[] { input, gamma, beta };
        CompiledTrainingPlan<Half> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var derived = engine.Reshape(engine.TensorMultiplyScalar(input, (Half)1), input.Shape.ToArray());
            var output = engine.BatchNorm(derived, gamma, beta, Epsilon, out _, out _);
            plan = scope.CompileTraining(parameters, engine.ReduceSum(engine.TensorMultiply(output, weights), null));
        }
        using (plan)
        {
            for (int step = 0; step < 2; step++)
            {
                input.SetFlat(3, (Half)(0.0003 + 0.0001 * step));
                input.IncrementVersion();
                engine.BatchNorm(input, gamma, beta, Epsilon, out var mean, out var variance);
                Assert.Equal((Half)0, variance.GetFlat(0)); // FP16 rounds away variance; backward must use its FP32 carry.
                var gradInput = engine.BatchNormBackward(weights, input, gamma, mean, variance, Epsilon, out var gradGamma, out var gradBeta);
                plan.Step();
                Assert.Equal(gradInput.ToArray(), plan.Gradients[0].ToArray());
                Assert.Equal(gradGamma.ToArray(), plan.Gradients[1].ToArray());
                Assert.Equal(gradBeta.ToArray(), plan.Gradients[2].ToArray());
            }
        }
    }
}
