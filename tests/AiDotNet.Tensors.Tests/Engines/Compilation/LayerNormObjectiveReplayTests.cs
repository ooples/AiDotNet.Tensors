using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class LayerNormObjectiveReplayTests
{
    private const double Epsilon = 1e-5;

    private static double Reference(double[] point, double[] weights, int features)
    {
        double loss = 0;
        for (int start = 0; start < 6; start += features)
        {
            double mean = 0, variance = 0;
            for (int feature = 0; feature < features; feature++) mean += point[start + feature];
            mean /= features;
            for (int feature = 0; feature < features; feature++)
            {
                double centered = point[start + feature] - mean;
                variance += centered * centered;
            }
            variance /= features;
            for (int feature = 0; feature < features; feature++)
                loss += weights[start + feature] * ((point[start + feature] - mean)
                    / Math.Sqrt(variance + Epsilon) * point[6 + feature] + point[6 + features + feature]);
        }
        return loss;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Double_objective_rebinds_statistics_scratch_and_replays_all_parameter_gradients(bool normalizeAllDimensions)
    {
        var engine = new CpuEngine();
        int features = normalizeAllDimensions ? 6 : 3;
        var shape = normalizeAllDimensions ? new[] { 2, 3 } : new[] { 3 };
        var input = new Tensor<double>(new[] { -1.2, 0.3, 2.8, 4.1, -0.7, 1.6 }, new[] { 2, 3 });
        var gamma = new Tensor<double>(shape);
        var beta = new Tensor<double>(shape);
        gamma.Fill(1.0);
        var parameters = new[] { input, gamma, beta };
        var weights = new Tensor<double>(new[] { 0.2, -0.5, 1.1, -0.3, 0.8, 0.4 }, new[] { 2, 3 });
        Tensor<double> Objective() => engine.ReduceSum(engine.TensorMultiply(
            engine.LayerNorm(input, gamma, beta, Epsilon, out _, out _), weights), null);

        using var recording = ClosureObjectiveGuard<double>.BeginRecording();
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            guard!.AttachTo(plan);
            for (int step = 0; step < 4; step++)
            {
                var point = new double[6 + 2 * features];
                for (int i = 0; i < 6; i++) point[i] = Math.Sin(i + 0.6 * step) * (1 + step) + 0.2 * i * step;
                for (int i = 0; i < features; i++)
                {
                    point[6 + i] = 0.7 + 0.1 * i - 0.15 * step;
                    point[6 + features + i] = 0.2 * step - 0.07 * i;
                }
                int offset = 0;
                foreach (var parameter in parameters)
                {
                    for (int i = 0; i < parameter.Length; i++) parameter.SetFlat(i, point[offset++]);
                    parameter.IncrementVersion();
                }
                var weightValues = new double[6];
                for (int i = 0; i < weightValues.Length; i++) weightValues[i] = Math.Cos(i - 0.4 * step);
                // A fresh leaf and fresh statistics holder must match the same original compiled objective.
                weights = new Tensor<double>(weightValues, new[] { 2, 3 });
                Assert.True(guard.TryRebind(Objective, parameters, out _) == ClosureGuardResult.Matched, guard.LastReason);
                Assert.Equal(Reference(point, weightValues, features), plan.Step().ToArray()[0], 10);
                offset = 0;
                for (int tensor = 0; tensor < parameters.Length; tensor++)
                {
                    var gradients = plan.Gradients[tensor].ToArray();
                    for (int index = 0; index < gradients.Length; index++, offset++)
                    {
                        const double delta = 1e-5;
                        double saved = point[offset];
                        point[offset] = saved + delta;
                        double plus = Reference(point, weightValues, features);
                        point[offset] = saved - delta;
                        double minus = Reference(point, weightValues, features);
                        point[offset] = saved;
                        double expected = (plus - minus) / (2 * delta);
                        Assert.True(Math.Abs(expected - gradients[index]) < 2e-6,
                            $"allDimensions={normalizeAllDimensions}, step={step}, parameter={offset}: expected {expected:R}, got {gradients[index]:R}");
                        Assert.Equal(saved, parameters[tensor].GetFlat(index));
                    }
                }
            }
        }
    }
}
