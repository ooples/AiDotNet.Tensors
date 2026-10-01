using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A double plan updates in double, so it must use the hyperparameters exactly as the caller configured them.
/// ConfigureOptimizer used to take them as float: a double model trained with lr = 1e-4, betas 0.9 / 0.999 and
/// eps = 1e-8 (none representable in float) drifted from the same update computed in double by the rounding of every
/// one of them, and a checkpoint stored the rounded values. These run the plan against a double reference that uses
/// the unrounded values; the tolerance is double precision, far below what float rounding moves.
/// </summary>
[Collection("DirectGpuSerial")]
public class DoublePlanHyperparameterPrecisionTests
{
    private const double Lr = 1e-4, B1 = 0.9, B2 = 0.999, Eps = 1e-8, Wd = 0.01;
    private const int Steps = 5;
    private const double Tolerance = 1e-15;

    private static readonly double[] Init = { 0.9, -0.4, 1.7, -2.3, 0.05, 0.6, -1.1, 0.3, 2.2, -0.7, 0.45 };

    // f(w) = sum(w^2), so the gradient is 2w, evaluated before each step as the plan's backward does.
    private static double[] Reference(OptimizerType type)
    {
        var w = (double[])Init.Clone();
        var m = new double[w.Length];
        var v = new double[w.Length];
        for (int t = 1; t <= Steps; t++)
        {
            double bc1 = 1 - Math.Pow(B1, t);
            double bc2 = 1 - Math.Pow(B2, t);
            for (int i = 0; i < w.Length; i++)
            {
                double g = 2 * w[i];
                if (type == OptimizerType.SGD)
                {
                    w[i] -= Lr * g;
                    continue;
                }
                if (type == OptimizerType.AdamW) w[i] -= Lr * Wd * w[i];
                m[i] = B1 * m[i] + (1 - B1) * g;
                v[i] = B2 * v[i] + (1 - B2) * g * g;
                w[i] -= Lr * (m[i] / bc1) / (Math.Sqrt(v[i] / bc2) + Eps);
            }
        }
        return w;
    }

    private static (Tensor<double> Weight, ICompiledTrainingPlan<double> Plan) Compile()
    {
        var engine = new CpuEngine();
        var weight = new Tensor<double>(new[] { Init.Length });
        for (int i = 0; i < Init.Length; i++) weight[i] = Init[i];
        using var scope = GraphMode.Enable();
        engine.ReduceSum(engine.TensorMultiply(weight, weight), null);
        return (weight, scope.CompileTraining(new[] { weight }));
    }

    private static void Configure(ICompiledTrainingPlan<double> plan, OptimizerType type) =>
        plan.ConfigureOptimizer(type, Lr, B1, B2, Eps, type == OptimizerType.AdamW ? Wd : 0);

    [Theory]
    [InlineData(OptimizerType.SGD)]
    [InlineData(OptimizerType.Adam)]
    [InlineData(OptimizerType.AdamW)]
    public void A_double_plan_updates_with_the_exact_hyperparameters(OptimizerType type)
    {
        var expected = Reference(type);
        var (weight, plan) = Compile();
        using (plan)
        {
            Configure(plan, type);
            for (int s = 0; s < Steps; s++) plan.Step();
        }

        for (int i = 0; i < Init.Length; i++)
            Assert.True(Math.Abs(expected[i] - weight[i]) <= Tolerance,
                $"{type}: w[{i}] expected {expected[i]:R}, got {weight[i]:R} (float-rounded hyperparameters move it ~1e-9)");
    }

    [Fact]
    public void Exported_optimizer_state_keeps_the_exact_hyperparameters()
    {
        var (_, plan) = Compile();
        using (plan)
        {
            Configure(plan, OptimizerType.AdamW);
            plan.Step();
            var checkpoint = Assert.IsType<CompiledTrainingPlan<double>>(plan).CaptureFusedOptimizerCheckpoint();
            Assert.NotNull(checkpoint);
            Assert.Equal(B1, checkpoint!.Beta1);
            Assert.Equal(B2, checkpoint.Beta2);
            Assert.Equal(Eps, checkpoint.Epsilon);
            Assert.Equal(Wd, checkpoint.WeightDecay);

            // The round trip through the exported bytes keeps them too.
            plan.ImportOptimizerState(plan.ExportOptimizerState() ?? throw new InvalidOperationException("no state"));
            var restored = Assert.IsType<CompiledTrainingPlan<double>>(plan).CaptureFusedOptimizerCheckpoint();
            Assert.NotNull(restored);
            Assert.Equal(B1, restored!.Beta1);
            Assert.Equal(Eps, restored.Epsilon);
            Assert.Equal(Wd, restored.WeightDecay);
        }
    }

    [Fact]
    public void Optimizer_state_exported_with_float_hyperparameters_still_imports()
    {
        var (_, plan) = Compile();
        using (plan)
        {
            Configure(plan, OptimizerType.Adam);
            plan.Step();
            byte[] current = plan.ExportOptimizerState() ?? throw new InvalidOperationException("no state");

            // Version 3 wrote beta1, beta2, epsilon and weight decay as float. For an ungrouped plan that is the only
            // layout difference: magic, version, then hasCheckpoint (bool), type (int), grouped (bool), step (int),
            // then the four values. Transcode the current export into that layout.
            const int header = 4 + 4, before = 1 + 4 + 1 + 4;
            int at = header + before;
            Assert.Equal(4, BitConverter.ToInt32(current, 4));
            var legacy = new byte[current.Length - (4 * sizeof(double)) + (4 * sizeof(float))];
            Buffer.BlockCopy(current, 0, legacy, 0, at);
            BitConverter.GetBytes(3).CopyTo(legacy, 4);
            for (int k = 0; k < 4; k++)
                BitConverter.GetBytes((float)BitConverter.ToDouble(current, at + (k * sizeof(double))))
                    .CopyTo(legacy, at + (k * sizeof(float)));
            Buffer.BlockCopy(current, at + (4 * sizeof(double)), legacy, at + (4 * sizeof(float)),
                current.Length - at - (4 * sizeof(double)));

            plan.ImportOptimizerState(legacy);
            var restored = Assert.IsType<CompiledTrainingPlan<double>>(plan).CaptureFusedOptimizerCheckpoint();
            Assert.NotNull(restored);
            Assert.Equal((double)(float)B1, restored!.Beta1);
            Assert.Equal((double)(float)Eps, restored.Epsilon);
            Assert.Equal(1, restored.OptimizerStep);
        }
    }
}