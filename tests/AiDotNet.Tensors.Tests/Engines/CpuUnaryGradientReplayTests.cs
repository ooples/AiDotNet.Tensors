using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CompilationGlobalState")]
public class CpuUnaryGradientReplayTests
{
    [Fact]
    public void ErfcReplayPreservesFloatRoundingAndDecimalOperations()
    {
        VerifyErfc(new[] { -3.125f, -0.0f, 0.3927f, 1.625f, 7.75f });
        VerifyErfc(new[] { -3.125m, 0m, 0.3927m, 1.625m, 7.75m });
    }

    private static void VerifyErfc<T>(T[] values)
    {
        var cpu = new CpuEngine();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        using var input = new Tensor<T>(values, new[] { values.Length });
        using var forward = cpu.TensorErfc(input);
        ICompiledTrainingPlan<T> plan;
        using (var scope = GraphMode.Enable())
        {
            _ = cpu.ReduceSum(cpu.TensorErfc(input), null);
            plan = scope.CompileTraining(new[] { input });
        }
        using (plan)
        {
            _ = plan.Step();
            var coefficient = ops.FromDouble(-2.0 / Math.Sqrt(Math.PI));
            for (int i = 0; i < values.Length; i++)
            {
                T x = values[i];
                Assert.Equal(AiDotNet.Tensors.Helpers.MathHelper.Erfc(x), forward[i]);
                Assert.Equal(ops.Multiply(coefficient, ops.Exp(ops.Negate(ops.Multiply(x, x)))), input.Grad![i]);
            }
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ErfcAndScaleContributeToTheSameInputAcrossSteps(bool erfcFirst)
    {
        var cpu = new CpuEngine();
        using var input = new Tensor<double>(new[] { 4 });
        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var first = erfcFirst ? cpu.TensorErfc(input) : cpu.TensorMultiplyScalar(input, 3.0);
            var second = erfcFirst ? cpu.TensorMultiplyScalar(input, 3.0) : cpu.TensorErfc(input);
            _ = cpu.ReduceSum(cpu.TensorAdd(first, second), null);
            plan = scope.CompileTraining(new[] { input });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            for (int step = 0; step < 3; step++)
            {
                for (int i = 0; i < input.Length; i++) input[i] = i * 0.5 - step;
                double expectedValue = 0;
                for (int i = 0; i < input.Length; i++)
                    expectedValue += AiDotNet.Tensors.Helpers.MathHelper.Erfc(input[i]) + 3 * input[i];
                Assert.Equal(expectedValue, plan.Step()[0], 12);
                for (int i = 0; i < input.Length; i++)
                {
                    double x = input[i];
                    double derivative = 3 - 2 / Math.Sqrt(Math.PI) * Math.Exp(-x * x);
                    Assert.Equal(derivative, input.Grad![i], 12);
                }
            }
        }
    }
}
