using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CompilationGlobalState")]
public class CpuUnaryGradientReplayTests
{
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
                plan.Step();
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
