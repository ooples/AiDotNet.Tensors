using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CompilationGlobalState")]
public class CpuTileReplayTests
{
    [Theory]
    [InlineData(3, 1, 1)]
    [InlineData(1, 5, 1)]
    [InlineData(1, 1, 7)]
    [InlineData(2, 1, 3)]
    [InlineData(1, 1, 1)]
    public void ReplayReadsCurrentInputAndAccumulatesPerCopyGradients(int a, int b, int c)
    {
        var cpu = new CpuEngine();
        int[] repeats = { a, b, c };
        int[] shape = { 2, 3, 4 };
        int[] resultShape = { 2 * a, 3 * b, 4 * c };
        using var input = new Tensor<double>(shape);
        using var weights = new Tensor<double>(resultShape);
        for (int i = 0; i < weights.Length; i++) weights[i] = i % 13 - 6;
        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            _ = cpu.ReduceSum(cpu.TensorMultiply(cpu.TensorTile(input, repeats), weights), null);
            plan = scope.CompileTraining(new[] { input });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            for (int step = 1; step <= 3; step++)
            {
                for (int i = 0; i < input.Length; i++) input[i] = i * step + 1;
                var expectedGradient = new double[input.Length];
                double expectedLoss = 0;
                for (int i = 0; i < weights.Length; i++)
                {
                    int row = i / resultShape[2];
                    int batch = row / resultShape[1] % shape[0];
                    int time = row % resultShape[1] % shape[1];
                    int channel = i % resultShape[2] % shape[2];
                    int source = (batch * shape[1] + time) * shape[2] + channel;
                    expectedLoss += input[source] * weights[i];
                    expectedGradient[source] += weights[i];
                }
                Assert.Equal(expectedLoss, plan.Step()[0]);
                Assert.Equal(expectedGradient, input.Grad!.ToArray());
            }
        }
    }
}
