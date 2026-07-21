using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public sealed class TensorSquashBackwardTests
{
    [Fact]
    public void NearZeroNorm_MatchesFiniteDifferenceOfRegularizedForward()
    {
        var engine = new CpuEngine();
        double[] values = [3e-9, -4e-9, 5e-9];
        double[] weights = [0.7, -1.1, 0.4];

        using var input = new Tensor<double>((double[])values.Clone(), [1, values.Length]);
        using var gradOutput = new Tensor<double>((double[])weights.Clone(), [1, weights.Length]);
        using var output = engine.TensorSquash(input, axis: -1);
        using var analytic = engine.TensorSquashBackward(gradOutput, input, output, axis: -1);

        const double step = 1e-12;
        for (int i = 0; i < values.Length; i++)
        {
            double original = values[i];
            values[i] = original + step;
            double plus = WeightedForward(engine, values, weights);
            values[i] = original - step;
            double minus = WeightedForward(engine, values, weights);
            values[i] = original;

            double numerical = (plus - minus) / (2 * step);
            double error = Math.Abs(analytic[i] - numerical);
            double tolerance = 1e-13 + 2e-5 * Math.Abs(numerical);
            Assert.True(error <= tolerance,
                $"gradient[{i}] analytic={analytic[i]:E16}, numerical={numerical:E16}, error={error:E3}");
        }
    }

    private static double WeightedForward(CpuEngine engine, double[] values, double[] weights)
    {
        using var input = new Tensor<double>((double[])values.Clone(), [1, values.Length]);
        using var output = engine.TensorSquash(input, axis: -1);
        double sum = 0;
        for (int i = 0; i < output.Length; i++)
            sum += output[i] * weights[i];
        return sum;
    }
}
