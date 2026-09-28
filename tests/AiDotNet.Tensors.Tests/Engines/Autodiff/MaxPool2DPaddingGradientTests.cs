using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// MaxPool2D with padding, on the CPU reference engine. The gradient routes each output's upstream value to the
/// input cell that won its window; away from ties that is the exact derivative, so a finite difference in double
/// is a tight oracle. Inputs are distinct so no window has a tie.
/// </summary>
public class MaxPool2DPaddingGradientTests : IDisposable
{
    // These tests set the process-wide engine; restore it so later tests see the engine they expect.
    private readonly IEngine _previousEngine = AiDotNetEngine.Current;

    public void Dispose() => AiDotNetEngine.Current = _previousEngine;

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void Padded_gradient_matches_finite_differences(int padding)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var x = new Tensor<double>(new[] { 2, 3, 5, 6 });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.37 * i) + 0.001 * i;

        Tensor<double> Forward(Tensor<double> input) => engine.MaxPool2D(input, poolSize: 3, stride: 2, padding: padding);

        var probe = Forward(x);
        var weight = new Tensor<double>(probe.Shape.ToArray());
        for (int i = 0; i < weight.Length; i++) weight[i] = 0.3 + 0.07 * (i % 13);

        double Loss(Tensor<double> input)
        {
            var y = Forward(input);
            double sum = 0;
            for (int i = 0; i < y.Length; i++) sum += y[i] * weight[i];
            return sum;
        }

        Tensor<double> analytic;
        using (var tape = new GradientTape<double>())
        {
            // The taped forward must be the same pool as the untaped one: it used to drop the padding.
            var taped = Forward(x);
            Assert.Equal(probe.Shape.ToArray(), taped.Shape.ToArray());
            for (int i = 0; i < probe.Length; i++) Assert.Equal(probe[i], taped[i]);
            var loss = engine.ReduceSum(engine.TensorMultiply(taped, weight), null);
            var grads = tape.ComputeGradients(loss, new[] { x });
            Assert.True(grads.TryGetValue(x, out var g) && g is not null, "MaxPool2D recorded no tape node.");
            analytic = g;
        }

        const double h = 1e-6;
        for (int i = 0; i < x.Length; i++)
        {
            double original = x[i];
            x[i] = original + h;
            double up = Loss(x);
            x[i] = original - h;
            double down = Loss(x);
            x[i] = original;
            double numeric = (up - down) / (2 * h);
            Assert.True(Math.Abs(numeric - analytic[i]) < 1e-6,
                $"cell {i}: analytic {analytic[i]:R} vs finite difference {numeric:R} (padding {padding})");
        }
    }
}