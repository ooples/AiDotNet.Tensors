using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// AvgPool2D with padding, on the CPU reference engine.
/// </summary>
/// <remarks>
/// The backward used to be recorded with pool size and stride only, so for padding &gt; 0 it walked unpadded
/// windows: gradient landed on the wrong cells and was divided by the full window even though the forward divided
/// by the covered count. The forward and the backward also had to agree on the divisor rule once it became a
/// parameter (<c>countIncludePad</c>). Average pooling is linear, so a finite difference in double is exact up to
/// rounding and makes a tight oracle for the gradient.
/// </remarks>
public class AvgPool2DPaddingGradientTests : IDisposable
{
    // These tests set the process-wide engine; restore it so later tests see the engine they expect.
    private readonly IEngine _previousEngine = AiDotNetEngine.Current;

    public void Dispose() => AiDotNetEngine.Current = _previousEngine;

    [Fact]
    public void Forward_divides_by_covered_cells_by_default_and_by_the_full_window_when_padding_counts()
    {
        var engine = new CpuEngine();
        var ones = new Tensor<double>(new[] { 1, 1, 2, 2 });
        for (int i = 0; i < ones.Length; i++) ones[i] = 1.0;

        // pool 2, stride 1, padding 1 over a 2x2 input -> 3x3 output. A corner window covers 1 real cell,
        // an edge window 2, the centre window all 4.
        var excluded = engine.AvgPool2D(ones, poolSize: 2, stride: 1, padding: 1);
        var included = engine.AvgPool2D(ones, poolSize: 2, stride: 1, padding: 1, countIncludePad: true);

        Assert.Equal(new[] { 1, 1, 3, 3 }, excluded.Shape.ToArray());
        for (int i = 0; i < excluded.Length; i++) Assert.Equal(1.0, excluded[i], 12);
        var expectedIncluded = new[] { 0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25 };
        for (int i = 0; i < included.Length; i++) Assert.Equal(expectedIncluded[i], included[i], 12);
    }

    [Theory]
    [InlineData(false, 1)]
    [InlineData(true, 1)]
    [InlineData(false, 2)]
    [InlineData(true, 2)]
    public void Padded_gradient_matches_finite_differences(bool countIncludePad, int padding)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var x = new Tensor<double>(new[] { 2, 3, 5, 6 });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.37 * i) + 0.1 * (i % 7);

        Tensor<double> Forward(Tensor<double> input) =>
            engine.AvgPool2D(input, poolSize: 3, stride: 2, padding: padding, countIncludePad: countIncludePad);

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
            var loss = engine.ReduceSum(engine.TensorMultiply(Forward(x), weight), null);
            var grads = tape.ComputeGradients(loss, new[] { x });
            Assert.True(grads.TryGetValue(x, out var g) && g is not null, "AvgPool2D recorded no tape node.");
            analytic = g;
        }

        const double h = 1e-3;
        for (int i = 0; i < x.Length; i++)
        {
            double original = x[i];
            x[i] = original + h;
            double up = Loss(x);
            x[i] = original - h;
            double down = Loss(x);
            x[i] = original;
            double numeric = (up - down) / (2 * h);
            Assert.True(Math.Abs(numeric - analytic[i]) < 1e-8,
                $"cell {i}: analytic {analytic[i]:R} vs finite difference {numeric:R} "
                + $"(padding {padding}, countIncludePad {countIncludePad})");
        }
    }
}