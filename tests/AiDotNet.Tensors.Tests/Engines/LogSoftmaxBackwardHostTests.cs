using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The fused host log-softmax backward (dx = g - exp(y) * rowsum(g) in one pass per row) against a double-precision
/// transcription, with the input consumed a second time so the gradient lands in an existing accumulator.
/// </summary>
public class LogSoftmaxBackwardHostTests
{
    [Theory]
    [InlineData(5, 37)]
    [InlineData(64, 1024)]
    public void TapeGradient_MatchesReference_WithASecondConsumer(int rows, int cols)
    {
        var rng = new Random(rows * 7 + cols);
        float[] Rand(int n, double scale) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale); return a; }
        var xa = Rand(rows * cols, 3); var ga = Rand(rows * cols, 1); var ha = Rand(rows * cols, 1);
        var x = new Tensor<float>(xa, new[] { rows, cols });
        var g = new Tensor<float>(ga, new[] { rows, cols });
        var h = new Tensor<float>(ha, new[] { rows, cols });
        var engine = new CpuEngine();

        using var tape = new GradientTape<float>();
        var y = engine.TensorLogSoftmax(x, axis: 1);
        var loss = engine.TensorAdd(engine.ReduceSum(engine.TensorMultiply(y, g), null),
                                    engine.ReduceSum(engine.TensorMultiply(x, h), null));
        var dx = tape.ComputeGradients(loss, new[] { x })[x].ToArray();

        double maxErr = 0;
        for (int r = 0; r < rows; r++)
        {
            double max = double.NegativeInfinity, sum = 0, gsum = 0;
            for (int c = 0; c < cols; c++) max = Math.Max(max, xa[r * cols + c]);
            for (int c = 0; c < cols; c++) { sum += Math.Exp(xa[r * cols + c] - max); gsum += ga[r * cols + c]; }
            for (int c = 0; c < cols; c++)
            {
                double soft = Math.Exp(xa[r * cols + c] - max) / sum;
                double expected = ga[r * cols + c] - soft * gsum + ha[r * cols + c];
                maxErr = Math.Max(maxErr, Math.Abs(expected - dx[r * cols + c]));
            }
        }
        Assert.True(maxErr < 1e-4, $"max |error| {maxErr:G4}");
    }
}
