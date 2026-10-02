using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// CPU normalization over values whose mean is large next to their spread.
/// </summary>
/// <remarks>
/// <para>
/// The fused single-pass LayerNorm took the variance as E[x^2] - E[x]^2 over the raw values. With a large
/// mean both terms round to the same float and the difference loses every significant digit: a float
/// LayerNorm over 1000 + U(0,1) was off by up to 167, and 60 + U(0,1) by 7e-3. The rows are now
/// accumulated relative to their first element. BatchNorm over [batch, features] summed raw values for
/// the mean and drifted ~6x past the precision of its inputs at 1e5 + U(0,1).
/// </para>
/// <para>
/// The bound is what the inputs themselves allow: a value near <c>offset</c> is only known to within half
/// a float spacing there, which after dividing by the standard deviation is the best any normalization
/// can do. Widths cover the scalar tail (7), the register-resident fs == 64 path and the generic two-pass
/// form (256), plus odd lengths that exercise every remainder loop.
/// </para>
/// </remarks>
public class CpuNormalizationLargeOffsetTests
{
    private readonly CpuEngine _engine = new();

    private static float[] Values(int count, double offset, int seed)
    {
        var rng = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(offset + rng.NextDouble());
        return values;
    }

    /// <summary>A few float spacings at <paramref name="offset"/>, divided by the U(0,1) standard deviation.</summary>
    private static double InputPrecisionBound(double offset)
    {
        double spacing = Math.Max(1e-7, Math.Abs(offset) * Math.Pow(2, -23));
        return 4 * spacing / Math.Sqrt(1.0 / 12.0) + 1e-5;
    }

    [Theory]
    [InlineData(0.0, 7)]
    [InlineData(0.0, 64)]
    [InlineData(0.0, 256)]
    [InlineData(1000.0, 7)]
    [InlineData(1000.0, 64)]
    [InlineData(1000.0, 77)]
    [InlineData(1000.0, 256)]
    [InlineData(100000.0, 64)]
    [InlineData(100000.0, 259)]
    public void FloatLayerNorm_StaysWithinTheInputPrecision(double offset, int width)
    {
        const int rows = 32;
        var data = Values(rows * width, offset, width);
        var input = new Tensor<float>(data, new[] { rows, width });
        var gamma = new Tensor<float>(new[] { width }); gamma.Fill(1f);
        var beta = new Tensor<float>(new[] { width });

        var output = _engine.LayerNorm(input, gamma, beta, 1e-5, out _, out _);

        double worst = 0;
        for (int r = 0; r < rows; r++)
        {
            double mean = 0, variance = 0;
            for (int j = 0; j < width; j++) mean += data[r * width + j];
            mean /= width;
            for (int j = 0; j < width; j++) variance += Math.Pow(data[r * width + j] - mean, 2);
            variance /= width;
            for (int j = 0; j < width; j++)
            {
                double expected = (data[r * width + j] - mean) / Math.Sqrt(variance + 1e-5);
                worst = Math.Max(worst, Math.Abs(output[r * width + j] - expected));
            }
        }

        Assert.True(worst <= InputPrecisionBound(offset),
            $"LayerNorm over {offset} + U(0,1), width {width}: worst error {worst:G4} exceeds {InputPrecisionBound(offset):G4}.");
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1000.0)]
    [InlineData(100000.0)]
    public void FloatBatchNorm2D_StaysWithinTheInputPrecision(double offset)
    {
        const int batch = 128, features = 16;
        var data = Values(batch * features, offset, 3);
        var input = new Tensor<float>(data, new[] { batch, features });
        var gamma = new Tensor<float>(new[] { features }); gamma.Fill(1f);
        var beta = new Tensor<float>(new[] { features });

        var output = _engine.BatchNorm(input, gamma, beta, 1e-5, out _, out _);

        double worst = 0;
        for (int f = 0; f < features; f++)
        {
            double mean = 0, variance = 0;
            for (int b = 0; b < batch; b++) mean += data[b * features + f];
            mean /= batch;
            for (int b = 0; b < batch; b++) variance += Math.Pow(data[b * features + f] - mean, 2);
            variance /= batch;
            for (int b = 0; b < batch; b++)
            {
                double expected = (data[b * features + f] - mean) / Math.Sqrt(variance + 1e-5);
                worst = Math.Max(worst, Math.Abs(output[b * features + f] - expected));
            }
        }

        Assert.True(worst <= InputPrecisionBound(offset),
            $"BatchNorm over {offset} + U(0,1): worst error {worst:G4} exceeds {InputPrecisionBound(offset):G4}.");
    }

    [Theory]
    [InlineData(7)]
    [InlineData(64)]
    [InlineData(256)]
    public void DoubleLayerNorm_KeepsDoublePrecisionAtALargeOffset(int width)
    {
        const int rows = 8;
        const double offset = 1e8;
        var rng = new Random(width);
        var data = new double[rows * width];
        for (int i = 0; i < data.Length; i++) data[i] = offset + rng.NextDouble();
        var input = new Tensor<double>(data, new[] { rows, width });
        var gamma = new Tensor<double>(new[] { width }); gamma.Fill(1.0);
        var beta = new Tensor<double>(new[] { width });

        var output = _engine.LayerNorm(input, gamma, beta, 1e-5, out _, out _);

        double worst = 0;
        for (int r = 0; r < rows; r++)
        {
            // Reference in shifted form so the reference itself does not cancel.
            double shift = data[r * width];
            double mean = 0;
            for (int j = 0; j < width; j++) mean += data[r * width + j] - shift;
            mean /= width;
            double variance = 0;
            for (int j = 0; j < width; j++) variance += Math.Pow(data[r * width + j] - shift - mean, 2);
            variance /= width;
            for (int j = 0; j < width; j++)
            {
                double expected = (data[r * width + j] - shift - mean) / Math.Sqrt(variance + 1e-5);
                worst = Math.Max(worst, Math.Abs(output[r * width + j] - expected));
            }
        }

        // Double spacing at 1e8 is ~1.5e-8; a few spacings over the U(0,1) standard deviation.
        Assert.True(worst < 1e-6, $"double LayerNorm over 1e8 + U(0,1), width {width}: worst error {worst:G4}.");
    }
}
