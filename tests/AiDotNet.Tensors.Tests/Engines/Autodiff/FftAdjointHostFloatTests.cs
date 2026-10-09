using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// The float32 host RFFT / IRFFT adjoints (parallel rows over the native FFT) against the double generic loop, on a
/// batch of rows with a weighted upstream gradient and a second consumer of the input so the gradient accumulates.
/// </summary>
public class FftAdjointHostFloatTests
{
    private static readonly CpuEngine Engine = new();

    private static (float[] f, double[] d) Grad(int rows, int cols, int seed, Func<Tensor<float>, Tensor<float>> ff, Func<Tensor<double>, Tensor<double>> fd, int outCols)
    {
        var rng = new Random(seed);
        var xa = new double[rows * cols]; var wa = new double[rows * outCols]; var ha = new double[rows * cols];
        for (int i = 0; i < xa.Length; i++) { xa[i] = rng.NextDouble() * 2 - 1; ha[i] = rng.NextDouble() * 2 - 1; }
        for (int i = 0; i < wa.Length; i++) wa[i] = rng.NextDouble() * 2 - 1;

        float[] F(double[] a) => Array.ConvertAll(a, v => (float)v);
        var xf = new Tensor<float>(F(xa), new[] { rows, cols });
        float[] gf;
        using (var tape = new GradientTape<float>())
        {
            var loss = Engine.TensorAdd(
                Engine.ReduceSum(Engine.TensorMultiply(ff(xf), new Tensor<float>(F(wa), new[] { rows, outCols })), null),
                Engine.ReduceSum(Engine.TensorMultiply(xf, new Tensor<float>(F(ha), new[] { rows, cols })), null));
            gf = tape.ComputeGradients(loss, new[] { xf })[xf].ToArray();
        }
        var xd = new Tensor<double>((double[])xa.Clone(), new[] { rows, cols });
        double[] gd;
        using (var tape = new GradientTape<double>())
        {
            var loss = Engine.TensorAdd(
                Engine.ReduceSum(Engine.TensorMultiply(fd(xd), new Tensor<double>(wa, new[] { rows, outCols })), null),
                Engine.ReduceSum(Engine.TensorMultiply(xd, new Tensor<double>(ha, new[] { rows, cols })), null));
            gd = tape.ComputeGradients(loss, new[] { xd })[xd].ToArray();
        }
        return (gf, gd);
    }

    private static void AssertClose(float[] f, double[] d)
    {
        Assert.Equal(d.Length, f.Length);
        double maxErr = 0, maxRef = 0;
        for (int i = 0; i < d.Length; i++) { maxErr = Math.Max(maxErr, Math.Abs(f[i] - d[i])); maxRef = Math.Max(maxRef, Math.Abs(d[i])); }
        Assert.True(maxErr <= 1e-4 * Math.Max(1, maxRef), $"max |error| {maxErr:G4} (max |ref| {maxRef:G4})");
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(37, 64)]
    [InlineData(16, 1024)]
    public void RfftGradient_MatchesDoubleReference(int rows, int n)
    {
        int outCols = (n / 2 + 1) * 2;
        var (f, d) = Grad(rows, n, rows * 13 + n, Engine.RFFT, Engine.RFFT, outCols);
        AssertClose(f, d);
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(37, 64)]
    [InlineData(16, 1024)]
    public void IrfftGradient_MatchesDoubleReference(int rows, int n)
    {
        int cols = (n / 2 + 1) * 2;
        var (f, d) = Grad(rows, cols, rows * 17 + n, x => Engine.IRFFT(x, n), x => Engine.IRFFT(x, n), n);
        AssertClose(f, d);
    }
}
