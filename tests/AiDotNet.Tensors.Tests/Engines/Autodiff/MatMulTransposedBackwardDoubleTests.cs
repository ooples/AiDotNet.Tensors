using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tape gradient of C = A B^T in float64 (the x W^T layer form), against the closed form dA = dC B, dB = dC^T A,
/// with a weighted upstream gradient and shapes large enough for the BLAS route.
/// </summary>
public class MatMulTransposedBackwardDoubleTests
{
    [Theory]
    [InlineData(3, 4, 5)]
    [InlineData(64, 32, 96)]
    public void TapeGradient_MatchesClosedForm(int m, int k, int n)
    {
        var rng = new Random(m + k + n);
        double[] R(int len) { var a = new double[len]; for (int i = 0; i < len; i++) a[i] = rng.NextDouble() * 2 - 1; return a; }
        var aD = R(m * k); var bD = R(n * k); var wD = R(m * n);
        var a = new Tensor<double>(aD, new[] { m, k });
        var b = new Tensor<double>(bD, new[] { n, k });
        var engine = new CpuEngine();
        double[] gA, gB;
        using (var tape = new GradientTape<double>())
        {
            var c = engine.TensorMatMulTransposed(a, b);
            var loss = engine.ReduceSum(engine.TensorMultiply(c, new Tensor<double>(wD, new[] { m, n })), null);
            var g = tape.ComputeGradients(loss, new[] { a, b });
            gA = g[a].ToArray(); gB = g[b].ToArray();
        }
        double err = 0;
        for (int i = 0; i < m; i++) for (int p = 0; p < k; p++)
        {
            double s = 0; for (int j = 0; j < n; j++) s += wD[i * n + j] * bD[j * k + p];
            err = Math.Max(err, Math.Abs(s - gA[i * k + p]));
        }
        for (int j = 0; j < n; j++) for (int p = 0; p < k; p++)
        {
            double s = 0; for (int i = 0; i < m; i++) s += wD[i * n + j] * aD[i * k + p];
            err = Math.Max(err, Math.Abs(s - gB[j * k + p]));
        }
        Assert.True(err < 1e-10, $"max |error| {err:E3}");
    }
}
