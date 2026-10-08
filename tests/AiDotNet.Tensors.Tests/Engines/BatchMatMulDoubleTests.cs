using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>Rank-3 float64 BatchMatMul (per-slice sequential DGEMM) against a scalar reference, incl. a permuted operand.</summary>
public class BatchMatMulDoubleTests
{
    [Theory]
    [InlineData(5, 24, 32, 24, false)]
    [InlineData(3, 7, 5, 9, false)]
    [InlineData(4, 24, 32, 24, true)]
    public void Rank3_MatchesReference(int batch, int m, int k, int n, bool transposedB)
    {
        var rng = new Random(batch * 100 + m + k + n);
        var a = new Tensor<double>(new[] { batch, m, k });
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        Tensor<double> b;
        Tensor<double> bSource = new Tensor<double>(transposedB ? new[] { batch, n, k } : new[] { batch, k, n });
        for (int i = 0; i < bSource.Length; i++) bSource[i] = rng.NextDouble() * 2 - 1;
        var engine = new CpuEngine();
        b = transposedB ? engine.TensorPermute(bSource, new[] { 0, 2, 1 }) : bSource;   // [batch, k, n] either way

        var c = engine.BatchMatMul(a, b);
        for (int s = 0; s < batch; s++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double e = 0;
                    for (int p = 0; p < k; p++) e += a[s, i, p] * b[s, p, j];
                    Assert.True(Math.Abs(e - c[s, i, j]) < 1e-12, $"[{s},{i},{j}] {c[s, i, j]} vs {e}");
                }
    }
}
