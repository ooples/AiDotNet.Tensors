using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the fused fp16-weight GEMM used by fp16-resident inference
/// (<see cref="CpuEngine.TensorMatMulFp16WeightB"/> / <see cref="CpuEngine.FusedLinearFp16WeightB"/>,
/// backed by <c>SimdGemm.SgemmFp16WeightB</c>). The convert-in-pack path must match the reference
/// "upcast the whole weight to fp32, then matmul" to within fp32 GEMM association differences — both
/// paths decode the SAME <see cref="Half"/> bit patterns, so only summation order differs.
/// Shapes exercise the parallel N path, trailing (&lt; Nr) columns, and the small sequential path.
/// </summary>
public class Fp16WeightGemmTests
{
    private static Tensor<float> RandomFloat(int rows, int cols, Random rng)
    {
        var t = new Tensor<float>(new[] { rows, cols });
        var d = t.GetCpuData();
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    // Half weight [k,n] plus its exact fp32 upcast (same bit patterns), so the reference matmul
    // and the fused path start from identical fp32 values — any diff is pure GEMM association.
    private static (Tensor<Half> half, Tensor<float> upcast) RandomHalfWeight(int k, int n, Random rng)
    {
        var half = new Tensor<Half>(new[] { k, n });
        var upcast = new Tensor<float>(new[] { k, n });
        var hd = half.GetCpuData();
        var ud = upcast.GetCpuData();
        for (int i = 0; i < hd.Length; i++)
        {
            var h = (Half)(float)(rng.NextDouble() * 2.0 - 1.0);
            hd[i] = h;
            ud[i] = (float)h;
        }
        return (half, upcast);
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> got)
    {
        var e = expected.GetCpuData();
        var g = got.GetCpuData();
        Assert.Equal(e.Length, g.Length);
        for (int i = 0; i < e.Length; i++)
        {
            double tol = 1e-3 + 1e-4 * Math.Abs(e[i]);
            Assert.True(Math.Abs(e[i] - g[i]) < tol,
                $"element {i}: fused={g[i]} vs upcast-reference={e[i]} (tol {tol})");
        }
    }

    [Theory]
    [InlineData(64, 128, 100)]  // parallel N path (n > Nr) + trailing columns (100 = 6*16 + 4)
    [InlineData(48, 96, 96)]    // parallel N path, n a multiple of Nr (no trailing)
    [InlineData(3, 5, 7)]       // small: sequential path, trailing-only columns
    [InlineData(1, 320, 1280)]  // single-row (m == 1), wide weight — MLP-like inference shape
    public void TensorMatMulFp16WeightB_MatchesUpcastThenMatMul(int m, int k, int n)
    {
        var engine = new CpuEngine();
        var rng = new Random(1234 + m * 31 + k * 7 + n);
        var a = RandomFloat(m, k, rng);
        var (wHalf, wUpcast) = RandomHalfWeight(k, n, rng);

        var reference = engine.TensorMatMul(a, wUpcast);
        var fused = engine.TensorMatMulFp16WeightB(a, wHalf);

        Assert.Equal(new[] { m, n }, fused.Shape.ToArray());
        AssertClose(reference, fused);
    }

    [Fact]
    public void FusedLinearFp16WeightB_WithBias_NoActivation_MatchesUpcastLinear()
    {
        var engine = new CpuEngine();
        var rng = new Random(99);
        int m = 32, k = 64, n = 48;
        var a = RandomFloat(m, k, rng);
        var (wHalf, wUpcast) = RandomHalfWeight(k, n, rng);
        var bias = new Tensor<float>(new[] { n });
        var bd = bias.GetCpuData();
        for (int j = 0; j < n; j++) bd[j] = (float)(rng.NextDouble() - 0.5);

        // Reference: upcast matmul + broadcast bias.
        var reference = engine.TensorMatMul(a, wUpcast);
        var rd = reference.GetCpuData();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) rd[i * n + j] += bd[j];

        var fused = engine.FusedLinearFp16WeightB(a, wHalf, bias, FusedActivationType.None);

        Assert.Equal(new[] { m, n }, fused.Shape.ToArray());
        AssertClose(reference, fused);
    }

    [Fact]
    public void FusedLinearFp16WeightB_RankOneInput_ReturnsRankOneVector()
    {
        var engine = new CpuEngine();
        var rng = new Random(7);
        int k = 40, n = 24;
        var a = RandomFloat(1, k, rng);
        var a1d = a.Reshape(new[] { k });   // rank-1 input path
        var (wHalf, wUpcast) = RandomHalfWeight(k, n, rng);

        var reference2d = engine.TensorMatMul(a, wUpcast);   // [1, n]
        var fused1d = engine.FusedLinearFp16WeightB(a1d, wHalf, null, FusedActivationType.None);

        Assert.Equal(new[] { n }, fused1d.Shape.ToArray());
        var e = reference2d.GetCpuData();
        var g = fused1d.GetCpuData();
        for (int j = 0; j < n; j++)
        {
            double tol = 1e-3 + 1e-4 * Math.Abs(e[j]);
            Assert.True(Math.Abs(e[j] - g[j]) < tol, $"element {j}: {g[j]} vs {e[j]}");
        }
    }
}
