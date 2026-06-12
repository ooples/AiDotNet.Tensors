using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Issue #471 — CPU <see cref="CpuEngine.TensorMatMul{T}(Tensor{T}, Tensor{T})"/> hard-crashes
/// the process (exit 255, no managed exception) on a large-K double matmul
/// <c>[d, V] × [V, d]</c> with V≈50257 (dense-vocab cross-covariance), via the managed
/// AVX2 <c>SimdGemm.Dgemm</c> fallback (OpenBLAS not loaded).
/// </summary>
public class Issue471CpuMatMulLargeKTests
{
    private readonly ITestOutputHelper _output;
    public Issue471CpuMatMulLargeKTests(ITestOutputHelper output) { _output = output; }

    // Env-gated raw repro: run with AIDOTNET_RUN_REPRO=1 to observe the crash / capture a dump.
    [Fact]
    public void Issue471_Repro_LargeK_DoubleMatMul()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_REPRO") != "1") return;
        var engine = new CpuEngine();
        const int d = 128;
        foreach (int V in new[] { 1000, 10000, 50257 })
        {
            Console.Error.WriteLine($"[REPRO] V={V} start");
            var a = FillSmall(new[] { d, V }, 1);
            var b = FillSmall(new[] { V, d }, 2);
            var c = engine.TensorMatMul(a, b);
            Console.Error.WriteLine($"[REPRO] V={V} done, c[0,0]={c.AsSpan()[0]:E3}");
        }
    }

    // The regression guard (runs in CI): the large-V matmul must complete and be correct.
    [Theory]
    [InlineData(1000)]
    [InlineData(10000)]
    [InlineData(50257)]   // the dense-vocab shape that crashed in #471
    public void CpuMatMul_LargeK_CompletesAndIsCorrect(int v)
    {
        var engine = new CpuEngine();
        const int d = 128;  // the #471 shape (M=N=128 hits the packed-tiled GEMM path)
        var a = FillSmall(new[] { d, v }, 1);
        var b = FillSmall(new[] { v, d }, 2);

        var c = engine.TensorMatMul(a, b);

        Assert.Equal(new[] { d, d }, c._shape);
        var cs = c.AsSpan();
        // Spot-check a few entries against a direct dot product over the V axis.
        var aS = a.AsSpan();
        var bS = b.AsSpan();
        foreach (var (i, j) in new[] { (0, 0), (1, 3), (d - 1, d - 1), (7, 2) })
        {
            double expected = 0;
            for (int p = 0; p < v; p++) expected += aS[i * v + p] * bS[p * d + j];
            Assert.True(Math.Abs(cs[i * d + j] - expected) <= 1e-6 * (1 + Math.Abs(expected)),
                $"c[{i},{j}]={cs[i * d + j]:E6} vs expected {expected:E6} (V={v})");
            Assert.True(!double.IsNaN(cs[i * d + j]) && !double.IsInfinity(cs[i * d + j]));
        }
    }

    // The cortex readout matmul: [d,d] × [d,V] → [d,V] (large N=V columns), a
    // DIFFERENT GEMM dispatch than the large-K orientation above. This is the
    // shape the HybridDepthStack actually runs (W = (gram+λI)⁻¹·Xᵀg, [d,d]×[d,V]).
    [Theory]
    [InlineData(1000)]
    [InlineData(10000)]
    [InlineData(50257)]
    public void CpuMatMul_LargeN_CompletesAndIsCorrect(int v)
    {
        var engine = new CpuEngine();
        const int d = 128;
        var a = FillSmall(new[] { d, d }, 1);   // [d, d]
        var b = FillSmall(new[] { d, v }, 2);   // [d, V]

        var c = engine.TensorMatMul(a, b);      // [d, V]

        Assert.Equal(new[] { d, v }, c._shape);
        var cs = c.AsSpan();
        var aS = a.AsSpan();
        var bS = b.AsSpan();
        foreach (var (i, j) in new[] { (0, 0), (1, 3), (d - 1, v - 1), (7, v / 2) })
        {
            double expected = 0;
            for (int p = 0; p < d; p++) expected += aS[i * d + p] * bS[p * v + j];
            Assert.True(Math.Abs(cs[i * v + j] - expected) <= 1e-6 * (1 + Math.Abs(expected)),
                $"c[{i},{j}]={cs[i * v + j]:E6} vs expected {expected:E6} (V={v})");
        }
    }

    // The EXACT cortex cross-covariance op (HybridDepthStackTokenLMPredictor line 399):
    //   cct = TensorMatMul(xtg, TensorTranspose(xtg))   with xtg = [d, V]
    // The second operand is a TRANSPOSE VIEW (non-contiguous), which routes through a
    // different matmul path than two contiguous operands.
    [Theory]
    [InlineData(1000)]
    [InlineData(10000)]
    [InlineData(50257)]
    public void CpuMatMul_TransposedViewOperand_LargeK_DoesNotCrash(int v)
    {
        var engine = new CpuEngine();
        const int d = 128;
        var xtg = FillSmall(new[] { d, v }, 1);              // [d, V] contiguous
        var xtgT = engine.TensorTranspose(xtg);             // [V, d] transpose view

        var cct = engine.TensorMatMul(xtg, xtgT);           // [d, d]

        Assert.Equal(new[] { d, d }, cct._shape);
        var cs = cct.AsSpan();
        var xs = xtg.AsSpan();
        // cct[i,j] = sum_p xtg[i,p] * xtg[j,p]  (since (xtgᵀ)[p,j] = xtg[j,p])
        foreach (var (i, j) in new[] { (0, 0), (1, 5), (d - 1, d - 1), (9, 2) })
        {
            double expected = 0;
            for (int p = 0; p < v; p++) expected += xs[i * v + p] * xs[j * v + p];
            Assert.True(Math.Abs(cs[i * d + j] - expected) <= 1e-6 * (1 + Math.Abs(expected)),
                $"cct[{i},{j}]={cs[i * d + j]:E6} vs expected {expected:E6} (V={v})");
        }
    }

    private static Tensor<double> FillSmall(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var s = t.AsWritableSpan();
        var rng = new Random(seed);
        for (int i = 0; i < s.Length; i++) s[i] = (rng.NextDouble() - 0.5) * 0.01;
        return t;
    }
}
