using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Verifies the new <c>ScaledDotProductAttentionDouble</c> fast path produces
/// numerically equivalent results to the previous scalar generic-T path.
/// <para>
/// The double SDPA path was the second-largest bottleneck for diffusion model
/// tests (after the TensorMatMul fallback). The previous implementation ran
/// two scalar virtual-dispatch triple-loops per <c>SDPA</c> call (Q·K^T then
/// weights·V), each ~75M FMAs at DiT-XL hot shapes. The new path mirrors the
/// existing float fast path: per-head SIMD-blocked DGEMM via
/// <c>MatrixMultiplyHelper.MultiplyBlocked</c>, separated by softmax.
/// </para>
/// <para>
/// Ground truth here is a hand-rolled scalar reference computed in the test —
/// not the previous engine path, so this stays valid even if the generic-T
/// fallback later changes.
/// </para>
/// </summary>
public class SdpaDoubleFastPathTests
{
    private readonly ITestOutputHelper _output;
    private readonly CpuEngine _engine = new();

    public SdpaDoubleFastPathTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static double[] ReferenceSdpa(
        double[] q, double[] k, double[] v,
        int batch, int heads, int seqQ, int seqK, int d_k, int d_v,
        double scale)
    {
        var output = new double[batch * heads * seqQ * d_v];
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < heads; h++)
            {
                var scores = new double[seqQ * seqK];
                int qOff = (b * heads + h) * seqQ * d_k;
                int kOff = (b * heads + h) * seqK * d_k;
                int vOff = (b * heads + h) * seqK * d_v;
                int oOff = (b * heads + h) * seqQ * d_v;

                // scores = Q @ K^T * scale
                for (int i = 0; i < seqQ; i++)
                {
                    for (int j = 0; j < seqK; j++)
                    {
                        double sum = 0;
                        for (int dd = 0; dd < d_k; dd++)
                            sum += q[qOff + i * d_k + dd] * k[kOff + j * d_k + dd];
                        scores[i * seqK + j] = sum * scale;
                    }
                }

                // softmax row-wise
                var weights = new double[seqQ * seqK];
                for (int i = 0; i < seqQ; i++)
                {
                    double maxVal = double.NegativeInfinity;
                    for (int j = 0; j < seqK; j++)
                        if (scores[i * seqK + j] > maxVal) maxVal = scores[i * seqK + j];
                    double sumExp = 0;
                    for (int j = 0; j < seqK; j++)
                    {
                        weights[i * seqK + j] = Math.Exp(scores[i * seqK + j] - maxVal);
                        sumExp += weights[i * seqK + j];
                    }
                    double inv = sumExp != 0 ? 1.0 / sumExp : 0;
                    for (int j = 0; j < seqK; j++)
                        weights[i * seqK + j] *= inv;
                }

                // output = weights @ V
                for (int i = 0; i < seqQ; i++)
                {
                    for (int j = 0; j < d_v; j++)
                    {
                        double sum = 0;
                        for (int kk = 0; kk < seqK; kk++)
                            sum += weights[i * seqK + kk] * v[vOff + kk * d_v + j];
                        output[oOff + i * d_v + j] = sum;
                    }
                }
            }
        }
        return output;
    }

    [Theory]
    [InlineData(1, 1, 4, 4, 8, 8)]      // tiny
    [InlineData(1, 2, 8, 8, 16, 16)]    // multi-head, small
    [InlineData(2, 4, 16, 16, 32, 32)]  // batched, mid
    [InlineData(1, 16, 64, 64, 128, 128)] // DiT-XL-ish per-head shape
    public void SdpaDouble_MatchesReference(int batch, int heads, int seqQ, int seqK, int d_k, int d_v)
    {
        var rng = new Random(42);
        var qData = new double[batch * heads * seqQ * d_k];
        var kData = new double[batch * heads * seqK * d_k];
        var vData = new double[batch * heads * seqK * d_v];
        for (int i = 0; i < qData.Length; i++) qData[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < kData.Length; i++) kData[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < vData.Length; i++) vData[i] = rng.NextDouble() * 2 - 1;

        var q = new Tensor<double>(qData, new[] { batch, heads, seqQ, d_k });
        var k = new Tensor<double>(kData, new[] { batch, heads, seqK, d_k });
        var v = new Tensor<double>(vData, new[] { batch, heads, seqK, d_v });

        double scale = 1.0 / Math.Sqrt(d_k);
        var actual = _engine.ScaledDotProductAttention(q, k, v, mask: null, scale: scale, out _);
        var actualData = actual.GetDataArray();
        var expected = ReferenceSdpa(qData, kData, vData, batch, heads, seqQ, seqK, d_k, d_v, scale);

        Assert.Equal(expected.Length, actualData.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            // Block-tiled summation reorders FMAs vs the naive sum, so we
            // tolerate small per-element drift proportional to d_k / seqK.
            Assert.Equal(expected[i], actualData[i], precision: 8);
        }
    }

    [Fact]
    public void SdpaDouble_DiTXL_BenchmarkCheck()
    {
        if (Environment.GetEnvironmentVariable("AIDN_RUN_GEMM_BENCH") != "1")
        {
            _output.WriteLine("skipped: set AIDN_RUN_GEMM_BENCH=1 to run.");
            return;
        }

        // Pika21 DiT default: hidden=2048, heads=16, headDim=128, seq=256, no batch
        const int batch = 1, heads = 16, seqQ = 256, seqK = 256, d_k = 128, d_v = 128;
        var rng = new Random(7);
        var qData = new double[batch * heads * seqQ * d_k];
        var kData = new double[batch * heads * seqK * d_k];
        var vData = new double[batch * heads * seqK * d_v];
        for (int i = 0; i < qData.Length; i++) qData[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < kData.Length; i++) kData[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < vData.Length; i++) vData[i] = rng.NextDouble() * 2 - 1;

        var q = new Tensor<double>(qData, new[] { batch, heads, seqQ, d_k });
        var k = new Tensor<double>(kData, new[] { batch, heads, seqK, d_k });
        var v = new Tensor<double>(vData, new[] { batch, heads, seqK, d_v });

        // Warm-up
        _ = _engine.ScaledDotProductAttention(q, k, v, mask: null, scale: 1.0 / Math.Sqrt(d_k), out _);

        var sw = Stopwatch.StartNew();
        const int iterations = 5;
        for (int i = 0; i < iterations; i++)
        {
            _ = _engine.ScaledDotProductAttention(q, k, v, mask: null, scale: 1.0 / Math.Sqrt(d_k), out _);
        }
        sw.Stop();
        double perCallMs = sw.Elapsed.TotalMilliseconds / iterations;
        // Two GEMMs per head: Q·K^T (seqQ × seqK × d_k) + W·V (seqQ × d_v × seqK)
        long fmasPerHead = (long)seqQ * seqK * d_k + (long)seqQ * d_v * seqK;
        long totalFmas = (long)batch * heads * fmasPerHead;
        double gflops = totalFmas * 2.0 / (perCallMs / 1000.0) / 1e9;
        _output.WriteLine($"SDPA double Pika21-shape per-call={perCallMs:F2} ms  ~{gflops:F1} GFLOPS");
    }
}
