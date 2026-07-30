using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Correctness for the block-Q8_0 GEMM (llama.cpp / ggml_vec_dot_q8_0_q8_0 layout).
/// Verifies the quantized matmul matches an fp32 reference within Q8_0 round-off, on
/// real decoder shapes.
/// </summary>
public sealed class Q8BlockGemmTests
{
    private static float[] Fp32Ref(float[] act, float[] wNK, int m, int k, int n)
    {
        // C[m,n] = Σ_k act[m,k] * W[n,k]  (W stored [N,K], i.e. out·inᵀ)
        var outp = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int kk = 0; kk < k; kk++) acc += (double)act[i * k + kk] * wNK[j * k + kk];
                outp[i * n + j] = (float)acc;
            }
        return outp;
    }

    [Theory]
    [InlineData(1, 576, 1536)]   // decode ffn
    [InlineData(128, 576, 1536)] // prefill ffn
    [InlineData(4, 64, 96)]      // small
    [InlineData(1, 576, 576)]    // decode attn proj
    public void Q8BlockGemm_MatchesFp32Reference_WithinQ8Roundoff(int m, int k, int n)
    {
        var rng = new Random(7);
        var act = new float[m * k];
        for (int i = 0; i < act.Length; i++) act[i] = (float)(rng.NextDouble() * 2 - 1);
        var wNK = new float[n * k];
        for (int i = 0; i < wNK.Length; i++) wNK[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

        // Quantize the weight to Q8_0 blocks (what GGUF already stores it as).
        var wq = new sbyte[n * k];
        var wsc = new float[n * (k / Q8BlockGemm.QK)];
        Q8BlockGemm.QuantizeRows(wNK, n, k, wq, wsc);

        var outp = new float[m * n];
        Q8BlockGemm.MatMul(act, wq, wsc, outp, m, k, n);

        var reference = Fp32Ref(act, wNK, m, k, n);

        // Q8_0 error: both weight (~0.4% relative) and activation quantization. Per-element
        // absolute error scales with sqrt(K)·|values|; a generous but meaningful bound.
        double refMax = 0;
        for (int i = 0; i < reference.Length; i++) refMax = Math.Max(refMax, Math.Abs(reference[i]));
        double tol = 0.03 * refMax + 1e-3;
        for (int i = 0; i < outp.Length; i++)
            Assert.True(Math.Abs(outp[i] - reference[i]) <= tol,
                $"[{i}] q8={outp[i]} ref={reference[i]} diff={Math.Abs(outp[i] - reference[i])} tol={tol}");
    }

    [Fact]
    public void QuantizeRows_MatchesGgufQ8_0_Formula()
    {
        // A single block of 32; scale = maxAbs/127, round-away-from-zero.
        var src = new float[32];
        var rng = new Random(3);
        for (int i = 0; i < 32; i++) src[i] = (float)(rng.NextDouble() * 4 - 2);
        var qs = new sbyte[32];
        var sc = new float[1];
        Q8BlockGemm.QuantizeRows(src, 1, 32, qs, sc);

        float maxAbs = 0;
        for (int i = 0; i < 32; i++) maxAbs = Math.Max(maxAbs, Math.Abs(src[i]));
        float expScale = maxAbs / 127f;
        Assert.Equal(expScale, sc[0], 5);
        for (int i = 0; i < 32; i++)
        {
            int expQ = (int)Math.Round(src[i] / expScale, MidpointRounding.AwayFromZero);
            expQ = Math.Max(-127, Math.Min(127, expQ));
            Assert.Equal(expQ, qs[i]);
        }
        // The max-abs element must map to ±127.
        Assert.Equal(127, Math.Abs((int)qs[Array.FindIndex(src, x => Math.Abs(x) == maxAbs)]));
    }
}
