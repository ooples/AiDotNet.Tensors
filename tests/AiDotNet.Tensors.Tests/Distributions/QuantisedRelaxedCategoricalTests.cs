// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Distributions;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Distributions;

/// <summary>
/// Acceptance tests for #263 sub-byte relaxed-categorical
/// distributions. Verifies round-trip fidelity, log_prob parity vs
/// the FP32 baseline, and sampler sanity.
/// </summary>
public class QuantisedRelaxedCategoricalTests
{
    private static (float[] logits, float[] temp) MakeBatch(int batch, int k, int seed)
    {
        var rng = new Random(seed);
        var logits = new float[batch * k];
        for (int i = 0; i < logits.Length; i++) logits[i] = (float)(rng.NextDouble() * 4 - 2);
        var temp = new float[batch];
        for (int b = 0; b < batch; b++) temp[b] = 0.5f + (float)rng.NextDouble();
        return (logits, temp);
    }

    [Fact]
    public void Int4_SampleProducesPackedValuesAndScale()
    {
        var (logits, temp) = MakeBatch(batch: 4, k: 8, seed: 1);
        var d = new RelaxedOneHotCategoricalInt4Distribution(logits, temp, 8);
        var rng = new Random(42);
        var quant = d.SampleInt4(rng);
        // 4 batches × 8 categories = 32 elements → 16 packed bytes.
        Assert.Equal(16, quant.PackedValues.Length);
        Assert.Equal(32, quant.TotalElements);
        Assert.Equal(4, quant.BatchSize);
        Assert.Equal(8, quant.K);
        Assert.NotNull(quant.Scale);
        Assert.True(quant.Scale.Scales.Length >= 1);
    }

    [Fact]
    public void Int4_DequantizeRoundTripBoundedByScale()
    {
        var (logits, temp) = MakeBatch(batch: 4, k: 8, seed: 2);
        var d = new RelaxedOneHotCategoricalInt4Distribution(logits, temp, 8, groupSize: 8);
        var rng = new Random(42);
        var quant = d.SampleInt4(rng);
        var dense = d.Dequantize(quant);
        // Each cell should be in [0, 1] (simplex membership) within
        // quantisation tolerance — every scale step is at most
        // scale.Scales[g] / 2 from the true value, and softmax
        // outputs are bounded in [0, 1].
        for (int i = 0; i < dense.Length; i++)
        {
            int g = i / quant.Scale.GroupSize;
            float tol = quant.Scale.Scales[g] / 2 + 1e-4f;
            Assert.InRange(dense[i], -tol, 1f + tol);
        }
        // Each row should approximately sum to 1 (relaxed simplex
        // constraint after quantisation; the int4 rounding can drift
        // from exact 1 but stays within ~0.2 for K=8 at group=8).
        for (int b = 0; b < 4; b++)
        {
            float sum = 0;
            for (int k = 0; k < 8; k++) sum += dense[b * 8 + k];
            Assert.InRange(sum, 0.5f, 1.5f);
        }
    }

    [Fact]
    public void Int4_LogProbDegradationUnderHalfNatPerDim()
    {
        // Generate a sample at FP32, compute its log-prob through the
        // FP32 distribution, then through the int4-quantised one. The
        // #263 acceptance threshold is 0.5 nats/dim — we measure the
        // delta on a small batch and assert.
        var (logits, temp) = MakeBatch(batch: 8, k: 16, seed: 3);
        var fp32 = new RelaxedOneHotCategoricalDistribution(logits, temp, 16);
        var int4 = new RelaxedOneHotCategoricalInt4Distribution(logits, temp, 16, groupSize: 16);
        var rng = new Random(999);

        // Use the FP32 sample so both distributions evaluate the same point.
        var sample = fp32.RSample(rng);
        var lpFp32 = fp32.LogProb(sample);
        var lpInt4 = int4.LogProb(sample);

        Assert.Equal(lpFp32.Length, lpInt4.Length);
        for (int b = 0; b < lpFp32.Length; b++)
        {
            float delta = MathF.Abs(lpFp32[b] - lpInt4[b]);
            // log_prob is the per-batch scalar; per-dim degradation
            // is delta / K. Assert delta < 0.5 · K to bound per-dim
            // < 0.5 nats.
            Assert.True(delta < 0.5f * 16, $"batch {b} delta {delta} exceeds 0.5 · K nats.");
        }
    }

    [Fact]
    public void Int4_RSampleIsApproximatelyOnSimplex()
    {
        // After dequant the sample should still be near-simplex: sum
        // close to 1, every cell in [0, 1] within quant tolerance.
        var (logits, temp) = MakeBatch(batch: 2, k: 16, seed: 4);
        var d = new RelaxedOneHotCategoricalInt4Distribution(logits, temp, 16, groupSize: 16);
        var sample = d.RSample(new Random(123));
        Assert.Equal(2 * 16, sample.Length);
        for (int b = 0; b < 2; b++)
        {
            float sum = 0;
            for (int k = 0; k < 16; k++) sum += sample[b * 16 + k];
            // Tighter bound here than the round-trip test because
            // groupSize=K means each row gets its own scale.
            Assert.InRange(sum, 0.7f, 1.3f);
        }
    }

    [Fact]
    public void Fp4_SampleProducesPackedHalfByteEncoding()
    {
        var (logits, temp) = MakeBatch(batch: 4, k: 8, seed: 5);
        var d = new RelaxedOneHotCategoricalFp4Distribution(logits, temp, 8);
        var packed = d.SampleFp4(new Random(42));
        // 4 × 8 = 32 elements → 16 packed bytes.
        Assert.Equal(16, packed.Length);
    }

    [Fact]
    public void Fp4_DequantizeReturnsValuesFromCodeTable()
    {
        var (logits, temp) = MakeBatch(batch: 2, k: 8, seed: 6);
        var d = new RelaxedOneHotCategoricalFp4Distribution(logits, temp, 8);
        var packed = d.SampleFp4(new Random(7));
        var dense = d.Dequantize(packed);
        Assert.Equal(16, dense.Length);
        // Every dequantised value must be one of the 16 FP4 codes.
        var table = new System.Collections.Generic.HashSet<float>(
            AiDotNet.Tensors.NumericOperations.Fp4E2M1.Table.ToArray());
        for (int i = 0; i < dense.Length; i++)
            Assert.Contains(dense[i], table);
    }

    [Fact]
    public void Fp4_LogProbAcceptsDequantisedSample()
    {
        var (logits, temp) = MakeBatch(batch: 4, k: 8, seed: 7);
        var d = new RelaxedOneHotCategoricalFp4Distribution(logits, temp, 8);
        var packed = d.SampleFp4(new Random(11));
        var dense = d.Dequantize(packed);
        var lp = d.LogProb(dense);
        Assert.Equal(4, lp.Length);
        // Most of the time the log-prob is finite. At the FP4 grid
        // boundaries (e.g. exact zero) the log-prob can become very
        // large negative due to the τ·log(y) term — that's expected
        // and matches PyTorch's FP32 behaviour at degenerate samples.
        // We accept finite OR very negative; we just reject NaN /
        // +inf.
        for (int b = 0; b < 4; b++)
            Assert.False(float.IsNaN(lp[b]) || float.IsPositiveInfinity(lp[b]),
                $"FP4 log-prob at batch {b} is invalid: {lp[b]}");
    }

    [Fact]
    public void Int4_MeanReturnsSoftmaxOfLogits()
    {
        // Logits = [0, 0, 0] → mean should be uniform 1/3.
        var logits = new float[] { 0f, 0f, 0f };
        var temp = new float[] { 1f };
        var d = new RelaxedOneHotCategoricalInt4Distribution(logits, temp, 3);
        var mean = d.Mean;
        Assert.Equal(3, mean.Length);
        for (int i = 0; i < 3; i++) Assert.Equal(1f / 3f, mean[i], 4);
    }

    [Fact]
    public void Int4_RejectsInvalidConfigurations()
    {
        Assert.Throws<ArgumentException>(() =>
            new RelaxedOneHotCategoricalInt4Distribution(new float[] { 1, 2 }, new float[] { 1f }, 2, groupSize: 0));
        Assert.Throws<ArgumentException>(() =>
            new RelaxedOneHotCategoricalInt4Distribution(new float[] { 1, 2 }, new float[] { 1f }, 2, groupSize: 3));
        Assert.Throws<ArgumentException>(() =>
            new RelaxedOneHotCategoricalInt4Distribution(new float[] { 1, 2 }, new float[] { 0f }, 2));
    }

    [Fact]
    public void Int4_VsFp32_HasSimilarDistributionShape()
    {
        // Both samplers should produce samples concentrated near the
        // softmax mode for low temperature. Empirical check that the
        // argmax of the dequantised int4 sample matches the argmax
        // of the FP32 sample on > 80% of batches over many trials.
        const int batch = 64;
        const int k = 8;
        const int trials = 32;
        var rng = new Random(0xBEEF);
        int agree = 0, total = 0;

        // Make logits that have a clear winner per row so the sampler's
        // mode is well-defined.
        var logits = new float[batch * k];
        for (int b = 0; b < batch; b++)
        {
            int winner = rng.Next(k);
            for (int i = 0; i < k; i++) logits[b * k + i] = i == winner ? 5f : 0f;
        }
        var temp = new float[batch];
        for (int i = 0; i < batch; i++) temp[i] = 0.3f;

        var fp32 = new RelaxedOneHotCategoricalDistribution(logits, temp, k);
        var int4 = new RelaxedOneHotCategoricalInt4Distribution(logits, temp, k, groupSize: 8);

        for (int t = 0; t < trials; t++)
        {
            var s1 = fp32.RSample(new Random(t * 13));
            var s2 = int4.RSample(new Random(t * 13));
            for (int b = 0; b < batch; b++)
            {
                int a1 = ArgMax(s1, b * k, k);
                int a2 = ArgMax(s2, b * k, k);
                if (a1 == a2) agree++;
                total++;
            }
        }
        double agreementRate = (double)agree / total;
        Assert.True(agreementRate > 0.8,
            $"Int4 vs FP32 argmax agreement {agreementRate:P} below 80% threshold.");
    }

    [Fact]
    public void Int4_RSampleTape_StraightThroughGradFlowsToLogits()
    {
        // Verify the STE path: the gradient through RSampleTape should
        // accumulate to logits as identity (the dequantise step is
        // treated as a no-op in backward).
        var (logitsArr, temp) = MakeBatch(batch: 4, k: 8, seed: 99);
        var d = new RelaxedOneHotCategoricalInt4Distribution(logitsArr, temp, 8, groupSize: 8);
        var logits = new Tensor<float>(new[] { 4, 8 });
        for (int i = 0; i < logits.Length; i++) logits.AsWritableSpan()[i] = logitsArr[i];

        using var tape = new GradientTape<float>();
        var sample = d.RSampleTape(logits, new Random(7));
        var engine = new CpuEngine();
        var loss = engine.ReduceSum(sample, null);
        var grads = tape.ComputeGradients(loss, new[] { logits });
        Assert.True(grads.ContainsKey(logits));
        // STE: dL/dlogits = 1 everywhere (since loss is sum of every cell).
        var g = grads[logits].AsSpan();
        for (int i = 0; i < g.Length; i++)
            Assert.Equal(1f, g[i], 4);
    }

    [Fact]
    public void Fp4_RSampleTape_StraightThroughGradFlowsToLogits()
    {
        var (logitsArr, temp) = MakeBatch(batch: 2, k: 8, seed: 199);
        var d = new RelaxedOneHotCategoricalFp4Distribution(logitsArr, temp, 8);
        var logits = new Tensor<float>(new[] { 2, 8 });
        for (int i = 0; i < logits.Length; i++) logits.AsWritableSpan()[i] = logitsArr[i];

        using var tape = new GradientTape<float>();
        var sample = d.RSampleTape(logits, new Random(11));
        var engine = new CpuEngine();
        var loss = engine.ReduceSum(sample, null);
        var grads = tape.ComputeGradients(loss, new[] { logits });
        Assert.True(grads.ContainsKey(logits));
        var g = grads[logits].AsSpan();
        for (int i = 0; i < g.Length; i++) Assert.Equal(1f, g[i], 4);
    }

    [Fact]
    public void Int4_DequantizeRejectsUndersizedPackedBuffer()
    {
        var (logits, temp) = MakeBatch(batch: 4, k: 8, seed: 777);
        var d = new RelaxedOneHotCategoricalInt4Distribution(logits, temp, 8, groupSize: 8);
        var truncated = new QuantisedCategoricalSample(
            new AiDotNet.Tensors.NumericOperations.PackedInt4[1],
            new AiDotNet.Tensors.NumericOperations.QuantizationScale(new float[] { 1f }, 8),
            batch: 4, k: 8);
        Assert.Throws<ArgumentException>(() => d.Dequantize(truncated));
    }

    [Fact]
    public void Fp4_DequantizeRejectsUndersizedPackedBuffer()
    {
        var (logits, temp) = MakeBatch(batch: 4, k: 8, seed: 778);
        var d = new RelaxedOneHotCategoricalFp4Distribution(logits, temp, 8);
        // 4 × 8 = 32 elements → 16 bytes expected; pass 1 byte.
        Assert.Throws<ArgumentException>(() => d.Dequantize(new byte[1]));
    }

    private static int ArgMax(float[] arr, int offset, int len)
    {
        int best = 0;
        float bestV = arr[offset];
        for (int i = 1; i < len; i++)
            if (arr[offset + i] > bestV) { bestV = arr[offset + i]; best = i; }
        return best;
    }
}
