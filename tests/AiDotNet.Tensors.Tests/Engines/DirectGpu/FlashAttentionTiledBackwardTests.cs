// Copyright (c) CrowdTrainer LLC. All rights reserved.
// #1528: the hand-written FlashAttention backward as two tiled, atomic-free kernels. With the declared route off they
// serve every FlashAttentionBackward call up to head dimension 128; checked here against an fp64 reference (the flash
// kernels' causal rule, key <= query, and an optional additive bias) and for bit-identical repeats.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public class FlashAttentionTiledBackwardTests
{
    private static CudaBackend OpenBackendOrSkip()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("CUDA driver was detected, but the CUDA backend failed to initialize.");
        }
        return backend;
    }

    private static float[] Fill(int count, int salt)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((((i * 37 + salt * 101) % 97) - 48) / 97.0 + 0.003 * (i % 7));
        return values;
    }

    // dQ, dK, dV of softmax(scale * Q K^T + bias) V against dO, in fp64, with the flash kernels' causal rule.
    private static (double[] Dq, double[] Dk, double[] Dv) Reference(float[] q, float[] k, float[] v, float[] dout,
        float[]? bias, int batch, int heads, int seqQ, int seqK, int d, float scale, bool causal)
    {
        var dq = new double[batch * heads * seqQ * d];
        var dk = new double[batch * heads * seqK * d];
        var dv = new double[batch * heads * seqK * d];
        for (int bh = 0; bh < batch * heads; bh++)
        {
            int b = bh / heads, h = bh % heads;
            for (int qi = 0; qi < seqQ; qi++)
            {
                var p = new double[seqK];
                double max = double.NegativeInfinity;
                for (int ki = 0; ki < seqK; ki++)
                {
                    if (causal && ki > qi) { p[ki] = double.NegativeInfinity; continue; }
                    double s = 0;
                    for (int c = 0; c < d; c++) s += (double)q[(bh * seqQ + qi) * d + c] * k[(bh * seqK + ki) * d + c];
                    s *= scale;
                    if (bias is not null) s += bias[(b * heads + h) * seqQ * seqK + qi * seqK + ki];
                    p[ki] = s;
                    max = Math.Max(max, s);
                }
                double sum = 0;
                for (int ki = 0; ki < seqK; ki++) { p[ki] = double.IsNegativeInfinity(p[ki]) ? 0 : Math.Exp(p[ki] - max); sum += p[ki]; }
                for (int ki = 0; ki < seqK; ki++) p[ki] /= sum;
                var dp = new double[seqK];
                double dot = 0;
                for (int ki = 0; ki < seqK; ki++)
                {
                    for (int c = 0; c < d; c++) dp[ki] += (double)dout[(bh * seqQ + qi) * d + c] * v[(bh * seqK + ki) * d + c];
                    dot += p[ki] * dp[ki];
                }
                for (int ki = 0; ki < seqK; ki++)
                {
                    double ds = p[ki] * (dp[ki] - dot) * scale;
                    for (int c = 0; c < d; c++)
                    {
                        dq[(bh * seqQ + qi) * d + c] += ds * k[(bh * seqK + ki) * d + c];
                        dk[(bh * seqK + ki) * d + c] += ds * q[(bh * seqQ + qi) * d + c];
                        dv[(bh * seqK + ki) * d + c] += p[ki] * dout[(bh * seqQ + qi) * d + c];
                    }
                }
            }
        }
        return (dq, dk, dv);
    }

    [SkippableTheory]
    [InlineData(1, 2, 5, 7, 33, false, false)]     // tile 32, everything partial
    [InlineData(2, 3, 64, 64, 64, true, false)]    // two full 32-tiles, causal
    [InlineData(1, 2, 40, 40, 96, true, false)]    // tile 16, causal across a partial tile
    [InlineData(2, 2, 17, 23, 128, false, true)]   // tile 16, unequal lengths, bias
    [InlineData(1, 1, 100, 100, 16, true, true)]   // many tiles, causal and bias
    public void Matches_fp64_and_repeats_bit_for_bit(int batch, int heads, int seqQ, int seqK, int d, bool causal, bool withBias)
    {
        using var backend = OpenBackendOrSkip();
        backend.DeclaredKernelsSwitch = false;   // the hand-written kernels, which the tiled backward now is
        float scale = (float)(1 / Math.Sqrt(d));
        float[] q = Fill(batch * heads * seqQ * d, 1), k = Fill(batch * heads * seqK * d, 2);
        float[] v = Fill(batch * heads * seqK * d, 3), dout = Fill(batch * heads * seqQ * d, 4);
        float[]? bias = withBias ? Fill(batch * heads * seqQ * seqK, 5) : null;
        using var qb = backend.AllocateBuffer(q);
        using var kb = backend.AllocateBuffer(k);
        using var vb = backend.AllocateBuffer(v);
        using var gob = backend.AllocateBuffer(dout);
        using var ob = backend.AllocateBuffer(q.Length);
        using var stats = backend.AllocateBuffer(batch * heads * seqQ);
        using var biasBuffer = bias is null ? null : backend.AllocateBuffer(bias);
        backend.FlashAttentionV2(qb, kb, vb, ob, stats, batch, heads, seqQ, seqK, d, scale, causal, biasBuffer,
            heads * seqQ * seqK);
        float[] Run(int which)
        {
            using var dq = backend.AllocateBuffer(q.Length);
            using var dk = backend.AllocateBuffer(k.Length);
            using var dv = backend.AllocateBuffer(v.Length);
            backend.FlashAttentionBackward(gob, qb, kb, vb, ob, stats, dq, dk, dv, batch, heads, seqQ, seqK, d, scale,
                causal, biasBuffer, heads * seqQ * seqK);
            backend.Synchronize();
            return backend.DownloadBuffer(which == 0 ? dq : which == 1 ? dk : dv);
        }
        var (rq, rk, rv) = Reference(q, k, v, dout, bias, batch, heads, seqQ, seqK, d, scale, causal);
        var refs = new[] { rq, rk, rv };
        for (int which = 0; which < 3; which++)
        {
            float[] first = Run(which), second = Run(which);
            for (int i = 0; i < first.Length; i++)
            {
                Assert.Equal(TestHelpers.MathCompat.SingleToInt32Bits(first[i]), TestHelpers.MathCompat.SingleToInt32Bits(second[i]));
                double r = refs[which][i];
                Assert.True(Math.Abs(first[i] - r) <= 2e-4 * (1 + Math.Abs(r)),
                    $"{"QKV"[which]}[{i}]: gpu {first[i]} vs fp64 {r}");
            }
        }
    }
}