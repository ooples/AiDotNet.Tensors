// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the Vulkan fused decode-attention kernel (P2, FlashDecoding), validated against a
// standard-attention CPU oracle. Skips without a Vulkan device + GLSL compiler (libshaderc).

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class FlashDecodeVulkanTests
{
    private static bool Ready
    {
        get
        {
            try { return VulkanBackend.Instance.IsAvailable && VulkanBackend.Instance.IsGlslCompilerAvailable; }
            catch { return false; }
        }
    }

    private static float[] DecodeOracle(float[] q, float[] k, float[] v, int heads, int kvHeads, int headDim, int seqLen, float scale)
    {
        var outp = new float[heads * headDim];
        for (int h = 0; h < heads; h++)
        {
            int kvHead = h / (heads / kvHeads);
            var logits = new float[seqLen];
            float max = float.NegativeInfinity;
            for (int t = 0; t < seqLen; t++)
            {
                long kb = ((long)t * kvHeads + kvHead) * headDim;
                float dot = 0f;
                for (int d = 0; d < headDim; d++) dot += q[h * headDim + d] * k[kb + d];
                logits[t] = dot * scale;
                if (logits[t] > max) max = logits[t];
            }
            float denom = 0f;
            for (int t = 0; t < seqLen; t++) { logits[t] = MathF.Exp(logits[t] - max); denom += logits[t]; }
            for (int t = 0; t < seqLen; t++)
            {
                long kb = ((long)t * kvHeads + kvHead) * headDim;
                float p = logits[t] / denom;
                for (int d = 0; d < headDim; d++) outp[h * headDim + d] += p * v[kb + d];
            }
        }
        return outp;
    }

    private static void RunAndCompare(int heads, int kvHeads, int headDim, int seqLen, int splits)
    {
        var backend = VulkanBackend.Instance;
        var rng = new Random(0xFDE + heads + kvHeads + seqLen + splits);
        int stepStride = kvHeads * headDim;
        var k = new float[seqLen * stepStride];
        var v = new float[seqLen * stepStride];
        for (int i = 0; i < k.Length; i++) { k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }
        var q = new float[heads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        float scale = 1.0f / MathF.Sqrt(headDim);

        var qBuf = backend.AllocateBuffer(q);
        var kBuf = backend.AllocateBuffer(k);
        var vBuf = backend.AllocateBuffer(v);
        var outBuf = backend.FlashDecode(qBuf, kBuf, vBuf, heads, kvHeads, headDim, seqLen, scale, splits);
        var actual = backend.DownloadBuffer(outBuf);
        qBuf.Dispose(); kBuf.Dispose(); vBuf.Dispose(); outBuf.Dispose();

        var expected = DecodeOracle(q, k, v, heads, kvHeads, headDim, seqLen, scale);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"flash-decode mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }

    [Theory]
    [InlineData(4, 64, 40, 8)]
    [InlineData(8, 32, 100, 8)]
    [InlineData(4, 64, 7, 8)]
    [InlineData(4, 64, 64, 1)]
    public void MhaDecode_MatchesOracle(int heads, int headDim, int seqLen, int splits)
    {
        if (!Ready) return;
        RunAndCompare(heads, heads, headDim, seqLen, splits);
    }

    [Theory]
    [InlineData(8, 2, 64, 50, 8)]
    [InlineData(8, 1, 64, 40, 8)]
    public void GqaDecode_MatchesOracle(int heads, int kvHeads, int headDim, int seqLen, int splits)
    {
        if (!Ready) return;
        RunAndCompare(heads, kvHeads, headDim, seqLen, splits);
    }
}

#endif
