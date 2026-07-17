// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the OpenCL fused decode-attention kernel (P2, FlashDecoding): single-query
// attention over contiguous K/V, sequence split across work-items and merged by online-softmax
// reduction. Validated against a standard-attention CPU oracle. Skips when no OpenCL device is present.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class FlashDecodeOpenClTests : IDisposable
{
    private readonly OpenClBackend? _backend;
    private readonly bool _ready;

    public FlashDecodeOpenClTests()
    {
        try { _backend = new OpenClBackend(); _ready = _backend.IsAvailable; }
        catch { _ready = false; }
    }

    public void Dispose() => _backend?.Dispose();

    private bool EnsureReady()
    {
        if (_ready) return true;
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException("GPU tests required but OpenCL was unavailable.");
        return false;
    }

    // Single-query standard-attention oracle over contiguous K/V [seqLen, kvHeads, headDim].
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

    private void RunAndCompare(int heads, int kvHeads, int headDim, int seqLen, int splits)
    {
        var backend = _backend!;
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
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"flash-decode mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }

    [Theory]
    [InlineData(4, 64, 40, 8)]    // MHA, default-ish splits
    [InlineData(8, 32, 100, 8)]
    [InlineData(2, 128, 20, 4)]
    [InlineData(4, 64, 7, 8)]     // splits > seqLen path (clamped)
    [InlineData(4, 64, 64, 1)]    // single split == serial reference
    public void MhaDecode_MatchesOracle(int heads, int headDim, int seqLen, int splits)
    {
        if (!EnsureReady()) return;
        RunAndCompare(heads, heads, headDim, seqLen, splits);
    }

    [Theory]
    [InlineData(8, 2, 64, 50, 8)]  // GQA: 8 query heads share 2 KV heads
    [InlineData(8, 1, 64, 40, 8)]  // MQA
    [InlineData(4, 4, 32, 30, 4)]  // kvHeads == heads (MHA via GQA path)
    public void GqaDecode_MatchesOracle(int heads, int kvHeads, int headDim, int seqLen, int splits)
    {
        if (!EnsureReady()) return;
        RunAndCompare(heads, kvHeads, headDim, seqLen, splits);
    }

    [Fact]
    public void DefaultSplits_MatchesOracle()
    {
        if (!EnsureReady()) return;
        RunAndCompare(heads: 6, kvHeads: 3, headDim: 48, seqLen: 77, splits: 0); // splits=0 → internal default
    }
}

#endif
