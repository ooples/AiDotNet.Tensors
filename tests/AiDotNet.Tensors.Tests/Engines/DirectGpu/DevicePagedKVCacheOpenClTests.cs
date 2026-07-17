// Copyright (c) AiDotNet. All rights reserved.
// End-to-end correctness tests for the on-device paged KV cache (P1): append tokens across block
// boundaries into the device pool, then run paged-attention through the uploaded block table and
// compare to a standard-attention CPU oracle. Skips when no OpenCL device is available.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class DevicePagedKVCacheOpenClTests : IDisposable
{
    private readonly OpenClBackend? _backend;
    private readonly bool _ready;

    public DevicePagedKVCacheOpenClTests()
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

    // Standard-attention oracle over a contiguous [seqLen, heads, headDim] K/V.
    private static float[] AttnOracle(float[] q, float[] k, float[] v, int heads, int headDim, int seqLen, float scale)
    {
        var outp = new float[heads * headDim];
        for (int h = 0; h < heads; h++)
        {
            var logits = new float[seqLen];
            float max = float.NegativeInfinity;
            for (int t = 0; t < seqLen; t++)
            {
                float dot = 0f;
                for (int d = 0; d < headDim; d++) dot += q[h * headDim + d] * k[(t * heads + h) * headDim + d];
                logits[t] = dot * scale;
                if (logits[t] > max) max = logits[t];
            }
            float denom = 0f;
            for (int t = 0; t < seqLen; t++) { logits[t] = MathF.Exp(logits[t] - max); denom += logits[t]; }
            for (int t = 0; t < seqLen; t++)
            {
                float p = logits[t] / denom;
                for (int d = 0; d < headDim; d++) outp[h * headDim + d] += p * v[(t * heads + h) * headDim + d];
            }
        }
        return outp;
    }

    [Theory]
    [InlineData(4, 64, 16, 40)]   // 40 tokens over 16-token blocks → 3 blocks, partial last
    [InlineData(2, 32, 8, 20)]
    [InlineData(8, 32, 32, 100)]
    public void Append_Then_Decode_MatchesOracle(int heads, int headDim, int blockSize, int seqLen)
    {
        if (!EnsureReady()) return;
        var backend = _backend!;
        var rng = new Random(0xBEE + heads + seqLen);
        int stepStride = heads * headDim;

        // Contiguous K/V for the whole sequence (the oracle input).
        var k = new float[seqLen * stepStride];
        var v = new float[seqLen * stepStride];
        for (int i = 0; i < k.Length; i++) { k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }
        var q = new float[stepStride];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        float scale = 1.0f / MathF.Sqrt(headDim);

        int maxBlocks = (seqLen + blockSize - 1) / blockSize + 4;
        using var cache = new DevicePagedKVCache(backend, maxBlocks, blockSize, heads, headDim);

        // Append in two chunks so we cross block boundaries mid-stream (continuous-batching style).
        int firstChunk = Math.Min(seqLen, blockSize + blockSize / 2);
        var k1 = k[..(firstChunk * stepStride)];
        var v1 = v[..(firstChunk * stepStride)];
        cache.Append(seqId: 7, k1, v1);
        if (seqLen > firstChunk)
        {
            var k2 = k[(firstChunk * stepStride)..];
            var v2 = v[(firstChunk * stepStride)..];
            cache.Append(seqId: 7, k2, v2);
        }
        Assert.Equal(seqLen, cache.GetLength(7));

        var qBuf = backend.AllocateBuffer(q);
        var btBuf = cache.GetBlockTableBuffer(7);
        var outBuf = backend.PagedAttentionDecode(qBuf, cache.KeyBlocks, cache.ValueBlocks, btBuf,
            heads, headDim, blockSize, cache.GetLength(7), scale);
        var actual = backend.DownloadBuffer(outBuf);
        qBuf.Dispose(); btBuf.Dispose(); outBuf.Dispose();

        var expected = AttnOracle(q, k, v, heads, headDim, seqLen, scale);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"decode mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }

    [Fact]
    public void Free_ReturnsBlocksToPool_AndReuses()
    {
        if (!EnsureReady()) return;
        var backend = _backend!;
        int heads = 2, headDim = 16, blockSize = 8, stepStride = heads * headDim;
        using var cache = new DevicePagedKVCache(backend, maxBlocks: 4, blockSize, heads, headDim);

        var k = new float[10 * stepStride];
        var v = new float[10 * stepStride];
        cache.Append(1, k, v);                     // 10 tokens → 2 blocks (8 + 2)
        Assert.Equal(2, cache.AllocatedBlocks);
        cache.Free(1);
        Assert.Equal(0, cache.AllocatedBlocks);
        Assert.Equal(0, cache.GetLength(1));

        cache.Append(2, k, v);                     // reuse freed blocks
        Assert.Equal(2, cache.AllocatedBlocks);
        Assert.Equal(10, cache.GetLength(2));
    }

    [Fact]
    public void ShareBlocks_TargetSeesSharedPrefix()
    {
        if (!EnsureReady()) return;
        var backend = _backend!;
        int heads = 2, headDim = 16, blockSize = 8, stepStride = heads * headDim;
        int prefixTokens = 12;
        using var cache = new DevicePagedKVCache(backend, maxBlocks: 8, blockSize, heads, headDim);

        var rng = new Random(0x5EED);
        var k = new float[prefixTokens * stepStride];
        var v = new float[prefixTokens * stepStride];
        for (int i = 0; i < k.Length; i++) { k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }
        cache.Append(1, k, v);
        cache.ShareBlocks(sourceSeqId: 1, targetSeqId: 2, prefixLen: prefixTokens);
        Assert.Equal(prefixTokens, cache.GetLength(2));

        var q = new float[stepStride];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        float scale = 1.0f / MathF.Sqrt(headDim);

        var qBuf = backend.AllocateBuffer(q);
        var btBuf = cache.GetBlockTableBuffer(2);
        var outBuf = backend.PagedAttentionDecode(qBuf, cache.KeyBlocks, cache.ValueBlocks, btBuf,
            heads, headDim, blockSize, prefixTokens, scale);
        var actual = backend.DownloadBuffer(outBuf);
        qBuf.Dispose(); btBuf.Dispose(); outBuf.Dispose();

        var expected = AttnOracle(q, k, v, heads, headDim, prefixTokens, scale);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"shared-prefix mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }
}

#endif
