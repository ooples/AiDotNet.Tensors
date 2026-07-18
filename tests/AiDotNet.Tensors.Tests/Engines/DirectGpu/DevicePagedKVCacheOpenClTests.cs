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

        float[] actual;
        IGpuBuffer? qBuf = null, btBuf = null, outBuf = null;
        try
        {
            qBuf = backend.AllocateBuffer(q);
            btBuf = cache.GetBlockTableBuffer(7);
            outBuf = backend.PagedAttentionDecode(qBuf, cache.KeyBlocks, cache.ValueBlocks, btBuf,
                heads, headDim, blockSize, cache.GetLength(7), scale);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            qBuf?.Dispose(); btBuf?.Dispose(); outBuf?.Dispose();
        }

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
    public void Backend_ExposesPagedAttentionCapability_AndInterfacePathMatchesOracle()
    {
        // The capability interface (IPagedAttentionBackend) is how higher layers (e.g. an inference/serving
        // engine) consume paged attention without depending on a concrete backend type. Verify the backend
        // advertises it and that dispatching through the interface produces the same result as the oracle.
        if (!EnsureReady()) return;
        var backend = _backend!;
        Assert.True(backend is IPagedAttentionBackend, "OpenClBackend must expose IPagedAttentionBackend.");
        var paged = (IPagedAttentionBackend)backend;

        const int heads = 2, headDim = 32, blockSize = 8, seqLen = 20;
        int stride = heads * headDim;
        var rng = new Random(123);
        var k = new float[seqLen * stride];
        var v = new float[seqLen * stride];
        for (int i = 0; i < k.Length; i++) { k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }
        var q = new float[stride];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        float scale = 1.0f / MathF.Sqrt(headDim);

        using var cache = new DevicePagedKVCache(backend, (seqLen + blockSize - 1) / blockSize + 2, blockSize, heads, headDim);
        cache.Append(1, k, v);

        float[] actual;
        IGpuBuffer? qBuf = null, btBuf = null, outBuf = null;
        try
        {
            qBuf = backend.AllocateBuffer(q);
            btBuf = cache.GetBlockTableBuffer(1);
            outBuf = paged.PagedAttentionDecode(qBuf, cache.KeyBlocks, cache.ValueBlocks, btBuf,
                heads, headDim, blockSize, cache.GetLength(1), scale);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally { qBuf?.Dispose(); btBuf?.Dispose(); outBuf?.Dispose(); }

        var expected = AttnOracle(q, k, v, heads, headDim, seqLen, scale);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float tol = 2e-3f + 2e-3f * Math.Abs(expected[i]);
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol,
                $"interface-path decode mismatch at {i}: expected {expected[i]}, got {actual[i]}");
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

        float[] actual;
        IGpuBuffer? qBuf = null, btBuf = null, outBuf = null;
        try
        {
            qBuf = backend.AllocateBuffer(q);
            btBuf = cache.GetBlockTableBuffer(2);
            outBuf = backend.PagedAttentionDecode(qBuf, cache.KeyBlocks, cache.ValueBlocks, btBuf,
                heads, headDim, blockSize, prefixTokens, scale);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            qBuf?.Dispose(); btBuf?.Dispose(); outBuf?.Dispose();
        }

        var expected = AttnOracle(q, k, v, heads, headDim, prefixTokens, scale);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"shared-prefix mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }

    [Fact]
    public void ShareBlocks_RefcountedFree_NoDoubleFree()
    {
        if (!EnsureReady()) return;
        var backend = _backend!;
        int heads = 2, headDim = 16, blockSize = 8, stepStride = heads * headDim;
        using var cache = new DevicePagedKVCache(backend, maxBlocks: 8, blockSize, heads, headDim);

        var k = new float[12 * stepStride];
        var v = new float[12 * stepStride];
        cache.Append(1, k, v);                                  // 12 tokens -> 2 blocks
        cache.ShareBlocks(1, 2, 12);                            // both blocks now refcount 2
        Assert.Equal(2, cache.AllocatedBlocks);

        cache.Free(1);                                          // holder 1 gone; blocks still held by 2
        Assert.Equal(2, cache.AllocatedBlocks);                // NOT returned yet (refcount 1)
        Assert.Equal(0, cache.GetLength(1));
        Assert.Equal(12, cache.GetLength(2));

        cache.Free(2);                                          // last holder -> blocks released
        Assert.Equal(0, cache.AllocatedBlocks);

        // The two physical blocks must be reusable exactly once (no duplicate ids on the free list).
        cache.Append(3, k, v);
        cache.Append(4, k, v);
        Assert.Equal(4, cache.AllocatedBlocks);                // 2 + 2 distinct blocks; no double-hand-out
    }

    [Fact]
    public void CowAppend_AfterShare_LeavesSourceIntact()
    {
        if (!EnsureReady()) return;
        var backend = _backend!;
        int heads = 2, headDim = 16, blockSize = 8, stepStride = heads * headDim;
        using var cache = new DevicePagedKVCache(backend, maxBlocks: 16, blockSize, heads, headDim);

        var rng = new Random(0xC0FFEE);
        int srcLen = 10;                                        // 2 blocks, last block partially filled (2 tokens)
        var k = new float[srcLen * stepStride];
        var v = new float[srcLen * stepStride];
        for (int i = 0; i < k.Length; i++) { k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }
        cache.Append(1, k, v);
        cache.ShareBlocks(1, 2, srcLen);                       // seq2 shares seq1's blocks incl. the partial tail

        // Append to seq2 -> must copy-on-write the shared partial tail block, not overwrite seq1's data.
        int extra = 5;
        var k2 = new float[extra * stepStride];
        var v2 = new float[extra * stepStride];
        for (int i = 0; i < k2.Length; i++) { k2[i] = (float)(rng.NextDouble() * 2 - 1); v2[i] = (float)(rng.NextDouble() * 2 - 1); }
        cache.Append(2, k2, v2);
        Assert.Equal(srcLen + extra, cache.GetLength(2));
        Assert.Equal(srcLen, cache.GetLength(1));

        var q = new float[stepStride];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        float scale = 1.0f / MathF.Sqrt(headDim);

        // seq1 output must still match the ORIGINAL srcLen K/V (unchanged by seq2's COW append).
        float[] seq1Actual;
        IGpuBuffer? q1 = null, bt1 = null, o1 = null;
        try
        {
            q1 = backend.AllocateBuffer(q);
            bt1 = cache.GetBlockTableBuffer(1);
            o1 = backend.PagedAttentionDecode(q1, cache.KeyBlocks, cache.ValueBlocks, bt1, heads, headDim, blockSize, srcLen, scale);
            seq1Actual = backend.DownloadBuffer(o1);
        }
        finally
        {
            q1?.Dispose(); bt1?.Dispose(); o1?.Dispose();
        }

        var seq1Expected = AttnOracle(q, k, v, heads, headDim, srcLen, scale);
        for (int i = 0; i < seq1Expected.Length; i++)
        {
            float e = seq1Expected[i], a = seq1Actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"COW corrupted source at {i}: expected {e}, got {a} (tol {tol})");
        }

        // seq2 output must match the concatenation of the shared prefix + its appended tokens.
        var kFull = new float[(srcLen + extra) * stepStride];
        var vFull = new float[(srcLen + extra) * stepStride];
        Array.Copy(k, 0, kFull, 0, k.Length); Array.Copy(k2, 0, kFull, k.Length, k2.Length);
        Array.Copy(v, 0, vFull, 0, v.Length); Array.Copy(v2, 0, vFull, v.Length, v2.Length);
        float[] seq2Actual;
        IGpuBuffer? q2 = null, bt2 = null, o2 = null;
        try
        {
            q2 = backend.AllocateBuffer(q);
            bt2 = cache.GetBlockTableBuffer(2);
            o2 = backend.PagedAttentionDecode(q2, cache.KeyBlocks, cache.ValueBlocks, bt2, heads, headDim, blockSize, srcLen + extra, scale);
            seq2Actual = backend.DownloadBuffer(o2);
        }
        finally
        {
            q2?.Dispose(); bt2?.Dispose(); o2?.Dispose();
        }

        var seq2Expected = AttnOracle(q, kFull, vFull, heads, headDim, srcLen + extra, scale);
        for (int i = 0; i < seq2Expected.Length; i++)
        {
            float e = seq2Expected[i], a = seq2Actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"COW seq2 mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }
}

#endif
