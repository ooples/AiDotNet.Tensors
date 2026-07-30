// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the Metal paged-attention decode kernel (P1), validated against a standard-
// attention CPU oracle. Skips off macOS; runs on Apple silicon / CI.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class PagedAttentionMetalTests : IDisposable
{
    private readonly MetalBackend? _backend;
    private readonly bool _ready;

    public PagedAttentionMetalTests()
    {
        try { _backend = new MetalBackend(); _ready = _backend.IsAvailable; }
        catch { _ready = false; }
    }

    public void Dispose() => _backend?.Dispose();

    [Fact]
    public void Probe_MetalAvailability()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_METAL") != "1") return;
        Assert.True(_ready, "Metal backend NOT available on this host");
    }

    private bool EnsureReady()
    {
        if (_ready) return true;
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_METAL"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException("GPU tests required but Metal was unavailable.");
        return false;
    }

    [Theory]
    [InlineData(4, 64, 16, 40)]
    [InlineData(2, 128, 8, 20)]
    [InlineData(8, 32, 32, 100)]
    public void PagedAttentionDecode_MatchesCpuOracle(int heads, int headDim, int blockSize, int seqLen)
    {
        if (!EnsureReady()) return;
        var backend = _backend!;
        var rng = new Random(0xA77 + heads + seqLen);

        int numLogicalBlocks = (seqLen + blockSize - 1) / blockSize;
        int maxBlocks = numLogicalBlocks + 5;
        int poolLen = maxBlocks * blockSize * heads * headDim;

        var kcache = new float[poolLen];
        var vcache = new float[poolLen];
        for (int i = 0; i < poolLen; i++) { kcache[i] = (float)(rng.NextDouble() * 2 - 1); vcache[i] = (float)(rng.NextDouble() * 2 - 1); }

        var q = new float[heads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);

        var blockTable = new int[numLogicalBlocks];
        var pool = new System.Collections.Generic.List<int>();
        for (int b = 0; b < maxBlocks; b++) pool.Add(b);
        for (int b = 0; b < numLogicalBlocks; b++) { int pick = rng.Next(pool.Count); blockTable[b] = pool[pick]; pool.RemoveAt(pick); }

        float scale = 1.0f / MathF.Sqrt(headDim);

        var expected = new float[heads * headDim];
        for (int h = 0; h < heads; h++)
        {
            var logits = new float[seqLen];
            float max = float.NegativeInfinity;
            for (int t = 0; t < seqLen; t++)
            {
                int blk = blockTable[t / blockSize], pos = t % blockSize;
                long baseIdx = (((long)blk * blockSize + pos) * heads + h) * headDim;
                float dot = 0f;
                for (int d = 0; d < headDim; d++) dot += q[h * headDim + d] * kcache[baseIdx + d];
                logits[t] = dot * scale;
                if (logits[t] > max) max = logits[t];
            }
            float denom = 0f;
            for (int t = 0; t < seqLen; t++) { logits[t] = MathF.Exp(logits[t] - max); denom += logits[t]; }
            for (int t = 0; t < seqLen; t++)
            {
                int blk = blockTable[t / blockSize], pos = t % blockSize;
                long baseIdx = (((long)blk * blockSize + pos) * heads + h) * headDim;
                float p = logits[t] / denom;
                for (int d = 0; d < headDim; d++) expected[h * headDim + d] += p * vcache[baseIdx + d];
            }
        }

        float[] actual;
        IGpuBuffer? qBuf = null, kBuf = null, vBuf = null, btBuf = null, outBuf = null;
        try
        {
            qBuf = backend.AllocateBuffer(q);
            kBuf = backend.AllocateBuffer(kcache);
            vBuf = backend.AllocateBuffer(vcache);
            btBuf = backend.AllocateIntBuffer(blockTable);
            outBuf = backend.PagedAttentionDecode(qBuf, kBuf, vBuf, btBuf, heads, headDim, blockSize, seqLen, scale);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            qBuf?.Dispose(); kBuf?.Dispose(); vBuf?.Dispose(); btBuf?.Dispose(); outBuf?.Dispose();
        }

        Assert.Equal(heads * headDim, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"paged-attn mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }

    [Theory]
    [InlineData(4, 64, 16, 8, 0)]
    [InlineData(2, 128, 8, 12, 5)]
    [InlineData(8, 32, 32, 20, 40)]
    public void PagedAttentionPrefill_MatchesCpuOracle(int heads, int headDim, int blockSize, int numQueries, int startPos)
    {
        if (!EnsureReady()) return;
        var backend = _backend!;
        var rng = new Random(0xB99 + heads + numQueries + startPos);

        int maxKeyLen = startPos + numQueries;
        int numLogicalBlocks = (maxKeyLen + blockSize - 1) / blockSize;
        int maxBlocks = numLogicalBlocks + 5;
        int poolLen = maxBlocks * blockSize * heads * headDim;

        var kcache = new float[poolLen];
        var vcache = new float[poolLen];
        for (int i = 0; i < poolLen; i++) { kcache[i] = (float)(rng.NextDouble() * 2 - 1); vcache[i] = (float)(rng.NextDouble() * 2 - 1); }

        var q = new float[numQueries * heads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);

        var blockTable = new int[numLogicalBlocks];
        var pool = new System.Collections.Generic.List<int>();
        for (int b = 0; b < maxBlocks; b++) pool.Add(b);
        for (int b = 0; b < numLogicalBlocks; b++) { int pick = rng.Next(pool.Count); blockTable[b] = pool[pick]; pool.RemoveAt(pick); }

        float scale = 1.0f / MathF.Sqrt(headDim);

        var expected = new float[numQueries * heads * headDim];
        for (int qi = 0; qi < numQueries; qi++)
        {
            int keyLen = startPos + qi + 1;
            for (int h = 0; h < heads; h++)
            {
                int qbase = (qi * heads + h) * headDim;
                var logits = new float[keyLen];
                float max = float.NegativeInfinity;
                for (int t = 0; t < keyLen; t++)
                {
                    int blk = blockTable[t / blockSize], pos = t % blockSize;
                    long kbase = (((long)blk * blockSize + pos) * heads + h) * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[qbase + d] * kcache[kbase + d];
                    logits[t] = dot * scale;
                    if (logits[t] > max) max = logits[t];
                }
                float denom = 0f;
                for (int t = 0; t < keyLen; t++) { logits[t] = MathF.Exp(logits[t] - max); denom += logits[t]; }
                for (int t = 0; t < keyLen; t++)
                {
                    int blk = blockTable[t / blockSize], pos = t % blockSize;
                    long kbase = (((long)blk * blockSize + pos) * heads + h) * headDim;
                    float p = logits[t] / denom;
                    for (int d = 0; d < headDim; d++) expected[qbase + d] += p * vcache[kbase + d];
                }
            }
        }

        float[] actual;
        IGpuBuffer? qBuf = null, kBuf = null, vBuf = null, btBuf = null, outBuf = null;
        try
        {
            qBuf = backend.AllocateBuffer(q);
            kBuf = backend.AllocateBuffer(kcache);
            vBuf = backend.AllocateBuffer(vcache);
            btBuf = backend.AllocateIntBuffer(blockTable);
            outBuf = backend.PagedAttentionPrefill(qBuf, kBuf, vBuf, btBuf, heads, headDim, blockSize, numQueries, startPos, scale);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            qBuf?.Dispose(); kBuf?.Dispose(); vBuf?.Dispose(); btBuf?.Dispose(); outBuf?.Dispose();
        }

        Assert.Equal(numQueries * heads * headDim, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"prefill mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }

    [Theory]
    [InlineData(8, 2, 64, 16, 40)]
    [InlineData(4, 4, 32, 8, 20)]
    [InlineData(8, 1, 64, 32, 50)]
    public void PagedAttentionDecodeGqa_MatchesCpuOracle(int heads, int kvHeads, int headDim, int blockSize, int seqLen)
    {
        if (!EnsureReady()) return;
        var backend = _backend!;
        var rng = new Random(0xC55 + heads + kvHeads + seqLen);
        int numLogicalBlocks = (seqLen + blockSize - 1) / blockSize;
        int maxBlocks = numLogicalBlocks + 5;
        int poolLen = maxBlocks * blockSize * kvHeads * headDim;
        var kcache = new float[poolLen];
        var vcache = new float[poolLen];
        for (int i = 0; i < poolLen; i++) { kcache[i] = (float)(rng.NextDouble() * 2 - 1); vcache[i] = (float)(rng.NextDouble() * 2 - 1); }
        var q = new float[heads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        var blockTable = new int[numLogicalBlocks];
        var pool = new System.Collections.Generic.List<int>();
        for (int b = 0; b < maxBlocks; b++) pool.Add(b);
        for (int b = 0; b < numLogicalBlocks; b++) { int pick = rng.Next(pool.Count); blockTable[b] = pool[pick]; pool.RemoveAt(pick); }
        float scale = 1.0f / MathF.Sqrt(headDim);

        var expected = new float[heads * headDim];
        for (int h = 0; h < heads; h++)
        {
            int kvHead = h / (heads / kvHeads);
            var logits = new float[seqLen];
            float max = float.NegativeInfinity;
            for (int t = 0; t < seqLen; t++)
            {
                int blk = blockTable[t / blockSize], pos = t % blockSize;
                long kbase = (((long)blk * blockSize + pos) * kvHeads + kvHead) * headDim;
                float dot = 0f;
                for (int d = 0; d < headDim; d++) dot += q[h * headDim + d] * kcache[kbase + d];
                logits[t] = dot * scale;
                if (logits[t] > max) max = logits[t];
            }
            float denom = 0f;
            for (int t = 0; t < seqLen; t++) { logits[t] = MathF.Exp(logits[t] - max); denom += logits[t]; }
            for (int t = 0; t < seqLen; t++)
            {
                int blk = blockTable[t / blockSize], pos = t % blockSize;
                long kbase = (((long)blk * blockSize + pos) * kvHeads + kvHead) * headDim;
                float p = logits[t] / denom;
                for (int d = 0; d < headDim; d++) expected[h * headDim + d] += p * vcache[kbase + d];
            }
        }

        float[] actual;
        IGpuBuffer? qBuf = null, kBuf = null, vBuf = null, btBuf = null, outBuf = null;
        try
        {
            qBuf = backend.AllocateBuffer(q);
            kBuf = backend.AllocateBuffer(kcache);
            vBuf = backend.AllocateBuffer(vcache);
            btBuf = backend.AllocateIntBuffer(blockTable);
            outBuf = backend.PagedAttentionDecodeGqa(qBuf, kBuf, vBuf, btBuf, heads, kvHeads, headDim, blockSize, seqLen, scale);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            qBuf?.Dispose(); kBuf?.Dispose(); vBuf?.Dispose(); btBuf?.Dispose(); outBuf?.Dispose();
        }

        Assert.Equal(heads * headDim, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"gqa-decode mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }

    [Theory]
    [InlineData(8, 2, 64, 16, 8, 0)]
    [InlineData(8, 1, 32, 8, 12, 5)]
    public void PagedAttentionPrefillGqa_MatchesCpuOracle(int heads, int kvHeads, int headDim, int blockSize, int numQueries, int startPos)
    {
        if (!EnsureReady()) return;
        var backend = _backend!;
        var rng = new Random(0xD66 + heads + kvHeads + numQueries + startPos);
        int maxKeyLen = startPos + numQueries;
        int numLogicalBlocks = (maxKeyLen + blockSize - 1) / blockSize;
        int maxBlocks = numLogicalBlocks + 5;
        int poolLen = maxBlocks * blockSize * kvHeads * headDim;
        var kcache = new float[poolLen];
        var vcache = new float[poolLen];
        for (int i = 0; i < poolLen; i++) { kcache[i] = (float)(rng.NextDouble() * 2 - 1); vcache[i] = (float)(rng.NextDouble() * 2 - 1); }
        var q = new float[numQueries * heads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        var blockTable = new int[numLogicalBlocks];
        var pool = new System.Collections.Generic.List<int>();
        for (int b = 0; b < maxBlocks; b++) pool.Add(b);
        for (int b = 0; b < numLogicalBlocks; b++) { int pick = rng.Next(pool.Count); blockTable[b] = pool[pick]; pool.RemoveAt(pick); }
        float scale = 1.0f / MathF.Sqrt(headDim);

        var expected = new float[numQueries * heads * headDim];
        for (int qi = 0; qi < numQueries; qi++)
        {
            int keyLen = startPos + qi + 1;
            for (int h = 0; h < heads; h++)
            {
                int kvHead = h / (heads / kvHeads);
                int qbase = (qi * heads + h) * headDim;
                var logits = new float[keyLen];
                float max = float.NegativeInfinity;
                for (int t = 0; t < keyLen; t++)
                {
                    int blk = blockTable[t / blockSize], pos = t % blockSize;
                    long kbase = (((long)blk * blockSize + pos) * kvHeads + kvHead) * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[qbase + d] * kcache[kbase + d];
                    logits[t] = dot * scale;
                    if (logits[t] > max) max = logits[t];
                }
                float denom = 0f;
                for (int t = 0; t < keyLen; t++) { logits[t] = MathF.Exp(logits[t] - max); denom += logits[t]; }
                for (int t = 0; t < keyLen; t++)
                {
                    int blk = blockTable[t / blockSize], pos = t % blockSize;
                    long kbase = (((long)blk * blockSize + pos) * kvHeads + kvHead) * headDim;
                    float p = logits[t] / denom;
                    for (int d = 0; d < headDim; d++) expected[qbase + d] += p * vcache[kbase + d];
                }
            }
        }

        float[] actual;
        IGpuBuffer? qBuf = null, kBuf = null, vBuf = null, btBuf = null, outBuf = null;
        try
        {
            qBuf = backend.AllocateBuffer(q);
            kBuf = backend.AllocateBuffer(kcache);
            vBuf = backend.AllocateBuffer(vcache);
            btBuf = backend.AllocateIntBuffer(blockTable);
            outBuf = backend.PagedAttentionPrefillGqa(qBuf, kBuf, vBuf, btBuf, heads, kvHeads, headDim, blockSize, numQueries, startPos, scale);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            qBuf?.Dispose(); kBuf?.Dispose(); vBuf?.Dispose(); btBuf?.Dispose(); outBuf?.Dispose();
        }

        Assert.Equal(numQueries * heads * headDim, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"gqa-prefill mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }
}

#endif
