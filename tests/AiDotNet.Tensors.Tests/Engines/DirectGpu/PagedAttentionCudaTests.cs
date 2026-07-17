// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the CUDA paged-attention decode kernel (P1), validated against a standard-
// attention CPU oracle. Skips without a CUDA device (this dev box is AMD); runs on NVIDIA / CI.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class PagedAttentionCudaTests : IDisposable
{
    private readonly CudaBackend? _backend;
    private readonly bool _ready;

    public PagedAttentionCudaTests()
    {
        try { _backend = new CudaBackend(); _ready = _backend.IsAvailable; }
        catch { _ready = false; }
    }

    public void Dispose() => _backend?.Dispose();

    [Fact]
    public void Probe_CudaAvailability()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_CUDA") != "1") return;
        Assert.True(_ready, "CUDA backend NOT available on this host");
    }

    [Theory]
    [InlineData(4, 64, 16, 40)]
    [InlineData(2, 128, 8, 20)]
    [InlineData(8, 32, 32, 100)]
    public void PagedAttentionDecode_MatchesCpuOracle(int heads, int headDim, int blockSize, int seqLen)
    {
        if (!_ready) return;
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

        var qBuf = backend.AllocateBuffer(q);
        var kBuf = backend.AllocateBuffer(kcache);
        var vBuf = backend.AllocateBuffer(vcache);
        var btBuf = backend.AllocateIntBuffer(blockTable);
        var outBuf = backend.PagedAttentionDecode(qBuf, kBuf, vBuf, btBuf, heads, headDim, blockSize, seqLen, scale);
        var actual = backend.DownloadBuffer(outBuf);
        qBuf.Dispose(); kBuf.Dispose(); vBuf.Dispose(); btBuf.Dispose(); outBuf.Dispose();

        Assert.Equal(heads * headDim, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"paged-attn mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }
}

#endif
