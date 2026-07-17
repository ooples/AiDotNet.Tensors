// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the OpenCL paged-attention decode kernel (P1): attention that gathers K/V
// through a block table, validated against a standard-attention CPU oracle. Skips when no OpenCL
// device is available, unless AIDOTNET_REQUIRE_GPU_TESTS=1.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class PagedAttentionOpenClTests : IDisposable
{
    private readonly OpenClBackend? _backend;
    private readonly bool _ready;
    private readonly Exception? _initException;

    public PagedAttentionOpenClTests()
    {
        try { _backend = new OpenClBackend(); _ready = _backend.IsAvailable; }
        catch (Exception ex) { _initException = ex; _ready = false; }
    }

    public void Dispose() => _backend?.Dispose();

    private bool EnsureReady()
    {
        if (_ready) return true;
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException("GPU tests required but OpenCL was unavailable.", _initException);
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

        // Shuffled physical block ids (exercises the indirection).
        var blockTable = new int[numLogicalBlocks];
        var pool = new System.Collections.Generic.List<int>();
        for (int b = 0; b < maxBlocks; b++) pool.Add(b);
        for (int b = 0; b < numLogicalBlocks; b++) { int pick = rng.Next(pool.Count); blockTable[b] = pool[pick]; pool.RemoveAt(pick); }

        float scale = 1.0f / MathF.Sqrt(headDim);

        // CPU oracle: standard attention gathering K/V via the block table.
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
            Assert.True(Math.Abs(e - a) <= tol, $"paged-attn mismatch at {i}: expected {e}, got {a} (tol {tol}, heads={heads}, seqLen={seqLen})");
        }
    }
}

#endif
