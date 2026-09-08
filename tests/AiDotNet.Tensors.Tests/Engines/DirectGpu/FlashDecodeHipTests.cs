// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the HIP fused decode-attention kernel (P2, FlashDecoding), validated against a
// standard-attention CPU oracle. Skips without a ROCm device; runs on a ROCm host / CI.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class FlashDecodeHipTests : IClassFixture<HipBackendTestFixture>
{
    private readonly HipBackendTestFixture _fixture;
    private bool IsReady => _fixture.IsAvailable;
    private HipBackend Backend => _fixture.Backend;

    public FlashDecodeHipTests(HipBackendTestFixture fixture)
    {
        _fixture = fixture;
    }

    [SkippableFact]
    public void Probe_HipAvailability()
    {
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_HIP") != "1",
            "HIP availability is asserted only in the required-HIP lane.");
        Assert.True(IsReady, "HIP/ROCm backend NOT available on this host");
        Assert.Empty(Backend.KernelCompilationFailures);
    }

    private bool EnsureReady()
    {
        if (IsReady) return true;
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_HIP"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException("GPU tests required but HIP/ROCm was unavailable.");
        return false;
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

    private void RunAndCompare(int heads, int kvHeads, int headDim, int seqLen, int splits)
    {
        var backend = Backend;
        var rng = new Random(0xFDE + heads + kvHeads + seqLen + splits);
        int stepStride = kvHeads * headDim;
        var k = new float[seqLen * stepStride];
        var v = new float[seqLen * stepStride];
        for (int i = 0; i < k.Length; i++) { k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }
        var q = new float[heads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        float scale = 1.0f / MathF.Sqrt(headDim);

        float[] actual;
        IGpuBuffer? qBuf = null, kBuf = null, vBuf = null, outBuf = null;
        try
        {
            qBuf = backend.AllocateBuffer(q);
            kBuf = backend.AllocateBuffer(k);
            vBuf = backend.AllocateBuffer(v);
            outBuf = backend.FlashDecode(qBuf, kBuf, vBuf, heads, kvHeads, headDim, seqLen, scale, splits);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            qBuf?.Dispose(); kBuf?.Dispose(); vBuf?.Dispose(); outBuf?.Dispose();
        }

        var expected = DecodeOracle(q, k, v, heads, kvHeads, headDim, seqLen, scale);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = 2e-3f + 2e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"flash-decode mismatch at {i}: expected {e}, got {a} (tol {tol})");
        }
    }

    [SkippableTheory]
    [InlineData(4, 64, 40, 8)]
    [InlineData(8, 32, 100, 8)]
    [InlineData(4, 64, 7, 8)]
    [InlineData(4, 64, 64, 1)]
    [InlineData(6, 48, 77, 0)]   // splits=0 -> internal default derivation
    public void MhaDecode_MatchesOracle(int heads, int headDim, int seqLen, int splits)
    {
        Skip.If(!EnsureReady(), "HIP/ROCm backend is unavailable.");
        RunAndCompare(heads, heads, headDim, seqLen, splits);
    }

    [SkippableTheory]
    [InlineData(8, 2, 64, 50, 8)]
    [InlineData(8, 1, 64, 40, 8)]
    public void GqaDecode_MatchesOracle(int heads, int kvHeads, int headDim, int seqLen, int splits)
    {
        Skip.If(!EnsureReady(), "HIP/ROCm backend is unavailable.");
        RunAndCompare(heads, kvHeads, headDim, seqLen, splits);
    }
}

#endif
