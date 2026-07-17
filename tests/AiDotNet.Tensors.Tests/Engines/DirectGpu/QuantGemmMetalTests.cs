// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the Metal weight-only fused dequant-GEMM (P0), validated against a CPU
// reference implementing the same contract as FusedDequantMatmulKernels. Skips when no Metal device
// is available (i.e. non-macOS hosts); runs on macOS / CI.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class QuantGemmMetalTests : IDisposable
{
    private const int M = 8, K = 128, N = 64, KN = K * N;

    private readonly MetalBackend? _backend;
    private readonly bool _ready;

    public QuantGemmMetalTests()
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

    private static float[] RandomAct(Random rng)
    {
        var a = new float[M * K];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    private static (float[] scales, int scaleCount, int gsArg) MakeScales(Random rng, int groupSize)
    {
        if (groupSize <= 0) return (new[] { 0.03f }, 1, KN);
        int totalGroups = (KN + groupSize - 1) / groupSize;
        var s = new float[totalGroups];
        for (int g = 0; g < totalGroups; g++) s[g] = 0.01f + (float)(rng.NextDouble() * 0.05);
        return (s, totalGroups, groupSize);
    }

    private static float[] Reference(float[] act, float[] decoded, float[] scales, int gsArg, int scaleCount)
    {
        var outp = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float acc = 0f;
                if (scaleCount == 1)
                {
                    for (int k = 0; k < K; k++) acc += act[i * K + k] * decoded[k * N + j];
                    acc *= scales[0];
                }
                else
                {
                    for (int k = 0; k < K; k++)
                    {
                        int flat = k * N + j;
                        acc += act[i * K + k] * decoded[flat] * scales[flat / gsArg];
                    }
                }
                outp[i * N + j] = acc;
            }
        return outp;
    }

    private void AssertClose(float[] expected, float[] actual, string tag)
    {
        Assert.Equal(M * N, actual.Length);
        for (int idx = 0; idx < expected.Length; idx++)
        {
            float e = expected[idx], a = actual[idx];
            float tol = 1e-2f + 1e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol, $"{tag} mismatch at {idx}: expected {e}, got {a} (tol {tol})");
        }
    }

    [Theory]
    [InlineData(-127, 128, 0)]
    [InlineData(-127, 128, 64)]
    [InlineData(-8, 8, 0)]
    [InlineData(-8, 8, 64)]
    public void DequantGemmInt_MatchesCpuOracle(int lo, int hi, int groupSize)
    {
        if (!_ready) return;
        var backend = _backend!;
        var rng = new Random(0x3000 + lo + groupSize);
        var act = RandomAct(rng);
        var w = new int[KN];
        for (int i = 0; i < KN; i++) w[i] = rng.Next(lo, hi);
        var (scales, scaleCount, gsArg) = MakeScales(rng, groupSize);
        var decoded = new float[KN];
        for (int i = 0; i < KN; i++) decoded[i] = w[i];
        var expected = Reference(act, decoded, scales, gsArg, scaleCount);

        var actBuf = backend.AllocateBuffer(act);
        var scaleBuf = backend.AllocateBuffer(scales);
        var wBuf = backend.AllocateIntBuffer(w);
        var outBuf = backend.DequantGemmInt(actBuf, wBuf, scaleBuf, M, K, N, gsArg, scaleCount);
        var actual = backend.DownloadBuffer(outBuf);
        actBuf.Dispose(); scaleBuf.Dispose(); wBuf.Dispose(); outBuf.Dispose();
        AssertClose(expected, actual, $"int(lo={lo},gs={groupSize})");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(64)]
    public void DequantGemmFp8E4M3_MatchesCpuOracle(int groupSize)
    {
        if (!_ready) return;
        var backend = _backend!;
        var rng = new Random(0xF800 + groupSize);
        var act = RandomAct(rng);
        var raws = new int[KN];
        var decoded = new float[KN];
        for (int i = 0; i < KN; i++)
        {
            var f8 = Float8E4M3.FromFloat((float)(rng.NextDouble() * 8.0 - 4.0));
            raws[i] = f8.RawValue;
            decoded[i] = f8.ToFloat();
        }
        var (scales, scaleCount, gsArg) = MakeScales(rng, groupSize);
        var expected = Reference(act, decoded, scales, gsArg, scaleCount);

        var actBuf = backend.AllocateBuffer(act);
        var scaleBuf = backend.AllocateBuffer(scales);
        var wBuf = backend.AllocateIntBuffer(raws);
        var outBuf = backend.DequantGemmFp8E4M3(actBuf, wBuf, scaleBuf, M, K, N, gsArg, scaleCount);
        var actual = backend.DownloadBuffer(outBuf);
        actBuf.Dispose(); scaleBuf.Dispose(); wBuf.Dispose(); outBuf.Dispose();
        AssertClose(expected, actual, $"fp8(gs={groupSize})");
    }
}

#endif
