// Copyright (c) AiDotNet. All rights reserved.
// Issue #338 Phase G.5: verify BF16 GEMM via Intel MKL.

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

public class Issue338BF16GemmTests
{
    private readonly ITestOutputHelper _output;
    public Issue338BF16GemmTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// Smoke test: BF16 GEMM produces results within BF16 precision (~3
    /// decimal digits) of the FP32 reference for a small example.
    /// </summary>
    [Fact]
    public void BF16Gemm_SmallMatrix_MatchesFp32WithinTolerance()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_BLAS_PROVIDER") != "mkl-bf16")
        {
            _output.WriteLine("Skip: AIDOTNET_BLAS_PROVIDER != mkl-bf16.");
            return;
        }
        if (!BlasProvider.IsMklBf16Available)
        {
            _output.WriteLine("Skip: BlasProvider.IsMklBf16Available is false (mkl_rt.3 not loadable).");
            return;
        }

        // Small 4x4 GEMM. C = A @ B
        const int M = 4, K = 4, N = 4;
        var a = new float[] {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        };
        var b = new float[] {
            0.1f, 0.2f, 0.3f, 0.4f,
            0.5f, 0.6f, 0.7f, 0.8f,
            0.9f, 1.0f, 1.1f, 1.2f,
            1.3f, 1.4f, 1.5f, 1.6f,
        };
        var cBf16 = new float[M * N];
        var cFp32 = new float[M * N];

        // BF16 path
        var aBf16 = new ushort[M * K];
        var bBf16 = new ushort[K * N];
        unsafe
        {
            fixed (float* aSrc = a)
            fixed (ushort* aDst = aBf16)
                BlasProvider.Fp32ToBf16Bulk(aSrc, aDst, M * K);
            fixed (float* bSrc = b)
            fixed (ushort* bDst = bBf16)
                BlasProvider.Fp32ToBf16Bulk(bSrc, bDst, K * N);
        }
        Assert.True(BlasProvider.TryGemmBf16(M, N, K,
            aBf16, 0, K, false, bBf16, 0, N, false, cBf16, 0, N));

        // FP32 reference
        Assert.True(BlasProvider.TryGemm(M, N, K, a, 0, K, b, 0, N, cFp32, 0, N));

        // BF16 has 7-bit mantissa → ~3 decimal digits of precision. Allow
        // up to 1% relative error per element.
        for (int i = 0; i < M * N; i++)
        {
            float relErr = MathF.Abs(cBf16[i] - cFp32[i]) / MathF.Max(MathF.Abs(cFp32[i]), 1e-6f);
            Assert.True(relErr < 0.01f,
                $"Element {i}: BF16={cBf16[i]:F6} vs FP32={cFp32[i]:F6}, rel err={relErr:E3}");
        }
        _output.WriteLine($"BF16 GEMM correctness: {M}x{K}x{N} all elements within 1% of FP32 reference.");
    }

    /// <summary>
    /// Per-call speed comparison at d=128 transformer shapes. BF16 GEMM
    /// should be 1.5-2× faster than FP32 on AVX-512-BF16 capable CPUs.
    /// On older CPUs MKL falls back internally to FP32; correctness still
    /// holds but no speedup.
    /// </summary>
    [Fact]
    [Trait("Category", "Perf")]
    public void BF16Gemm_AB_AtTransformerShapes()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_GATES") != "1") return;
        if (Environment.ProcessorCount < 16) return;
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64) return;
        if (Environment.GetEnvironmentVariable("AIDOTNET_BLAS_PROVIDER") != "mkl-bf16") return;
        if (!BlasProvider.IsMklBf16Available)
        {
            _output.WriteLine("Skip: BlasProvider.IsMklBf16Available is false.");
            return;
        }

        const int M = 2048, K = 128, N = 128;
        const int iters = 200;

        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var cFp32 = new float[M * N];
        var cBf16 = new float[M * N];

        var aBf16 = new ushort[M * K];
        var bBf16 = new ushort[K * N];
        unsafe
        {
            fixed (float* aSrc = a)
            fixed (ushort* aDst = aBf16)
                BlasProvider.Fp32ToBf16Bulk(aSrc, aDst, M * K);
            fixed (float* bSrc = b)
            fixed (ushort* bDst = bBf16)
                BlasProvider.Fp32ToBf16Bulk(bSrc, bDst, K * N);
        }

        // Warmup
        for (int i = 0; i < 5; i++)
        {
            BlasProvider.TryGemm(M, N, K, a, 0, K, b, 0, N, cFp32, 0, N);
            BlasProvider.TryGemmBf16(M, N, K, aBf16, 0, K, false, bBf16, 0, N, false, cBf16, 0, N);
        }

        // FP32 measurement — assert success so a dispatch miss can't be
        // reported as a misleadingly fast "GEMM" timing.
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            Assert.True(BlasProvider.TryGemm(M, N, K, a, 0, K, b, 0, N, cFp32, 0, N),
                "BlasProvider.TryGemm failed during timed FP32 measurement");
        sw.Stop();
        double fp32Ms = sw.Elapsed.TotalMilliseconds / iters;

        // BF16 measurement (including A and B already-converted; that's
        // the realistic in-context cost when weights are pre-cached).
        sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            Assert.True(BlasProvider.TryGemmBf16(M, N, K, aBf16, 0, K, false, bBf16, 0, N, false, cBf16, 0, N),
                "BlasProvider.TryGemmBf16 failed during timed BF16 measurement");
        sw.Stop();
        double bf16Ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"# Issue #338 BF16 GEMM at M={M}, K={K}, N={N}");
        _output.WriteLine($"# FP32 sgemm: {fp32Ms:F3} ms/call");
        _output.WriteLine($"# BF16 gemm: {bf16Ms:F3} ms/call");
        _output.WriteLine($"# ratio BF16/FP32: {bf16Ms / fp32Ms:F2}× ({(fp32Ms / bf16Ms):F2}× speedup)");
    }
}
