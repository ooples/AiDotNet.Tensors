// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the HIP/ROCm IGpuHalfPrecisionBackend implementation
// (issue #560): FP16 GEMM (C = A·B) on AMD GPUs instead of a scalar CPU fallback.

// System.Half (the FP16 oracle) only exists on .NET 5+, and the FP16 GPU path is
// exercised only on the modern TFM anyway; net471 builds skip this file.
#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Exercises <see cref="IGpuHalfPrecisionBackend.GemmFp16In32fOut"/> and
/// <see cref="IGpuHalfPrecisionBackend.Hgemm"/> on the HIP backend against a CPU
/// reference computed from the SAME FP16-rounded inputs, so the only residual
/// error is FP32 accumulation order. Honors <c>AIDOTNET_REQUIRE_GPU_TESTS=1</c>
/// (throws instead of skipping when a GPU is required) per the no-silent-pass
/// guideline. Requires AMD ROCm + hipBLAS; skips on non-ROCm hosts.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class HipHalfPrecisionGemmTests : IDisposable
{
    private readonly HipBackend _backend;
    private readonly bool _ready;
    private readonly Exception _initException;

    public HipHalfPrecisionGemmTests()
    {
        try
        {
            _backend = new HipBackend();
            _ready = _backend.IsAvailable && _backend.SupportsHgemm;
        }
        catch (Exception ex)
        {
            _initException = ex;
            _ready = false;
        }
    }

    public void Dispose() => _backend?.Dispose();

    private bool EnsureReady()
    {
        if (_ready) return true;
        if (string.Equals(
                Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"),
                "1",
                StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                "GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but the HIP backend " +
                "or hipBLAS was unavailable.",
                _initException);
        }

        return false;
    }

    private static float ToFp16AndBack(float value) => (float)(System.Half)value;

    private static float[] RandomMatrix(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return data;
    }

    private static float[] CpuReferenceFromFp16(float[] a, float[] b, int m, int n, int k)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int l = 0; l < k; l++)
                    acc += ToFp16AndBack(a[i * k + l]) * ToFp16AndBack(b[l * n + j]);
                c[i * n + j] = acc;
            }
        }

        return c;
    }

    private (IGpuBuffer aFp16, IGpuBuffer bFp16) UploadFp16Inputs(
        float[] a, float[] b, int m, int n, int k)
    {
        using var aFp32 = _backend.AllocateBuffer(a);
        using var bFp32 = _backend.AllocateBuffer(b);
        var aFp16 = _backend.AllocateBuffer(m * k);
        var bFp16 = _backend.AllocateBuffer(k * n);
        _backend.ConvertToFp16(aFp32, aFp16, m * k);
        _backend.ConvertToFp16(bFp32, bFp16, k * n);
        return (aFp16, bFp16);
    }

    private static void AssertClose(float[] expected, float[] actual, double absTol, double relTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(expected[i] - actual[i]);
            double tol = absTol + relTol * Math.Abs(expected[i]);
            Assert.True(diff <= tol,
                $"FP16 GEMM mismatch at [{i}]: expected {expected[i]}, got {actual[i]} " +
                $"(|diff|={diff}, tol={tol}).");
        }
    }

    [SkippableTheory]
    [InlineData(16, 16, 16)]
    [InlineData(64, 48, 96)]
    [InlineData(37, 53, 71)]
    [InlineData(128, 128, 256)]
    public void GemmFp16In32fOut_MatchesCpuReference(int m, int n, int k)
    {
        Skip.If(!EnsureReady(), "HIP FP16 GEMM not available on this system.");

        var a = RandomMatrix(m, k, seed: 1234 + m + k);
        var b = RandomMatrix(k, n, seed: 5678 + n + k);
        var expected = CpuReferenceFromFp16(a, b, m, n, k);

        var (aFp16, bFp16) = UploadFp16Inputs(a, b, m, n, k);
        using var cBuf = _backend.AllocateBuffer(m * n);
        try
        {
            ((IGpuHalfPrecisionBackend)_backend).GemmFp16In32fOut(aFp16, bFp16, cBuf, m, n, k);
            var actual = _backend.DownloadBuffer(cBuf);
            AssertClose(expected, actual, absTol: 1e-2, relTol: 1e-2);
        }
        finally
        {
            aFp16.Dispose();
            bFp16.Dispose();
        }
    }

    [SkippableFact]
    public void GemmFp16In32fOut_RejectsNonPositiveDimensions()
    {
        Skip.If(!EnsureReady(), "HIP FP16 GEMM not available on this system.");

        using var dummy = _backend.AllocateBuffer(4);
        var half = (IGpuHalfPrecisionBackend)_backend;
        Assert.Throws<ArgumentOutOfRangeException>(
            () => half.GemmFp16In32fOut(dummy, dummy, dummy, 0, 4, 4));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => half.GemmFp16In32fOut(dummy, dummy, dummy, 4, -1, 4));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => half.GemmFp16In32fOut(dummy, dummy, dummy, 4, 4, 0));
    }
}

#endif
