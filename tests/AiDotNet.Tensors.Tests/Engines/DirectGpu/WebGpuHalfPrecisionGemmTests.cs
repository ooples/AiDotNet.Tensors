// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the WebGPU IGpuHalfPrecisionBackend implementation
// (issue #560): FP16 GEMM (C = A·B) on the GPU instead of a scalar CPU fallback.


// WebGpuBackend only exists on .NET 7+; net471 excludes this test.
#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Exercises <see cref="IGpuHalfPrecisionBackend.GemmFp16In32fOut"/> and
/// <see cref="IGpuHalfPrecisionBackend.Hgemm"/> on the WebGPU backend.
/// <para>
/// WebGPU models FP16 as f32 with a truncated mantissa, so the oracle downloads
/// the exact truncated f32 operands the GPU multiplies and recomputes the
/// reference from those — the only residual error is f32 accumulation order,
/// proving the GPU FP16 GEMM is numerically correct rather than just
/// non-crashing. Honors <c>AIDOTNET_REQUIRE_GPU_TESTS=1</c> (throws instead of
/// skipping when a GPU is required) per the no-silent-pass guideline.
/// </para>
/// </summary>
public sealed class WebGpuHalfPrecisionGemmTests : IDisposable
{
    private readonly WebGpuBackend _backend;
    private readonly bool _ready;
    private readonly Exception _initException;

    public WebGpuHalfPrecisionGemmTests()
    {
        try
        {
            _backend = new WebGpuBackend();
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
                "GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but the WebGPU " +
                "backend was unavailable.",
                _initException);
        }

        return false;
    }

    private static float[] RandomMatrix(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0); // [-1, 1)
        return data;
    }

    // C = A·B in f32 from the EXACT (FP16-truncated) operands the GPU used.
    private static float[] CpuReference(float[] a, float[] b, int m, int n, int k)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int l = 0; l < k; l++)
                    acc += a[i * k + l] * b[l * n + j];
                c[i * n + j] = acc;
            }
        }

        return c;
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

    // Upload FP32, truncate to FP16 precision on the device, and return both the
    // FP16 buffers and the EXACT truncated host values the GPU will multiply.
    private (IGpuBuffer aFp16, IGpuBuffer bFp16, float[] aTrunc, float[] bTrunc) PrepareFp16Inputs(
        float[] a, float[] b, int m, int n, int k)
    {
        using var aFp32 = _backend.AllocateBuffer(a);
        using var bFp32 = _backend.AllocateBuffer(b);
        var aFp16 = _backend.AllocateBuffer(m * k);
        var bFp16 = _backend.AllocateBuffer(k * n);
        _backend.ConvertToFp16(aFp32, aFp16, m * k);
        _backend.ConvertToFp16(bFp32, bFp16, k * n);

        var aTrunc = _backend.DownloadBuffer(aFp16);
        var bTrunc = _backend.DownloadBuffer(bFp16);
        return (aFp16, bFp16, aTrunc, bTrunc);
    }

    [SkippableTheory]
    [InlineData(16, 16, 16)]
    [InlineData(64, 48, 96)]
    [InlineData(37, 53, 71)]
    [InlineData(128, 128, 256)]
    public void GemmFp16In32fOut_MatchesCpuReference(int m, int n, int k)
    {
        Skip.If(!EnsureReady(), "WebGPU FP16 GEMM not available on this system.");

        var a = RandomMatrix(m, k, seed: 1234 + m + k);
        var b = RandomMatrix(k, n, seed: 5678 + n + k);

        var (aFp16, bFp16, aTrunc, bTrunc) = PrepareFp16Inputs(a, b, m, n, k);
        using var cBuf = _backend.AllocateBuffer(m * n);
        try
        {
            ((IGpuHalfPrecisionBackend)_backend).GemmFp16In32fOut(aFp16, bFp16, cBuf, m, n, k);
            var actual = _backend.DownloadBuffer(cBuf);
            var expected = CpuReference(aTrunc, bTrunc, m, n, k);
            AssertClose(expected, actual, absTol: 1e-2, relTol: 1e-2);
        }
        finally
        {
            aFp16.Dispose();
            bFp16.Dispose();
        }
    }

    [SkippableFact]
    public void Hgemm_Fp16Output_MatchesCpuReferenceWithinHalfPrecision()
    {
        Skip.If(!EnsureReady(), "WebGPU FP16 GEMM not available on this system.");

        const int m = 64, n = 64, k = 128;
        var a = RandomMatrix(m, k, seed: 24);
        var b = RandomMatrix(k, n, seed: 42);

        var (aFp16, bFp16, aTrunc, bTrunc) = PrepareFp16Inputs(a, b, m, n, k);
        using var cFp16 = _backend.AllocateBuffer(m * n);
        try
        {
            ((IGpuHalfPrecisionBackend)_backend).Hgemm(aFp16, bFp16, cFp16, m, n, k);
            // WebGPU FP16 output is already f32-storage (truncated), so download directly.
            var actual = _backend.DownloadBuffer(cFp16);
            var expected = CpuReference(aTrunc, bTrunc, m, n, k);
            // The result is rounded to FP16 precision, so the dominant error is the
            // final mantissa truncation of the accumulated value.
            AssertClose(expected, actual, absTol: 5e-2, relTol: 3e-2);
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
        Skip.If(!EnsureReady(), "WebGPU FP16 GEMM not available on this system.");

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
