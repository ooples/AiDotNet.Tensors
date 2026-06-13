// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the OpenCL IGpuHalfPrecisionBackend implementation
// (issue #560): FP16 GEMM (C = A·B) on the GPU instead of a scalar CPU fallback.

// System.Half (the FP16 oracle) only exists on .NET 5+, and the FP16 GPU path is
// exercised only on the modern TFM anyway; net471 builds skip this file.
#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Exercises <see cref="IGpuHalfPrecisionBackend.GemmFp16In32fOut"/> and
/// <see cref="IGpuHalfPrecisionBackend.Hgemm"/> on the OpenCL backend against a
/// CPU reference computed from the SAME FP16-rounded inputs, so the only
/// residual error is FP32 accumulation order — proving the GPU FP16 kernel is
/// numerically correct, not just non-crashing.
/// <para>
/// Tests run on any OpenCL device (the kernels use the core <c>vload_half</c>
/// built-in, not <c>cl_khr_fp16</c>). When a GPU is genuinely unavailable they
/// skip — unless <c>AIDOTNET_REQUIRE_GPU_TESTS=1</c> is set, in which case they
/// throw so a CI lane that is supposed to have a GPU surfaces the failure loudly
/// rather than passing as a no-op.
/// </para>
/// </summary>
public sealed class OpenClHalfPrecisionGemmTests : IDisposable
{
    private readonly OpenClBackend _backend;
    private readonly bool _ready;
    private readonly Exception _initException;

    public OpenClHalfPrecisionGemmTests()
    {
        try
        {
            _backend = new OpenClBackend();
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
                "GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but the OpenCL " +
                "backend or its FP16 GEMM kernels were unavailable.",
                _initException);
        }

        return false;
    }

    // Round-trip a value through IEEE-754 binary16 exactly as the GPU does, so
    // the CPU oracle and the GPU kernel start from identical FP16 inputs.
    private static float ToFp16AndBack(float value) => (float)(System.Half)value;

    private static float[] RandomMatrix(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0); // [-1, 1)
        return data;
    }

    // C = A·B in FP32 from FP16-rounded inputs (row-major).
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
        // Upload FP32, then convert to FP16 on the device (cl_khr_fp16 path) —
        // the same two ConvertToFp16 calls the engine's MatrixMultiply makes.
        var aFp32 = _backend.AllocateBuffer(a);
        var bFp32 = _backend.AllocateBuffer(b);

        // FP16 buffers hold M*K / K*N halfs (2 bytes each). Allocating that many
        // floats over-allocates (4 bytes each) which is harmless and mirrors the
        // CUDA engine path's AllocateOutputBuffer(M*K) sizing.
        var aFp16 = _backend.AllocateBuffer(m * k);
        var bFp16 = _backend.AllocateBuffer(k * n);
        _backend.ConvertToFp16(aFp32, aFp16, m * k);
        _backend.ConvertToFp16(bFp32, bFp16, k * n);

        aFp32.Dispose();
        bFp32.Dispose();
        return (aFp16, bFp16);
    }

    private static void AssertClose(float[] expected, float[] actual, double absTol, double relTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(expected[i] - actual[i]);
            double tol = absTol + relTol * Math.Abs(expected[i]);
            worst = Math.Max(worst, diff - tol);
            Assert.True(diff <= tol,
                $"FP16 GEMM mismatch at [{i}]: expected {expected[i]}, got {actual[i]} " +
                $"(|diff|={diff}, tol={tol}).");
        }
    }

    [SkippableTheory]
    [InlineData(16, 16, 16)]    // exactly one tile
    [InlineData(64, 48, 96)]    // aligned multi-tile
    [InlineData(37, 53, 71)]    // ragged (exercises out-of-range tile guards)
    [InlineData(128, 128, 256)] // larger, deeper K
    public void GemmFp16In32fOut_MatchesCpuReference(int m, int n, int k)
    {
        Skip.If(!EnsureReady(), "OpenCL FP16 GEMM not available on this system.");

        var a = RandomMatrix(m, k, seed: 1234 + m + k);
        var b = RandomMatrix(k, n, seed: 5678 + n + k);
        var expected = CpuReferenceFromFp16(a, b, m, n, k);

        var (aFp16, bFp16) = UploadFp16Inputs(a, b, m, n, k);
        using var cBuf = _backend.AllocateBuffer(m * n);
        try
        {
            ((IGpuHalfPrecisionBackend)_backend).GemmFp16In32fOut(aFp16, bFp16, cBuf, m, n, k);
            var actual = _backend.DownloadBuffer(cBuf);
            // Inputs already FP16-rounded on both sides; remaining error is just
            // FP32 accumulation order across the tiled K loop.
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
        Skip.If(!EnsureReady(), "OpenCL FP16 GEMM not available on this system.");

        const int m = 64, n = 64, k = 128;
        var a = RandomMatrix(m, k, seed: 24);
        var b = RandomMatrix(k, n, seed: 42);
        var expected = CpuReferenceFromFp16(a, b, m, n, k);

        var (aFp16, bFp16) = UploadFp16Inputs(a, b, m, n, k);
        // FP16 output buffer (M*N halfs); over-allocate as floats.
        using var cFp16 = _backend.AllocateBuffer(m * n);
        // Convert the FP16 result back to FP32 for comparison.
        using var cFp32 = _backend.AllocateBuffer(m * n);
        try
        {
            ((IGpuHalfPrecisionBackend)_backend).Hgemm(aFp16, bFp16, cFp16, m, n, k);
            _backend.ConvertToFp32(cFp16, cFp32, m * n);
            var actual = _backend.DownloadBuffer(cFp32);
            // The result itself is rounded to FP16, so the dominant error is the
            // final half-rounding of the accumulated value (~1e-3 relative).
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
        Skip.If(!EnsureReady(), "OpenCL FP16 GEMM not available on this system.");

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
