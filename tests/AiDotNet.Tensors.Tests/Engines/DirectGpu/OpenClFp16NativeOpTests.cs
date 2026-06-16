// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the OpenCL FP16-NATIVE op kernels (issue #558): GELU / ReLU / residual-add over
// half-stored activations, computed in FP32 in-register. Validates the GPU counterpart of the CPU
// FP16-native emit — the activation buffers the kernel reads AND writes are genuinely Half (2 bytes).

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Exercises <see cref="OpenClBackend.Fp16Gelu"/> / <see cref="OpenClBackend.Fp16Relu"/> /
/// <see cref="OpenClBackend.Fp16Add"/> on genuinely HALF buffers (FP32 → ConvertToFp16 → kernel →
/// ConvertToFp32 → readback) against a CPU reference computed from the same FP16-rounded inputs, so the
/// only residual error is FP32 round-off. Honors <c>AIDOTNET_REQUIRE_GPU_TESTS=1</c> (throws instead of
/// skipping when a GPU is required) per the no-silent-pass guideline.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class OpenClFp16NativeOpTests : IDisposable
{
    private readonly OpenClBackend _backend;
    private readonly bool _ready;
    private readonly Exception _initException;

    public OpenClFp16NativeOpTests()
    {
        try
        {
            _backend = new OpenClBackend();
            _ready = _backend.IsAvailable && _backend.SupportsFp16NativeOps;
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
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException(
                "GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but the OpenCL FP16-native op kernels were unavailable.",
                _initException);
        return false;
    }

    private static float ToFp16AndBack(float v) => (float)(System.Half)v;

    private static float[] RandomVec(int n, int seed)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 4.0); // span FP16 normal range
        return d;
    }

    private IGpuBuffer ToFp16(float[] x)
    {
        using var f32 = _backend.AllocateBuffer(x);
        var f16 = _backend.AllocateBuffer(x.Length); // over-allocated as floats; harmless
        _backend.ConvertToFp16(f32, f16, x.Length);
        return f16;
    }

    private float[] FromFp16(IGpuBuffer f16, int n)
    {
        using var f32 = _backend.AllocateBuffer(n);
        _backend.ConvertToFp32(f16, f32, n);
        return _backend.DownloadBuffer(f32);
    }

    private static void AssertClose(float[] expected, float[] actual, double absTol, double relTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"non-finite at {i}: {actual[i]}");
            double tol = absTol + relTol * Math.Abs(expected[i]);
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol,
                $"mismatch at [{i}]: expected {expected[i]}, got {actual[i]} (tol {tol}).");
        }
    }

    [SkippableTheory]
    [InlineData(256)]
    [InlineData(2048)]
    [InlineData(1023)]
    public void Fp16Gelu_OnHalfBuffers_MatchesFp32(int n)
    {
        Skip.If(!EnsureReady(), "OpenCL FP16-native ops not available.");
        var x = RandomVec(n, 11 + n);
        var expected = new float[n];
        for (int i = 0; i < n; i++)
        {
            float xi = ToFp16AndBack(x[i]);
            float z = 0.7978845608028654f * (xi + 0.044715f * xi * xi * xi);
            expected[i] = 0.5f * xi * (1f + MathF.Tanh(z));
        }

        using var inB = ToFp16(x);
        using var outB = _backend.AllocateBuffer(n);
        _backend.Fp16Gelu(inB, outB, n);
        AssertClose(expected, FromFp16(outB, n), absTol: 3e-2, relTol: 4e-2);
    }

    [SkippableTheory]
    [InlineData(256)]
    [InlineData(2048)]
    [InlineData(1023)]
    public void Fp16Relu_OnHalfBuffers_MatchesFp32(int n)
    {
        Skip.If(!EnsureReady(), "OpenCL FP16-native ops not available.");
        var x = RandomVec(n, 23 + n);
        var expected = new float[n];
        for (int i = 0; i < n; i++) { float xi = ToFp16AndBack(x[i]); expected[i] = xi > 0 ? xi : 0; }

        using var inB = ToFp16(x);
        using var outB = _backend.AllocateBuffer(n);
        _backend.Fp16Relu(inB, outB, n);
        AssertClose(expected, FromFp16(outB, n), absTol: 1e-3, relTol: 1e-3);
    }

    [SkippableTheory]
    [InlineData(256)]
    [InlineData(2048)]
    public void Fp16Add_OnHalfBuffers_MatchesFp32(int n)
    {
        Skip.If(!EnsureReady(), "OpenCL FP16-native ops not available.");
        var a = RandomVec(n, 31 + n);
        var b = RandomVec(n, 97 + n);
        var expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = ToFp16AndBack(a[i]) + ToFp16AndBack(b[i]);

        using var aB = ToFp16(a);
        using var bB = ToFp16(b);
        using var outB = _backend.AllocateBuffer(n);
        _backend.Fp16Add(aB, bB, outB, n);
        AssertClose(expected, FromFp16(outB, n), absTol: 2e-2, relTol: 2e-2);
    }

    [SkippableFact]
    public void Fp16NativeOps_RejectNonPositiveLength()
    {
        Skip.If(!EnsureReady(), "OpenCL FP16-native ops not available.");
        using var dummy = _backend.AllocateBuffer(4);
        Assert.Throws<ArgumentOutOfRangeException>(() => _backend.Fp16Gelu(dummy, dummy, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => _backend.Fp16Relu(dummy, dummy, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => _backend.Fp16Add(dummy, dummy, dummy, 0));
    }
}

#endif
