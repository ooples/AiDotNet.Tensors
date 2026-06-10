// GPU-residency audit: parity tests for ops that were silently falling back to the
// CpuEngine base on a GPU device (causing host<->device ping-pong). Each new GPU
// override must produce bit-comparable results to the trusted CPU reference across a
// shape sweep. Runs only when a GPU is present; CPU-only hosts early-return (no-op pass).

#if !NETFRAMEWORK
#nullable disable

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class GpuMissingKernelsParityTests : IDisposable
{
    private readonly CpuEngine _cpu = new CpuEngine();
    private readonly DirectGpuTensorEngine _gpu;
    private readonly bool _gpuReady;
    private readonly Exception _gpuInitException;

    public GpuMissingKernelsParityTests()
    {
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _gpuReady = _gpu.IsGpuAvailable;
        }
        catch (Exception ex)
        {
            _gpuInitException = ex;
            _gpuReady = false;
        }
    }

    public void Dispose() => _gpu?.Dispose();

    private bool EnsureGpuReady()
    {
        if (_gpuReady) return true;
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException(
                "GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but DirectGpu init failed or no GPU is available.",
                _gpuInitException);
        return false;
    }

    // float32 GEMM at moderate K accumulates a few 1e-6 of abs error; 1e-3 cleanly
    // separates a correct result from a broken kernel.
    private const float Tol = 1e-3f;

    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var rng = new Random(seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, shape);
    }

    private static void AssertMatch(Tensor<float> gpu, Tensor<float> cpu, string op)
    {
        Assert.Equal(cpu.Shape.ToArray(), gpu.Shape.ToArray());
        var sa = gpu.ToArray();
        var sb = cpu.ToArray();
        double m = 0;
        for (int i = 0; i < sa.Length; i++)
        {
            Assert.False(float.IsNaN(sa[i]) || float.IsInfinity(sa[i]), $"{op}: GPU non-finite {sa[i]} at {i}");
            double e = Math.Abs((double)sa[i] - sb[i]);
            if (e > m) m = e;
        }
        Assert.True(m < Tol, $"{op}: GPU vs CPU max_abs_err {m:E3} exceeded {Tol:E3}");
    }

    public static IEnumerable<object[]> AddMMSizes() => new List<object[]>
    {
        new object[] { 1, 1, 1 },
        new object[] { 4, 4, 4 },
        new object[] { 16, 32, 8 },
        new object[] { 32, 32, 32 },
        new object[] { 33, 17, 65 },
        new object[] { 64, 128, 32 },
        new object[] { 128, 256, 64 },
        new object[] { 1, 256, 256 },
        new object[] { 256, 1, 256 },
        new object[] { 200, 50, 120 },
    };

    [Theory]
    [MemberData(nameof(AddMMSizes))]
    public void TensorAddMM_DefaultAlphaBeta_GpuMatchesCpu(int m, int k, int n)
    {
        if (!EnsureGpuReady()) return;
        var input = Rand(1, m, n);
        var a = Rand(2, m, k);
        var b = Rand(3, k, n);

        var cpu = _cpu.TensorAddMM(input, a, b);
        var gpu = _gpu.TensorAddMM(input, a, b);
        AssertMatch(gpu, cpu, $"TensorAddMM[{m},{k},{n}]");
    }

    [Theory]
    [MemberData(nameof(AddMMSizes))]
    public void TensorAddMM_AlphaBeta_GpuMatchesCpu(int m, int k, int n)
    {
        if (!EnsureGpuReady()) return;
        var input = Rand(11, m, n);
        var a = Rand(12, m, k);
        var b = Rand(13, k, n);
        const float alpha = 0.75f, beta = -1.5f;

        var cpu = _cpu.TensorAddMM(input, a, b, alpha, beta);
        var gpu = _gpu.TensorAddMM(input, a, b, alpha, beta);
        AssertMatch(gpu, cpu, $"TensorAddMM_ab[{m},{k},{n}]");
    }
}
#endif
