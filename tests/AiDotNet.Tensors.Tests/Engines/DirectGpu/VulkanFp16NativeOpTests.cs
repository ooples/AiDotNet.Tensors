// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Tests for the Vulkan FP16-NATIVE op kernels (issue #558): GELU / ReLU / residual-add over packed-half
// activations, computed in FP32 in-register. GPU counterpart of the CPU FP16-native emit.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Verifies <see cref="VulkanBackend.Fp16Gelu"/> / <see cref="VulkanBackend.Fp16Relu"/> /
/// <see cref="VulkanBackend.Fp16Add"/> on genuinely packed-half buffers (FP32 → ConvertToFp16 → kernel →
/// ConvertToFp32 → readback) against a CPU reference from the same FP16-rounded inputs. The kernels are
/// runtime-compiled GLSL, so they only run with libshaderc present; gating on IsGlslCompilerAvailable +
/// SupportsFp16NativeOps keeps the test honest (GPU path or skip, never silent CPU-pass). Honors
/// <c>AIDOTNET_REQUIRE_GPU_TESTS=1</c>.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class VulkanFp16NativeOpTests
{
    private readonly VulkanBackend _backend;
    private readonly bool _ready;

    public VulkanFp16NativeOpTests()
    {
        try
        {
            _backend = VulkanBackend.Instance;
            _ready = _backend.Initialize() && _backend.IsGlslCompilerAvailable && _backend.SupportsFp16NativeOps;
        }
        catch
        {
            _ready = false;
        }
    }

    private bool EnsureReady()
    {
        if (_ready) return true;
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException(
                "GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but the Vulkan FP16-native op kernels were unavailable (Vulkan device + libshaderc required).");
        return false;
    }

    private static float ToFp16AndBack(float v) => (float)(System.Half)v;

    private static float[] RandomVec(int n, int seed)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 4.0);
        return d;
    }

    private IGpuBuffer ToFp16(float[] x)
    {
        using var f32 = _backend.AllocateBuffer(x);
        var f16 = _backend.AllocateBuffer(x.Length);
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
        Skip.If(!EnsureReady(), "Vulkan FP16-native ops not available.");
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
        Skip.If(!EnsureReady(), "Vulkan FP16-native ops not available.");
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
        Skip.If(!EnsureReady(), "Vulkan FP16-native ops not available.");
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

    // ---- Row softmax and layer norm, including the column counts that are not a whole number of words (#1230).

    private static float[] SoftmaxReference(float[] x, int rows, int cols)
    {
        var expected = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            double max = double.NegativeInfinity, sum = 0;
            for (int c = 0; c < cols; c++) max = Math.Max(max, ToFp16AndBack(x[r * cols + c]));
            for (int c = 0; c < cols; c++) sum += Math.Exp(ToFp16AndBack(x[r * cols + c]) - max);
            for (int c = 0; c < cols; c++) expected[r * cols + c] = (float)(Math.Exp(ToFp16AndBack(x[r * cols + c]) - max) / sum);
        }
        return expected;
    }

    [SkippableTheory]
    [InlineData(4, 64)]      // even: the single-dispatch kernel
    [InlineData(1, 1)]
    [InlineData(1, 7)]
    [InlineData(5, 3)]       // odd rows x odd cols: the last word has an unused half
    [InlineData(6, 33)]
    [InlineData(3, 257)]     // more columns than one work-group has invocations
    [InlineData(7, 1023)]
    public void Fp16Softmax_AnyColumnCount_MatchesFp32(int rows, int cols)
    {
        Skip.If(!EnsureReady(), "Vulkan FP16-native ops not available.");
        int n = rows * cols;
        var x = RandomVec(n, 401 + n);
        var expected = SoftmaxReference(x, rows, cols);

        using var inB = ToFp16(x);
        using var outB = _backend.AllocateBuffer(n);
        _backend.Fp16Softmax(inB, outB, rows, cols);
        var actual = FromFp16(outB, n);
        AssertClose(expected, actual, absTol: 2e-3, relTol: 2e-2);
        for (int r = 0; r < rows; r++)
        {
            double rowSum = 0;
            for (int c = 0; c < cols; c++) rowSum += actual[r * cols + c];
            Assert.True(Math.Abs(rowSum - 1.0) < 2e-2, $"row {r} sums to {rowSum}");
        }
    }

    [SkippableTheory]
    [InlineData(4, 64)]
    [InlineData(1, 3)]
    [InlineData(5, 3)]
    [InlineData(6, 33)]
    [InlineData(3, 257)]
    [InlineData(7, 1023)]
    public void Fp16LayerNorm_AnyColumnCount_MatchesFp32(int rows, int cols)
    {
        Skip.If(!EnsureReady(), "Vulkan FP16-native ops not available.");
        const float eps = 1e-5f;
        int n = rows * cols;
        var x = RandomVec(n, 811 + n);
        var gamma = RandomVec(cols, 5 + cols);
        var beta = RandomVec(cols, 9 + cols);
        var expected = new float[n];
        var expectedMean = new float[rows];
        var expectedVar = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            double mean = 0, variance = 0;
            for (int c = 0; c < cols; c++) mean += ToFp16AndBack(x[r * cols + c]);
            mean /= cols;
            for (int c = 0; c < cols; c++) { double d = ToFp16AndBack(x[r * cols + c]) - mean; variance += d * d; }
            variance /= cols;
            expectedMean[r] = (float)mean;
            expectedVar[r] = (float)variance;
            for (int c = 0; c < cols; c++)
                expected[r * cols + c] = (float)((ToFp16AndBack(x[r * cols + c]) - mean) / Math.Sqrt(variance + eps)
                    * ToFp16AndBack(gamma[c]) + ToFp16AndBack(beta[c]));
        }

        using var inB = ToFp16(x);
        using var gammaB = ToFp16(gamma);
        using var betaB = ToFp16(beta);
        using var outB = _backend.AllocateBuffer(n);
        using var meanB = _backend.AllocateBuffer(rows);
        using var varB = _backend.AllocateBuffer(rows);
        _backend.Fp16LayerNorm(inB, gammaB, betaB, outB, meanB, varB, rows, cols, eps);
        AssertClose(expected, FromFp16(outB, n), absTol: 2e-2, relTol: 2e-2);
        // The statistics come back as FP32 for the backward pass.
        AssertClose(expectedMean, _backend.DownloadBuffer(meanB).AsSpan(0, rows).ToArray(), absTol: 1e-4, relTol: 1e-4);
        AssertClose(expectedVar, _backend.DownloadBuffer(varB).AsSpan(0, rows).ToArray(), absTol: 1e-4, relTol: 1e-4);
    }

    [SkippableFact]
    public void Fp16Softmax_OddColumns_LeavesTheNeighbouringRowsHalfAlone()
    {
        // With 3 columns, row 0 ends in the low half of word 1 and row 1 starts in its high half. A kernel that
        // wrote a row at a time would have two writers for that word. Rows with very different contents make a
        // lost or misplaced half obvious.
        Skip.If(!EnsureReady(), "Vulkan FP16-native ops not available.");
        const int rows = 64, cols = 3;
        var x = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) x[r * cols + c] = (r % 2 == 0) ? c * 3f : -c * 3f;
        var expected = SoftmaxReference(x, rows, cols);

        using var inB = ToFp16(x);
        using var outB = _backend.AllocateBuffer(rows * cols);
        for (int repeat = 0; repeat < 20; repeat++)
        {
            _backend.Fp16Softmax(inB, outB, rows, cols);
            AssertClose(expected, FromFp16(outB, rows * cols), absTol: 2e-3, relTol: 2e-2);
        }
    }

    [SkippableFact]
    public void Fp16NativeOps_RejectNonPositiveLength()
    {
        Skip.If(!EnsureReady(), "Vulkan FP16-native ops not available.");
        using var dummy = _backend.AllocateBuffer(4);
        Assert.Throws<ArgumentOutOfRangeException>(() => _backend.Fp16Gelu(dummy, dummy, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => _backend.Fp16Relu(dummy, dummy, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => _backend.Fp16Add(dummy, dummy, dummy, 0));
    }
}

#endif
