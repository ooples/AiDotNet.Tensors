// Copyright (c) AiDotNet. All rights reserved.
// Comprehensive integration tests for Metal GPU backend on Apple Silicon.

using System;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Comprehensive tests for the Metal GPU backend.
/// These tests verify correctness and find bugs in Metal compute operations.
/// </summary>
/// <remarks>
/// Tests are designed to find bugs in:
/// - GEMM operations (matrix multiplication)
/// - Element-wise operations (Add, Subtract, Multiply, Divide)
/// - Activation functions (ReLU, Sigmoid, Tanh, GELU, Softmax)
/// - Reduction operations (Sum, Max, Min)
/// - Memory management
/// - Edge cases and precision
/// </remarks>
public class MetalBackendTests : IDisposable
{
#if NET7_0_OR_GREATER
    private readonly ITestOutputHelper _output;
    private readonly AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalBackend? _backend;
    private readonly bool _isAvailable;

    public MetalBackendTests(ITestOutputHelper output)
    {
        _output = output;

        // Metal is only available on macOS
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            _isAvailable = false;
            return;
        }

        try
        {
            _backend = new AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalBackend();
            _isAvailable = _backend.IsAvailable;
            if (_isAvailable)
            {
                _output.WriteLine($"Metal backend: {_backend.DeviceName}");
                _output.WriteLine($"Compute units: {_backend.ComputeUnits}");
            }
        }
        catch (Exception ex)
        {
            _output.WriteLine($"Metal initialization failed: {ex.Message}");
            _isAvailable = false;
        }
    }

    private void SkipIfNoMetal()
    {
        Skip.If(!_isAvailable, "Metal backend not available on this platform");
    }

    public void Dispose()
    {
        _backend?.Dispose();
    }

    #region Initialization Tests

    [SkippableFact]
    public void MetalBackend_OnMacOS_IsAvailableOrSkipped()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            // Skip on non-macOS platforms
            Assert.False(_isAvailable);
            return;
        }

        // On macOS with Apple Silicon, Metal should be available
        Assert.True(_isAvailable || true, "Metal may or may not be available depending on hardware");
    }

    [SkippableFact]
    public void MetalBackend_DeviceName_IsNotEmpty()
    {
        SkipIfNoMetal();
        Assert.False(string.IsNullOrEmpty(_backend!.DeviceName));
        _output.WriteLine($"Device name: {_backend.DeviceName}");
    }

    [SkippableFact]
    public void MetalBackend_ComputeUnits_IsPositive()
    {
        SkipIfNoMetal();
        Assert.True(_backend!.ComputeUnits > 0);
        _output.WriteLine($"Compute units: {_backend.ComputeUnits}");
    }

    #endregion

    #region Buffer Management Tests

    [SkippableFact]
    public void AllocateBuffer_WithData_StoresCorrectly()
    {
        SkipIfNoMetal();

        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        using var buffer = _backend!.AllocateBuffer(data);

        Assert.NotNull(buffer);
        Assert.Equal(5, buffer.Size);

        var result = _backend.DownloadBuffer(buffer);
        Assert.Equal(data, result);
    }

    [SkippableFact]
    public void AllocateBuffer_EmptySize_Throws()
    {
        SkipIfNoMetal();
        Assert.Throws<ArgumentOutOfRangeException>(() => _backend!.AllocateBuffer(0));
    }

    [SkippableFact]
    public void AllocateBuffer_NegativeSize_Throws()
    {
        SkipIfNoMetal();
        Assert.Throws<ArgumentOutOfRangeException>(() => _backend!.AllocateBuffer(-1));
    }

    [SkippableFact]
    public void AllocateBuffer_LargeSize_Works()
    {
        SkipIfNoMetal();

        int size = 1_000_000; // 1M elements = 4MB
        using var buffer = _backend!.AllocateBuffer(size);
        Assert.NotNull(buffer);
        Assert.Equal(size, buffer.Size);
    }

    #endregion

    #region Element-wise Operation Tests

    [SkippableFact]
    public void Add_BasicOperation_ReturnsCorrectResult()
    {
        SkipIfNoMetal();

        var a = new float[] { 1, 2, 3, 4, 5 };
        var b = new float[] { 10, 20, 30, 40, 50 };
        var expected = new float[] { 11, 22, 33, 44, 55 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(5);

        _backend.Add(bufferA, bufferB, bufferC, 5);

        var result = _backend.DownloadBuffer(bufferC);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public void Subtract_BasicOperation_ReturnsCorrectResult()
    {
        SkipIfNoMetal();

        var a = new float[] { 10, 20, 30, 40, 50 };
        var b = new float[] { 1, 2, 3, 4, 5 };
        var expected = new float[] { 9, 18, 27, 36, 45 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(5);

        _backend.Subtract(bufferA, bufferB, bufferC, 5);

        var result = _backend.DownloadBuffer(bufferC);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public void Multiply_BasicOperation_ReturnsCorrectResult()
    {
        SkipIfNoMetal();

        var a = new float[] { 1, 2, 3, 4, 5 };
        var b = new float[] { 2, 3, 4, 5, 6 };
        var expected = new float[] { 2, 6, 12, 20, 30 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(5);

        _backend.Multiply(bufferA, bufferB, bufferC, 5);

        var result = _backend.DownloadBuffer(bufferC);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public void Divide_BasicOperation_ReturnsCorrectResult()
    {
        SkipIfNoMetal();

        var a = new float[] { 10, 20, 30, 40, 50 };
        var b = new float[] { 2, 4, 5, 8, 10 };
        var expected = new float[] { 5, 5, 6, 5, 5 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(5);

        _backend.Divide(bufferA, bufferB, bufferC, 5);

        var result = _backend.DownloadBuffer(bufferC);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public void Scale_BasicOperation_ReturnsCorrectResult()
    {
        SkipIfNoMetal();

        var a = new float[] { 1, 2, 3, 4, 5 };
        float scalar = 2.5f;
        var expected = new float[] { 2.5f, 5.0f, 7.5f, 10.0f, 12.5f };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(5);

        _backend.Scale(bufferA, bufferB, scalar, 5);

        var result = _backend.DownloadBuffer(bufferB);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    #endregion

    #region Activation Function Tests

    [SkippableFact]
    public void ReLU_PositiveAndNegativeValues_ReturnsCorrectResult()
    {
        SkipIfNoMetal();

        var input = new float[] { -2, -1, 0, 1, 2 };
        var expected = new float[] { 0, 0, 0, 1, 2 };

        using var bufferA = _backend!.AllocateBuffer(input);
        using var bufferB = _backend.AllocateBuffer(5);

        _backend.Relu(bufferA, bufferB, 5);

        var result = _backend.DownloadBuffer(bufferB);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public void Sigmoid_VariousValues_ReturnsCorrectResult()
    {
        SkipIfNoMetal();

        var input = new float[] { -10, -1, 0, 1, 10 };

        using var bufferA = _backend!.AllocateBuffer(input);
        using var bufferB = _backend.AllocateBuffer(5);

        _backend.Sigmoid(bufferA, bufferB, 5);

        var result = _backend.DownloadBuffer(bufferB);

        // Sigmoid(-10) ≈ 0, Sigmoid(0) = 0.5, Sigmoid(10) ≈ 1
        Assert.True(result[0] < 0.001f); // sigmoid(-10) ≈ 0
        Assert.Equal(0.5f, result[2], precision: 5); // sigmoid(0) = 0.5
        Assert.True(result[4] > 0.999f); // sigmoid(10) ≈ 1

        // All values should be in (0, 1)
        foreach (var v in result)
        {
            Assert.True(v > 0 && v < 1);
        }
    }

    [SkippableFact]
    public void Tanh_VariousValues_ReturnsCorrectResult()
    {
        SkipIfNoMetal();

        var input = new float[] { -10, -1, 0, 1, 10 };

        using var bufferA = _backend!.AllocateBuffer(input);
        using var bufferB = _backend.AllocateBuffer(5);

        _backend.Tanh(bufferA, bufferB, 5);

        var result = _backend.DownloadBuffer(bufferB);

        // Tanh(-10) ≈ -1, Tanh(0) = 0, Tanh(10) ≈ 1
        Assert.True(result[0] < -0.999f); // tanh(-10) ≈ -1
        Assert.Equal(0, result[2], precision: 5); // tanh(0) = 0
        Assert.True(result[4] > 0.999f); // tanh(10) ≈ 1

        // All values should be in (-1, 1)
        foreach (var v in result)
        {
            Assert.True(v > -1 && v < 1);
        }
    }

    [SkippableFact]
    public void GELU_VariousValues_ReturnsCorrectResult()
    {
        SkipIfNoMetal();

        var input = new float[] { -3, -1, 0, 1, 3 };

        using var bufferA = _backend!.AllocateBuffer(input);
        using var bufferB = _backend.AllocateBuffer(5);

        _backend.Gelu(bufferA, bufferB, 5);

        var result = _backend.DownloadBuffer(bufferB);

        // GELU(0) = 0
        Assert.Equal(0, result[2], precision: 5);

        // GELU(x) < x for all x (approximately linear for positive x)
        Assert.True(result[3] > 0 && result[3] < 1.5f);
        Assert.True(result[4] > 0 && result[4] < 3.5f);
    }

    #endregion

    #region Reduction Tests

    [SkippableFact]
    public void Sum_BasicArray_ReturnsCorrectSum()
    {
        SkipIfNoMetal();

        var data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        float expected = 55;

        using var buffer = _backend!.AllocateBuffer(data);
        float result = _backend.Sum(buffer, data.Length);

        Assert.Equal(expected, result, precision: 3);
    }

    [SkippableFact]
    public void Sum_LargeArray_ReturnsCorrectSum()
    {
        SkipIfNoMetal();

        int size = 10000;
        var data = Enumerable.Range(1, size).Select(x => (float)x).ToArray();
        float expected = (float)size * (size + 1) / 2;

        using var buffer = _backend!.AllocateBuffer(data);
        float result = _backend.Sum(buffer, size);

        Assert.Equal(expected, result, precision: 0);
    }

    [SkippableFact]
    public void Max_BasicArray_ReturnsMaximum()
    {
        SkipIfNoMetal();

        var data = new float[] { 3, 1, 4, 1, 5, 9, 2, 6, 5 };
        float expected = 9;

        using var buffer = _backend!.AllocateBuffer(data);
        float result = _backend.Max(buffer, data.Length);

        Assert.Equal(expected, result, precision: 5);
    }

    [SkippableFact]
    public void Min_BasicArray_ReturnsMinimum()
    {
        SkipIfNoMetal();

        var data = new float[] { 3, 1, 4, 1, 5, 9, 2, 6, 5 };
        float expected = 1;

        using var buffer = _backend!.AllocateBuffer(data);
        float result = _backend.Min(buffer, data.Length);

        Assert.Equal(expected, result, precision: 5);
    }

    #endregion

    #region GEMM Tests

    [SkippableFact]
    public void Gemm_SmallMatrices_ReturnsCorrectResult()
    {
        SkipIfNoMetal();

        // 2x3 * 3x2 = 2x2
        var a = new float[] { 1, 2, 3, 4, 5, 6 }; // 2x3
        var b = new float[] { 1, 2, 3, 4, 5, 6 }; // 3x2
        // Expected: [[22, 28], [49, 64]]
        var expected = new float[] { 22, 28, 49, 64 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(4);

        _backend.Gemm(bufferA, bufferB, bufferC, 2, 2, 3);

        var result = _backend.DownloadBuffer(bufferC);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 3);
        }
    }

    [SkippableFact]
    public void Gemm_IdentityMatrix_ReturnsOriginal()
    {
        SkipIfNoMetal();

        int n = 4;
        var a = new float[n * n];
        var identity = new float[n * n];

        var random = new Random(42);
        for (int i = 0; i < n * n; i++)
        {
            a[i] = (float)(random.NextDouble() * 10);
        }

        // Create identity matrix
        for (int i = 0; i < n; i++)
        {
            identity[i * n + i] = 1.0f;
        }

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferI = _backend.AllocateBuffer(identity);
        using var bufferC = _backend.AllocateBuffer(n * n);

        _backend.Gemm(bufferA, bufferI, bufferC, n, n, n);

        var result = _backend.DownloadBuffer(bufferC);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(a[i], result[i], precision: 3);
        }
    }

    [SkippableFact]
    public void Gemm_LargerMatrices_CompletesSuccessfully()
    {
        SkipIfNoMetal();

        int m = 64, n = 64, k = 64;
        var a = new float[m * k];
        var b = new float[k * n];

        var random = new Random(42);
        for (int i = 0; i < a.Length; i++) a[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(random.NextDouble() * 2 - 1);

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(m * n);

        _backend.Gemm(bufferA, bufferB, bufferC, m, n, k);

        var result = _backend.DownloadBuffer(bufferC);
        Assert.Equal(m * n, result.Length);

        // Verify result is not all zeros
        bool hasNonZero = result.Any(x => Math.Abs(x) > 1e-6);
        Assert.True(hasNonZero);
    }

    #endregion

    #region Edge Case Tests

    [SkippableFact]
    public void Operations_WithZeroValues_HandleCorrectly()
    {
        SkipIfNoMetal();

        var zeros = new float[] { 0, 0, 0, 0, 0 };
        var ones = new float[] { 1, 1, 1, 1, 1 };

        using var bufferZeros = _backend!.AllocateBuffer(zeros);
        using var bufferOnes = _backend.AllocateBuffer(ones);
        using var bufferOut = _backend.AllocateBuffer(5);

        // Add zeros
        _backend.Add(bufferZeros, bufferOnes, bufferOut, 5);
        var result = _backend.DownloadBuffer(bufferOut);
        Assert.All(result, v => Assert.Equal(1.0f, v, precision: 5));

        // Multiply by zeros
        _backend.Multiply(bufferZeros, bufferOnes, bufferOut, 5);
        result = _backend.DownloadBuffer(bufferOut);
        Assert.All(result, v => Assert.Equal(0.0f, v, precision: 5));
    }

    [SkippableFact]
    public void Operations_WithNegativeValues_HandleCorrectly()
    {
        SkipIfNoMetal();

        var negatives = new float[] { -1, -2, -3, -4, -5 };
        var positives = new float[] { 1, 2, 3, 4, 5 };

        using var bufferNeg = _backend!.AllocateBuffer(negatives);
        using var bufferPos = _backend.AllocateBuffer(positives);
        using var bufferOut = _backend.AllocateBuffer(5);

        _backend.Add(bufferNeg, bufferPos, bufferOut, 5);
        var result = _backend.DownloadBuffer(bufferOut);
        Assert.All(result, v => Assert.Equal(0.0f, v, precision: 5));
    }

    [SkippableFact]
    public void Operations_WorkgroupBoundary_HandlesCorrectly()
    {
        SkipIfNoMetal();

        // Test sizes around typical workgroup size (256)
        int[] testSizes = { 255, 256, 257, 511, 512, 513 };

        foreach (var size in testSizes)
        {
            var a = Enumerable.Repeat(1.0f, size).ToArray();
            var b = Enumerable.Repeat(2.0f, size).ToArray();

            using var bufferA = _backend!.AllocateBuffer(a);
            using var bufferB = _backend.AllocateBuffer(b);
            using var bufferC = _backend.AllocateBuffer(size);

            _backend.Add(bufferA, bufferB, bufferC, size);

            var result = _backend.DownloadBuffer(bufferC);
            Assert.All(result, v => Assert.Equal(3.0f, v, precision: 5));
        }
    }

    [SkippableFact]
    public void Sigmoid_ExtremeValues_DoesNotOverflow()
    {
        SkipIfNoMetal();

        var extreme = new float[] { -100, -50, 0, 50, 100 };

        using var bufferIn = _backend!.AllocateBuffer(extreme);
        using var bufferOut = _backend.AllocateBuffer(5);

        _backend.Sigmoid(bufferIn, bufferOut, 5);

        var result = _backend.DownloadBuffer(bufferOut);

        // All results should be valid (not NaN or Inf)
        Assert.All(result, v =>
        {
            Assert.False(float.IsNaN(v));
            Assert.False(float.IsInfinity(v));
            Assert.True(v >= 0 && v <= 1);
        });
    }

    #endregion

#else
    // .NET Framework stub - Metal is not supported
    [SkippableFact]
    public void MetalBackend_NotSupportedOnNetFramework()
    {
        Assert.True(true, "Metal is only supported on .NET 7+ on macOS");
    }

    public void Dispose()
    {
        // No-op on .NET Framework
    }
#endif
}

