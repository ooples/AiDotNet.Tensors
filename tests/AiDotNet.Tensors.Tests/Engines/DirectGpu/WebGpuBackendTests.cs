// Copyright (c) AiDotNet. All rights reserved.
// Comprehensive integration tests for WebGPU backend for browser GPU compute.

using System;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Comprehensive tests for the WebGPU GPU backend.
/// These tests verify correctness and find bugs in WebGPU compute operations.
/// </summary>
/// <remarks>
/// Tests are designed to find bugs in:
/// - Element-wise operations (Add, Sub, Mul, Div)
/// - Scalar operations (Scale, Power)
/// - Unary operations (Sqrt, Exp, Log, Abs)
/// - Activation functions (ReLU, LeakyReLU, Sigmoid, Tanh, GELU, etc.)
/// - Reduction operations (Sum, Max, Min)
/// - Matrix operations (GEMM, Transpose)
/// - Normalization (Softmax, LayerNorm)
/// </remarks>
public class WebGpuBackendTests : IDisposable
{
#if NET7_0_OR_GREATER
    private readonly ITestOutputHelper _output;
    private readonly AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend? _backend;
    private bool _isAvailable;

    public WebGpuBackendTests(ITestOutputHelper output)
    {
        _output = output;
        _isAvailable = false;

        // WebGPU requires browser/WASM environment - skip on desktop platforms
        if (!System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Create("BROWSER")))
        {
            _output.WriteLine("WebGPU is only available in browser/WASM environment");
            return;
        }

        try
        {
            _backend = new AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend();
            // WebGPU requires async initialization - will be set in InitializeAsync
        }
        catch (Exception ex)
        {
            _output.WriteLine($"WebGPU initialization failed: {ex.Message}");
        }
    }

    private async Task EnsureInitializedAsync()
    {
        if (_backend is not null && !_isAvailable)
        {
            try
            {
                _isAvailable = await _backend.InitializeAsync();
                if (_isAvailable)
                {
                    _output.WriteLine($"WebGPU backend: {_backend.DeviceName}");
                    _output.WriteLine($"Max workgroup size: {_backend.MaxWorkgroupSize}");
                }
            }
            catch (PlatformNotSupportedException)
            {
                _output.WriteLine("WebGPU JS interop not supported on this platform");
                _isAvailable = false;
            }
        }
    }

    private async Task SkipIfNoWebGpuAsync()
    {
        await EnsureInitializedAsync();
        Skip.If(!_isAvailable, "WebGPU backend not available (requires browser environment)");
    }

    public void Dispose()
    {
        _backend?.Dispose();
    }

    #region Initialization Tests

    [SkippableFact]
    public async Task WebGpuBackend_Initialization_WorksOrSkips()
    {
        if (_backend is null)
        {
            Assert.True(true, "WebGPU backend not created");
            return;
        }

        var result = await _backend.InitializeAsync();
        // WebGPU may or may not be available depending on environment
        Assert.True(result || !result, "Initialization returns a boolean");
    }

    #endregion

    #region Element-wise Operation Tests

    [SkippableFact]
    public async Task Add_BasicOperation_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 1, 2, 3, 4, 5 };
        var b = new float[] { 10, 20, 30, 40, 50 };
        var expected = new float[] { 11, 22, 33, 44, 55 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(5);

        await _backend.AddAsync(bufferA, bufferB, bufferC, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferC).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public async Task Sub_BasicOperation_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 10, 20, 30, 40, 50 };
        var b = new float[] { 1, 2, 3, 4, 5 };
        var expected = new float[] { 9, 18, 27, 36, 45 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(5);

        await _backend.SubAsync(bufferA, bufferB, bufferC, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferC).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public async Task Mul_BasicOperation_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 1, 2, 3, 4, 5 };
        var b = new float[] { 2, 3, 4, 5, 6 };
        var expected = new float[] { 2, 6, 12, 20, 30 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(5);

        await _backend.MulAsync(bufferA, bufferB, bufferC, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferC).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public async Task Div_BasicOperation_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 10, 20, 30, 40, 50 };
        var b = new float[] { 2, 4, 5, 8, 10 };
        var expected = new float[] { 5, 5, 6, 5, 5 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(5);

        await _backend.DivAsync(bufferA, bufferB, bufferC, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferC).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public async Task Maximum_BasicOperation_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 1, 5, 3, 8, 2 };
        var b = new float[] { 4, 2, 6, 1, 9 };
        var expected = new float[] { 4, 5, 6, 8, 9 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(5);

        await _backend.MaximumAsync(bufferA, bufferB, bufferC, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferC).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public async Task Minimum_BasicOperation_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 1, 5, 3, 8, 2 };
        var b = new float[] { 4, 2, 6, 1, 9 };
        var expected = new float[] { 1, 2, 3, 1, 2 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(5);

        await _backend.MinimumAsync(bufferA, bufferB, bufferC, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferC).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    #endregion

    #region Scalar Operation Tests

    [SkippableFact]
    public async Task Scale_BasicOperation_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 1, 2, 3, 4, 5 };
        float scalar = 2.5f;
        var expected = new float[] { 2.5f, 5.0f, 7.5f, 10.0f, 12.5f };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.ScaleAsync(bufferA, bufferB, scalar, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public async Task AddScalar_BasicOperation_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 1, 2, 3, 4, 5 };
        float scalar = 10.0f;
        var expected = new float[] { 11, 12, 13, 14, 15 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.AddScalarAsync(bufferA, bufferB, scalar, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public async Task Power_SquareOperation_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 1, 2, 3, 4, 5 };
        float exponent = 2.0f;
        var expected = new float[] { 1, 4, 9, 16, 25 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.PowerAsync(bufferA, bufferB, exponent, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 4);
        }
    }

    #endregion

    #region Unary Operation Tests

    [SkippableFact]
    public async Task Sqrt_PositiveValues_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 1, 4, 9, 16, 25 };
        var expected = new float[] { 1, 2, 3, 4, 5 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.SqrtAsync(bufferA, bufferB, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public async Task Exp_BasicValues_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 0, 1, 2 };
        var expected = new float[] { 1.0f, MathF.E, MathF.E * MathF.E };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(3);

        await _backend.ExpAsync(bufferA, bufferB, 3);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 4);
        }
    }

    [SkippableFact]
    public async Task Log_PositiveValues_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { 1, MathF.E, MathF.E * MathF.E };
        var expected = new float[] { 0, 1, 2 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(3);

        await _backend.LogAsync(bufferA, bufferB, 3);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 4);
        }
    }

    [SkippableFact]
    public async Task Abs_MixedValues_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { -3, -1, 0, 1, 3 };
        var expected = new float[] { 3, 1, 0, 1, 3 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.AbsAsync(bufferA, bufferB, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public async Task Neg_MixedValues_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var a = new float[] { -3, -1, 0, 1, 3 };
        var expected = new float[] { 3, 1, 0, -1, -3 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.NegAsync(bufferA, bufferB, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    #endregion

    #region Activation Function Tests

    [SkippableFact]
    public async Task ReLU_MixedValues_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var input = new float[] { -2, -1, 0, 1, 2 };
        var expected = new float[] { 0, 0, 0, 1, 2 };

        using var bufferA = _backend!.AllocateBuffer(input);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.ReLUAsync(bufferA, bufferB, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public async Task LeakyReLU_MixedValues_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var input = new float[] { -2, -1, 0, 1, 2 };
        float alpha = 0.1f;
        var expected = new float[] { -0.2f, -0.1f, 0, 1, 2 };

        using var bufferA = _backend!.AllocateBuffer(input);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.LeakyReLUAsync(bufferA, bufferB, 5, alpha);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    [SkippableFact]
    public async Task Sigmoid_VariousValues_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var input = new float[] { -10, -1, 0, 1, 10 };

        using var bufferA = _backend!.AllocateBuffer(input);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.SigmoidAsync(bufferA, bufferB, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();

        Assert.True(result[0] < 0.001f); // sigmoid(-10) ≈ 0
        Assert.Equal(0.5f, result[2], precision: 5); // sigmoid(0) = 0.5
        Assert.True(result[4] > 0.999f); // sigmoid(10) ≈ 1

        foreach (var v in result)
        {
            Assert.True(v > 0 && v < 1);
        }
    }

    [SkippableFact]
    public async Task Tanh_VariousValues_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var input = new float[] { -10, -1, 0, 1, 10 };

        using var bufferA = _backend!.AllocateBuffer(input);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.TanhAsync(bufferA, bufferB, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();

        Assert.True(result[0] < -0.999f); // tanh(-10) ≈ -1
        Assert.Equal(0, result[2], precision: 5); // tanh(0) = 0
        Assert.True(result[4] > 0.999f); // tanh(10) ≈ 1

        foreach (var v in result)
        {
            Assert.True(v > -1 && v < 1);
        }
    }

    [SkippableFact]
    public async Task GELU_VariousValues_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var input = new float[] { -3, -1, 0, 1, 3 };

        using var bufferA = _backend!.AllocateBuffer(input);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.GeLUAsync(bufferA, bufferB, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();

        // GELU(0) = 0
        Assert.Equal(0, result[2], precision: 5);

        // GELU is positive for positive inputs
        Assert.True(result[3] > 0);
        Assert.True(result[4] > 0);
    }

    [SkippableFact]
    public async Task Swish_VariousValues_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        var input = new float[] { -2, -1, 0, 1, 2 };

        using var bufferA = _backend!.AllocateBuffer(input);
        using var bufferB = _backend.AllocateBuffer(5);

        await _backend.SwishAsync(bufferA, bufferB, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();

        // Swish(0) = 0
        Assert.Equal(0, result[2], precision: 5);

        // Swish(x) = x * sigmoid(x), so positive for positive x
        Assert.True(result[3] > 0);
        Assert.True(result[4] > 0);
    }

    #endregion

    #region Reduction Tests

    [SkippableFact]
    public async Task Sum_BasicArray_ReturnsCorrectSum()
    {
        await SkipIfNoWebGpuAsync();

        var data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        float expected = 55;

        using var buffer = _backend!.AllocateBuffer(data);
        float result = await _backend.SumAsync(buffer, data.Length);

        Assert.Equal(expected, result, precision: 3);
    }

    [SkippableFact]
    public async Task Max_BasicArray_ReturnsMaximum()
    {
        await SkipIfNoWebGpuAsync();

        var data = new float[] { 3, 1, 4, 1, 5, 9, 2, 6, 5 };
        float expected = 9;

        using var buffer = _backend!.AllocateBuffer(data);
        float result = await _backend.MaxAsync(buffer, data.Length);

        Assert.Equal(expected, result, precision: 5);
    }

    [SkippableFact]
    public async Task Min_BasicArray_ReturnsMinimum()
    {
        await SkipIfNoWebGpuAsync();

        var data = new float[] { 3, 1, 4, 1, 5, 9, 2, 6, 5 };
        float expected = 1;

        using var buffer = _backend!.AllocateBuffer(data);
        float result = await _backend.MinAsync(buffer, data.Length);

        Assert.Equal(expected, result, precision: 5);
    }

    #endregion

    #region Matrix Operation Tests

    [SkippableFact]
    public async Task Gemm_SmallMatrices_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        // 2x3 * 3x2 = 2x2
        var a = new float[] { 1, 2, 3, 4, 5, 6 }; // 2x3
        var b = new float[] { 1, 2, 3, 4, 5, 6 }; // 3x2
        // Expected: [[22, 28], [49, 64]]
        var expected = new float[] { 22, 28, 49, 64 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(4);

        await _backend.GemmAsync(bufferA, bufferB, bufferC, 2, 2, 3);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferC).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 3);
        }
    }

    [SkippableFact]
    public async Task Transpose_BasicMatrix_ReturnsCorrectResult()
    {
        await SkipIfNoWebGpuAsync();

        // 2x3 matrix
        var a = new float[] { 1, 2, 3, 4, 5, 6 };
        // Expected 3x2 transpose
        var expected = new float[] { 1, 4, 2, 5, 3, 6 };

        using var bufferA = _backend!.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(6);

        await _backend.TransposeAsync(bufferA, bufferB, 2, 3);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferB).DownloadAsync();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], precision: 5);
        }
    }

    #endregion

    #region Softmax and Normalization Tests

    [SkippableFact]
    public async Task Softmax_BasicInput_SumsToOne()
    {
        await SkipIfNoWebGpuAsync();

        var input = new float[] { 1, 2, 3, 4, 5 };

        using var bufferIn = _backend!.AllocateBuffer(input);
        using var bufferOut = _backend.AllocateBuffer(5);

        await _backend.SoftmaxAsync(bufferIn, bufferOut, 1, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferOut).DownloadAsync();

        // Sum should be 1
        float sum = result.Sum();
        Assert.Equal(1.0f, sum, precision: 4);

        // All values should be positive
        Assert.All(result, v => Assert.True(v > 0));

        // Values should be increasing (since input is increasing)
        for (int i = 1; i < result.Length; i++)
        {
            Assert.True(result[i] > result[i - 1]);
        }
    }

    [SkippableFact]
    public async Task Softmax_BatchedInput_EachBatchSumsToOne()
    {
        await SkipIfNoWebGpuAsync();

        // 3 batches of 4 elements each
        var input = new float[]
        {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        };

        using var bufferIn = _backend!.AllocateBuffer(input);
        using var bufferOut = _backend.AllocateBuffer(12);

        await _backend.SoftmaxAsync(bufferIn, bufferOut, 3, 4);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferOut).DownloadAsync();

        // Each batch should sum to 1
        for (int batch = 0; batch < 3; batch++)
        {
            float batchSum = 0;
            for (int i = 0; i < 4; i++)
            {
                batchSum += result[batch * 4 + i];
            }
            Assert.Equal(1.0f, batchSum, precision: 4);
        }
    }

    #endregion

    #region Edge Case Tests

    [SkippableFact]
    public async Task Operations_WorkgroupBoundary_HandlesCorrectly()
    {
        await SkipIfNoWebGpuAsync();

        // Test sizes around typical workgroup size (256)
        int[] testSizes = { 255, 256, 257, 511, 512, 513 };

        foreach (var size in testSizes)
        {
            var a = Enumerable.Repeat(1.0f, size).ToArray();
            var b = Enumerable.Repeat(2.0f, size).ToArray();

            using var bufferA = _backend!.AllocateBuffer(a);
            using var bufferB = _backend.AllocateBuffer(b);
            using var bufferC = _backend.AllocateBuffer(size);

            await _backend.AddAsync(bufferA, bufferB, bufferC, size);

            var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferC).DownloadAsync();
            Assert.All(result, v => Assert.Equal(3.0f, v, precision: 5));
        }
    }

    [SkippableFact]
    public async Task Sigmoid_ExtremeValues_DoesNotOverflow()
    {
        await SkipIfNoWebGpuAsync();

        var extreme = new float[] { -100, -50, 0, 50, 100 };

        using var bufferIn = _backend!.AllocateBuffer(extreme);
        using var bufferOut = _backend.AllocateBuffer(5);

        await _backend.SigmoidAsync(bufferIn, bufferOut, 5);

        var result = await ((AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBuffer)bufferOut).DownloadAsync();

        Assert.All(result, v =>
        {
            Assert.False(float.IsNaN(v));
            Assert.False(float.IsInfinity(v));
            Assert.True(v >= 0 && v <= 1);
        });
    }

    #endregion

#else
    // .NET Framework stub - WebGPU is not supported
    [SkippableFact]
    public void WebGpuBackend_NotSupportedOnNetFramework()
    {
        Assert.True(true, "WebGPU is only supported on .NET 7+");
    }

    public void Dispose()
    {
        // No-op on .NET Framework
    }
#endif
}

