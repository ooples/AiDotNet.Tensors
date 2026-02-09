// Copyright (c) AiDotNet. All rights reserved.
// Comprehensive integration tests for Vulkan GPU backend.
// Goal: Find as many bugs as possible through edge cases and stress testing.

using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Integration tests for the Vulkan GPU backend.
/// Tests are designed to expose bugs in:
/// - Memory allocation/deallocation
/// - Data transfer (CPU to GPU and back)
/// - Shader execution correctness
/// - Numerical precision
/// - Edge cases and boundary conditions
/// - Thread safety
/// </summary>
public class VulkanBackendTests : IDisposable
{
    private readonly bool _isVulkanAvailable;
    private readonly VulkanBackend? _backend;

    public VulkanBackendTests()
    {
        try
        {
            _backend = VulkanBackend.Instance;
            _isVulkanAvailable = _backend.Initialize();
        }
        catch
        {
            _isVulkanAvailable = false;
        }
    }

    public void Dispose()
    {
        // Don't dispose singleton
    }

    private void SkipIfNoVulkan()
    {
        if (!_isVulkanAvailable)
        {
            throw new SkipException("Vulkan not available on this system");
        }
    }

    #region Initialization Tests

    [Fact]
    public void Initialize_CanBeCalledMultipleTimes_DoesNotCrash()
    {
        SkipIfNoVulkan();

        // Bug pattern: Double initialization causing resource leaks or crashes
        for (int i = 0; i < 10; i++)
        {
            var result = _backend!.Initialize();
            Assert.True(result);
        }
    }

    [Fact]
    public void IsAvailable_ReturnsConsistentValue()
    {
        SkipIfNoVulkan();

        // Bug pattern: Race condition in availability check
        var results = new bool[100];
        Parallel.For(0, 100, i =>
        {
            results[i] = _backend!.IsAvailable;
        });

        Assert.All(results, r => Assert.True(r));
    }

    [Fact]
    public void DeviceName_IsNotNullOrEmpty()
    {
        SkipIfNoVulkan();

        // Bug pattern: Null pointer in device name extraction
        Assert.False(string.IsNullOrEmpty(_backend!.DeviceName));
    }

    [Fact]
    public void VendorName_IsNotNullOrEmpty()
    {
        SkipIfNoVulkan();

        Assert.False(string.IsNullOrEmpty(_backend!.VendorName));
    }

    #endregion

    #region Vector Addition Tests

    [Fact]
    public void Add_SingleElement_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        // Bug pattern: Off-by-one errors with minimal sizes
        var a = new float[] { 5.0f };
        var b = new float[] { 3.0f };
        var result = new float[1];

        _backend!.Add(a, b, result);

        Assert.Equal(8.0f, result[0], 5);
    }

    [Fact]
    public void Add_TwoElements_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        var a = new float[] { 1.0f, 2.0f };
        var b = new float[] { 3.0f, 4.0f };
        var result = new float[2];

        _backend!.Add(a, b, result);

        Assert.Equal(4.0f, result[0], 5);
        Assert.Equal(6.0f, result[1], 5);
    }

    [Fact]
    public void Add_ExactlyWorkgroupSize_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        // Bug pattern: Boundary at workgroup size (256)
        const int size = 256;
        var a = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(0, size).Select(i => (float)i * 2).ToArray();
        var result = new float[size];

        _backend!.Add(a, b, result);

        for (int i = 0; i < size; i++)
        {
            Assert.Equal(i + i * 2, result[i], 5);
        }
    }

    [Fact]
    public void Add_OneMoreThanWorkgroupSize_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        // Bug pattern: First element of second workgroup incorrect
        const int size = 257;
        var a = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(0, size).Select(i => 1.0f).ToArray();
        var result = new float[size];

        _backend!.Add(a, b, result);

        for (int i = 0; i < size; i++)
        {
            Assert.Equal(i + 1.0f, result[i], 5);
        }
    }

    [Fact]
    public void Add_OneLessThanWorkgroupSize_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        // Bug pattern: Last element not computed
        const int size = 255;
        var a = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(0, size).Select(i => 1.0f).ToArray();
        var result = new float[size];

        _backend!.Add(a, b, result);

        for (int i = 0; i < size; i++)
        {
            Assert.Equal(i + 1.0f, result[i], 5);
        }
    }

    [Fact]
    public void Add_MultipleWorkgroups_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        // Bug pattern: Workgroup boundaries cause errors
        const int size = 1024; // 4 workgroups
        var a = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(0, size).Select(i => (float)(size - i)).ToArray();
        var result = new float[size];

        _backend!.Add(a, b, result);

        for (int i = 0; i < size; i++)
        {
            Assert.Equal(size, result[i], 5);
        }
    }

    [Fact]
    public void Add_NonPowerOfTwo_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        // Bug pattern: Non-power-of-two sizes not handled
        const int size = 1000;
        var a = Enumerable.Range(0, size).Select(i => 1.5f).ToArray();
        var b = Enumerable.Range(0, size).Select(i => 2.5f).ToArray();
        var result = new float[size];

        _backend!.Add(a, b, result);

        Assert.All(result, r => Assert.Equal(4.0f, r, 5));
    }

    [Fact]
    public void Add_LargeArray_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        // Bug pattern: Buffer size limits
        const int size = 1_000_000;
        var a = Enumerable.Range(0, size).Select(i => 1.0f).ToArray();
        var b = Enumerable.Range(0, size).Select(i => 2.0f).ToArray();
        var result = new float[size];

        _backend!.Add(a, b, result);

        Assert.All(result, r => Assert.Equal(3.0f, r, 5));
    }

    [Fact]
    public void Add_ZeroValues_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        var a = new float[] { 0.0f, 0.0f, 0.0f };
        var b = new float[] { 0.0f, 0.0f, 0.0f };
        var result = new float[3];

        _backend!.Add(a, b, result);

        Assert.All(result, r => Assert.Equal(0.0f, r, 5));
    }

    [Fact]
    public void Add_NegativeValues_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        var a = new float[] { -1.0f, -2.0f, -3.0f };
        var b = new float[] { -4.0f, -5.0f, -6.0f };
        var result = new float[3];

        _backend!.Add(a, b, result);

        Assert.Equal(-5.0f, result[0], 5);
        Assert.Equal(-7.0f, result[1], 5);
        Assert.Equal(-9.0f, result[2], 5);
    }

    [Fact]
    public void Add_MixedSigns_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        var a = new float[] { 10.0f, -10.0f, 0.0f };
        var b = new float[] { -3.0f, 3.0f, 0.0f };
        var result = new float[3];

        _backend!.Add(a, b, result);

        Assert.Equal(7.0f, result[0], 5);
        Assert.Equal(-7.0f, result[1], 5);
        Assert.Equal(0.0f, result[2], 5);
    }

    [Fact]
    public void Add_VerySmallValues_MaintainsPrecision()
    {
        SkipIfNoVulkan();

        // Bug pattern: Precision loss with small values
        var a = new float[] { 1e-7f, 1e-8f, 1e-9f };
        var b = new float[] { 1e-7f, 1e-8f, 1e-9f };
        var result = new float[3];

        _backend!.Add(a, b, result);

        Assert.Equal(2e-7f, result[0], 10);
        Assert.Equal(2e-8f, result[1], 10);
        Assert.Equal(2e-9f, result[2], 10);
    }

    [Fact]
    public void Add_VeryLargeValues_DoesNotOverflow()
    {
        SkipIfNoVulkan();

        // Bug pattern: Overflow handling
        var a = new float[] { 1e38f, -1e38f };
        var b = new float[] { 1e37f, -1e37f };
        var result = new float[2];

        _backend!.Add(a, b, result);

        Assert.False(float.IsInfinity(result[0]));
        Assert.False(float.IsInfinity(result[1]));
    }

    #endregion

    #region Vector Subtraction Tests

    [Fact]
    public void Subtract_BasicValues_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        var a = new float[] { 10.0f, 20.0f, 30.0f };
        var b = new float[] { 3.0f, 5.0f, 7.0f };
        var result = new float[3];

        _backend!.Subtract(a, b, result);

        Assert.Equal(7.0f, result[0], 5);
        Assert.Equal(15.0f, result[1], 5);
        Assert.Equal(23.0f, result[2], 5);
    }

    [Fact]
    public void Subtract_ResultNegative_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        var a = new float[] { 1.0f, 2.0f, 3.0f };
        var b = new float[] { 10.0f, 20.0f, 30.0f };
        var result = new float[3];

        _backend!.Subtract(a, b, result);

        Assert.Equal(-9.0f, result[0], 5);
        Assert.Equal(-18.0f, result[1], 5);
        Assert.Equal(-27.0f, result[2], 5);
    }

    [Fact]
    public void Subtract_SameValues_ReturnsZero()
    {
        SkipIfNoVulkan();

        var a = new float[] { 5.0f, 10.0f, 15.0f };
        var b = new float[] { 5.0f, 10.0f, 15.0f };
        var result = new float[3];

        _backend!.Subtract(a, b, result);

        Assert.All(result, r => Assert.Equal(0.0f, r, 5));
    }

    #endregion

    #region Vector Multiplication Tests

    [Fact]
    public void Multiply_BasicValues_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        var a = new float[] { 2.0f, 3.0f, 4.0f };
        var b = new float[] { 5.0f, 6.0f, 7.0f };
        var result = new float[3];

        _backend!.Multiply(a, b, result);

        Assert.Equal(10.0f, result[0], 5);
        Assert.Equal(18.0f, result[1], 5);
        Assert.Equal(28.0f, result[2], 5);
    }

    [Fact]
    public void Multiply_ByZero_ReturnsZero()
    {
        SkipIfNoVulkan();

        var a = new float[] { 100.0f, 200.0f, 300.0f };
        var b = new float[] { 0.0f, 0.0f, 0.0f };
        var result = new float[3];

        _backend!.Multiply(a, b, result);

        Assert.All(result, r => Assert.Equal(0.0f, r, 5));
    }

    [Fact]
    public void Multiply_ByOne_ReturnsSame()
    {
        SkipIfNoVulkan();

        var a = new float[] { 1.5f, 2.5f, 3.5f };
        var b = new float[] { 1.0f, 1.0f, 1.0f };
        var result = new float[3];

        _backend!.Multiply(a, b, result);

        Assert.Equal(1.5f, result[0], 5);
        Assert.Equal(2.5f, result[1], 5);
        Assert.Equal(3.5f, result[2], 5);
    }

    [Fact]
    public void Multiply_NegativeValues_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        var a = new float[] { -2.0f, 3.0f, -4.0f };
        var b = new float[] { 5.0f, -6.0f, -7.0f };
        var result = new float[3];

        _backend!.Multiply(a, b, result);

        Assert.Equal(-10.0f, result[0], 5);
        Assert.Equal(-18.0f, result[1], 5);
        Assert.Equal(28.0f, result[2], 5);
    }

    #endregion

    #region Vector Division Tests

    [Fact]
    public void Divide_BasicValues_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        var a = new float[] { 10.0f, 20.0f, 30.0f };
        var b = new float[] { 2.0f, 4.0f, 5.0f };
        var result = new float[3];

        _backend!.Divide(a, b, result);

        Assert.Equal(5.0f, result[0], 5);
        Assert.Equal(5.0f, result[1], 5);
        Assert.Equal(6.0f, result[2], 5);
    }

    [Fact]
    public void Divide_ByOne_ReturnsSame()
    {
        SkipIfNoVulkan();

        var a = new float[] { 1.5f, 2.5f, 3.5f };
        var b = new float[] { 1.0f, 1.0f, 1.0f };
        var result = new float[3];

        _backend!.Divide(a, b, result);

        Assert.Equal(1.5f, result[0], 5);
        Assert.Equal(2.5f, result[1], 5);
        Assert.Equal(3.5f, result[2], 5);
    }

    [Fact]
    public void Divide_ByZero_ReturnsInfinity()
    {
        SkipIfNoVulkan();

        // GPU division by zero should return infinity, not crash
        var a = new float[] { 1.0f, -1.0f };
        var b = new float[] { 0.0f, 0.0f };
        var result = new float[2];

        _backend!.Divide(a, b, result);

        Assert.True(float.IsPositiveInfinity(result[0]) || float.IsNaN(result[0]));
        Assert.True(float.IsNegativeInfinity(result[1]) || float.IsNaN(result[1]));
    }

    [Fact]
    public void Divide_ZeroByZero_ReturnsNaN()
    {
        SkipIfNoVulkan();

        var a = new float[] { 0.0f };
        var b = new float[] { 0.0f };
        var result = new float[1];

        _backend!.Divide(a, b, result);

        Assert.True(float.IsNaN(result[0]));
    }

    [Fact]
    public void Divide_FractionalResults_MaintainsPrecision()
    {
        SkipIfNoVulkan();

        var a = new float[] { 1.0f, 1.0f, 1.0f };
        var b = new float[] { 3.0f, 7.0f, 11.0f };
        var result = new float[3];

        _backend!.Divide(a, b, result);

        Assert.Equal(1.0f / 3.0f, result[0], 5);
        Assert.Equal(1.0f / 7.0f, result[1], 5);
        Assert.Equal(1.0f / 11.0f, result[2], 5);
    }

    #endregion

    #region Scalar Multiplication Tests

    [Fact]
    public void ScalarMultiply_ByZero_ReturnsZero()
    {
        SkipIfNoVulkan();

        var input = new float[] { 1.0f, 2.0f, 3.0f };
        var result = new float[3];

        _backend!.ScalarMultiply(input, 0.0f, result);

        Assert.All(result, r => Assert.Equal(0.0f, r, 5));
    }

    [Fact]
    public void ScalarMultiply_ByOne_ReturnsSame()
    {
        SkipIfNoVulkan();

        var input = new float[] { 1.5f, 2.5f, 3.5f };
        var result = new float[3];

        _backend!.ScalarMultiply(input, 1.0f, result);

        Assert.Equal(1.5f, result[0], 5);
        Assert.Equal(2.5f, result[1], 5);
        Assert.Equal(3.5f, result[2], 5);
    }

    [Fact]
    public void ScalarMultiply_ByNegative_NegatesValues()
    {
        SkipIfNoVulkan();

        var input = new float[] { 1.0f, -2.0f, 3.0f };
        var result = new float[3];

        _backend!.ScalarMultiply(input, -1.0f, result);

        Assert.Equal(-1.0f, result[0], 5);
        Assert.Equal(2.0f, result[1], 5);
        Assert.Equal(-3.0f, result[2], 5);
    }

    [Fact]
    public void ScalarMultiply_ByFraction_ScalesDown()
    {
        SkipIfNoVulkan();

        var input = new float[] { 10.0f, 20.0f, 30.0f };
        var result = new float[3];

        _backend!.ScalarMultiply(input, 0.5f, result);

        Assert.Equal(5.0f, result[0], 5);
        Assert.Equal(10.0f, result[1], 5);
        Assert.Equal(15.0f, result[2], 5);
    }

    [Fact]
    public void ScalarMultiply_LargeArray_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        const int size = 100000;
        var input = Enumerable.Range(0, size).Select(i => 2.0f).ToArray();
        var result = new float[size];

        _backend!.ScalarMultiply(input, 3.0f, result);

        Assert.All(result, r => Assert.Equal(6.0f, r, 5));
    }

    #endregion

    #region ReLU Activation Tests

    [Fact]
    public void ReLU_PositiveValues_ReturnsSame()
    {
        SkipIfNoVulkan();

        var input = new float[] { 1.0f, 2.0f, 3.0f };
        var result = new float[3];

        _backend!.ReLU(input, result);

        Assert.Equal(1.0f, result[0], 5);
        Assert.Equal(2.0f, result[1], 5);
        Assert.Equal(3.0f, result[2], 5);
    }

    [Fact]
    public void ReLU_NegativeValues_ReturnsZero()
    {
        SkipIfNoVulkan();

        var input = new float[] { -1.0f, -2.0f, -3.0f };
        var result = new float[3];

        _backend!.ReLU(input, result);

        Assert.All(result, r => Assert.Equal(0.0f, r, 5));
    }

    [Fact]
    public void ReLU_Zero_ReturnsZero()
    {
        SkipIfNoVulkan();

        var input = new float[] { 0.0f, 0.0f, 0.0f };
        var result = new float[3];

        _backend!.ReLU(input, result);

        Assert.All(result, r => Assert.Equal(0.0f, r, 5));
    }

    [Fact]
    public void ReLU_MixedValues_ComputesCorrectly()
    {
        SkipIfNoVulkan();

        var input = new float[] { -2.0f, 0.0f, 3.0f, -0.5f, 1.5f };
        var result = new float[5];

        _backend!.ReLU(input, result);

        Assert.Equal(0.0f, result[0], 5);
        Assert.Equal(0.0f, result[1], 5);
        Assert.Equal(3.0f, result[2], 5);
        Assert.Equal(0.0f, result[3], 5);
        Assert.Equal(1.5f, result[4], 5);
    }

    [Fact]
    public void ReLU_VerySmallNegative_ReturnsZero()
    {
        SkipIfNoVulkan();

        // Bug pattern: Floating point comparison issues
        var input = new float[] { -1e-10f, -1e-20f, -1e-30f };
        var result = new float[3];

        _backend!.ReLU(input, result);

        Assert.All(result, r => Assert.Equal(0.0f, r, 5));
    }

    #endregion

    #region Sigmoid Activation Tests

    [Fact]
    public void Sigmoid_Zero_ReturnsHalf()
    {
        SkipIfNoVulkan();

        var input = new float[] { 0.0f };
        var result = new float[1];

        _backend!.Sigmoid(input, result);

        Assert.Equal(0.5f, result[0], 4);
    }

    [Fact]
    public void Sigmoid_LargePositive_ReturnsNearOne()
    {
        SkipIfNoVulkan();

        var input = new float[] { 10.0f, 20.0f, 100.0f };
        var result = new float[3];

        _backend!.Sigmoid(input, result);

        Assert.True(result[0] > 0.9999f);
        Assert.True(result[1] > 0.9999f);
        Assert.True(result[2] > 0.9999f);
    }

    [Fact]
    public void Sigmoid_LargeNegative_ReturnsNearZero()
    {
        SkipIfNoVulkan();

        var input = new float[] { -10.0f, -20.0f, -100.0f };
        var result = new float[3];

        _backend!.Sigmoid(input, result);

        Assert.True(result[0] < 0.0001f);
        Assert.True(result[1] < 0.0001f);
        Assert.True(result[2] < 0.0001f);
    }

    [Fact]
    public void Sigmoid_OutputRange_BetweenZeroAndOne()
    {
        SkipIfNoVulkan();

        var input = Enumerable.Range(-50, 100).Select(i => (float)i).ToArray();
        var result = new float[100];

        _backend!.Sigmoid(input, result);

        Assert.All(result, r =>
        {
            Assert.True(r >= 0.0f);
            Assert.True(r <= 1.0f);
        });
    }

    [Fact]
    public void Sigmoid_Symmetric_AroundHalf()
    {
        SkipIfNoVulkan();

        var positive = new float[] { 1.0f, 2.0f, 3.0f };
        var negative = new float[] { -1.0f, -2.0f, -3.0f };
        var resultPos = new float[3];
        var resultNeg = new float[3];

        _backend!.Sigmoid(positive, resultPos);
        _backend!.Sigmoid(negative, resultNeg);

        for (int i = 0; i < 3; i++)
        {
            // sigmoid(x) + sigmoid(-x) = 1
            Assert.Equal(1.0f, resultPos[i] + resultNeg[i], 4);
        }
    }

    #endregion

    #region Tanh Activation Tests

    [Fact]
    public void Tanh_Zero_ReturnsZero()
    {
        SkipIfNoVulkan();

        var input = new float[] { 0.0f };
        var result = new float[1];

        _backend!.Tanh(input, result);

        Assert.Equal(0.0f, result[0], 5);
    }

    [Fact]
    public void Tanh_LargePositive_ReturnsNearOne()
    {
        SkipIfNoVulkan();

        var input = new float[] { 10.0f, 20.0f };
        var result = new float[2];

        _backend!.Tanh(input, result);

        Assert.True(result[0] > 0.9999f);
        Assert.True(result[1] > 0.9999f);
    }

    [Fact]
    public void Tanh_LargeNegative_ReturnsNearNegativeOne()
    {
        SkipIfNoVulkan();

        var input = new float[] { -10.0f, -20.0f };
        var result = new float[2];

        _backend!.Tanh(input, result);

        Assert.True(result[0] < -0.9999f);
        Assert.True(result[1] < -0.9999f);
    }

    [Fact]
    public void Tanh_OutputRange_BetweenNegativeOneAndOne()
    {
        SkipIfNoVulkan();

        var input = Enumerable.Range(-50, 100).Select(i => (float)i).ToArray();
        var result = new float[100];

        _backend!.Tanh(input, result);

        Assert.All(result, r =>
        {
            Assert.True(r >= -1.0f);
            Assert.True(r <= 1.0f);
        });
    }

    [Fact]
    public void Tanh_OddFunction_Symmetric()
    {
        SkipIfNoVulkan();

        var positive = new float[] { 0.5f, 1.0f, 2.0f };
        var negative = new float[] { -0.5f, -1.0f, -2.0f };
        var resultPos = new float[3];
        var resultNeg = new float[3];

        _backend!.Tanh(positive, resultPos);
        _backend!.Tanh(negative, resultNeg);

        for (int i = 0; i < 3; i++)
        {
            // tanh(-x) = -tanh(x)
            Assert.Equal(-resultPos[i], resultNeg[i], 4);
        }
    }

    #endregion

    #region Memory Management Tests

    [Fact]
    public void RepeatedOperations_DoNotLeakMemory()
    {
        SkipIfNoVulkan();

        // Bug pattern: Memory leak from repeated buffer allocations
        const int size = 10000;
        var a = Enumerable.Range(0, size).Select(i => 1.0f).ToArray();
        var b = Enumerable.Range(0, size).Select(i => 2.0f).ToArray();
        var result = new float[size];

        // Perform many operations
        for (int i = 0; i < 100; i++)
        {
            _backend!.Add(a, b, result);
        }

        // If we got here without out-of-memory, test passes
        Assert.True(true);
    }

    [Fact]
    public void DifferentSizedOperations_HandleCorrectly()
    {
        SkipIfNoVulkan();

        // Bug pattern: Buffer reuse issues with different sizes
        int[] sizes = { 10, 1000, 100, 10000, 50, 500 };

        foreach (var size in sizes)
        {
            var a = Enumerable.Range(0, size).Select(i => 1.0f).ToArray();
            var b = Enumerable.Range(0, size).Select(i => 2.0f).ToArray();
            var result = new float[size];

            _backend!.Add(a, b, result);

            Assert.All(result, r => Assert.Equal(3.0f, r, 5));
        }
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public void ConcurrentOperations_ProduceCorrectResults()
    {
        SkipIfNoVulkan();

        // Bug pattern: Race conditions in shared state
        const int threadCount = 10;
        const int size = 1000;
        var results = new bool[threadCount];

        Parallel.For(0, threadCount, i =>
        {
            var a = Enumerable.Range(0, size).Select(j => (float)i).ToArray();
            var b = Enumerable.Range(0, size).Select(j => 1.0f).ToArray();
            var result = new float[size];

            _backend!.Add(a, b, result);

            results[i] = result.All(r => Math.Abs(r - (i + 1.0f)) < 0.001f);
        });

        Assert.All(results, r => Assert.True(r));
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void SpecialFloatValues_HandleCorrectly()
    {
        SkipIfNoVulkan();

        // Bug pattern: NaN/Infinity propagation
        var a = new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity };
        var b = new float[] { 1.0f, 1.0f, 1.0f };
        var result = new float[3];

        _backend!.Add(a, b, result);

        Assert.True(float.IsNaN(result[0]));
        Assert.True(float.IsPositiveInfinity(result[1]));
        Assert.True(float.IsNegativeInfinity(result[2]));
    }

    [Fact]
    public void DenormalizedNumbers_HandleCorrectly()
    {
        SkipIfNoVulkan();

        // Bug pattern: Flush-to-zero behavior differences
        float denormal = float.Epsilon / 2;
        var a = new float[] { denormal, denormal, denormal };
        var b = new float[] { denormal, denormal, denormal };
        var result = new float[3];

        _backend!.Add(a, b, result);

        // Result should be either 2*denormal or 0 (if flushed)
        Assert.All(result, r => Assert.True(r == 0.0f || r == 2 * denormal));
    }

    [Fact]
    public void MaxFloatValues_HandleCorrectly()
    {
        SkipIfNoVulkan();

        var a = new float[] { float.MaxValue };
        var b = new float[] { float.MaxValue };
        var result = new float[1];

        _backend!.Add(a, b, result);

        // Should overflow to infinity
        Assert.True(float.IsPositiveInfinity(result[0]));
    }

    [Fact]
    public void MinFloatValues_HandleCorrectly()
    {
        SkipIfNoVulkan();

        var a = new float[] { float.MinValue };
        var b = new float[] { float.MinValue };
        var result = new float[1];

        _backend!.Add(a, b, result);

        // Should overflow to negative infinity
        Assert.True(float.IsNegativeInfinity(result[0]));
    }

    #endregion
}

/// <summary>
/// Exception used to skip tests when Vulkan is not available.
/// </summary>
public class SkipException : Exception
{
    public SkipException(string message) : base(message) { }
}
