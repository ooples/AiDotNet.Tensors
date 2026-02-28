// Copyright (c) AiDotNet. All rights reserved.
// Tests that verify GPU results match CPU reference implementation.
// Goal: Find precision bugs and numerical inconsistencies between GPU and CPU.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Tests comparing GPU backend results against CPU reference implementations.
/// These tests are designed to catch:
/// - Precision differences between GPU and CPU floating point
/// - Shader implementation bugs
/// - Data transfer corruption
/// - Memory layout mismatches
/// </summary>
public class GpuCpuConsistencyTests
{
    private readonly bool _isVulkanAvailable;
    private readonly VulkanBackend? _backend;
    private readonly bool _isDirectGpuAvailable;
    private const float Tolerance = 1e-5f;
    private const float RelativeTolerance = 1e-4f;

    public GpuCpuConsistencyTests()
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

        // Probe DirectGpuEngine availability (CUDA, OpenCL, HIP) separately from Vulkan.
        // Tests that use DirectGpuTensorEngine should check this instead of Vulkan.
        try
        {
            using var probe = new DirectGpuTensorEngine();
            _isDirectGpuAvailable = probe.IsGpuAvailable;
        }
        catch
        {
            _isDirectGpuAvailable = false;
        }
    }

    /// <summary>
    /// Skips the test if Vulkan is not available. Use for tests that call VulkanBackend directly.
    /// </summary>
    private void SkipIfNoGpu()
    {
        Skip.If(!_isVulkanAvailable, "Vulkan GPU backend not available on this system");
    }

    /// <summary>
    /// Skips the test if no DirectGpu backend (CUDA, OpenCL, or HIP) is available.
    /// Use for tests that use DirectGpuTensorEngine.
    /// </summary>
    private void SkipIfNoDirectGpu()
    {
        Skip.If(!_isDirectGpuAvailable, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system");
    }

    private static bool AreClose(float expected, float actual, float tolerance = Tolerance)
    {
        if (float.IsNaN(expected) && float.IsNaN(actual)) return true;
        if (float.IsInfinity(expected) && float.IsInfinity(actual))
            return Math.Sign(expected) == Math.Sign(actual);
        return Math.Abs(expected - actual) <= tolerance;
    }

    private static bool AreCloseRelative(float expected, float actual, float relativeTolerance = RelativeTolerance)
    {
        if (float.IsNaN(expected) && float.IsNaN(actual)) return true;
        if (float.IsInfinity(expected) && float.IsInfinity(actual))
            return Math.Sign(expected) == Math.Sign(actual);
        if (Math.Abs(expected) < Tolerance) return Math.Abs(actual) < Tolerance;
        return Math.Abs((expected - actual) / expected) <= relativeTolerance;
    }

    #region Addition Consistency Tests

    [SkippableFact]
    public void Add_RandomData_MatchesCpu()
    {
        SkipIfNoGpu();

        var random = new Random(42);
        const int size = 10000;

        var a = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 200 - 100)).ToArray();
        var b = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 200 - 100)).ToArray();

        // CPU reference
        var cpuResult = new float[size];
        for (int i = 0; i < size; i++)
            cpuResult[i] = a[i] + b[i];

        // GPU result
        var gpuResult = new float[size];
        _backend!.Add(a, b, gpuResult);

        // Compare
        for (int i = 0; i < size; i++)
        {
            Assert.True(AreClose(cpuResult[i], gpuResult[i]),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}");
        }
    }

    [SkippableFact]
    public void Add_ExtremeValues_MatchesCpu()
    {
        SkipIfNoGpu();

        var testCases = new (float a, float b)[]
        {
            (float.MaxValue / 2, float.MaxValue / 2),
            (float.MinValue / 2, float.MinValue / 2),
            (1e-38f, 1e-38f),
            (-1e-38f, -1e-38f),
            (float.Epsilon, float.Epsilon),
            (1e30f, -1e30f),
            (1e-30f, -1e-30f),
        };

        var a = testCases.Select(t => t.a).ToArray();
        var b = testCases.Select(t => t.b).ToArray();

        var cpuResult = new float[testCases.Length];
        for (int i = 0; i < testCases.Length; i++)
            cpuResult[i] = a[i] + b[i];

        var gpuResult = new float[testCases.Length];
        _backend!.Add(a, b, gpuResult);

        for (int i = 0; i < testCases.Length; i++)
        {
            Assert.True(AreClose(cpuResult[i], gpuResult[i]),
                $"Mismatch for ({a[i]}, {b[i]}): CPU={cpuResult[i]}, GPU={gpuResult[i]}");
        }
    }

    #endregion

    #region Subtraction Consistency Tests

    [SkippableFact]
    public void Subtract_RandomData_MatchesCpu()
    {
        SkipIfNoGpu();

        var random = new Random(43);
        const int size = 10000;

        var a = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 200 - 100)).ToArray();
        var b = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 200 - 100)).ToArray();

        var cpuResult = new float[size];
        for (int i = 0; i < size; i++)
            cpuResult[i] = a[i] - b[i];

        var gpuResult = new float[size];
        _backend!.Subtract(a, b, gpuResult);

        for (int i = 0; i < size; i++)
        {
            Assert.True(AreClose(cpuResult[i], gpuResult[i]),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}");
        }
    }

    [SkippableFact]
    public void Subtract_CancellationPrecision_MatchesCpu()
    {
        SkipIfNoGpu();

        // Bug pattern: Catastrophic cancellation
        var a = new float[] { 1.0000001f, 1.0000002f, 1.0000003f };
        var b = new float[] { 1.0000000f, 1.0000001f, 1.0000002f };

        var cpuResult = new float[3];
        for (int i = 0; i < 3; i++)
            cpuResult[i] = a[i] - b[i];

        var gpuResult = new float[3];
        _backend!.Subtract(a, b, gpuResult);

        for (int i = 0; i < 3; i++)
        {
            Assert.True(AreCloseRelative(cpuResult[i], gpuResult[i]),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}");
        }
    }

    #endregion

    #region Multiplication Consistency Tests

    [SkippableFact]
    public void Multiply_RandomData_MatchesCpu()
    {
        SkipIfNoGpu();

        var random = new Random(44);
        const int size = 10000;

        var a = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 20 - 10)).ToArray();
        var b = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 20 - 10)).ToArray();

        var cpuResult = new float[size];
        for (int i = 0; i < size; i++)
            cpuResult[i] = a[i] * b[i];

        var gpuResult = new float[size];
        _backend!.Multiply(a, b, gpuResult);

        for (int i = 0; i < size; i++)
        {
            Assert.True(AreCloseRelative(cpuResult[i], gpuResult[i]),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}");
        }
    }

    [SkippableFact]
    public void Multiply_OverflowCases_MatchesCpu()
    {
        SkipIfNoGpu();

        var a = new float[] { 1e20f, 1e-20f, 1e30f, 1e-30f };
        var b = new float[] { 1e20f, 1e20f, 1e10f, 1e-10f };

        var cpuResult = new float[4];
        for (int i = 0; i < 4; i++)
            cpuResult[i] = a[i] * b[i];

        var gpuResult = new float[4];
        _backend!.Multiply(a, b, gpuResult);

        for (int i = 0; i < 4; i++)
        {
            Assert.True(AreClose(cpuResult[i], gpuResult[i]) || AreCloseRelative(cpuResult[i], gpuResult[i]),
                $"Mismatch for ({a[i]} * {b[i]}): CPU={cpuResult[i]}, GPU={gpuResult[i]}");
        }
    }

    #endregion

    #region Division Consistency Tests

    [SkippableFact]
    public void Divide_RandomData_MatchesCpu()
    {
        SkipIfNoGpu();

        var random = new Random(45);
        const int size = 10000;

        var a = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 200 - 100)).ToArray();
        // Avoid division by very small numbers
        var b = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 18 + 1) * (random.Next(2) == 0 ? 1 : -1)).ToArray();

        var cpuResult = new float[size];
        for (int i = 0; i < size; i++)
            cpuResult[i] = a[i] / b[i];

        var gpuResult = new float[size];
        _backend!.Divide(a, b, gpuResult);

        for (int i = 0; i < size; i++)
        {
            Assert.True(AreCloseRelative(cpuResult[i], gpuResult[i]),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}, a={a[i]}, b={b[i]}");
        }
    }

    [SkippableFact]
    public void Divide_SpecialCases_MatchesCpu()
    {
        SkipIfNoGpu();

        var testCases = new (float a, float b)[]
        {
            (1.0f, 3.0f),        // Non-terminating decimal
            (1.0f, 7.0f),        // Non-terminating decimal
            (22.0f, 7.0f),       // Pi approximation
            (1.0f, 0.1f),        // 0.1 is not exact in float
            (0.1f, 0.1f),        // Should be 1.0
            (1e10f, 1e-10f),     // Large result
            (1e-10f, 1e10f),     // Small result
        };

        var a = testCases.Select(t => t.a).ToArray();
        var b = testCases.Select(t => t.b).ToArray();

        var cpuResult = new float[testCases.Length];
        for (int i = 0; i < testCases.Length; i++)
            cpuResult[i] = a[i] / b[i];

        var gpuResult = new float[testCases.Length];
        _backend!.Divide(a, b, gpuResult);

        for (int i = 0; i < testCases.Length; i++)
        {
            Assert.True(AreCloseRelative(cpuResult[i], gpuResult[i]),
                $"Mismatch for ({a[i]} / {b[i]}): CPU={cpuResult[i]}, GPU={gpuResult[i]}");
        }
    }

    #endregion

    #region Scalar Multiply Consistency Tests

    [SkippableFact]
    public void ScalarMultiply_RandomData_MatchesCpu()
    {
        SkipIfNoGpu();

        var random = new Random(46);
        const int size = 10000;
        float scalar = (float)(random.NextDouble() * 10 - 5);

        var input = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 200 - 100)).ToArray();

        var cpuResult = new float[size];
        for (int i = 0; i < size; i++)
            cpuResult[i] = input[i] * scalar;

        var gpuResult = new float[size];
        _backend!.ScalarMultiply(input, scalar, gpuResult);

        for (int i = 0; i < size; i++)
        {
            Assert.True(AreCloseRelative(cpuResult[i], gpuResult[i]),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}, scalar={scalar}");
        }
    }

    [SkippableFact]
    public void ScalarMultiply_SpecialScalars_MatchesCpu()
    {
        SkipIfNoGpu();

        float[] scalars = { 0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 1e10f, 1e-10f, -0.001f };
        var input = new float[] { 1.0f, -1.0f, 0.5f, 100.0f, 0.0f };

        foreach (var scalar in scalars)
        {
            var cpuResult = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
                cpuResult[i] = input[i] * scalar;

            var gpuResult = new float[input.Length];
            _backend!.ScalarMultiply(input, scalar, gpuResult);

            for (int i = 0; i < input.Length; i++)
            {
                Assert.True(AreClose(cpuResult[i], gpuResult[i]),
                    $"Mismatch for input[{i}]={input[i]} * {scalar}: CPU={cpuResult[i]}, GPU={gpuResult[i]}");
            }
        }
    }

    #endregion

    #region ReLU Consistency Tests

    [SkippableFact]
    public void ReLU_RandomData_MatchesCpu()
    {
        SkipIfNoGpu();

        var random = new Random(47);
        const int size = 10000;

        var input = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 200 - 100)).ToArray();

        var cpuResult = new float[size];
        for (int i = 0; i < size; i++)
            cpuResult[i] = Math.Max(0, input[i]);

        var gpuResult = new float[size];
        _backend!.ReLU(input, gpuResult);

        for (int i = 0; i < size; i++)
        {
            Assert.True(AreClose(cpuResult[i], gpuResult[i]),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}, input={input[i]}");
        }
    }

    [SkippableFact]
    public void ReLU_NearZero_MatchesCpu()
    {
        SkipIfNoGpu();

        // Bug pattern: Comparison precision at zero boundary
        var input = new float[]
        {
            1e-10f, -1e-10f, 1e-20f, -1e-20f,
            float.Epsilon, -float.Epsilon,
            float.Epsilon / 2, -float.Epsilon / 2,
            0.0f, -0.0f
        };

        var cpuResult = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            cpuResult[i] = Math.Max(0, input[i]);

        var gpuResult = new float[input.Length];
        _backend!.ReLU(input, gpuResult);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.True(AreClose(cpuResult[i], gpuResult[i]),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}, input={input[i]}");
        }
    }

    #endregion

    #region Sigmoid Consistency Tests

    [SkippableFact]
    public void Sigmoid_RandomData_MatchesCpu()
    {
        SkipIfNoGpu();

        var random = new Random(48);
        const int size = 10000;

        var input = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 20 - 10)).ToArray();

        var cpuResult = new float[size];
        for (int i = 0; i < size; i++)
            cpuResult[i] = 1.0f / (1.0f + MathF.Exp(-input[i]));

        var gpuResult = new float[size];
        _backend!.Sigmoid(input, gpuResult);

        for (int i = 0; i < size; i++)
        {
            Assert.True(AreClose(cpuResult[i], gpuResult[i], 1e-4f),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}, input={input[i]}");
        }
    }

    [SkippableFact]
    public void Sigmoid_ExtremeValues_MatchesCpu()
    {
        SkipIfNoGpu();

        // Bug pattern: Overflow in exp()
        var input = new float[] { -100.0f, -50.0f, -10.0f, 0.0f, 10.0f, 50.0f, 100.0f };

        var cpuResult = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            cpuResult[i] = 1.0f / (1.0f + MathF.Exp(-input[i]));

        var gpuResult = new float[input.Length];
        _backend!.Sigmoid(input, gpuResult);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.True(AreClose(cpuResult[i], gpuResult[i], 1e-4f),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}, input={input[i]}");
        }
    }

    #endregion

    #region Tanh Consistency Tests

    [SkippableFact]
    public void Tanh_RandomData_MatchesCpu()
    {
        SkipIfNoGpu();

        var random = new Random(49);
        const int size = 10000;

        var input = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 20 - 10)).ToArray();

        var cpuResult = new float[size];
        for (int i = 0; i < size; i++)
            cpuResult[i] = MathF.Tanh(input[i]);

        var gpuResult = new float[size];
        _backend!.Tanh(input, gpuResult);

        for (int i = 0; i < size; i++)
        {
            Assert.True(AreClose(cpuResult[i], gpuResult[i], 1e-4f),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}, input={input[i]}");
        }
    }

    [SkippableFact]
    public void Tanh_ExtremeValues_MatchesCpu()
    {
        SkipIfNoGpu();

        var input = new float[] { -100.0f, -20.0f, -5.0f, 0.0f, 5.0f, 20.0f, 100.0f };

        var cpuResult = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            cpuResult[i] = MathF.Tanh(input[i]);

        var gpuResult = new float[input.Length];
        _backend!.Tanh(input, gpuResult);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.True(AreClose(cpuResult[i], gpuResult[i], 1e-4f),
                $"Mismatch at index {i}: CPU={cpuResult[i]}, GPU={gpuResult[i]}, input={input[i]}");
        }
    }

    #endregion

    #region Data Transfer Integrity Tests

    [SkippableFact]
    public void DataTransfer_PreservesExactBitPattern()
    {
        SkipIfNoGpu();

        // Bug pattern: Data corruption during CPU-GPU transfer
        var input = new float[]
        {
            0.0f, -0.0f,
            1.0f, -1.0f,
            float.MaxValue, float.MinValue,
            float.Epsilon, -float.Epsilon,
            float.PositiveInfinity, float.NegativeInfinity,
            float.NaN,
            CreateSignalingNaN(), // Signaling NaN
            1.23456789f,
            9.87654321f,
        };

        // Use identity operation (multiply by 1)
        var ones = Enumerable.Repeat(1.0f, input.Length).ToArray();
        var result = new float[input.Length];

        _backend!.Multiply(input, ones, result);

        for (int i = 0; i < input.Length; i++)
        {
            if (float.IsNaN(input[i]))
            {
                Assert.True(float.IsNaN(result[i]), $"NaN not preserved at index {i}");
            }
            else
            {
                Assert.Equal(input[i], result[i]);
            }
        }
    }

    [SkippableFact]
    public void DataTransfer_LargeArray_PreservesAllValues()
    {
        SkipIfNoGpu();

        const int size = 1_000_000;
        var input = new float[size];
        for (int i = 0; i < size; i++)
        {
            // Create predictable but varied values
            input[i] = MathF.Sin(i * 0.001f) * 1000;
        }

        var ones = Enumerable.Repeat(1.0f, size).ToArray();
        var result = new float[size];

        _backend!.Multiply(input, ones, result);

        for (int i = 0; i < size; i++)
        {
            Assert.True(AreClose(input[i], result[i]),
                $"Mismatch at index {i}: expected={input[i]}, actual={result[i]}");
        }
    }

    #endregion

    #region Determinism Tests

    [SkippableFact]
    public void Operations_AreDeterministic()
    {
        SkipIfNoGpu();

        // Bug pattern: Non-deterministic results from GPU
        var random = new Random(50);
        const int size = 10000;

        var a = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 100)).ToArray();
        var b = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 100)).ToArray();

        var result1 = new float[size];
        var result2 = new float[size];
        var result3 = new float[size];

        _backend!.Add(a, b, result1);
        _backend!.Add(a, b, result2);
        _backend!.Add(a, b, result3);

        for (int i = 0; i < size; i++)
        {
            Assert.Equal(result1[i], result2[i]);
            Assert.Equal(result2[i], result3[i]);
        }
    }

    #endregion

    #region Associativity/Commutativity Tests

    [SkippableFact]
    public void Add_IsCommutative()
    {
        SkipIfNoGpu();

        var a = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var b = new float[] { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };

        var result1 = new float[5];
        var result2 = new float[5];

        _backend!.Add(a, b, result1);
        _backend!.Add(b, a, result2);

        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(result1[i], result2[i]);
        }
    }

    [SkippableFact]
    public void Multiply_IsCommutative()
    {
        SkipIfNoGpu();

        var a = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var b = new float[] { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };

        var result1 = new float[5];
        var result2 = new float[5];

        _backend!.Multiply(a, b, result1);
        _backend!.Multiply(b, a, result2);

        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(result1[i], result2[i]);
        }
    }

    #endregion

    #region TensorLerp GPU vs CPU Consistency Tests

    [SkippableFact]
    public void TensorLerp_GpuMatchesCpu()
    {
        SkipIfNoDirectGpu();

        var random = new Random(42);
        const int size = 1000;

        var aData = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 200 - 100)).ToArray();
        var bData = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 200 - 100)).ToArray();

        var a = new Tensor<float>(new[] { size });
        var b = new Tensor<float>(new[] { size });
        for (int i = 0; i < size; i++)
        {
            a.SetFlat(i, aData[i]);
            b.SetFlat(i, bData[i]);
        }

        float t = 0.3f;

        // CPU reference
        var cpuEngine = new CpuEngine();
        var cpuResult = ((IEngine)cpuEngine).TensorLerp(a, b, t);

        // GPU via DirectGpuTensorEngine (falls back to CPU if no GPU)
        using var gpuEngine = new DirectGpuTensorEngine();
        var gpuResult = ((IEngine)gpuEngine).TensorLerp(a, b, t);

        // Compare
        Assert.Equal(cpuResult.Shape, gpuResult.Shape);
        for (int i = 0; i < size; i++)
        {
            Assert.True(AreClose(cpuResult.GetFlat(i), gpuResult.GetFlat(i)),
                $"TensorLerp mismatch at index {i}: CPU={cpuResult.GetFlat(i)}, GPU={gpuResult.GetFlat(i)}");
        }
    }

    [SkippableFact]
    public void TensorLerp_BoundaryInterpolation_GpuMatchesCpu()
    {
        SkipIfNoDirectGpu();

        var a = new Tensor<float>(new[] { 4 });
        var b = new Tensor<float>(new[] { 4 });
        a.SetFlat(0, 1f); a.SetFlat(1, 2f); a.SetFlat(2, 3f); a.SetFlat(3, 4f);
        b.SetFlat(0, 10f); b.SetFlat(1, 20f); b.SetFlat(2, 30f); b.SetFlat(3, 40f);

        var cpuEngine = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();

        // t=0 should return a, t=1 should return b
        var cpuAt0 = ((IEngine)cpuEngine).TensorLerp(a, b, 0f);
        var gpuAt0 = ((IEngine)gpuEngine).TensorLerp(a, b, 0f);

        var cpuAt1 = ((IEngine)cpuEngine).TensorLerp(a, b, 1f);
        var gpuAt1 = ((IEngine)gpuEngine).TensorLerp(a, b, 1f);

        for (int i = 0; i < 4; i++)
        {
            Assert.True(AreClose(cpuAt0.GetFlat(i), gpuAt0.GetFlat(i)),
                $"TensorLerp(t=0) mismatch at {i}: CPU={cpuAt0.GetFlat(i)}, GPU={gpuAt0.GetFlat(i)}");
            Assert.True(AreClose(cpuAt1.GetFlat(i), gpuAt1.GetFlat(i)),
                $"TensorLerp(t=1) mismatch at {i}: CPU={cpuAt1.GetFlat(i)}, GPU={gpuAt1.GetFlat(i)}");
        }
    }

    #endregion

    #region TensorAddScaled GPU vs CPU Consistency Tests

    [SkippableFact]
    public void TensorAddScaled_GpuMatchesCpu()
    {
        SkipIfNoDirectGpu();

        var random = new Random(42);
        const int size = 1000;

        var aData = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 200 - 100)).ToArray();
        var bData = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 200 - 100)).ToArray();

        var a = new Tensor<float>(new[] { size });
        var b = new Tensor<float>(new[] { size });
        for (int i = 0; i < size; i++)
        {
            a.SetFlat(i, aData[i]);
            b.SetFlat(i, bData[i]);
        }

        float scaleA = 0.7f;
        float scaleB = 0.3f;

        // CPU reference
        var cpuEngine = new CpuEngine();
        var cpuResult = ((IEngine)cpuEngine).TensorAddScaled(a, b, scaleA, scaleB);

        // GPU via DirectGpuTensorEngine
        using var gpuEngine = new DirectGpuTensorEngine();
        var gpuResult = ((IEngine)gpuEngine).TensorAddScaled(a, b, scaleA, scaleB);

        // Compare
        Assert.Equal(cpuResult.Shape, gpuResult.Shape);
        for (int i = 0; i < size; i++)
        {
            Assert.True(AreClose(cpuResult.GetFlat(i), gpuResult.GetFlat(i)),
                $"TensorAddScaled mismatch at index {i}: CPU={cpuResult.GetFlat(i)}, GPU={gpuResult.GetFlat(i)}");
        }
    }

    [SkippableFact]
    public void TensorAddScaled_DiffusionNoiseMixing_GpuMatchesCpu()
    {
        SkipIfNoDirectGpu();

        // Simulate diffusion noise mixing: alpha * signal + sigma * noise
        var random = new Random(123);
        const int size = 512;

        var signal = new Tensor<float>(new[] { size });
        var noise = new Tensor<float>(new[] { size });
        for (int i = 0; i < size; i++)
        {
            signal.SetFlat(i, (float)(random.NextDouble() * 2 - 1));
            noise.SetFlat(i, (float)(random.NextDouble() * 2 - 1));
        }

        float alpha = 0.95f;  // signal weight
        float sigma = 0.05f;  // noise weight

        var cpuEngine = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();

        var cpuResult = ((IEngine)cpuEngine).TensorAddScaled(signal, noise, alpha, sigma);
        var gpuResult = ((IEngine)gpuEngine).TensorAddScaled(signal, noise, alpha, sigma);

        Assert.Equal(cpuResult.Shape, gpuResult.Shape);
        for (int i = 0; i < size; i++)
        {
            Assert.True(AreClose(cpuResult.GetFlat(i), gpuResult.GetFlat(i)),
                $"Diffusion mixing mismatch at {i}: CPU={cpuResult.GetFlat(i)}, GPU={gpuResult.GetFlat(i)}");
        }
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates a signaling NaN value.
    /// Uses unsafe code to avoid BitConverter.Int32BitsToSingle which is not available in .NET Framework 4.7.1.
    /// </summary>
    private static unsafe float CreateSignalingNaN()
    {
        int bits = 0x7F800001; // Signaling NaN bit pattern
        return *(float*)&bits;
    }

    #endregion
}
