// Copyright (c) AiDotNet. All rights reserved.
// Tests that verify GPU results match CPU reference implementation.
// Goal: Find precision bugs and numerical inconsistencies between GPU and CPU.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
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
[Collection("VulkanGlobalState")]
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

    private static bool IsSubnormal(float value)
    {
        // A subnormal (denormalized) float has exponent bits all zero but non-zero mantissa
        return value != 0.0f && MathF.Abs(value) < 1.175494351e-38f; // smallest positive normal float
    }

    private static bool AreCloseRelative(float expected, float actual, float relativeTolerance = RelativeTolerance)
    {
        if (float.IsNaN(expected) && float.IsNaN(actual)) return true;
        if (float.IsInfinity(expected) && float.IsInfinity(actual))
            return Math.Sign(expected) == Math.Sign(actual);
        if (Math.Abs(expected) < Tolerance) return Math.Abs(actual) < Tolerance;
        return Math.Abs((expected - actual) / expected) <= relativeTolerance;
    }

    private static float DeterministicValue(int value)
    {
        unchecked
        {
            uint x = (uint)value * 747_796_405u + 2_891_336_453u;
            x = ((x >> (int)((x >> 28) + 4)) ^ x) * 277_803_737u;
            x = (x >> 22) ^ x;
            return (x & 0xFFFF) / 65_535f - 0.5f;
        }
    }

    [SkippableFact]
    public void HardsigmoidBackward_IsBitIdenticalAtBoundariesAndStaysResident()
    {
        SkipIfNoDirectGpu();

        float[] inputValues =
        {
            -3.0f,
            MathF.BitIncrement(-3.0f),
            -1.0f,
            0.0f,
            MathF.BitDecrement(3.0f),
            3.0f
        };
        float[] gradientValues = { 0.1f, -0.2f, 0.3f, -0.4f, 0.7f, -0.9f };
        var input = new Tensor<float>(inputValues, new[] { inputValues.Length });
        var gradient = new Tensor<float>(gradientValues, new[] { gradientValues.Length });
        var cpu = new CpuEngine();
        using var gpu = new DirectGpuTensorEngine();

        float[] expected = cpu.HardsigmoidBackward(gradient, input).GetDataArray();
        GpuLaunchProbe.Reset();
        Tensor<float> gpuResult = gpu.HardsigmoidBackward(gradient, input);

        Assert.True(GpuLaunchProbe.Count > 0);
        Assert.Equal(0, GpuLaunchProbe.Readbacks);

        float[] actual = gpuResult.GetDataArray();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(
                MathCompat.SingleToInt32Bits(expected[i]),
                MathCompat.SingleToInt32Bits(actual[i]));
        }
    }

    [SkippableFact]
    public void RReluEvaluation_ForwardBackward_AreBitIdenticalAndStayResident()
    {
        SkipIfNoDirectGpu();

        float[] inputValues = { -3.0f, -0.75f, -0.0f, 0.0f, 0.25f, 2.0f };
        var input = new Tensor<float>(inputValues, new[] { inputValues.Length });
        const double lower = 0.125;
        const double upper = 0.333;
        float slope = (float)((lower + upper) / 2.0);
        using var gpu = new DirectGpuTensorEngine();
        using var tape = new GradientTape<float>();
        tape.BindEngineIfUnset(gpu);

        GpuLaunchProbe.Reset();
        Tensor<float> output = gpu.TensorRReLU(input, lower, upper, training: false);
        var gradients = tape.ComputeGradients(output, sources: new[] { input });

        Assert.True(GpuLaunchProbe.Count > 0);
        Assert.Equal(0, GpuLaunchProbe.Readbacks);
        Assert.True(gradients.TryGetValue(input, out var gradient));

        float[] actualOutput = output.GetDataArray();
        float[] actualGradient = gradient.GetDataArray();
        for (int i = 0; i < inputValues.Length; i++)
        {
            float expectedOutput = inputValues[i] >= 0.0f ? inputValues[i] : slope * inputValues[i];
            float expectedGradient = inputValues[i] >= 0.0f ? 1.0f : slope;
            Assert.Equal(MathCompat.SingleToInt32Bits(expectedOutput), MathCompat.SingleToInt32Bits(actualOutput[i]));
            Assert.Equal(MathCompat.SingleToInt32Bits(expectedGradient), MathCompat.SingleToInt32Bits(actualGradient[i]));
        }
    }

    [SkippableFact]
    public void BroadcastBinary_AcceleratedPaths_RecordAutodiff()
    {
        SkipIfNoDirectGpu();
        using var gpu = new DirectGpuTensorEngine();
        var operations = new (string Name, Func<Tensor<float>, Tensor<float>, Tensor<float>> Run)[]
        {
            ("add", gpu.TensorBroadcastAdd),
            ("subtract", gpu.TensorBroadcastSubtract),
            ("multiply", gpu.TensorBroadcastMultiply),
            ("divide", gpu.TensorBroadcastDivide)
        };

        foreach (var operation in operations)
        {
            var a = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });
            var b = new Tensor<float>(new[] { 2f, 3f, 4f }, new[] { 1, 3 });
            using var tape = new GradientTape<float>();
            tape.BindEngineIfUnset(gpu);

            var output = operation.Run(a, b);
            Assert.True(tape.EntryCount > 0, $"Broadcast {operation.Name} did not record its GPU forward.");

            var loss = gpu.TensorMeanDiff(output);
            var gradients = tape.ComputeGradients(loss, sources: new[] { a, b });
            Assert.True(gradients.ContainsKey(a), $"Broadcast {operation.Name} omitted the left gradient.");
            Assert.True(gradients.ContainsKey(b), $"Broadcast {operation.Name} omitted the right gradient.");
        }
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
            else if (IsSubnormal(input[i]))
            {
                // GPUs operate in FTZ (Flush-To-Zero) mode by default, which flushes
                // denormalized/subnormal floats to zero. This is standard GPU hardware
                // behavior per IEEE 754 relaxed mode and cannot be changed in code.
                Assert.True(result[i] == 0.0f || result[i] == input[i],
                    $"Subnormal at index {i}: expected 0 or {input[i]}, got {result[i]}");
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

    #region Scaled Dot-Product Attention GPU vs CPU Consistency Tests

    [SkippableFact]
    public void ScaledDotProductAttention_CrossAttentionStaysResidentAndMatchesCpu()
    {
        SkipIfNoDirectGpu();

        const int batch = 2, heads = 2, seqQ = 2, seqK = 3, dimension = 3;
        const double scale = 0.75;
        var query = new Tensor<float>(
            Enumerable.Range(0, batch * heads * seqQ * dimension).Select(i => -0.2f + 0.03f * i).ToArray(),
            new[] { batch, heads, seqQ, dimension });
        var key = new Tensor<float>(
            Enumerable.Range(0, batch * heads * seqK * dimension).Select(i => 0.5f - 0.025f * i).ToArray(),
            new[] { batch, heads, seqK, dimension });
        var value = new Tensor<float>(
            Enumerable.Range(0, batch * heads * seqK * dimension).Select(i => -0.1f + 0.04f * i).ToArray(),
            new[] { batch, heads, seqK, dimension });

        var cpu = new CpuEngine();
        var expected = cpu.ScaledDotProductAttention(query, key, value, null, scale, out var expectedWeights);
        var gradOutput = new Tensor<float>(
            Enumerable.Range(0, batch * heads * seqQ * dimension).Select(i => 0.3f - 0.02f * i).ToArray(),
            new[] { batch, heads, seqQ, dimension });
        cpu.ScaledDotProductAttentionBackward(gradOutput, query, key, value, expectedWeights, scale,
            out var expectedGradQuery, out var expectedGradKey, out var expectedGradValue);

        using var gpu = new DirectGpuTensorEngine();
        using var scope = gpu.BeginGpuScope();
        var actual = ((IEngine)gpu).ScaledDotProductAttention(
            query, key, value, null, scale, out var actualWeights);
        ((IEngine)gpu).ScaledDotProductAttentionBackward(
            gradOutput, query, key, value, actualWeights, scale,
            out var actualGradQuery, out var actualGradKey, out var actualGradValue);

        Assert.Equal(new[] { batch, heads, seqQ, dimension }, actual.Shape.ToArray());
        Assert.Equal(new[] { batch, heads, seqQ, seqK }, actualWeights.Shape.ToArray());
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actual.DataVector),
            "Cross-attention output must remain GPU-resident before host access.");
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualWeights.DataVector),
            "Cross-attention weights must remain GPU-resident before host access.");
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualGradQuery.DataVector),
            "Cross-attention query gradients must remain GPU-resident before host access.");
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualGradKey.DataVector),
            "Cross-attention key gradients must remain GPU-resident before host access.");
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualGradValue.DataVector),
            "Cross-attention value gradients must remain GPU-resident before host access.");

        var expectedData = expected.AsSpan();
        var actualData = actual.AsSpan();
        for (int i = 0; i < expectedData.Length; i++)
            Assert.True(MathF.Abs(expectedData[i] - actualData[i]) <= 2e-5f,
                $"Cross-attention output mismatch at {i}: CPU={expectedData[i]}, GPU={actualData[i]}");

        var expectedWeightData = expectedWeights.AsSpan();
        var actualWeightData = actualWeights.AsSpan();
        for (int i = 0; i < expectedWeightData.Length; i++)
            Assert.True(MathF.Abs(expectedWeightData[i] - actualWeightData[i]) <= 2e-5f,
                $"Cross-attention weight mismatch at {i}: CPU={expectedWeightData[i]}, GPU={actualWeightData[i]}");

        Assert.Equal(expectedGradQuery.Shape, actualGradQuery.Shape);
        Assert.Equal(expectedGradKey.Shape, actualGradKey.Shape);
        Assert.Equal(expectedGradValue.Shape, actualGradValue.Shape);
        AssertTensorClose(expectedGradQuery, actualGradQuery, "query gradient");
        AssertTensorClose(expectedGradKey, actualGradKey, "key gradient");
        AssertTensorClose(expectedGradValue, actualGradValue, "value gradient");
    }

    [SkippableFact]
    public void ScaledDotProductAttention_FullBooleanMaskStaysResidentAndMatchesCpu()
    {
        SkipIfNoDirectGpu();

        const int batch = 2, heads = 2, seqQ = 2, seqK = 3, dimension = 3;
        const double scale = 0.75;
        var query = new Tensor<float>(
            Enumerable.Range(0, batch * heads * seqQ * dimension).Select(i => -0.2f + 0.03f * i).ToArray(),
            new[] { batch, heads, seqQ, dimension });
        var key = new Tensor<float>(
            Enumerable.Range(0, batch * heads * seqK * dimension).Select(i => 0.5f - 0.025f * i).ToArray(),
            new[] { batch, heads, seqK, dimension });
        var value = new Tensor<float>(
            Enumerable.Range(0, batch * heads * seqK * dimension).Select(i => -0.1f + 0.04f * i).ToArray(),
            new[] { batch, heads, seqK, dimension });
        var mask = new Tensor<bool>(
            new[]
            {
                true, false, true, false, true, true,
                true, true, false, true, false, false,
                false, true, true, true, true, false,
                true, false, false, false, true, false
            },
            new[] { batch, heads, seqQ, seqK });
        var gradOutput = new Tensor<float>(
            Enumerable.Range(0, batch * heads * seqQ * dimension).Select(i => 0.3f - 0.02f * i).ToArray(),
            new[] { batch, heads, seqQ, dimension });

        var cpu = new CpuEngine();
        var expected = cpu.ScaledDotProductAttention(query, key, value, mask, scale, out var expectedWeights);
        cpu.ScaledDotProductAttentionBackward(gradOutput, query, key, value, expectedWeights, scale,
            out var expectedGradQuery, out var expectedGradKey, out var expectedGradValue);

        using var gpu = new DirectGpuTensorEngine();
        using var scope = gpu.BeginGpuScope();
        var actual = ((IEngine)gpu).ScaledDotProductAttention(
            query, key, value, mask, scale, out var actualWeights);
        ((IEngine)gpu).ScaledDotProductAttentionBackward(
            gradOutput, query, key, value, actualWeights, scale,
            out var actualGradQuery, out var actualGradKey, out var actualGradValue);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actual.DataVector),
            "Masked attention output must remain GPU-resident before host access.");
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualWeights.DataVector),
            "Masked attention weights must remain GPU-resident before host access.");
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualGradQuery.DataVector),
            "Masked attention query gradients must remain GPU-resident before host access.");
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualGradKey.DataVector),
            "Masked attention key gradients must remain GPU-resident before host access.");
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualGradValue.DataVector),
            "Masked attention value gradients must remain GPU-resident before host access.");

        AssertTensorClose(expected, actual, "masked output");
        AssertTensorClose(expectedWeights, actualWeights, "masked weights");
        AssertTensorClose(expectedGradQuery, actualGradQuery, "masked query gradient");
        AssertTensorClose(expectedGradKey, actualGradKey, "masked key gradient");
        AssertTensorClose(expectedGradValue, actualGradValue, "masked value gradient");
    }

    private static void AssertTensorClose(Tensor<float> expected, Tensor<float> actual, string name)
    {
        var expectedData = expected.AsSpan();
        var actualData = actual.AsSpan();
        Assert.Equal(expectedData.Length, actualData.Length);
        for (int i = 0; i < expectedData.Length; i++)
            Assert.True(MathF.Abs(expectedData[i] - actualData[i]) <= 2e-5f,
                $"Cross-attention {name} mismatch at {i}: CPU={expectedData[i]}, GPU={actualData[i]}");
    }

    #endregion

    #region TensorMatMul GPU vs CPU Consistency Tests

    [SkippableFact]
    public void TensorMatMul_GemvAboveClBlastIndirectThreshold_GpuMatchesCpu()
    {
        SkipIfNoDirectGpu();

        const int rows = 512;
        const int cols = 64;
        var weights = Enumerable.Range(0, rows * cols)
            .Select(i => DeterministicValue(i + 304))
            .ToArray();
        var query = Enumerable.Range(0, cols)
            .Select(i => DeterministicValue(i + 10_000))
            .ToArray();

        var a = new Tensor<float>(weights, new[] { rows, cols });
        var b = new Tensor<float>(query, new[] { cols, 1 });

        var cpuEngine = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        var cpuResult = cpuEngine.TensorMatMul(a, b);
        var gpuResult = gpuEngine.TensorMatMul(a, b);

        Assert.Equal(cpuResult.Shape, gpuResult.Shape);
        for (int i = 0; i < cpuResult.Length; i++)
        {
            float expected = cpuResult.GetFlat(i);
            float actual = gpuResult.GetFlat(i);
            float tolerance = 1e-4f + MathF.Abs(expected) * 1e-4f;
            Assert.True(MathF.Abs(expected - actual) <= tolerance,
                $"TensorMatMul GEMV mismatch at {i}: CPU={expected}, GPU={actual}");
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

    [SkippableFact]
    public void NchwcReorder_RoundTripStaysResidentAndIsBitExact()
    {
        SkipIfNoDirectGpu();
        const int n = 1, c = 16, h = 2, w = 3;
        var data = Enumerable.Range(0, n * c * h * w)
            .Select(i => Int32BitsToSingleCompat(unchecked((int)(0x3F000000u + (uint)i))))
            .ToArray();
        var source = new Tensor<float>((float[])data.Clone(), new[] { n, c, h, w });
        var cpu = new CpuEngine();
        using var gpu = new DirectGpuTensorEngine();

        var expectedPacked = cpu.ReorderToNchwc(source, TensorLayout.Nchwc8);
        var actualPacked = gpu.ReorderToNchwc(source, TensorLayout.Nchwc8);
        Assert.Equal(TensorLayout.Nchwc8, actualPacked.Layout);
        Assert.NotNull(actualPacked.TryGetGpuBuffer());

        var actualRoundTrip = gpu.ReorderToNchw(actualPacked);
        Assert.Equal(TensorLayout.Nchw, actualRoundTrip.Layout);
        Assert.NotNull(actualRoundTrip.TryGetGpuBuffer());
        Assert.Equal(expectedPacked.GetDataArray(), actualPacked.GetDataArray());
        Assert.Equal(data, actualRoundTrip.GetDataArray());
    }

    [SkippableFact]
    public void IntegerSelectionOps_StayResidentAndMatchCpu()
    {
        SkipIfNoDirectGpu();
        var values = new Tensor<float>(new float[]
        {
            5, 1, 7, 3,
            -2, 9, 4, 8,
            6, 0, -1, 2
        }, new[] { 3, 4 });
        var boundaries = new Tensor<float>(new float[] { -2, 0, 3, 8 }, new[] { 4 });
        var probes = new Tensor<float>(new float[] { -3, -2, 1, 8, 10 }, new[] { 5 });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var expectedMax = cpu.TensorArgMax(values, 0);
        var expectedMin = cpu.TensorArgMin(values, 1);
        var expectedBuckets = cpu.TensorBucketize(probes, boundaries, right: true);
        var actualMax = gpu.TensorArgMax(values, 0);
        var actualMin = gpu.TensorArgMin(values, 1);
        var actualBuckets = gpu.TensorBucketize(probes, boundaries, right: true);

        Assert.NotNull(actualMax.TryGetGpuBuffer());
        Assert.NotNull(actualMin.TryGetGpuBuffer());
        Assert.NotNull(actualBuckets.TryGetGpuBuffer());
        Assert.Equal(expectedMax.GetDataArray(), actualMax.GetDataArray());
        Assert.Equal(expectedMin.GetDataArray(), actualMin.GetDataArray());
        Assert.Equal(expectedBuckets.GetDataArray(), actualBuckets.GetDataArray());
    }

    [SkippableFact]
    public void GenericEmbeddingForwardBackward_StayResidentAndMatchCpu()
    {
        SkipIfNoDirectGpu();
        var tableData = Enumerable.Range(0, 32).Select(i => (i - 11) * 0.125f).ToArray();
        var table = new Tensor<float>((float[])tableData.Clone(), new[] { 8, 4 });
        var indices = new Tensor<int>(new[] { 5, 1, 7, 0 }, new[] { 2, 2 });
        var gradient = new Tensor<float>(Enumerable.Range(0, 16).Select(i => (i + 1) * 0.25f).ToArray(),
            new[] { 2, 2, 4 });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var expected = cpu.TensorEmbeddingLookup<float, int>(table, indices);
        var expectedGradient = cpu.TensorEmbeddingLookupBackward<float, int>(gradient, indices, 8, 4);
        var actual = gpu.TensorEmbeddingLookup<float, int>(table, indices);
        var actualGradient = gpu.TensorEmbeddingLookupBackward<float, int>(gradient, indices, 8, 4);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actual.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualGradient.DataVector));
        Assert.Equal(expected.GetDataArray(), actual.GetDataArray());
        Assert.Equal(expectedGradient.GetDataArray(), actualGradient.GetDataArray());
    }

    [SkippableFact]
    public void PitchShift_CompositionStaysResidentAndMatchesCpu()
    {
        SkipIfNoDirectGpu();
        var samples = Enumerable.Range(0, 256)
            .Select(i => MathF.Sin(2f * MathF.PI * i / 16f) + 0.25f * MathF.Cos(2f * MathF.PI * i / 7f))
            .ToArray();
        var waveform = new Tensor<float>(samples, new[] { samples.Length });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var expected = cpu.PitchShift(waveform, 16_000, 3.0, nFft: 16, hopLength: 4);
        var actual = gpu.PitchShift(waveform, 16_000, 3.0, nFft: 16, hopLength: 4);

        Assert.NotNull(actual.TryGetGpuBuffer());
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var expectedData = expected.GetDataArray();
        var actualData = actual.GetDataArray();
        for (int i = 0; i < expectedData.Length; i++)
            Assert.True(AreCloseRelative(expectedData[i], actualData[i], 5e-3f),
                $"PitchShift mismatch at {i}: CPU={expectedData[i]}, GPU={actualData[i]}");
    }

    [SkippableFact]
    public void GumbelSoftmax_SoftAndHardStayResidentAndRespectRowContract()
    {
        SkipIfNoDirectGpu();
        var logits = new Tensor<float>(Enumerable.Range(0, 15).Select(i => (i - 7) * 0.2f).ToArray(),
            new[] { 3, 5 });
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var soft = gpu.GumbelSoftmax(logits, temperature: 0.75, hard: false, axis: -1);
        var hard = gpu.GumbelSoftmax(logits, temperature: 0.75, hard: true, axis: -1);

        Assert.NotNull(soft.TryGetGpuBuffer());
        Assert.NotNull(hard.TryGetGpuBuffer());
        var softData = soft.GetDataArray();
        var hardData = hard.GetDataArray();
        for (int row = 0; row < 3; row++)
        {
            float softSum = 0f;
            float hardSum = 0f;
            for (int column = 0; column < 5; column++)
            {
                softSum += softData[row * 5 + column];
                hardSum += hardData[row * 5 + column];
                Assert.True(hardData[row * 5 + column] is 0f or 1f);
            }
            Assert.True(AreClose(1f, softSum, 1e-5f));
            Assert.Equal(1f, hardSum);
        }
    }

    [SkippableFact]
    public void PackedByteGatherScatter_StayResidentAndAreBitExact()
    {
        SkipIfNoDirectGpu();
        var packed = new Tensor<byte>(Enumerable.Range(0, 12).Select(i => (byte)(i * 17 + 3)).ToArray(),
            new[] { 2, 3, 2 });
        var indices = new Tensor<int>(new[] { 2, 0 }, new[] { 2 });
        var source = new Tensor<byte>(new byte[] { 250, 249, 240, 239, 230, 229, 220, 219 },
            new[] { 2, 2, 2 });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var expectedGather = cpu.TensorGatherPacked(packed, indices, axis: 1, valuesPerByte: 2);
        var expectedScatter = cpu.TensorScatterPacked(packed, indices, source, axis: 1, valuesPerByte: 2);
        var actualGather = gpu.TensorGatherPacked(packed, indices, axis: 1, valuesPerByte: 2);
        var actualScatter = gpu.TensorScatterPacked(packed, indices, source, axis: 1, valuesPerByte: 2);

        Assert.NotNull(actualGather.TryGetGpuBuffer());
        Assert.NotNull(actualScatter.TryGetGpuBuffer());
        Assert.Equal(expectedGather.GetDataArray(), actualGather.GetDataArray());
        Assert.Equal(expectedScatter.GetDataArray(), actualScatter.GetDataArray());
    }

    [SkippableFact]
    public void ImportanceSampling_StaysResidentAndIsBitExactForBinaryCdf()
    {
        SkipIfNoDirectGpu();
        var tValues = new Tensor<float>(new float[]
        {
            0, 1, 2, 3,
            -2, -1, 0, 1
        }, new[] { 2, 4 });
        var weights = new Tensor<float>(new float[]
        {
            0, 0, 0, 0,
            1, 2, 1, 0
        }, new[] { 2, 4 });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var expected = cpu.ImportanceSampling(tValues, weights, numFineSamples: 8);
        var actual = gpu.ImportanceSampling(tValues, weights, numFineSamples: 8);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actual.DataVector));
        Assert.Equal(new[] { 2, 8 }, actual.Shape.ToArray());
        Assert.Equal(expected.GetDataArray(), actual.GetDataArray());
    }

    [SkippableFact]
    public void BinCount_FromResidentSearchSortedStaysResidentAndIsBitExact()
    {
        SkipIfNoDirectGpu();
        var boundaries = new Tensor<float>(new float[] { -2, 0, 3, 8 }, new[] { 4 });
        var probes = new Tensor<float>(new float[] { -3, -2, -1, 0, 1, 2, 3, 7, 8, 9 },
            new[] { 10 });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var expectedIndices = cpu.TensorSearchSorted(boundaries, probes, right: true);
        var expected = cpu.TensorBinCount(expectedIndices, minLength: 7);
        var residentIndices = gpu.TensorSearchSorted(boundaries, probes, right: true);
        var actual = gpu.TensorBinCount(residentIndices, minLength: 7);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
            residentIndices.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actual.DataVector));
        Assert.Equal(expected.GetDataArray(), actual.GetDataArray());
    }

    [SkippableFact]
    public void NmsAndBatchedNms_StayResidentAndAreBitExact()
    {
        SkipIfNoDirectGpu();
        var boxes = new Tensor<float>(new float[]
        {
            0, 0, 10, 10,
            1, 1, 11, 11,
            20, 20, 30, 30,
            21, 21, 31, 31,
            0, 0, 5, 5,
            2, 2, 12, 12
        }, new[] { 6, 4 });
        var scores = new Tensor<float>(new float[] { 0.9f, 0.8f, 0.95f, 0.7f, 0.6f, 0.85f },
            new[] { 6 });
        var classIds = new Tensor<int>(new[] { 0, 0, 1, 1, 0, 1 }, new[] { 6 });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var expected = cpu.Nms(boxes, scores, 0.5);
        var expectedBatched = cpu.BatchedNms(boxes, scores, classIds, 0.5);
        var actual = gpu.Nms(boxes, scores, 0.5);
        var actualBatched = gpu.BatchedNms(boxes, scores, classIds, 0.5);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actual.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
            actualBatched.DataVector));
        Assert.Equal(expected.GetDataArray(), actual.GetDataArray());
        Assert.Equal(expectedBatched.GetDataArray(), actualBatched.GetDataArray());
    }

    [SkippableFact]
    public void GenerateSpiralIndices_StaysResidentAndIsBitExact()
    {
        SkipIfNoDirectGpu();
        var vertices = new Tensor<float>(new float[]
        {
             0,  0, 0,
             1,  0, 0,
             0,  1, 0,
            -1,  0, 0,
             0, -1, 0
        }, new[] { 5, 3 });
        var faces = new Tensor<int>(new[]
        {
            0, 1, 2,
            0, 2, 3,
            0, 3, 4,
            0, 4, 1
        }, new[] { 4, 3 });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var expected = cpu.GenerateSpiralIndices(vertices, faces, spiralLength: 4);
        var actual = gpu.GenerateSpiralIndices(vertices, faces, spiralLength: 4);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actual.DataVector));
        Assert.Equal(expected.GetDataArray(), actual.GetDataArray());
    }

    [SkippableFact]
    public void NativeComplexFftNd_RoundTripKeepsSplitComplexResultResident()
    {
        SkipIfNoDirectGpu();
        var values = Enumerable.Range(0, 4 * 3 * 8)
            .Select(i => DeterministicValue(i + 1))
            .ToArray();
        var input = new Tensor<float>(values, new[] { 4, 3, 8 });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var expectedSpectrum = cpu.NativeComplexFFTND(input, new[] { 0, 2 });
        var spectrum = gpu.NativeComplexFFTND(input, new[] { 0, 2 });
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
            spectrum.DataVector));

        // The inverse must consume both complex planes directly from the resident
        // buffer. If it decomposes through Complex<T> on the host, this pending
        // registration is cleared before the assertion below.
        var recovered = gpu.NativeComplexIFFTNDReal(spectrum, new[] { 0, 2 });
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
            spectrum.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
            recovered.DataVector));

        var expected = expectedSpectrum.GetDataArray();
        var actual = spectrum.GetDataArray();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(AreClose(expected[i].Real, actual[i].Real, 2e-4f),
                $"FFT real mismatch at {i}: expected {expected[i].Real}, actual {actual[i].Real}");
            Assert.True(AreClose(expected[i].Imaginary, actual[i].Imaginary, 2e-4f),
                $"FFT imaginary mismatch at {i}: expected {expected[i].Imaginary}, actual {actual[i].Imaginary}");
        }

        var roundTrip = recovered.GetDataArray();
        for (int i = 0; i < values.Length; i++)
            Assert.True(AreClose(values[i], roundTrip[i], 2e-4f),
                $"IFFT round-trip mismatch at {i}: expected {values[i]}, actual {roundTrip[i]}");
    }

    [SkippableFact]
    public void NativeComplexOperatorChain_ConsumesAndProducesResidentSplitPlanes()
    {
        SkipIfNoDirectGpu();
        var values = Enumerable.Range(0, 16)
            .Select(i => DeterministicValue(i + 101))
            .ToArray();
        var input = new Tensor<float>(values, new[] { 16 });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var cpuSpectrum = cpu.NativeComplexFFT(input);
        var cpuScaled = cpu.NativeComplexScale(cpuSpectrum, 0.25f);
        var cpuConjugate = cpu.NativeComplexConjugate(cpuSpectrum);
        var expected = cpu.NativeComplexMagnitude(cpu.NativeComplexAdd(cpuScaled, cpuConjugate));

        var spectrum = gpu.NativeComplexFFT(input);
        var scaled = gpu.NativeComplexScale(spectrum, 0.25f);
        var conjugate = gpu.NativeComplexConjugate(spectrum);
        var sum = gpu.NativeComplexAdd(scaled, conjugate);
        var actual = gpu.NativeComplexMagnitude(sum);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(spectrum.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(scaled.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(conjugate.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(sum.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actual.DataVector));

        var expectedValues = expected.GetDataArray();
        var actualValues = actual.GetDataArray();
        for (int i = 0; i < expectedValues.Length; i++)
            Assert.True(AreClose(expectedValues[i], actualValues[i], 2e-4f),
                $"Complex chain mismatch at {i}: expected {expectedValues[i]}, actual {actualValues[i]}");
    }

    [SkippableFact]
    public void StftIstft_RoundTripConsumesResidentSpectrogram()
    {
        SkipIfNoDirectGpu();
        const int nFft = 16;
        const int hopLength = 4;
        var values = Enumerable.Range(0, 2 * 48)
            .Select(i => DeterministicValue(i + 301))
            .ToArray();
        var input = new Tensor<float>(values, new[] { 2, 48 });
        var windowValues = Enumerable.Range(0, nFft)
            .Select(i => 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / nFft))
            .ToArray();
        var window = new Tensor<float>(windowValues, new[] { nFft });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        cpu.STFT(input, nFft, hopLength, window, center: true, out var expectedMagnitude, out var expectedPhase);
        gpu.STFT(input, nFft, hopLength, window, center: true, out var magnitude, out var phase);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(magnitude.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(phase.DataVector));

        var expectedOutput = cpu.ISTFT(expectedMagnitude, expectedPhase, nFft, hopLength, window, center: true);
        var output = gpu.ISTFT(magnitude, phase, nFft, hopLength, window, center: true);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(magnitude.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(phase.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(output.DataVector));
        Assert.Equal(expectedOutput.Shape.ToArray(), output.Shape.ToArray());

        var expected = expectedOutput.GetDataArray();
        var actual = output.GetDataArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.True(AreClose(expected[i], actual[i], 5e-4f),
                $"STFT/ISTFT mismatch at {i}: expected {expected[i]}, actual {actual[i]}");
    }

    [SkippableFact]
    public void MelSpectrogram_ProducesResidentCpuParityResult()
    {
        SkipIfNoDirectGpu();
        const int nFft = 16;
        const int hopLength = 8;
        const int nMels = 8;
        var input = new Tensor<float>(Enumerable.Range(0, 128)
            .Select(i => DeterministicValue(i + 501)).ToArray(), new[] { 1, 128 });
        var window = new Tensor<float>(Enumerable.Range(0, nFft)
            .Select(i => 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / nFft)).ToArray(), new[] { nFft });
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var expected = cpu.MelSpectrogram(input, 16000, nFft, hopLength, nMels, 0f, 8000f, window, true);
        var actual = gpu.MelSpectrogram(input, 16000, nFft, hopLength, nMels, 0f, 8000f, window, true);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actual.DataVector));
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var expectedValues = expected.GetDataArray();
        var actualValues = actual.GetDataArray();
        for (int i = 0; i < expectedValues.Length; i++)
            Assert.True(AreClose(expectedValues[i], actualValues[i], 2e-3f),
                $"Mel spectrogram mismatch at {i}: expected {expectedValues[i]}, actual {actualValues[i]}");
    }

    [SkippableFact]
    public void SplitPlaneFftApis_KeepBatched1DAnd2DOutputsResident()
    {
        SkipIfNoDirectGpu();
        IEngine cpu = new CpuEngine();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;

        var real1D = new Tensor<float>(Enumerable.Range(0, 3 * 8)
            .Select(i => DeterministicValue(i + 701)).ToArray(), new[] { 3, 8 });
        var imag1D = new Tensor<float>(Enumerable.Range(0, 3 * 8)
            .Select(i => DeterministicValue(i + 801)).ToArray(), new[] { 3, 8 });
        cpu.FFT(real1D, imag1D, out var expectedReal1D, out var expectedImag1D);
        gpu.FFT(real1D, imag1D, out var actualReal1D, out var actualImag1D);
        gpu.IFFT(actualReal1D, actualImag1D, out var recoveredReal1D, out var recoveredImag1D);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualReal1D.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualImag1D.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(recoveredReal1D.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(recoveredImag1D.DataVector));
        AssertFftPlaneClose(expectedReal1D, actualReal1D, "batched FFT real");
        AssertFftPlaneClose(expectedImag1D, actualImag1D, "batched FFT imaginary");
        AssertFftPlaneClose(real1D, recoveredReal1D, "batched IFFT real");
        AssertFftPlaneClose(imag1D, recoveredImag1D, "batched IFFT imaginary");

        var real2D = new Tensor<float>(Enumerable.Range(0, 2 * 4 * 4)
            .Select(i => DeterministicValue(i + 901)).ToArray(), new[] { 2, 4, 4 });
        var imag2D = new Tensor<float>(Enumerable.Range(0, 2 * 4 * 4)
            .Select(i => DeterministicValue(i + 1001)).ToArray(), new[] { 2, 4, 4 });
        cpu.FFT2D(real2D, imag2D, out var expectedReal2D, out var expectedImag2D);
        gpu.FFT2D(real2D, imag2D, out var actualReal2D, out var actualImag2D);
        gpu.IFFT2D(actualReal2D, actualImag2D, out var recoveredReal2D, out var recoveredImag2D);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualReal2D.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualImag2D.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(recoveredReal2D.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(recoveredImag2D.DataVector));
        AssertFftPlaneClose(expectedReal2D, actualReal2D, "batched FFT2D real");
        AssertFftPlaneClose(expectedImag2D, actualImag2D, "batched FFT2D imaginary");
        AssertFftPlaneClose(real2D, recoveredReal2D, "batched IFFT2D real");
        AssertFftPlaneClose(imag2D, recoveredImag2D, "batched IFFT2D imaginary");
    }

    private static void AssertFftPlaneClose(Tensor<float> expected, Tensor<float> actual, string operation)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var expectedValues = expected.GetDataArray();
        var actualValues = actual.GetDataArray();
        for (int i = 0; i < expectedValues.Length; i++)
            Assert.True(AreClose(expectedValues[i], actualValues[i], 2e-4f),
                $"{operation} mismatch at {i}: expected {expectedValues[i]}, actual {actualValues[i]}");
    }

    [SkippableFact]
    public void GriffinLim_IterationsRemainResidentAndDeterministic()
    {
        SkipIfNoDirectGpu();
        const int nFft = 8;
        const int hopLength = 2;
        var input = new Tensor<float>(Enumerable.Range(0, 32)
            .Select(i => DeterministicValue(i + 1101)).ToArray(), new[] { 32 });
        var window = new Tensor<float>(Enumerable.Range(0, nFft)
            .Select(i => 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / nFft)).ToArray(), new[] { nFft });
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;
        gpu.STFT(input, nFft, hopLength, window, center: true, out var magnitude, out _);

        var first = gpu.GriffinLim(magnitude, nFft, hopLength, window,
            iterations: 2, momentum: 0.5, length: null);
        var second = gpu.GriffinLim(magnitude, nFft, hopLength, window,
            iterations: 2, momentum: 0.5, length: null);

        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(magnitude.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(first.DataVector));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(second.DataVector));
        var firstValues = first.GetDataArray();
        var secondValues = second.GetDataArray();
        Assert.Equal(firstValues, secondValues);
        Assert.All(firstValues, value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
    }

    [SkippableFact]
    public void FusedBiasDropout_KeepsOutputAndMaskResident()
    {
        SkipIfNoDirectGpu();
        using var gpu = new DirectGpuTensorEngine();
        var source = new Tensor<float>(new[] { -2f, -1f, 0f, 1f, 2f, 3f }, new[] { 2, 3 });
        var biasSource = new Tensor<float>(new[] { 0.5f, -1.5f, 2f }, new[] { 3 });
        var input = gpu.TensorAddScalar(source, 0f);
        var bias = gpu.TensorAddScalar(biasSource, 0f);

        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var inference = gpu.FusedBiasDropout(input, bias, 0.25, training: false,
                out var inferenceMask);
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(input.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(bias.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(inference.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(inferenceMask.DataVector));
            Assert.Equal(new[] { -1.5f, -2.5f, 2f, 1.5f, 0.5f, 5f }, inference.GetDataArray());
            Assert.All(inferenceMask.GetDataArray(), value => Assert.Equal(1f, value));

            var training = gpu.FusedBiasDropout(input, bias, 0.25, training: true,
                out var trainingMask);
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(training.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(trainingMask.DataVector));

            float[] biased = { -1.5f, -2.5f, 2f, 1.5f, 0.5f, 5f };
            float scale = 1f / 0.75f;
            var values = training.GetDataArray();
            var maskValues = trainingMask.GetDataArray();
            for (int i = 0; i < values.Length; i++)
            {
                Assert.True(maskValues[i] == 0f || maskValues[i] == scale,
                    $"Unexpected mask[{i}]={maskValues[i]:R}; expected 0 or {scale:R}.");
                Assert.Equal(maskValues[i] == 0f ? 0f : biased[i] * scale, values[i]);
            }
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = false;
        }
    }

    [SkippableFact]
    public void InterleavedFft_DeinterleaveTransformAndReassemblyStayResident()
    {
        SkipIfNoDirectGpu();
        using var gpu = new DirectGpuTensorEngine();
        var data = new float[32];
        for (int i = 0; i < data.Length / 2; i++)
        {
            data[2 * i] = DeterministicValue(i + 301);
            data[2 * i + 1] = DeterministicValue(i + 701);
        }
        var input = new Tensor<float>(data, new[] { 2, 16 });

        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var spectrum = gpu.TryBackendFft(input, inverse: false);
            Assert.NotNull(spectrum);
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                spectrum!.DataVector));

            var recovered = gpu.TryBackendFft(spectrum, inverse: true);
            Assert.NotNull(recovered);
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                spectrum.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                recovered!.DataVector));

            var actual = recovered.GetDataArray();
            for (int i = 0; i < data.Length; i++)
                Assert.True(AreClose(data[i], actual[i], 2e-5f),
                    $"Interleaved FFT round-trip mismatch at {i}: expected {data[i]}, got {actual[i]}.");
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = false;
        }
    }

    [SkippableFact]
    public void AdvancedFusedKernels_WriteResidentDestinationTensors()
    {
        SkipIfNoDirectGpu();
        using var gpu = new DirectGpuTensorEngine();
        var input = new Tensor<float>(new[] { 1f, -2f, 0.5f, 3f, -1f, 2f }, new[] { 2, 3 });
        var baseOutput = new Tensor<float>(new[] { 0.25f, -0.5f, 1f, 2f }, new[] { 2, 2 });
        var loraA = new Tensor<float>(new[] { 0.5f, -1f, 2f }, new[] { 3, 1 });
        var loraB = new Tensor<float>(new[] { 1.5f, -0.25f }, new[] { 1, 2 });
        var expectedLora = new Tensor<float>(new[] { 2, 2 });
        AiDotNet.Tensors.Helpers.CpuFusedOperations.FusedLoRAForward(
            input, baseOutput, loraA, loraB, 0.75f, expectedLora);

        var epsilon = new Tensor<float>(new[] { -0.5f, 0.25f, 1f, -1.5f, 0.75f, 0.1f }, new[] { 2, 3 });
        var expectedDdim = new Tensor<float>(new[] { 2, 3 });
        AiDotNet.Tensors.Helpers.CpuFusedOperations.FusedDDIMStep(
            input, epsilon, 0.64f, 0.81f, expectedDdim);

        int[] rowOffsets = { 0, 2, 3 };
        int[] columnIndices = { 0, 2, 1 };
        var sparseValues = new Tensor<float>(new[] { 1f, -0.5f, 2f }, new[] { 3 });
        var sparseBias = new Tensor<float>(new[] { 0.1f, -0.2f }, new[] { 2 });
        var expectedSparse = new Tensor<float>(new[] { 2, 2 });
        AiDotNet.Tensors.Helpers.CpuFusedOperations.FusedSparseLinear(
            input, rowOffsets, columnIndices, sparseValues, sparseBias,
            FusedActivationType.None, expectedSparse);

        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var actualLora = gpu.FusedLoRAForward(input, baseOutput, loraA, loraB, 0.75f);
            var actualDdim = gpu.FusedDDIMStep(input, epsilon, 0.64f, 0.81f);
            var actualSparse = gpu.FusedSparseLinear(input, rowOffsets, columnIndices,
                sparseValues, sparseBias, FusedActivationType.None);

            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                actualLora.DataVector.GetBackingArrayUnsafe()!));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                actualDdim.DataVector.GetBackingArrayUnsafe()!));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                actualSparse.DataVector.GetBackingArrayUnsafe()!));
            AssertTensorClose(expectedLora, actualLora, "fused LoRA");
            AssertTensorClose(expectedDdim, actualDdim, "fused DDIM");
            AssertTensorClose(expectedSparse, actualSparse, "fused sparse linear");
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = false;
        }
    }

    [SkippableFact]
    public void TensorIsIn_SortAndLookupStayResident()
    {
        SkipIfNoDirectGpu();
        using var gpu = new DirectGpuTensorEngine();
        var elementsSource = new Tensor<float>(
            new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });
        var testSource = new Tensor<float>(
            new[] { 5f, 2f, 9f, 4f }, new[] { 4 });
        var elements = gpu.TensorAddScalar(elementsSource, 0f);
        var testElements = gpu.TensorAddScalar(testSource, 0f);

        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var selected = gpu.TensorIsIn(elements, testElements);
            var inverted = gpu.TensorIsIn(elements, testElements, invert: true);

            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                elements.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                testElements.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                selected.DataVector.GetBackingArrayUnsafe()!));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                inverted.DataVector.GetBackingArrayUnsafe()!));
            Assert.Equal(new[] { false, true, false, true, true, false },
                selected.GetDataArray().Select(value => (bool)value).ToArray());
            Assert.Equal(new[] { true, false, true, false, false, true },
                inverted.GetDataArray().Select(value => (bool)value).ToArray());
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = false;
        }
    }

    [SkippableFact]
    public void TensorMaskedSelect_ResidentMaskCompactsOnDevice()
    {
        SkipIfNoDirectGpu();
        using var gpu = new DirectGpuTensorEngine();
        var values = gpu.TensorAddScalar(new Tensor<float>(
            new[] { -3f, -2f, -1f, 0f, 1f, 2f }, new[] { 2, 3 }), 0f);
        var maskValues = gpu.TensorAddScalar(new Tensor<float>(
            new[] { 1f, 0f, 1f, 0f, 0f, 1f }, new[] { 2, 3 }), 0f);
        var mask = gpu.TensorEqScalar(maskValues, 1f);

        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var selected = gpu.TensorMaskedSelect(values, mask);

            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                values.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                mask.DataVector.GetBackingArrayUnsafe()!));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                selected.DataVector));
            Assert.Equal(new[] { -3f, -1f, 2f }, selected.GetDataArray());
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = false;
        }
    }

    [SkippableFact]
    public void TensorMode_ReducesResidentInputWithCpuTieSemantics()
    {
        SkipIfNoDirectGpu();
        using var gpu = new DirectGpuTensorEngine();
        var cpu = new CpuEngine();
        var source = new Tensor<float>(
            new[] { float.NaN, 3f, 2f, 3f, 2f, -1f }, new[] { 2, 3 });
        var input = gpu.TensorAddScalar(source, 0f);
        var expected = cpu.TensorMode(source);

        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var actual = gpu.TensorMode(input);

            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                input.DataVector));
            Assert.Equal(expected.Count, actual.Count);
            Assert.Equal(expected.Value, actual.Value);
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = false;
        }
    }

    [SkippableFact]
    public void ResidentIndices_WriteOperationsStayOnDeviceAndPreserveOrdering()
    {
        SkipIfNoDirectGpu();
        using var gpu = new DirectGpuTensorEngine();
        var sortInput = gpu.TensorAddScalar(new Tensor<float>(
            new[] { 30f, 10f, 40f, 20f }, new[] { 4 }), 0f);
        var (_, indices) = gpu.TensorSort(sortInput);
        var destination = gpu.TensorAddScalar(new Tensor<float>(
            new[] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f }, new[] { 4, 2 }), 0f);
        var source = gpu.TensorAddScalar(new Tensor<float>(
            new[] { 10f, 11f, 20f, 21f, 30f, 31f, 40f, 41f }, new[] { 4, 2 }), 0f);
        var flatDestination = gpu.TensorAddScalar(new Tensor<float>(
            new[] { 0f, 1f, 2f, 3f }, new[] { 4 }), 0f);
        var flatSource = gpu.TensorAddScalar(new Tensor<float>(
            new[] { 10f, 20f, 30f, 40f }, new[] { 4 }), 0f);
        var maskValues = gpu.TensorAddScalar(new Tensor<float>(
            new[] { 1f, 0f, 1f, 0f, 1f, 0f }, new[] { 2, 3 }), 0f);
        var mask = gpu.TensorEqScalar(maskValues, 1f);
        var maskedDestination = gpu.TensorAddScalar(new Tensor<float>(
            new[] { 0f, 1f, 2f, 3f, 4f, 5f }, new[] { 2, 3 }), 0f);
        var maskedSource = gpu.TensorAddScalar(new Tensor<float>(
            new[] { 10f, 20f, 30f, 40f }, new[] { 4 }), 0f);
        var (_, rowIndices) = gpu.TensorSort(gpu.TensorAddScalar(new Tensor<float>(
            new[] { 1f, 2f, 0f }, new[] { 3 }), 0f));
        var (_, columnIndices) = gpu.TensorSort(gpu.TensorAddScalar(new Tensor<float>(
            new[] { 2f, 0f, 1f }, new[] { 3 }), 0f));
        var indexPutDestination = gpu.TensorAddScalar(new Tensor<float>(
            Enumerable.Range(0, 12).Select(value => (float)value).ToArray(), new[] { 3, 4 }), 0f);
        var indexPutSource = gpu.TensorAddScalar(new Tensor<float>(
            new[] { 90f, 80f, 70f }, new[] { 3 }), 0f);

        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            IEngine engine = gpu;
            var copied = gpu.TensorIndexCopy(destination, 0, indices, source);
            var filled = gpu.TensorIndexFill(destination, 0, indices, 9f);
            var put = gpu.TensorPut(flatDestination, indices, flatSource);
            var taken = gpu.TensorTake(flatSource, indices);
            var selected = gpu.TensorIndexSelect(destination, indices, 0);
            var gathered = engine.Gather(destination, indices, 0);
            var tensorGathered = gpu.TensorGather(destination, indices, 0);
            var scatterValues = gpu.TensorAddScalar(new Tensor<float>(
                new[] { 10f, 11f, 20f, 21f, 30f, 31f, 40f, 41f }, new[] { 4, 2 }), 0f);
            var scatter = engine.Scatter(destination, indices, scatterValues, 0);
            var rowScatterIndices = engine.Reshape(indices, new[] { 2, 2 });
            var rowScatter = engine.TensorScatter(
                engine.Reshape(destination, new[] { 2, 4 }), rowScatterIndices,
                new Tensor<float>(new[] { 100f, 101f, 102f, 103f }, new[] { 2, 2 }), 1);
            var scatterAdd = engine.ScatterAdd(source, indices, 0, 5);
            var scatterMax = engine.ScatterMax(source, indices, out var scatterArgmax, 0, 5);
            var scatterMean = gpu.ScatterMean(source, indices, out var scatterMeanCounts, 0, 5);
            var scatterSoftmax = engine.ScatterSoftmax(source, indices, 0, 5);
            var scatterGradOutput = gpu.TensorAddScalar(new Tensor<float>(
                Enumerable.Range(0, 10).Select(value => (float)value).ToArray(),
                new[] { 5, 2 }), 0f);
            var scatterAddGradient = engine.ScatterAddBackward(
                scatterGradOutput, indices, new[] { 4, 2 }, 0);
            var scatterMeanGradient = engine.ScatterMeanBackward(
                scatterGradOutput, indices, scatterMeanCounts!, new[] { 4, 2 }, 0);
            var scatterMaxGradient = engine.ScatterMaxBackward(
                scatterGradOutput, scatterArgmax!, new[] { 4, 2 }, 0);
            var scatterSoftmaxGradient = engine.ScatterSoftmaxBackward(
                source, scatterSoftmax, indices, 0);
            var scattered = gpu.TensorMaskedScatter(maskedDestination, mask, maskedSource);
            var maskedFilled = gpu.TensorMaskedFill(maskedDestination, mask, 99f);
            var indexPut = gpu.TensorIndexPut(indexPutDestination,
                new[] { rowIndices, columnIndices }, indexPutSource);
            var indexAdded = gpu.TensorIndexAdd(
                new Tensor<float>(new[] { 0f, 1f, 2f, 3f, 4f, 5f }, new[] { 3, 2 }),
                0, rowIndices,
                new Tensor<float>(new[] { 10f, 11f, 20f, 21f, 30f, 31f }, new[] { 3, 2 }));
            var packed = new Tensor<byte>(Enumerable.Range(0, 8).Select(value => (byte)value).ToArray(),
                new[] { 1, 4, 2 });
            var packedSource = new Tensor<byte>(
                new byte[] { 100, 101, 110, 111, 120, 121, 130, 131 }, new[] { 1, 4, 2 });
            var packedGather = engine.TensorGatherPacked(packed, indices, 1, valuesPerByte: 2);
            var packedScatter = engine.TensorScatterPacked(
                packed, indices, packedSource, 1, valuesPerByte: 2);
            var embeddingTable = gpu.TensorAddScalar(new Tensor<float>(new[]
            {
                0f, 1f,
                10f, 11f,
                20f, 21f,
                30f, 31f,
                40f, 41f,
            }, new[] { 5, 2 }), 0f);
            var embedded = gpu.Embedding(indices, embeddingTable);
            var interfaceEmbedded = engine.Embedding(indices, embeddingTable);

            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                indices.DataVector.GetBackingArrayUnsafe()!));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                rowIndices.DataVector.GetBackingArrayUnsafe()!));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                columnIndices.DataVector.GetBackingArrayUnsafe()!));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                mask.DataVector.GetBackingArrayUnsafe()!));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(copied.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(filled.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(put.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(taken.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(selected.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(gathered.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(tensorGathered.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(scatter.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(rowScatter.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(scatterAdd.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(scatterMax.DataVector));
            Assert.NotNull(scatterArgmax);
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                scatterArgmax!.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(scatterMean.DataVector));
            Assert.NotNull(scatterMeanCounts);
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                scatterMeanCounts!.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                scatterAddGradient.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                scatterMeanGradient.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                scatterMaxGradient.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                scatterSoftmaxGradient.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(scattered.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(maskedFilled.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(indexPut.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(indexAdded.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(packedGather.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(packedScatter.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(embedded.DataVector));
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                interfaceEmbedded.DataVector));

            Assert.Equal(new[] { 30f, 31f, 10f, 11f, 40f, 41f, 20f, 21f }, copied.GetDataArray());
            Assert.Equal(Enumerable.Repeat(9f, 8).ToArray(), filled.GetDataArray());
            Assert.Equal(new[] { 30f, 10f, 40f, 20f }, put.GetDataArray());
            Assert.Equal(new[] { 20f, 40f, 10f, 30f }, taken.GetDataArray());
            Assert.Equal(new[] { 2f, 3f, 6f, 7f, 0f, 1f, 4f, 5f }, selected.GetDataArray());
            Assert.Equal(selected.GetDataArray(), gathered.GetDataArray());
            Assert.Equal(selected.GetDataArray(), tensorGathered.GetDataArray());
            Assert.Equal(new[] { 30f, 31f, 10f, 11f, 40f, 41f, 20f, 21f },
                scatter.GetDataArray());
            Assert.Equal(new[] { 0f, 100f, 2f, 101f, 102f, 5f, 103f, 7f },
                rowScatter.GetDataArray());
            Assert.Equal(new[]
            {
                30f, 31f, 10f, 11f, 40f, 41f, 20f, 21f, 0f, 0f
            }, scatterAdd.GetDataArray());
            Assert.Equal(new[]
            {
                30f, 31f, 10f, 11f, 40f, 41f, 20f, 21f,
                float.NegativeInfinity, float.NegativeInfinity
            }, scatterMax.GetDataArray());
            Assert.Equal(new[] { 2, 2, 0, 0, 3, 3, 1, 1, -1, -1 },
                scatterArgmax.GetDataArray());
            Assert.Equal(new[]
            {
                30f, 31f, 10f, 11f, 40f, 41f, 20f, 21f, 0f, 0f
            }, scatterMean.GetDataArray());
            Assert.Equal(new[] { 1, 1, 1, 1, 0 }, scatterMeanCounts.GetDataArray());
            var expectedScatterGradient = new[] { 2f, 3f, 6f, 7f, 0f, 1f, 4f, 5f };
            Assert.Equal(expectedScatterGradient, scatterAddGradient.GetDataArray());
            Assert.Equal(expectedScatterGradient, scatterMeanGradient.GetDataArray());
            Assert.Equal(expectedScatterGradient, scatterMaxGradient.GetDataArray());
            Assert.Equal(new float[8], scatterSoftmaxGradient.GetDataArray());
            Assert.Equal(new[] { 10f, 1f, 20f, 3f, 30f, 5f }, scattered.GetDataArray());
            Assert.Equal(new[] { 99f, 1f, 99f, 3f, 99f, 5f }, maskedFilled.GetDataArray());
            Assert.Equal(new[] { 0f, 1f, 80f, 3f, 70f, 5f, 6f, 7f, 8f, 90f, 10f, 11f },
                indexPut.GetDataArray());
            Assert.Equal(new[] { 20f, 22f, 32f, 34f, 14f, 16f }, indexAdded.GetDataArray());
            Assert.Equal(new byte[] { 2, 3, 6, 7, 0, 1, 4, 5 }, packedGather.GetDataArray());
            Assert.Equal(new byte[] { 120, 121, 100, 101, 130, 131, 110, 111 },
                packedScatter.GetDataArray());
            Assert.Equal(new[] { 10f, 11f, 30f, 31f, 0f, 1f, 20f, 21f },
                embedded.GetDataArray());
            Assert.Equal(embedded.GetDataArray(), interfaceEmbedded.GetDataArray());

            var repeated = new Tensor<int>(new[] { 2, 0, 2 }, new[] { 3 });
            var duplicatePut = gpu.TensorPut(
                new Tensor<float>(new[] { 0f, 1f, 2f }, new[] { 3 }), repeated,
                new Tensor<float>(new[] { 10f, 20f, 30f }, new[] { 3 }));
            var duplicateCopy = gpu.TensorIndexCopy(
                new Tensor<float>(new[] { 0f, 1f, 2f, 3f, 4f, 5f }, new[] { 3, 2 }), 0, repeated,
                new Tensor<float>(new[] { 10f, 11f, 20f, 21f, 30f, 31f }, new[] { 3, 2 }));
            var duplicateAdd = gpu.TensorIndexAdd(
                new Tensor<float>(new[] { 0f, 1f, 2f, 3f, 4f, 5f }, new[] { 3, 2 }), 0, repeated,
                new Tensor<float>(new[] { 10f, 11f, 20f, 21f, 30f, 31f }, new[] { 3, 2 }));
            var tieMax = engine.ScatterMax(
                new Tensor<float>(new[] { 5f, 7f, 5f, 9f, float.NegativeInfinity, 4f },
                    new[] { 3, 2 }),
                new Tensor<int>(new[] { 0, 0, 1 }, new[] { 3 }), out var tieArgmax, 0, 2);
            Assert.Equal(new[] { 20f, 1f, 30f }, duplicatePut.GetDataArray());
            Assert.Equal(new[] { 20f, 21f, 2f, 3f, 30f, 31f }, duplicateCopy.GetDataArray());
            Assert.Equal(new[] { 20f, 22f, 2f, 3f, 44f, 47f }, duplicateAdd.GetDataArray());
            Assert.Equal(new[] { 5f, 9f, float.NegativeInfinity, 4f }, tieMax.GetDataArray());
            Assert.NotNull(tieArgmax);
            Assert.Equal(new[] { 0, 1, -1, 2 }, tieArgmax!.GetDataArray());
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = false;
        }
    }

    [SkippableFact]
    public void GraphAttention_BatchedResidentEdgesHaveNoInternalReadbacksAndMatchCpu()
    {
        SkipIfNoDirectGpu();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;
        IEngine cpu = new CpuEngine();
        var (_, sourceIndices) = gpuEngine.TensorSort(gpuEngine.TensorAddScalar(
            new Tensor<float>(new[] { 30f, 10f, 40f, 20f }, new[] { 4 }), 0f));
        var (_, targetIndices) = gpuEngine.TensorSort(gpuEngine.TensorAddScalar(
            new Tensor<float>(new[] { 20f, 30f, 10f, 40f }, new[] { 4 }), 0f));
        var nodes = new Tensor<float>(Enumerable.Range(0, 24)
            .Select(value => (value - 11) * 0.125f).ToArray(), new[] { 2, 4, 3 });
        var sourceWeights = new Tensor<float>(new[] { 0.25f, -0.5f, 0.75f }, new[] { 3 });
        var targetWeights = new Tensor<float>(new[] { -0.4f, 0.2f, 0.6f }, new[] { 3 });

        GpuLaunchProbe.Reset();
        var actual = gpu.GraphAttention(nodes, sourceIndices, targetIndices,
            sourceWeights, targetWeights, 0.2, out var actualCoefficients);

        Assert.Equal(0, GpuLaunchProbe.Readbacks);
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
            sourceIndices.DataVector.GetBackingArrayUnsafe()!));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
            targetIndices.DataVector.GetBackingArrayUnsafe()!));
        var actualBacking = actual.DataVector.GetBackingArrayUnsafe();
        var coefficientBacking = actualCoefficients.DataVector.GetBackingArrayUnsafe();
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actual.DataVector)
            || (actualBacking is not null &&
                AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actualBacking)));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                actualCoefficients.DataVector)
            || (coefficientBacking is not null &&
                AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(coefficientBacking)));

        var expected = cpu.GraphAttention(nodes,
            new Tensor<int>(new[] { 1, 3, 0, 2 }, new[] { 4 }),
            new Tensor<int>(new[] { 2, 0, 1, 3 }, new[] { 4 }),
            sourceWeights, targetWeights, 0.2, out var expectedCoefficients);
        var expectedData = expected.GetDataArray();
        var actualData = actual.GetDataArray();
        for (int i = 0; i < expectedData.Length; i++)
            Assert.True(AreCloseRelative(expectedData[i], actualData[i], 1e-4f),
                $"GraphAttention mismatch at {i}: CPU={expectedData[i]}, GPU={actualData[i]}");
        var expectedCoefficientData = expectedCoefficients.GetDataArray();
        var actualCoefficientData = actualCoefficients.GetDataArray();
        for (int i = 0; i < expectedCoefficientData.Length; i++)
            Assert.True(AreCloseRelative(expectedCoefficientData[i], actualCoefficientData[i], 1e-4f),
                $"GraphAttention coefficient mismatch at {i}: CPU={expectedCoefficientData[i]}, GPU={actualCoefficientData[i]}");
    }

    [SkippableFact]
    public void UniformMeshLaplacian_ResidentFacesHaveNoInternalReadbacksAndMatchCpuExactly()
    {
        SkipIfNoDirectGpu();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;
        IEngine cpu = new CpuEngine();
        var boundaries = new Tensor<float>(new[] { 0f, 1f, 2f, 3f }, new[] { 4 });
        var faceValues = new Tensor<float>(new[] { 0f, 1f, 2f, 0f, 2f, 3f }, new[] { 6 });
        var residentFaceData = gpu.TensorSearchSorted(boundaries, faceValues, right: false);
        var residentFaces = gpu.Reshape(residentFaceData, new[] { 2, 3 });
        var hostFaces = new Tensor<int>(new[] { 0, 1, 2, 0, 2, 3 }, new[] { 2, 3 });
        var vertices = new Tensor<float>(new[]
        {
            0f, 0f, 0f,
            1f, 0f, 0f,
            1f, 1f, 0f,
            0f, 1f, 0f,
        }, new[] { 4, 3 });

        GpuLaunchProbe.Reset();
        var actual = gpu.ComputeMeshLaplacian(
            vertices, residentFaces, LaplacianType.Uniform);

        Assert.Equal(0, GpuLaunchProbe.Readbacks);
        var faceBacking = residentFaces.DataVector.GetBackingArrayUnsafe();
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                residentFaces.DataVector)
            || (faceBacking is not null &&
                AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(faceBacking)));
        Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(actual.DataVector));

        var expected = cpu.ComputeMeshLaplacian(vertices, hostFaces, LaplacianType.Uniform);
        Assert.Equal(expected.GetDataArray(), actual.GetDataArray());
    }

    [SkippableFact]
    public void StandaloneNormalizationGpu_ResidentStateHasNoInternalReadbacksAndMatchesCpu()
    {
        SkipIfNoDirectGpu();
        using var gpuEngine = new DirectGpuTensorEngine();
        IEngine gpu = gpuEngine;
        IEngine cpu = new CpuEngine();

        var input = new Tensor<float>(new[]
        {
            -2f, -1f, 0f, 1f,
            2f, 3f, 4f, 5f,
        }, new[] { 2, 4 });
        var gamma = new Tensor<float>(new[] { 0.5f, 1f, 1.5f, 2f }, new[] { 4 });
        var beta = new Tensor<float>(new[] { -1f, 0f, 1f, 2f }, new[] { 4 });
        var gradOutput = new Tensor<float>(new[]
        {
            0.25f, -0.5f, 0.75f, -1f,
            1.25f, -1.5f, 1.75f, -2f,
        }, new[] { 2, 4 });
        var residentInput = gpu.TensorAddScalar(input, 0f);
        var residentGamma = gpu.TensorAddScalar(gamma, 0f);
        var residentBeta = gpu.TensorAddScalar(beta, 0f);
        var residentGradOutput = gpu.TensorAddScalar(gradOutput, 0f);

        var batchInput = new Tensor<float>(new[]
        {
            -3f, -2f, -1f, 0f,
            1f, 2f, 3f, 4f,
            5f, 6f, 7f, 8f,
            9f, 10f, 11f, 12f,
        }, new[] { 2, 2, 2, 2 });
        var batchGamma = new Tensor<float>(new[] { 0.75f, 1.25f }, new[] { 2 });
        var batchBeta = new Tensor<float>(new[] { -0.5f, 0.5f }, new[] { 2 });
        var residentBatchInput = gpu.TensorAddScalar(batchInput, 0f);
        var residentBatchGamma = gpu.TensorAddScalar(batchGamma, 0f);
        var residentBatchBeta = gpu.TensorAddScalar(batchBeta, 0f);
        Tensor<float> runningMean = gpu.TensorAddScalar(
            new Tensor<float>(new[] { 0f, 0f }, new[] { 2 }), 0f);
        Tensor<float> runningVar = gpu.TensorAddScalar(
            new Tensor<float>(new[] { 1f, 1f }, new[] { 2 }), 0f);

        GpuLaunchProbe.Reset();
        var (layerOutput, saveMean, saveInvVar) = gpuEngine.LayerNormGpu(
            residentInput, residentGamma, residentBeta, 1e-5);
        var (gradInput, gradGamma, gradBeta) = gpuEngine.LayerNormBackwardGpu(
            residentGradOutput, residentInput, residentGamma, saveMean, saveInvVar, 1e-5);
        var (batchOutput, batchSaveMean, batchSaveVar) = gpuEngine.FusedBatchNormGpu(
            residentBatchInput, residentBatchGamma, residentBatchBeta,
            ref runningMean, ref runningVar, 1e-5, 0.1, training: true);

        Assert.Equal(0, GpuLaunchProbe.Readbacks);
        foreach (var tensor in new[]
        {
            layerOutput, saveMean, saveInvVar, gradInput, gradGamma, gradBeta,
            batchOutput, batchSaveMean!, batchSaveVar!, runningMean, runningVar,
        })
        {
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(
                tensor.DataVector));
        }

        var expectedLayerOutput = cpu.LayerNorm(
            input, gamma, beta, 1e-5, out var expectedMean, out var expectedVariance);
        var expectedGradInput = cpu.LayerNormBackward(
            gradOutput, input, gamma, expectedMean, expectedVariance, 1e-5,
            out var expectedGradGamma, out var expectedGradBeta);
        var filledGradOutput = new Tensor<float>(input.Shape.ToArray());
        gpu.TensorFill(filledGradOutput, 1f);
        GpuLaunchProbe.Reset();
        var publicGradInput = gpu.LayerNormBackward(
            filledGradOutput, input, gamma, expectedMean, expectedVariance, 1e-5,
            out var publicGradGamma, out var publicGradBeta);
        var expectedPublicGradInput = cpu.LayerNormBackward(
            filledGradOutput, input, gamma, expectedMean, expectedVariance, 1e-5,
            out var expectedPublicGradGamma, out var expectedPublicGradBeta);
        Assert.Equal(0, GpuLaunchProbe.Readbacks);
        var expectedBatchOutput = cpu.BatchNorm(
            batchInput, batchGamma, batchBeta, 1e-5, out _, out _);

        AssertClose(expectedLayerOutput, layerOutput, 1e-4f);
        AssertClose(expectedGradInput, gradInput, 1e-4f);
        AssertClose(expectedGradGamma, gradGamma, 1e-4f);
        AssertClose(expectedGradBeta, gradBeta, 1e-4f);
        AssertClose(expectedPublicGradInput, publicGradInput, 1e-4f);
        AssertClose(expectedPublicGradGamma, publicGradGamma, 1e-4f);
        AssertClose(expectedPublicGradBeta, publicGradBeta, 1e-4f);
        AssertClose(expectedBatchOutput, batchOutput, 1e-4f);
    }

    #endregion

    #region Helper Methods

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float tolerance)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var expectedData = expected.GetDataArray();
        var actualData = actual.GetDataArray();
        for (int i = 0; i < expectedData.Length; i++)
            Assert.True(AreCloseRelative(expectedData[i], actualData[i], tolerance),
                $"Tensor mismatch at {i}: CPU={expectedData[i]}, GPU={actualData[i]}");
    }

    /// <summary>
    /// Creates a signaling NaN value.
    /// Uses unsafe code to avoid BitConverter.Int32BitsToSingle which is not available in .NET Framework 4.7.1.
    /// </summary>
    private static unsafe float CreateSignalingNaN()
    {
        int bits = 0x7F800001; // Signaling NaN bit pattern
        return *(float*)&bits;
    }

    private static unsafe float Int32BitsToSingleCompat(int bits) => *(float*)&bits;

    #endregion
}
