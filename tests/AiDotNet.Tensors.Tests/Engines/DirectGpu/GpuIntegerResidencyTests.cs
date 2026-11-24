// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Integer tensors on the device through the engine: a chain over resident operands stays there, computed in the
/// tensor's own type, and its values are the CPU engine's bit for bit, overflow included.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuIntegerResidencyTests
{
    private const int Large = DirectGpuTensorEngine.IntegerElementwiseMinLength + 4097;

    private static (DirectGpuTensorEngine Engine, IDirectGpuBackend Backend)? Device(GpuElementType type)
    {
        if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpu) return null;
        if (gpu.GetBackend() is not { } backend) return null;
        if (backend is not IGpuIntegerElementwise elementwise || !elementwise.SupportsIntegerElementwise(type)) return null;
        if (backend is not IGpuIntegerReductions reductions || !reductions.SupportsIntegerReductions(type)) return null;
        return (gpu, backend);
    }

    private static long[] Raw(int length, int seed)
    {
        var values = new long[length];
        ulong state = (ulong)seed * 0x9E3779B97F4A7C15UL + 1UL;
        for (int i = 0; i < length; i++)
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values[i] = i % 3 == 0 ? (long)(state % 2001UL) - 1000 : unchecked((long)state);
        }

        return values;
    }

    private static void RunChain<T>(GpuElementType type, Func<long, T> narrow, T scalar, bool explicitPreserve)
        where T : unmanaged, IEquatable<T>
    {
        var device = Device(type);
        Skip.If(device is null, $"No device with {type} kernels on this host.");
        var (gpu, _) = device.Value;
        var cpu = (IEngine)new CpuEngine();
        IEngine engine = gpu;
        using var policy = explicitPreserve ? new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve) : null;
        Skip.If(!explicitPreserve && GpuExecutionPolicyScope.ExactTypesSpeedFirstByDefault, "This process opted out of the exact-type default.");

        var xValues = Array.ConvertAll(Raw(Large, 11), v => narrow(v));
        var yValues = Array.ConvertAll(Raw(Large, 12), v => narrow(v));
        var x = new Tensor<T>((T[])xValues.Clone(), new[] { Large });
        var y = new Tensor<T>((T[])yValues.Clone(), new[] { Large });
        var hostX = new Tensor<T>((T[])xValues.Clone(), new[] { Large });
        var hostY = new Tensor<T>((T[])yValues.Clone(), new[] { Large });
        gpu.RegisterPersistentTensor(x, PersistentTensorRole.Weights);
        gpu.RegisterPersistentTensor(y, PersistentTensorRole.Weights);
        try
        {
            var sum = engine.TensorAdd(x, y);
            Assert.True(gpu.IsTypedResident(sum), $"The {type} sum was not left on the device.");
            var product = engine.TensorMultiply(sum, x);
            var difference = engine.TensorSubtract(product, y);
            var capped = engine.TensorMin(engine.TensorMax(difference, x), sum);
            Assert.True(gpu.IsTypedResident(capped), $"The {type} min/max result was not left on the device.");
            var negated = engine.TensorNegate(capped);
            var shifted = engine.TensorAddScalar(negated, scalar);
            var scaled = engine.TensorMultiplyScalar(shifted, scalar);
            var lowered = engine.TensorSubtractScalar(scaled, scalar);
            Assert.True(gpu.IsTypedResident(lowered), $"The {type} chain's result was not left on the device.");
            T total = engine.TensorSum(lowered);

            var expected = cpu.TensorSubtractScalar(cpu.TensorMultiplyScalar(cpu.TensorAddScalar(cpu.TensorNegate(
                cpu.TensorMin(cpu.TensorMax(cpu.TensorSubtract(cpu.TensorMultiply(cpu.TensorAdd(hostX, hostY), hostX), hostY), hostX),
                    cpu.TensorAdd(hostX, hostY))), scalar), scalar), scalar);
            Assert.Equal(cpu.TensorSum(expected), total);
            var actualValues = lowered.GetReadOnlyDataArray();
            var expectedValues = expected.GetReadOnlyDataArray();
            for (int i = 0; i < Large; i++)
                Assert.True(expectedValues[i].Equals(actualValues[i]), $"{type} chain at {i}: CPU {expectedValues[i]}, device {actualValues[i]}.");
        }
        finally
        {
            gpu.UnregisterPersistentTensor(x);
            gpu.UnregisterPersistentTensor(y);
            gpu.ClearTypedResidents();
        }
    }

    [SkippableFact]
    public void Int32_chain_stays_on_the_device_and_equals_the_cpu_engine()
        => RunChain(GpuElementType.Int32, v => unchecked((int)v), 7919, explicitPreserve: false);

    [SkippableFact]
    public void Int64_chain_stays_on_the_device_and_equals_the_cpu_engine()
        => RunChain(GpuElementType.Int64, v => v, 0x1_0000_0003L, explicitPreserve: false);

    [SkippableFact]
    public void Int16_chain_stays_on_the_device_under_a_preserving_policy()
        => RunChain(GpuElementType.Int16, v => unchecked((short)v), (short)311, explicitPreserve: true);

    [SkippableFact]
    public void Int8_chain_stays_on_the_device_under_a_preserving_policy()
        => RunChain(GpuElementType.Int8, v => unchecked((sbyte)v), (sbyte)-13, explicitPreserve: true);

    [SkippableFact]
    public void Plain_host_operands_stay_on_the_cpu_and_a_host_write_invalidates_the_device_copy()
    {
        var device = Device(GpuElementType.Int32);
        Skip.If(device is null, "No device with Int32 kernels on this host.");
        Skip.If(GpuExecutionPolicyScope.ExactTypesSpeedFirstByDefault, "This process opted out of the exact-type default.");
        var (gpu, _) = device.Value;
        IEngine engine = gpu;
        var values = Array.ConvertAll(Raw(Large, 21), v => unchecked((int)v));
        try
        {
            // Two host tensors: uploading both costs more than the CPU engine's add.
            var a = new Tensor<int>((int[])values.Clone(), new[] { Large });
            var b = new Tensor<int>((int[])values.Clone(), new[] { Large });
            var hostSum = engine.TensorAdd(a, b);
            Assert.False(gpu.IsTypedResident(hostSum));
            Assert.Equal(unchecked(values[5] + values[5]), hostSum[5]);

            // A short tensor is the CPU engine's whatever is resident.
            var small = new Tensor<int>(new[] { 1, 2, 3 }, new[] { 3 });
            gpu.RegisterPersistentTensor(small, PersistentTensorRole.Weights);
            Assert.False(gpu.IsTypedResident(engine.TensorAdd(small, small)));
            gpu.UnregisterPersistentTensor(small);

            // A speed-first scope does not take the integer route.
            var p = new Tensor<int>((int[])values.Clone(), new[] { Large });
            gpu.RegisterPersistentTensor(p, PersistentTensorRole.Weights);
            var first = engine.TensorAdd(p, p);
            Assert.True(gpu.IsTypedResident(first));
            Assert.Equal(unchecked(values[9] + values[9]), first[9]);

            // The host writes the registered tensor and says so: the next op sees the new values.
            p.AsWritableSpan()[9] = 41;
            gpu.InvalidatePersistentTensor(p);
            var second = engine.TensorAdd(p, p);
            Assert.Equal(82, second[9]);
            gpu.UnregisterPersistentTensor(p);
        }
        finally
        {
            gpu.ClearTypedResidents();
        }
    }

    [SkippableFact]
    public void BFloat16_divide_and_functions_stay_on_the_device_and_equal_the_function_rounded_once()
    {
        const GpuElementType Type = GpuElementType.BFloat16;
        Skip.If(AiDotNetEngine.Current is not DirectGpuTensorEngine, "No GPU engine on this host.");
        var gpu = (DirectGpuTensorEngine)AiDotNetEngine.Current;
        Skip.If(gpu.GetBackend() is not IGpuNarrowFloatElementwise narrow || gpu.GetBackend() is not { } held || !held.SupportsElementType(Type)
            || !narrow.SupportsNarrowFloatBinary(GpuNarrowFloatBinaryOperation.Divide, Type)
            || !narrow.SupportsNarrowFloatUnary(GpuNarrowFloatUnaryOperation.Sigmoid, Type), "No device with BFloat16 kernels on this host.");
        IEngine engine = gpu;
        using var policy = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);

        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(37);
        var xValues = new BFloat16[Large];
        var yValues = new BFloat16[Large];
        for (int i = 0; i < Large; i++)
        {
            xValues[i] = BFloat16.FromFloat((float)((random.NextDouble() - 0.5) * 8.0));
            // Positive and away from zero: a divisor, and an operand of the root and the logarithm.
            yValues[i] = BFloat16.FromFloat((float)(0.25 + random.NextDouble() * 6.0));
        }

        var x = new Tensor<BFloat16>((BFloat16[])xValues.Clone(), new[] { Large });
        var y = new Tensor<BFloat16>((BFloat16[])yValues.Clone(), new[] { Large });
        gpu.RegisterPersistentTensor(x, PersistentTensorRole.Weights);
        gpu.RegisterPersistentTensor(y, PersistentTensorRole.Weights);
        try
        {
            var results = new (string Name, Tensor<BFloat16> Result, Func<double, double, double> Function, bool Exact)[]
            {
                ("Divide", engine.TensorDivide(x, y), (a, b) => (float)a / (float)b, true),
                ("Abs", engine.TensorAbs(x), (a, _) => Math.Abs(a), true),
                ("Sqrt", engine.TensorSqrt(y), (_, b) => Math.Sqrt(b), false),
                ("Exp", engine.TensorExp(x), (a, _) => Math.Exp(a), false),
                ("Log", engine.TensorLog(y), (_, b) => Math.Log(b), false),
                ("Tanh", engine.TensorTanh(x), (a, _) => Math.Tanh(a), false),
                ("Sigmoid", engine.TensorSigmoid(x), (a, _) => 1.0 / (1.0 + Math.Exp(-a)), false),
            };
            foreach (var (name, result, function, exact) in results)
            {
                Assert.True(gpu.IsTypedResident(result), $"The BFloat16 {name} was not left on the device.");
                var values = result.GetReadOnlyDataArray();
                for (int i = 0; i < Large; i++)
                {
                    float expected = (float)function((float)xValues[i], (float)yValues[i]);
                    float got = (float)values[i];
                    if (exact)
                        Assert.True(BFloat16.FromFloat(expected).RawValue == values[i].RawValue,
                            $"{name} at {i} of ({(float)xValues[i]:R}, {(float)yValues[i]:R}): expected {expected:R}, device {got:R}.");
                    else
                        // Two units in BFloat16's last place (7 significand bits).
                        Assert.True(Math.Abs(expected - got) <= 1e-6f + Math.Abs(expected) / 64f,
                            $"{name} at {i} of ({(float)xValues[i]:R}, {(float)yValues[i]:R}): expected {expected:R}, device {got:R}.");
                }
            }
        }
        finally
        {
            gpu.ClearTypedResidents();
        }
    }

    [SkippableFact]
    public void BFloat16_chain_stays_on_the_device_and_equals_the_cpu_engine()
    {
        const GpuElementType Type = GpuElementType.BFloat16;
        Skip.If(AiDotNetEngine.Current is not DirectGpuTensorEngine, "No GPU engine on this host.");
        var gpu = (DirectGpuTensorEngine)AiDotNetEngine.Current;
        Skip.If(gpu.GetBackend() is not IGpuNarrowFloatElementwise narrow || gpu.GetBackend() is not { } held || !held.SupportsElementType(Type)
            || !narrow.SupportsNarrowFloatBinary(GpuNarrowFloatBinaryOperation.Multiply, Type)
            || !narrow.SupportsNarrowFloatUnary(GpuNarrowFloatUnaryOperation.Gelu, Type), "No device with BFloat16 kernels on this host.");
        var cpu = (IEngine)new CpuEngine();
        IEngine engine = gpu;
        using var policy = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);

        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(31);
        var xValues = new BFloat16[Large];
        var yValues = new BFloat16[Large];
        for (int i = 0; i < Large; i++)
        {
            xValues[i] = BFloat16.FromFloat((float)((random.NextDouble() - 0.5) * 8.0));
            yValues[i] = BFloat16.FromFloat((float)((random.NextDouble() - 0.5) * 8.0));
        }

        var x = new Tensor<BFloat16>((BFloat16[])xValues.Clone(), new[] { Large });
        var y = new Tensor<BFloat16>((BFloat16[])yValues.Clone(), new[] { Large });
        var hostX = new Tensor<BFloat16>((BFloat16[])xValues.Clone(), new[] { Large });
        var hostY = new Tensor<BFloat16>((BFloat16[])yValues.Clone(), new[] { Large });
        gpu.RegisterPersistentTensor(x, PersistentTensorRole.Weights);
        gpu.RegisterPersistentTensor(y, PersistentTensorRole.Weights);
        try
        {
            var sum = engine.TensorAdd(x, y);
            Assert.True(gpu.IsTypedResident(sum), "The BFloat16 sum was not left on the device.");
            var negated = engine.TensorNegate(engine.TensorSubtract(engine.TensorMultiply(sum, x), y));
            var rectified = engine.TensorReLU(negated);
            var smooth = engine.TensorGELU(negated);
            Assert.True(gpu.IsTypedResident(rectified), "The BFloat16 chain's result was not left on the device.");
            Assert.True(gpu.IsTypedResident(smooth), "The BFloat16 GELU was not left on the device.");

            var hostNegated = cpu.TensorNegate(cpu.TensorSubtract(cpu.TensorMultiply(cpu.TensorAdd(hostX, hostY), hostX), hostY));
            var exact = negated.GetReadOnlyDataArray();
            var exactHost = hostNegated.GetReadOnlyDataArray();
            var relu = rectified.GetReadOnlyDataArray();
            var reluHost = cpu.TensorReLU(hostNegated).GetReadOnlyDataArray();
            var gelu = smooth.GetReadOnlyDataArray();
            for (int i = 0; i < Large; i++)
            {
                Assert.True(exactHost[i].RawValue == exact[i].RawValue, $"Arithmetic at {i}: CPU {(float)exactHost[i]:R}, device {(float)exact[i]:R}.");
                // The value: the zero of a negative input may carry either sign.
                Assert.True((float)reluHost[i] == (float)relu[i], $"ReLU at {i}: CPU {(float)reluHost[i]:R}, device {(float)relu[i]:R}.");
                // GELU against its formula evaluated in float and rounded once, which is what the device does. (The CPU
                // engine rounds every intermediate to BFloat16 and drifts several percent from the true value.)
                float v = (float)exactHost[i];
                float z = Math.Min(20f, Math.Max(-20f, 0.7978845608028654f * (v + 0.044715f * v * v * v)));
                float expected = 0.5f * v * (1f + (float)Math.Tanh(z));
                // Two units in BFloat16's last place (7 significand bits), plus the cancellation floor near zero.
                Assert.True(Math.Abs(expected - (float)gelu[i]) <= 1e-4f + Math.Abs(expected) / 64f,
                    $"GELU at {i} of {(float)exactHost[i]:R}: CPU {expected:R}, device {(float)gelu[i]:R}.");
            }
        }
        finally
        {
            gpu.UnregisterPersistentTensor(x);
            gpu.UnregisterPersistentTensor(y);
            gpu.ClearTypedResidents();
        }
    }
}
