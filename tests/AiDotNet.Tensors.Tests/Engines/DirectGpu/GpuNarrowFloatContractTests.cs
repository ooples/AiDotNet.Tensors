// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The 16-bit float kernels (<see cref="GpuElementType.Float16"/>, <see cref="GpuElementType.BFloat16"/>) on typed
/// buffers: each element widened to a float, computed there and rounded back as the host conversions round, for every
/// (operation, type) pair the backend says it has.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuNarrowFloatContractTests
{
    private static readonly GpuElementType[] Types = { GpuElementType.Float16, GpuElementType.BFloat16 };

    private static readonly int[] Sizes = { 1, 255, 257, 4097 };

    private static IDirectGpuBackend? Device()
        => AiDotNetEngine.Current is DirectGpuTensorEngine gpu && gpu.GetBackend() is IGpuNarrowFloatElementwise { } backend
            ? (IDirectGpuBackend)backend
            : null;

    private static float Widen(GpuElementType type, ushort bits)
        => type == GpuElementType.Float16 ? (float)BitConverter.UInt16BitsToHalf(bits) : BFloat16.ToFloat(BFloat16.FromRawBits(bits));

    private static ushort Narrow(GpuElementType type, float value)
        => type == GpuElementType.Float16 ? BitConverter.HalfToUInt16Bits((Half)value) : BFloat16.FromFloat(value).RawValue;

    // Bit patterns across the whole type: zeros of both signs, subnormals, both infinities, the largest finite values;
    // no NaN operands (their payloads are not part of the contract).
    private static ushort[] Patterns(GpuElementType type, int length, int seed, bool moderate = false)
    {
        var values = new ushort[length];
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        ushort[] special = type == GpuElementType.Float16
            ? new ushort[] { 0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x3C00, 0xBC00 }
            : new ushort[] { 0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x3F80, 0xBF80 };
        for (int i = 0; i < length; i++)
        {
            if (moderate)
            {
                values[i] = Narrow(type, (float)((random.NextDouble() - 0.5) * 16.0));
                continue;
            }

            ushort bits = i % 7 == 3 ? special[(i / 7) % special.Length] : (ushort)random.Next(0, 65536);
            if (float.IsNaN(Widen(type, bits))) bits = special[i % special.Length];
            values[i] = bits;
        }

        return values;
    }

    private static IGpuBuffer Upload(IDirectGpuBackend backend, GpuElementType type, ushort[] bits)
    {
        var buffer = backend.AllocateBuffer(bits.Length, type);
        try
        {
            if (type == GpuElementType.Float16) backend.UploadBuffer<Half>(buffer, Array.ConvertAll(bits, BitConverter.UInt16BitsToHalf));
            else backend.UploadBuffer<BFloat16>(buffer, Array.ConvertAll(bits, BFloat16.FromRawBits));
        }
        catch
        {
            buffer.Dispose();
            throw;
        }

        return buffer;
    }

    private static ushort[] Download(IDirectGpuBackend backend, GpuElementType type, IGpuBuffer buffer, int length)
    {
        if (type == GpuElementType.Float16)
        {
            var halves = new Half[length];
            backend.DownloadBuffer<Half>(buffer, halves);
            return Array.ConvertAll(halves, BitConverter.HalfToUInt16Bits);
        }

        var values = new BFloat16[length];
        backend.DownloadBuffer<BFloat16>(buffer, values);
        return Array.ConvertAll(values, v => v.RawValue);
    }

    private static void AssertSame(GpuElementType type, float expected, ushort actual, string what)
    {
        if (float.IsNaN(expected))
        {
            Assert.True(float.IsNaN(Widen(type, actual)), $"{what}: host NaN, device 0x{actual:X4}.");
            return;
        }

        ushort bits = Narrow(type, expected);
        Assert.True(bits == actual, $"{what}: host 0x{bits:X4} ({expected:R}), device 0x{actual:X4}.");
    }

    [SkippableTheory]
    [InlineData(GpuNarrowFloatBinaryOperation.Add)]
    [InlineData(GpuNarrowFloatBinaryOperation.Subtract)]
    [InlineData(GpuNarrowFloatBinaryOperation.Multiply)]
    public void Binary_kernels_round_as_the_host_conversion_does(GpuNarrowFloatBinaryOperation operation)
    {
        var backend = Device();
        Skip.If(backend is null, "No GPU backend on this host.");
        var elementwise = (IGpuNarrowFloatElementwise)backend;
        int ran = 0;
        foreach (var type in Types)
        {
            if (!backend.SupportsElementType(type) || !elementwise.SupportsNarrowFloatBinary(operation, type)) continue;
            foreach (int size in Sizes)
            {
                var left = Patterns(type, size, 1);
                var right = Patterns(type, size, 2);
                using var a = Upload(backend, type, left);
                using var b = Upload(backend, type, right);
                using var output = backend.AllocateBuffer(size, type);
                Assert.Equal(size, output.Size);
                elementwise.BinaryNarrowFloat(operation, type, a, b, output, size);
                var actual = Download(backend, type, output, size);
                for (int i = 0; i < size; i++)
                {
                    float x = Widen(type, left[i]), y = Widen(type, right[i]);
                    float expected = operation switch
                    {
                        GpuNarrowFloatBinaryOperation.Add => x + y,
                        GpuNarrowFloatBinaryOperation.Subtract => x - y,
                        _ => x * y,
                    };
                    AssertSame(type, expected, actual[i], $"{operation}<{type}> at {i} of {size} (0x{left[i]:X4}, 0x{right[i]:X4})");
                }

                // The output may be an operand.
                elementwise.BinaryNarrowFloat(operation, type, a, b, a, size);
                Assert.Equal(actual, Download(backend, type, a, size));
                // A float read of a 16-bit buffer is refused rather than returning half the bytes as floats.
                Assert.Throws<InvalidOperationException>(() => backend.DownloadBuffer(output));
                ran++;
            }
        }

        Skip.If(ran == 0, $"This device has no {operation} kernel for a 16-bit float type it holds.");
    }

    [SkippableTheory]
    [InlineData(GpuNarrowFloatUnaryOperation.Negate)]
    [InlineData(GpuNarrowFloatUnaryOperation.Relu)]
    [InlineData(GpuNarrowFloatUnaryOperation.Gelu)]
    public void Unary_kernels_equal_the_host(GpuNarrowFloatUnaryOperation operation)
    {
        var backend = Device();
        Skip.If(backend is null, "No GPU backend on this host.");
        var elementwise = (IGpuNarrowFloatElementwise)backend;
        int ran = 0;
        foreach (var type in Types)
        {
            if (!backend.SupportsElementType(type) || !elementwise.SupportsNarrowFloatUnary(operation, type)) continue;
            // One unit in the last place of the type, relative: 10 significand bits for half, 7 for bfloat16.
            float ulp = type == GpuElementType.Float16 ? 1f / 1024f : 1f / 128f;
            foreach (int size in Sizes)
            {
                var values = Patterns(type, size, 3, moderate: operation == GpuNarrowFloatUnaryOperation.Gelu);
                using var a = Upload(backend, type, values);
                using var output = backend.AllocateBuffer(size, type);
                elementwise.UnaryNarrowFloat(operation, type, a, output, size);
                var actual = Download(backend, type, output, size);
                for (int i = 0; i < size; i++)
                {
                    float x = Widen(type, values[i]);
                    float got = Widen(type, actual[i]);
                    string what = $"{operation}<{type}> at {i} of {size} (0x{values[i]:X4} = {x:R}) gave 0x{actual[i]:X4} = {got:R}";
                    switch (operation)
                    {
                        case GpuNarrowFloatUnaryOperation.Negate:
                            Assert.True((ushort)(values[i] ^ 0x8000) == actual[i], what);
                            break;
                        case GpuNarrowFloatUnaryOperation.Relu:
                            // The value, not the bits: the zero of a negative input may carry either sign.
                            Assert.True((x > 0f ? x : 0f) == got, what);
                            break;
                        default:
                            float z = Math.Min(20f, Math.Max(-20f, 0.7978845608028654f * (x + 0.044715f * x * x * x)));
                            float expected = 0.5f * x * (1f + MathF.Tanh(z));
                            Assert.True(Math.Abs(expected - got) <= 1e-6f + 2f * ulp * Math.Abs(expected), what + $", host {expected:R}");
                            break;
                    }
                }

                ran++;
            }
        }

        Skip.If(ran == 0, $"This device has no {operation} kernel for a 16-bit float type it holds.");
    }

    [SkippableFact]
    public void Matrix_multiply_accumulates_in_float_and_stores_the_type()
    {
        var backend = Device();
        Skip.If(backend is null, "No GPU backend on this host.");
        Skip.If(backend is not IGpuNarrowFloatMatrixMultiply, "No 16-bit float matrix multiply on this device.");
        var multiply = (IGpuNarrowFloatMatrixMultiply)backend;
        const int M = 5, K = 7, N = 3;
        int ran = 0;
        foreach (var type in Types)
        {
            if (!backend.SupportsElementType(type) || !multiply.SupportsNarrowFloatMatrixMultiply(type)) continue;
            Assert.True(backend.SupportsOperation(GpuOperationFamily.MatrixMultiply, type));
            // Small integers: every product and partial sum is exact in the type, whatever order the device adds in.
            var left = new ushort[M * K];
            var right = new ushort[K * N];
            for (int i = 0; i < left.Length; i++) left[i] = Narrow(type, (i * 5 % 7) - 3);
            for (int i = 0; i < right.Length; i++) right[i] = Narrow(type, (i * 3 % 5) - 2);
            using var a = Upload(backend, type, left);
            using var b = Upload(backend, type, right);
            using var c = backend.AllocateBuffer(M * N, type);
            multiply.MatrixMultiplyNarrowFloat(type, a, b, c, M, N, K);
            var actual = Download(backend, type, c, M * N);
            for (int row = 0; row < M; row++)
            {
                for (int column = 0; column < N; column++)
                {
                    float expected = 0f;
                    for (int j = 0; j < K; j++) expected += Widen(type, left[row * K + j]) * Widen(type, right[j * N + column]);
                    Assert.Equal(expected, Widen(type, actual[row * N + column]));
                }
            }

            Assert.Throws<ArgumentException>(() => multiply.MatrixMultiplyNarrowFloat(type, a, b, a, M, N, K));
            Assert.Throws<ArgumentOutOfRangeException>(() => multiply.MatrixMultiplyNarrowFloat(type, a, b, c, 0, N, K));
            ran++;
        }

        Skip.If(ran == 0, "This device multiplies no 16-bit float type it holds.");
    }

    [SkippableFact]
    public void The_capability_table_and_argument_checks_agree_with_the_interfaces()
    {
        var backend = Device();
        Skip.If(backend is null, "No GPU backend on this host.");
        var elementwise = (IGpuNarrowFloatElementwise)backend;
        foreach (var type in Types)
        {
            Assert.Equal(backend.SupportsElementType(type) && elementwise.SupportsNarrowFloatBinary(GpuNarrowFloatBinaryOperation.Add, type),
                backend.SupportsOperation(GpuOperationFamily.Elementwise, type));
            Assert.False(backend.SupportsOperation(GpuOperationFamily.Reduction, type));
        }

        Assert.False(elementwise.SupportsNarrowFloatBinary(GpuNarrowFloatBinaryOperation.Add, GpuElementType.Float32));
        Assert.False(elementwise.SupportsNarrowFloatUnary(GpuNarrowFloatUnaryOperation.Relu, GpuElementType.Int16));
        Skip.If(!backend.SupportsElementType(GpuElementType.Float16) || !backend.SupportsElementType(GpuElementType.BFloat16),
            "The argument checks below need both 16-bit float types held.");
        using var halves = backend.AllocateBuffer(16, GpuElementType.Float16);
        using var brains = backend.AllocateBuffer(16, GpuElementType.BFloat16);
        const GpuNarrowFloatBinaryOperation Add = GpuNarrowFloatBinaryOperation.Add;
        // A buffer typed as the other 16-bit float is refused, as is a count past a typed buffer or a wider type.
        Assert.Throws<ArgumentException>(() => elementwise.BinaryNarrowFloat(Add, GpuElementType.Float16, halves, brains, halves, 16));
        Assert.Throws<ArgumentException>(() => elementwise.UnaryNarrowFloat(GpuNarrowFloatUnaryOperation.Negate, GpuElementType.BFloat16, halves, brains, 16));
        Assert.Throws<ArgumentOutOfRangeException>(() => elementwise.BinaryNarrowFloat(Add, GpuElementType.Float16, halves, halves, halves, 17));
        Assert.Throws<ArgumentOutOfRangeException>(() => elementwise.BinaryNarrowFloat(Add, GpuElementType.Float16, halves, halves, halves, -1));
        Assert.Throws<ArgumentException>(() => elementwise.BinaryNarrowFloat(Add, GpuElementType.Float32, halves, halves, halves, 16));
    }
}

#endif
