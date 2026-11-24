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
public abstract class GpuNarrowFloatContractTests
{
    /// <summary>The backend under test, or null when its device is absent on this host.</summary>
    protected abstract IDirectGpuBackend? Device();

    /// <summary>Why <see cref="Device"/> returned null.</summary>
    protected abstract string SkipReason { get; }

    /// <summary>
    /// True for a backend that must run every operation on both types: a kernel it lacks then fails the test instead
    /// of skipping it.
    /// </summary>
    protected virtual bool RunsEveryOperation => false;

    private void RequireRan(int ran, int expected, string what)
    {
        if (RunsEveryOperation) Assert.True(ran == expected, $"{what}: {ran} of {expected} cases ran.");
        else Skip.If(ran == 0, $"This device has no {what} for a 16-bit float type it holds.");
    }

    private static readonly GpuElementType[] Types = { GpuElementType.Float16, GpuElementType.BFloat16 };

    private static readonly int[] Sizes = { 1, 2, 3, 255, 257, 4097, 70_001 };

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
        Skip.If(backend is null, SkipReason);
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

        RequireRan(ran, Types.Length * Sizes.Length, $"{operation} kernel");
    }

    [SkippableTheory]
    [InlineData(GpuNarrowFloatUnaryOperation.Negate)]
    [InlineData(GpuNarrowFloatUnaryOperation.Relu)]
    [InlineData(GpuNarrowFloatUnaryOperation.Gelu)]
    public void Unary_kernels_equal_the_host(GpuNarrowFloatUnaryOperation operation)
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
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

        RequireRan(ran, Types.Length * Sizes.Length, $"{operation} kernel");
    }

    // Past the tanh clamp GELU is the input itself (or a zero): large finite inputs, whose cube overflows binary32
    // for BFloat16, must come back as the host computes them and not as whatever an overflowed intermediate leaves.
    [SkippableFact]
    public void Gelu_of_a_large_finite_input_is_the_input_or_zero()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        var elementwise = (IGpuNarrowFloatElementwise)backend;
        int ran = 0;
        foreach (var type in Types)
        {
            if (!backend.SupportsElementType(type) || !elementwise.SupportsNarrowFloatUnary(GpuNarrowFloatUnaryOperation.Gelu, type)) continue;
            ushort huge = type == GpuElementType.Float16 ? (ushort)0x7BFF : (ushort)0x6780;   // 65504, and 2^80
            var values = new[] { huge, (ushort)(huge | 0x8000), Narrow(type, 9f), Narrow(type, -9f), Narrow(type, 100f), Narrow(type, -100f) };
            using var a = Upload(backend, type, values);
            using var output = backend.AllocateBuffer(values.Length, type);
            elementwise.UnaryNarrowFloat(GpuNarrowFloatUnaryOperation.Gelu, type, a, output, values.Length);
            var actual = Download(backend, type, output, values.Length);
            for (int i = 0; i < values.Length; i++)
            {
                float x = Widen(type, values[i]);
                float got = Widen(type, actual[i]);
                Assert.True((x > 0f ? x : 0f) == got, $"Gelu<{type}> of {x:R} (0x{values[i]:X4}) gave {got:R} (0x{actual[i]:X4})");
            }

            ran++;
        }

        RequireRan(ran, Types.Length, "Gelu kernel");
    }

    [SkippableFact]
    public void Matrix_multiply_accumulates_in_float_and_stores_the_type()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        if (RunsEveryOperation) Assert.IsAssignableFrom<IGpuNarrowFloatMatrixMultiply>(backend);
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

            // 512 ones times 512 ones: a sum kept in BFloat16 (8 significand bits) stops growing at 256, where adding
            // one no longer changes it. Only a float accumulator reaches 512.
            const int Long = 512;
            var ones = new ushort[Long];
            for (int i = 0; i < Long; i++) ones[i] = Narrow(type, 1f);
            using var rowOfOnes = Upload(backend, type, ones);
            using var columnOfOnes = Upload(backend, type, ones);
            using var dot = backend.AllocateBuffer(1, type);
            multiply.MatrixMultiplyNarrowFloat(type, rowOfOnes, columnOfOnes, dot, 1, 1, Long);
            Assert.Equal((float)Long, Widen(type, Download(backend, type, dot, 1)[0]));

            // One by one, bit for bit against the host: the smallest pattern of the type (for BFloat16 a binary32
            // subnormal, which a device may flush before a native multiply) times a large value, and an ordinary pair.
            foreach (var (x, y) in new[] { ((ushort)0x0001, type == GpuElementType.Float16 ? (ushort)0x7800 : (ushort)0x7E80),
                         (Narrow(type, 1.5f), Narrow(type, -2.25f)) })
            {
                using var one = Upload(backend, type, new[] { x });
                using var other = Upload(backend, type, new[] { y });
                using var product = backend.AllocateBuffer(1, type);
                multiply.MatrixMultiplyNarrowFloat(type, one, other, product, 1, 1, 1);
                ushort expectedBits = Narrow(type, Widen(type, x) * Widen(type, y));
                ushort gotBits = Download(backend, type, product, 1)[0];
                Assert.True(expectedBits == gotBits, $"{type} 0x{x:X4} times 0x{y:X4} gave 0x{gotBits:X4}, host 0x{expectedBits:X4}");
            }

            Assert.Throws<ArgumentException>(() => multiply.MatrixMultiplyNarrowFloat(type, a, b, a, M, N, K));
            Assert.Throws<ArgumentOutOfRangeException>(() => multiply.MatrixMultiplyNarrowFloat(type, a, b, c, 0, N, K));
            ran++;
        }

        RequireRan(ran, Types.Length, "matrix multiply");
    }

    [SkippableFact]
    public void The_capability_table_and_argument_checks_agree_with_the_interfaces()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
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

/// <summary>The backend the engine picked on this host (OpenCL, CUDA or HIP where one is present).</summary>
[Collection("DirectGpuSerial")]
public sealed class EngineBackendNarrowFloatTests : GpuNarrowFloatContractTests
{
    protected override IDirectGpuBackend? Device()
        => AiDotNetEngine.Current is DirectGpuTensorEngine gpu && gpu.GetBackend() is IGpuNarrowFloatElementwise { } backend
            ? (IDirectGpuBackend)backend
            : null;

    protected override string SkipReason => "No GPU backend on this host.";
}

/// <summary>Vulkan: the packed GLSL kernels, both types and every operation.</summary>
[Collection("VulkanGlobalState")]
public sealed class VulkanNarrowFloatTests : GpuNarrowFloatContractTests
{
    private readonly IDirectGpuBackend? _backend;

    public VulkanNarrowFloatTests()
    {
        try
        {
            var backend = AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanBackend.Instance;
            // Gated on what the kernels need (a device and the GLSL compiler), not on the capability under test.
            if (backend.Initialize() && backend.IsGlslCompilerAvailable) _backend = backend;
        }
        catch (Exception unavailable) when (unavailable is DllNotFoundException or InvalidOperationException or NotSupportedException)
        {
            _backend = null;
        }
    }

    protected override IDirectGpuBackend? Device() => _backend;

    protected override string SkipReason => "No Vulkan device with libshaderc.";

    protected override bool RunsEveryOperation => true;
}

/// <summary>Metal: the packed kernels in Metal Shading Language, both types and every operation.</summary>
[Collection("DirectGpuSerial")]
public sealed class MetalNarrowFloatTests : GpuNarrowFloatContractTests, IDisposable
{
    private readonly AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalBackend? _backend;

    public MetalNarrowFloatTests()
    {
        bool required = string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_METAL"), "1", StringComparison.Ordinal);
        if (!System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX))
        {
            if (required) throw new InvalidOperationException("AIDOTNET_REQUIRE_METAL=1 on a host that is not macOS.");
            return;
        }

        var backend = new AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalBackend();
        if (backend.IsAvailable)
        {
            _backend = backend;
            return;
        }

        backend.Dispose();
        if (required) throw new InvalidOperationException("AIDOTNET_REQUIRE_METAL=1 but the Metal backend is not available.");
    }

    protected override IDirectGpuBackend? Device() => _backend;

    protected override string SkipReason => "Metal exists only on macOS with a Metal device.";

    protected override bool RunsEveryOperation => true;

    public void Dispose() => _backend?.Dispose();
}

#if NET7_0_OR_GREATER
/// <summary>WebGPU on Dawn: the packed WGSL kernels, both types and every operation.</summary>
[Collection("DirectGpuSerial")]
public sealed class WebGpuNarrowFloatTests : GpuNarrowFloatContractTests
{
    protected override IDirectGpuBackend? Device() => DawnHostedWebGpu.BackendOrNull();

    protected override string SkipReason => "Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).";

    protected override bool RunsEveryOperation => true;
}
#endif

/// <summary>
/// The host GELU of the 16-bit float types is the FP32 result rounded once, which is what the device kernels store.
/// Evaluated in the storage type instead, every intermediate is rounded: several percent off in BFloat16, and NaN
/// for +infinity.
/// </summary>
public class NarrowFloatHostGeluTests
{
    private static readonly float[] Inputs =
    {
        // The two largest lead, so the vector path (the first eight elements) takes them and not only the scalar tail.
        float.PositiveInfinity, 3.2964854E+38f, -6f, -3f, -1f, -0.5f, -0.0f, 0f, 0.25f, 0.5f, 1f, 2f, 3.5f, 8f, 1e30f, 1.5f, -2f,
    };

    private static float[] Fp32()
    {
        var expected = new float[Inputs.Length];
        AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<float>().GELU(Inputs, expected);
        return expected;
    }

    [Fact]
    public void BFloat16_gelu_is_the_fp32_result_rounded_once()
    {
        var x = Array.ConvertAll(Inputs, BFloat16.FromFloat);
        var actual = new BFloat16[x.Length];
        AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<BFloat16>().GELU(x, actual);
        var expected = new float[x.Length];
        AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<float>().GELU(Array.ConvertAll(x, v => (float)v), expected);
        for (int i = 0; i < x.Length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits((float)BFloat16.FromFloat(expected[i])), BitConverter.SingleToInt32Bits((float)actual[i]));
        Assert.True(float.IsPositiveInfinity((float)actual[0]), "GELU(+infinity) is +infinity.");
    }

    [Fact]
    public void Half_gelu_is_the_fp32_result_rounded_once()
    {
        var x = Array.ConvertAll(Inputs, v => (Half)v);
        var actual = new Half[x.Length];
        AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>().GELU(x, actual);
        var expected = new float[x.Length];
        AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<float>().GELU(Array.ConvertAll(x, v => (float)v), expected);
        for (int i = 0; i < x.Length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits((float)(Half)expected[i]), BitConverter.SingleToInt32Bits((float)actual[i]));
    }

    [Fact]
    public void Fp32_gelu_of_a_value_above_half_the_largest_float_is_finite()
    {
        var expected = Fp32();
        Assert.Equal(3.2964854E+38f, expected[1]);
        Assert.True(float.IsPositiveInfinity(expected[0]));
    }
}

#endif
