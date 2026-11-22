// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// What every backend's double-precision storage and kernels owe, driven on the backend itself so it is exercised
/// wherever that backend's device exists, whichever backend the engine picked. The kernels are held to the host's
/// results (bit for bit where the operation is one IEEE operation) and the buffer to counting doubles, not words.
/// </summary>
public abstract class GpuFloat64BackendContractTests
{
    /// <summary>The backend under test, or null when its device (or its double support) is absent on this host.</summary>
    protected abstract IDirectGpuBackend? Device();

    /// <summary>Why <see cref="Device"/> returned null.</summary>
    protected abstract string SkipReason { get; }

    private static IGpuFloat64Elementwise Elementwise(IDirectGpuBackend backend)
        => backend as IGpuFloat64Elementwise ?? throw new Xunit.Sdk.XunitException($"{backend.BackendName} has no double element-wise kernels.");

    private static IGpuFloat64Reductions Reductions(IDirectGpuBackend backend)
        => backend as IGpuFloat64Reductions ?? throw new Xunit.Sdk.XunitException($"{backend.BackendName} has no double reductions.");

    private static IGpuFloat64MatrixMultiply MatrixMultiply(IDirectGpuBackend backend)
        => backend as IGpuFloat64MatrixMultiply ?? throw new Xunit.Sdk.XunitException($"{backend.BackendName} has no double matrix multiply.");

    [SkippableFact]
    public void The_backend_reports_its_double_capabilities()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        Assert.True(backend.SupportsElementType(GpuElementType.Float64));
        Assert.True(Elementwise(backend).SupportsFloat64Elementwise);
        Assert.True(Reductions(backend).SupportsFloat64Reductions);
        Assert.True(MatrixMultiply(backend).SupportsFloat64MatrixMultiply);
        Assert.True(backend.SupportsOperation(GpuOperationFamily.Elementwise, GpuElementType.Float64));
    }

    // Full 53-bit mantissas, mixed signs and magnitudes: a float round trip, a contracted multiply-add or a
    // reciprocal-multiply division would each change the last bits.
    private static double[] Values(int length, int seed)
    {
        var values = new double[length];
        ulong state = (ulong)seed * 0x9E3779B97F4A7C15UL + 1UL;
        for (int i = 0; i < length; i++)
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            double unit = (state >> 11) * (1.0 / 9007199254740992.0);
            values[i] = (unit - 0.5) * Math.Pow(10.0, (i % 13) - 6) + ((i % 7 == 0) ? 3.0 : 0.0);
            if (values[i] == 0.0) values[i] = 1.0;
        }

        return values;
    }

    private static void AssertBitEqual(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.DoubleToInt64Bits(expected[i]) == BitConverter.DoubleToInt64Bits(actual[i]),
                $"Element {i}: host {expected[i]:R}, device {actual[i]:R}.");
    }

    private static long UlpDistance(double a, double b)
    {
        if (double.IsNaN(a) || double.IsNaN(b)) return double.IsNaN(a) && double.IsNaN(b) ? 0 : long.MaxValue;
        long x = BitConverter.DoubleToInt64Bits(a), y = BitConverter.DoubleToInt64Bits(b);
        if ((x < 0) != (y < 0)) return a == b ? 0 : long.MaxValue;
        return Math.Abs(x - y);
    }

    [SkippableFact]
    public void A_double_buffer_counts_doubles_and_round_trips_them()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        const int length = 1027;
        var values = Values(length, 1);
        using var buffer = backend.AllocateBuffer(length, GpuElementType.Float64);
        Assert.Equal(GpuElementType.Float64, buffer.ElementType);
        Assert.Equal(length, buffer.Size);
        Assert.Equal(8L * length, buffer.SizeInBytes);

        backend.UploadBuffer<double>(buffer, values);
        var back = new double[length];
        backend.DownloadBuffer<double>(buffer, back);
        AssertBitEqual(values, back);

        // The float readers count 4-byte words: they refuse the buffer rather than return half of it reinterpreted.
        Assert.Throws<InvalidOperationException>(() => backend.DownloadBuffer(buffer));
        Assert.Throws<InvalidOperationException>(() => backend.DownloadBuffer(buffer, new float[length]));
    }

    [SkippableFact]
    public void A_device_copy_moves_whole_doubles_and_refuses_a_float_destination()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        const int length = 300;
        var values = Values(length, 2);
        using var source = backend.AllocateBuffer(length, GpuElementType.Float64);
        using var destination = backend.AllocateBuffer(length, GpuElementType.Float64);
        using var floats = backend.AllocateBuffer(length);
        backend.UploadBuffer<double>(source, values);
        backend.UploadBuffer<double>(destination, new double[length]);

        backend.Copy(source, 100, destination, 50, 120);
        var back = new double[length];
        backend.DownloadBuffer<double>(destination, back);
        for (int i = 0; i < length; i++)
        {
            double expected = i >= 50 && i < 170 ? values[i + 50] : 0.0;
            Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(back[i]), $"Element {i}.");
        }

        Assert.Throws<InvalidOperationException>(() => backend.Copy(source, 0, floats, 0, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.Copy(source, 200, destination, 0, 101));
    }

    [SkippableTheory]
    [InlineData(GpuFloat64BinaryOperation.Add)]
    [InlineData(GpuFloat64BinaryOperation.Subtract)]
    [InlineData(GpuFloat64BinaryOperation.Multiply)]
    [InlineData(GpuFloat64BinaryOperation.Divide)]
    public void Arithmetic_equals_the_host_bit_for_bit_with_a_buffer_or_a_scalar_on_the_right(GpuFloat64BinaryOperation operation)
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        const int length = 4099;
        const double scalar = 1.0 / 3.0 + 1e-9;
        var left = Values(length, 10 + (int)operation);
        var right = Values(length, 20 + (int)operation);
        static double Apply(GpuFloat64BinaryOperation op, double x, double y) => op switch
        {
            GpuFloat64BinaryOperation.Add => x + y,
            GpuFloat64BinaryOperation.Subtract => x - y,
            GpuFloat64BinaryOperation.Multiply => x * y,
            _ => x / y,
        };
        var expected = new double[length];
        var expectedScalar = new double[length];
        for (int i = 0; i < length; i++)
        {
            expected[i] = Apply(operation, left[i], right[i]);
            expectedScalar[i] = Apply(operation, left[i], scalar);
        }

        using var a = backend.AllocateBuffer(length, GpuElementType.Float64);
        using var b = backend.AllocateBuffer(length, GpuElementType.Float64);
        using var output = backend.AllocateBuffer(length, GpuElementType.Float64);
        using var floats = backend.AllocateBuffer(length);
        backend.UploadBuffer<double>(a, left);
        backend.UploadBuffer<double>(b, right);
        var actual = new double[length];

        Elementwise(backend).BinaryFloat64(operation, a, b, output, length);
        backend.DownloadBuffer<double>(output, actual);
        AssertBitEqual(expected, actual);

        Elementwise(backend).ScalarFloat64(operation, a, scalar, output, length);
        backend.DownloadBuffer<double>(output, actual);
        AssertBitEqual(expectedScalar, actual);

        Assert.Throws<ArgumentException>(() => Elementwise(backend).BinaryFloat64(operation, a, floats, output, length));
        Assert.Throws<ArgumentException>(() => Elementwise(backend).ScalarFloat64(operation, floats, scalar, output, length));
        Assert.Throws<ArgumentOutOfRangeException>(() => Elementwise(backend).BinaryFloat64(operation, a, b, output, length + 1));
    }

    [SkippableTheory]
    [InlineData(GpuFloat64UnaryOperation.Negate, 0)]
    [InlineData(GpuFloat64UnaryOperation.Abs, 0)]
    [InlineData(GpuFloat64UnaryOperation.Relu, 0)]
    [InlineData(GpuFloat64UnaryOperation.Sqrt, 0)]
    [InlineData(GpuFloat64UnaryOperation.Exp, 16)]
    [InlineData(GpuFloat64UnaryOperation.Log, 16)]
    [InlineData(GpuFloat64UnaryOperation.Tanh, 16)]
    [InlineData(GpuFloat64UnaryOperation.Sigmoid, 16)]
    public void Unary_kernels_match_the_host(GpuFloat64UnaryOperation operation, int ulps)
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        const int length = 2053;
        var input = Values(length, 30 + (int)operation);
        if (operation is GpuFloat64UnaryOperation.Sqrt or GpuFloat64UnaryOperation.Log)
            for (int i = 0; i < length; i++) input[i] = Math.Abs(input[i]);
        var expected = new double[length];
        for (int i = 0; i < length; i++)
            expected[i] = operation switch
            {
                GpuFloat64UnaryOperation.Negate => -input[i],
                GpuFloat64UnaryOperation.Abs => Math.Abs(input[i]),
                GpuFloat64UnaryOperation.Relu => Math.Max(input[i], 0.0),
                GpuFloat64UnaryOperation.Sqrt => Math.Sqrt(input[i]),
                GpuFloat64UnaryOperation.Exp => Math.Exp(input[i]),
                GpuFloat64UnaryOperation.Log => Math.Log(input[i]),
                GpuFloat64UnaryOperation.Tanh => Math.Tanh(input[i]),
                _ => 1.0 / (1.0 + Math.Exp(-input[i])),
            };

        using var a = backend.AllocateBuffer(length, GpuElementType.Float64);
        using var output = backend.AllocateBuffer(length, GpuElementType.Float64);
        backend.UploadBuffer<double>(a, input);
        Elementwise(backend).UnaryFloat64(operation, a, output, length);
        var actual = new double[length];
        backend.DownloadBuffer<double>(output, actual);
        for (int i = 0; i < length; i++)
            Assert.True(UlpDistance(expected[i], actual[i]) <= ulps,
                $"{operation} element {i} ({input[i]:R}): host {expected[i]:R}, device {actual[i]:R}.");
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(4097)]
    [InlineData(100003)]
    public void Sum_and_dot_agree_with_the_host_to_rounding(int length)
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        var left = Values(length, 40);
        var right = Values(length, 41);
        double sum = 0.0, dot = 0.0, sumMagnitude = 0.0, dotMagnitude = 0.0;
        for (int i = 0; i < length; i++)
        {
            sum += left[i];
            dot += left[i] * right[i];
            sumMagnitude += Math.Abs(left[i]);
            dotMagnitude += Math.Abs(left[i] * right[i]);
        }

        using var a = backend.AllocateBuffer(length, GpuElementType.Float64);
        using var b = backend.AllocateBuffer(length, GpuElementType.Float64);
        backend.UploadBuffer<double>(a, left);
        backend.UploadBuffer<double>(b, right);
        // A blocked sum orders the additions differently from the host's left-to-right one: both are within the
        // classic n * eps * sum|x| of the exact value. A float-precision reduction misses this by nine digits.
        double bound = 2.0 * length * 2.220446049250313e-16;
        Assert.True(Math.Abs(Reductions(backend).SumFloat64(a, length) - sum) <= bound * sumMagnitude, $"Sum of {length}.");
        Assert.True(Math.Abs(Reductions(backend).DotFloat64(a, b, length) - dot) <= bound * dotMagnitude, $"Dot of {length}.");
        Assert.Equal(0.0, Reductions(backend).SumFloat64(a, 0));
        Assert.Equal(0.0, Reductions(backend).DotFloat64(a, b, 0));
    }

    [SkippableFact]
    public void Matrix_multiply_matches_the_host_and_refuses_an_aliased_product()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        const int m = 17, k = 23, n = 9;
        var left = Values(m * k, 50);
        var right = Values(k * n, 51);
        using var a = backend.AllocateBuffer(m * k, GpuElementType.Float64);
        using var b = backend.AllocateBuffer(k * n, GpuElementType.Float64);
        using var c = backend.AllocateBuffer(m * n, GpuElementType.Float64);
        backend.UploadBuffer<double>(a, left);
        backend.UploadBuffer<double>(b, right);
        MatrixMultiply(backend).MatrixMultiplyFloat64(a, b, c, m, n, k);
        var actual = new double[m * n];
        backend.DownloadBuffer<double>(c, actual);
        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                double expected = 0.0, magnitude = 0.0;
                for (int j = 0; j < k; j++)
                {
                    double product = left[row * k + j] * right[j * n + col];
                    expected += product;
                    magnitude += Math.Abs(product);
                }

                // The kernel accumulates in the host's order with uncontracted products: the same value.
                Assert.True(Math.Abs(actual[row * n + col] - expected) <= 4.0 * k * 2.220446049250313e-16 * magnitude,
                    $"C[{row},{col}]: host {expected:R}, device {actual[row * n + col]:R}.");
            }
        }

        using var square = backend.AllocateBuffer(9, GpuElementType.Float64);
        using var other = backend.AllocateBuffer(9, GpuElementType.Float64);
        Assert.Throws<ArgumentException>(() => MatrixMultiply(backend).MatrixMultiplyFloat64(square, other, square, 3, 3, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => MatrixMultiply(backend).MatrixMultiplyFloat64(square, other, c, 0, 3, 3));
    }
}

/// <summary>Vulkan: GLSL double kernels behind shaderFloat64 and the runtime GLSL compiler.</summary>
[Collection("VulkanGlobalState")]
public sealed class VulkanFloat64Tests : GpuFloat64BackendContractTests
{
    private readonly IDirectGpuBackend? _backend;

    public VulkanFloat64Tests()
    {
        try
        {
            var backend = AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanBackend.Instance;
            if (backend.Initialize() && backend.SupportsElementType(GpuElementType.Float64)) _backend = backend;
        }
        catch (Exception unavailable) when (unavailable is DllNotFoundException or InvalidOperationException or NotSupportedException)
        {
            _backend = null;
        }
    }

    protected override IDirectGpuBackend? Device() => _backend;

    protected override string SkipReason => "No Vulkan device with shaderFloat64 and libshaderc.";
}

#if NET7_0_OR_GREATER
/// <summary>WebGPU on Dawn: binary64 computed in integer arithmetic, WGSL having no 64-bit float.</summary>
[Collection("DirectGpuSerial")]
public sealed class WebGpuFloat64Tests : GpuFloat64BackendContractTests
{
    protected override IDirectGpuBackend? Device() => DawnHostedWebGpu.BackendOrNull();

    protected override string SkipReason => "Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).";
}
#endif

#endif
