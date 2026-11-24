// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// What every backend's integer kernels owe, driven on the backend itself: results in the buffers' own element type,
/// equal to unchecked host arithmetic bit for bit, for every type the backend says it computes.
/// </summary>
public abstract class GpuIntegerBackendContractTests
{
    /// <summary>The backend under test, or null when its device is absent on this host.</summary>
    protected abstract IDirectGpuBackend? Device();

    /// <summary>Why <see cref="Device"/> returned null.</summary>
    protected abstract string SkipReason { get; }

    private static readonly int[] Sizes = { 1, 2, 63, 64, 65, 255, 256, 257, 4097, 70_001 };

    private static IGpuIntegerElementwise Elementwise(IDirectGpuBackend backend)
        => backend as IGpuIntegerElementwise ?? throw new Xunit.Sdk.XunitException($"{backend.BackendName} has no integer element-wise kernels.");

    private static IGpuIntegerReductions Reductions(IDirectGpuBackend backend)
        => backend as IGpuIntegerReductions ?? throw new Xunit.Sdk.XunitException($"{backend.BackendName} has no integer reductions.");

    // Values across the whole range of the type, the extremes included, so sums, products and negations overflow.
    private static long[] Values(GpuElementType type, int length, int seed)
    {
        var values = new long[length];
        ulong state = (ulong)seed * 0x9E3779B97F4A7C15UL + 1UL;
        long[] extremes = { 0, 1, -1, long.MinValue, long.MaxValue, int.MinValue, int.MaxValue, short.MinValue, sbyte.MinValue, 255, 127, 128 };
        for (int i = 0; i < length; i++)
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            long raw = i % 11 == 3 ? extremes[(i / 11) % extremes.Length] : i % 5 == 0 ? (long)(state % 201UL) - 100 : unchecked((long)state);
            values[i] = type == GpuElementType.Bool ? (long)(state & 1UL) : GpuIntegerLayout.Wrap(raw, type);
        }

        return values;
    }

    private static IGpuBuffer Upload(IDirectGpuBackend backend, GpuElementType type, long[] values)
    {
        var buffer = backend.AllocateBuffer(values.Length, type);
        try
        {
            switch (type)
            {
                case GpuElementType.Int8: backend.UploadBuffer<sbyte>(buffer, Array.ConvertAll(values, v => unchecked((sbyte)v))); break;
                case GpuElementType.UInt8: backend.UploadBuffer<byte>(buffer, Array.ConvertAll(values, v => unchecked((byte)v))); break;
                case GpuElementType.Int16: backend.UploadBuffer<short>(buffer, Array.ConvertAll(values, v => unchecked((short)v))); break;
                case GpuElementType.Int32: backend.UploadBuffer<int>(buffer, Array.ConvertAll(values, v => unchecked((int)v))); break;
                case GpuElementType.Int64: backend.UploadBuffer<long>(buffer, values); break;
                case GpuElementType.Bool: backend.UploadBuffer<bool>(buffer, Array.ConvertAll(values, v => v != 0)); break;
                default: throw new ArgumentOutOfRangeException(nameof(type));
            }
        }
        catch
        {
            buffer.Dispose();
            throw;
        }

        return buffer;
    }

    private static long[] Download(IDirectGpuBackend backend, GpuElementType type, IGpuBuffer buffer, int length)
    {
        switch (type)
        {
            case GpuElementType.Int8: { var d = new sbyte[length]; backend.DownloadBuffer<sbyte>(buffer, d); return Array.ConvertAll(d, v => (long)v); }
            case GpuElementType.UInt8: { var d = new byte[length]; backend.DownloadBuffer<byte>(buffer, d); return Array.ConvertAll(d, v => (long)v); }
            case GpuElementType.Int16: { var d = new short[length]; backend.DownloadBuffer<short>(buffer, d); return Array.ConvertAll(d, v => (long)v); }
            case GpuElementType.Int32: { var d = new int[length]; backend.DownloadBuffer<int>(buffer, d); return Array.ConvertAll(d, v => (long)v); }
            case GpuElementType.Int64: { var d = new long[length]; backend.DownloadBuffer<long>(buffer, d); return d; }
            case GpuElementType.Bool:
            {
                // Read as bytes: the kernel must store exactly 0 or 1, which a bool read would not show.
                var raw = backend.DownloadByteBuffer(buffer, length);
                return Array.ConvertAll(raw, v => (long)v);
            }
            default: throw new ArgumentOutOfRangeException(nameof(type));
        }
    }

    private static long Host(GpuIntegerBinaryOperation operation, GpuElementType type, long x, long y)
    {
        if (type == GpuElementType.Bool)
        {
            bool p = x != 0, q = y != 0;
            return operation switch
            {
                GpuIntegerBinaryOperation.And => p && q,
                GpuIntegerBinaryOperation.Or => p || q,
                _ => p != q,
            } ? 1 : 0;
        }

        long wide = operation switch
        {
            GpuIntegerBinaryOperation.Add => unchecked(x + y),
            GpuIntegerBinaryOperation.Subtract => unchecked(x - y),
            GpuIntegerBinaryOperation.Multiply => unchecked(x * y),
            GpuIntegerBinaryOperation.Min => Math.Min(x, y),
            GpuIntegerBinaryOperation.Max => Math.Max(x, y),
            GpuIntegerBinaryOperation.And => x & y,
            GpuIntegerBinaryOperation.Or => x | y,
            // Defined everywhere: a zero divisor gives 0, and over -1 the quotient is the wrapping negation (the host
            // operators raise for both in 64 bits).
            GpuIntegerBinaryOperation.Divide => y == 0 ? 0 : y == -1 ? unchecked(-x) : x / y,
            GpuIntegerBinaryOperation.Remainder => y == 0 || y == -1 ? 0 : x % y,
            GpuIntegerBinaryOperation.Equal => x == y ? 1 : 0,
            GpuIntegerBinaryOperation.NotEqual => x != y ? 1 : 0,
            GpuIntegerBinaryOperation.Greater => x > y ? 1 : 0,
            GpuIntegerBinaryOperation.Less => x < y ? 1 : 0,
            _ => x ^ y,
        };
        return GpuIntegerLayout.Wrap(wide, type);
    }

    private static long Host(GpuIntegerUnaryOperation operation, GpuElementType type, long x)
        => type == GpuElementType.Bool ? (x == 0 ? 1 : 0)
            : GpuIntegerLayout.Wrap(operation switch
            {
                GpuIntegerUnaryOperation.Negate => unchecked(-x),
                // The most negative value is its own absolute value, as it is its own negation.
                GpuIntegerUnaryOperation.Abs => x < 0 ? unchecked(-x) : x,
                GpuIntegerUnaryOperation.Sign => Math.Sign(x),
                _ => ~x,
            }, type);

    private static readonly GpuElementType[] Types =
    {
        GpuElementType.Int8, GpuElementType.UInt8, GpuElementType.Int16, GpuElementType.Int32, GpuElementType.Int64, GpuElementType.Bool,
    };

    [SkippableFact]
    public void The_backend_computes_int32_and_reports_each_type_consistently()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        var elementwise = Elementwise(backend);
        var reductions = Reductions(backend);
        Assert.True(elementwise.SupportsIntegerElementwise(GpuElementType.Int32));
        Assert.True(reductions.SupportsIntegerReductions(GpuElementType.Int32));
        Assert.False(elementwise.SupportsIntegerElementwise(GpuElementType.Float32));
        Assert.False(elementwise.SupportsIntegerElementwise(GpuElementType.Float64));
        Assert.False(reductions.SupportsIntegerReductions(GpuElementType.Bool));
        foreach (var type in Types)
        {
            // A type the device computes is one it can hold; the capability table says the same.
            if (elementwise.SupportsIntegerElementwise(type)) Assert.True(backend.SupportsElementType(type));
            Assert.Equal(elementwise.SupportsIntegerElementwise(type), backend.SupportsOperation(GpuOperationFamily.Elementwise, type));
            Assert.Equal(reductions.SupportsIntegerReductions(type), backend.SupportsOperation(GpuOperationFamily.Reduction, type));
            Assert.False(backend.SupportsOperation(GpuOperationFamily.MatrixMultiply, type));
        }
    }

    [SkippableTheory]
    [InlineData(GpuIntegerBinaryOperation.Add)]
    [InlineData(GpuIntegerBinaryOperation.Subtract)]
    [InlineData(GpuIntegerBinaryOperation.Multiply)]
    [InlineData(GpuIntegerBinaryOperation.Min)]
    [InlineData(GpuIntegerBinaryOperation.Max)]
    [InlineData(GpuIntegerBinaryOperation.And)]
    [InlineData(GpuIntegerBinaryOperation.Or)]
    [InlineData(GpuIntegerBinaryOperation.Xor)]
    [InlineData(GpuIntegerBinaryOperation.Divide)]
    [InlineData(GpuIntegerBinaryOperation.Remainder)]
    [InlineData(GpuIntegerBinaryOperation.Equal)]
    [InlineData(GpuIntegerBinaryOperation.NotEqual)]
    [InlineData(GpuIntegerBinaryOperation.Greater)]
    [InlineData(GpuIntegerBinaryOperation.Less)]
    public void Binary_and_scalar_kernels_equal_the_host(GpuIntegerBinaryOperation operation)
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        var elementwise = Elementwise(backend);
        int ran = 0;
        foreach (var type in Types)
        {
            if (!elementwise.SupportsIntegerElementwise(type) || !GpuIntegerLayout.Defined(operation, type)) continue;
            foreach (int size in Sizes)
            {
                var left = Values(type, size, 1);
                var right = Values(type, size, 2);
                using var a = Upload(backend, type, left);
                using var b = Upload(backend, type, right);
                using var output = backend.AllocateBuffer(size, type);
                elementwise.BinaryInteger(operation, a, b, output, size);
                var actual = Download(backend, type, output, size);
                for (int i = 0; i < size; i++)
                    Assert.True(Host(operation, type, left[i], right[i]) == actual[i],
                        $"{operation}<{type}> at {i} of {size}: {left[i]}, {right[i]} gave {actual[i]}, host {Host(operation, type, left[i], right[i])}.");

                // The scalar is truncated to the element type, as a host scalar of that type is.
                foreach (long scalar in new[] { 0L, 1L, -1L, 3L, -77L, long.MinValue, 0x7FFF_FFFF_0000_0081L })
                {
                    long truncated = type == GpuElementType.Bool ? (GpuIntegerLayout.Wrap(scalar, GpuElementType.UInt8) != 0 ? 1 : 0) : GpuIntegerLayout.Wrap(scalar, type);
                    elementwise.ScalarInteger(operation, a, scalar, output, size);
                    actual = Download(backend, type, output, size);
                    for (int i = 0; i < size; i += Math.Max(1, size / 97))
                        Assert.True(Host(operation, type, left[i], truncated) == actual[i],
                            $"{operation}<{type}> scalar {scalar} at {i} of {size}: {left[i]} gave {actual[i]}.");
                }

                ran++;
            }
        }

        Assert.True(ran > 0, "No element type ran.");
    }

    [SkippableTheory]
    [InlineData(GpuIntegerUnaryOperation.Negate)]
    [InlineData(GpuIntegerUnaryOperation.Not)]
    [InlineData(GpuIntegerUnaryOperation.Abs)]
    [InlineData(GpuIntegerUnaryOperation.Sign)]
    public void Unary_kernels_equal_the_host(GpuIntegerUnaryOperation operation)
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        var elementwise = Elementwise(backend);
        int ran = 0;
        foreach (var type in Types)
        {
            if (!elementwise.SupportsIntegerElementwise(type) || !GpuIntegerLayout.Defined(operation, type)) continue;
            foreach (int size in Sizes)
            {
                var values = Values(type, size, 3);
                using var a = Upload(backend, type, values);
                using var output = backend.AllocateBuffer(size, type);
                elementwise.UnaryInteger(operation, a, output, size);
                var actual = Download(backend, type, output, size);
                for (int i = 0; i < size; i++)
                    Assert.True(Host(operation, type, values[i]) == actual[i], $"{operation}<{type}> at {i} of {size}: {values[i]} gave {actual[i]}.");
                ran++;
            }
        }

        Assert.True(ran > 0, "No element type ran.");
    }

    [SkippableFact]
    public void Sum_wraps_in_the_element_type_as_the_host_does()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        var reductions = Reductions(backend);
        int ran = 0;
        foreach (var type in Types)
        {
            if (!reductions.SupportsIntegerReductions(type)) continue;
            foreach (int size in Sizes)
            {
                var values = Values(type, size, 4);
                long expected = 0;
                foreach (long value in values) expected = unchecked(expected + value);
                expected = GpuIntegerLayout.Wrap(expected, type);
                using var a = Upload(backend, type, values);
                Assert.Equal(expected, reductions.SumInteger(a, size));
                // A prefix: the elements past it are not read.
                int prefix = size / 2;
                long prefixExpected = 0;
                for (int i = 0; i < prefix; i++) prefixExpected = unchecked(prefixExpected + values[i]);
                Assert.Equal(GpuIntegerLayout.Wrap(prefixExpected, type), reductions.SumInteger(a, prefix));
                ran++;
            }
        }

        Assert.True(ran > 0, "No element type ran.");
    }

    // By the element type's own order: the values hold both extremes of every type, and a prefix must not read past
    // itself (the extreme of the whole buffer usually lies outside a short prefix).
    [SkippableFact]
    public void Min_and_max_are_the_extremes_in_the_element_types_order()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        var reductions = Reductions(backend);
        int ran = 0;
        foreach (var type in Types)
        {
            if (!reductions.SupportsIntegerReductions(type)) continue;
            foreach (int size in Sizes)
            {
                var values = Values(type, size, 6);
                using var a = Upload(backend, type, values);
                foreach (int count in new[] { size, Math.Max(1, size / 2), 1 })
                {
                    long smallest = values[0], largest = values[0];
                    for (int i = 1; i < count; i++)
                    {
                        smallest = Math.Min(smallest, values[i]);
                        largest = Math.Max(largest, values[i]);
                    }

                    Assert.True(smallest == reductions.MinInteger(a, count), $"Min<{type}> of the first {count} of {size}: host {smallest}, device {reductions.MinInteger(a, count)}.");
                    Assert.True(largest == reductions.MaxInteger(a, count), $"Max<{type}> of the first {count} of {size}: host {largest}, device {reductions.MaxInteger(a, count)}.");
                }

                Assert.Throws<ArgumentException>(() => reductions.MinInteger(a, 0));
                Assert.Throws<ArgumentException>(() => reductions.MaxInteger(a, 0));
                ran++;
            }
        }

        Assert.True(ran > 0, "No element type ran.");
    }

    // A device copy counts elements of the buffers' type. An 8- or 16-bit element shares a 4-byte word with its
    // neighbours on the word-addressed backends, so every offset within a word is tried, in both directions, and the
    // elements either side of the range must keep what they held.
    [SkippableFact]
    public void A_device_copy_moves_exactly_the_requested_elements_of_every_type()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        int ran = 0;
        foreach (var type in Types)
        {
            if (!backend.SupportsElementType(type)) continue;
            ran++;
            const int Length = 41;
            var values = Values(type, Length, 91);
            long fill = type == GpuElementType.Bool ? 1 : GpuIntegerLayout.Wrap(-7, type);
            foreach (int from in new[] { 0, 1, 2, 3, 5 })
            foreach (int to in new[] { 0, 1, 2, 3, 6 })
            foreach (int count in new[] { 1, 2, 3, 4, 5, 9, 30 })
            {
                using var source = Upload(backend, type, values);
                using var target = Upload(backend, type, Enumerable.Repeat(fill, Length).ToArray());
                backend.Copy(source, from, target, to, count);
                var actual = Download(backend, type, target, Length);
                for (int i = 0; i < Length; i++)
                {
                    long expected = i >= to && i < to + count ? values[from + i - to] : fill;
                    Assert.True(expected == actual[i], $"{type} copy of {count} from {from} to {to}: element {i} is {actual[i]}, not {expected}.");
                }

                // Within one buffer, the ranges overlapping: the destination gets what the source held before the copy.
                using var same = Upload(backend, type, values);
                backend.Copy(same, from, same, to, count);
                actual = Download(backend, type, same, Length);
                for (int i = 0; i < Length; i++)
                {
                    long expected = i >= to && i < to + count ? values[from + i - to] : values[i];
                    Assert.True(expected == actual[i], $"{type} copy within a buffer of {count} from {from} to {to}: element {i} is {actual[i]}, not {expected}.");
                }
            }

            // A range copied onto itself changes nothing.
            using var self = Upload(backend, type, values);
            backend.Copy(self, 3, self, 3, 17);
            Assert.Equal(values, Download(backend, type, self, Length));

            using var whole = Upload(backend, type, values);
            using var copy = backend.AllocateBuffer(Length, type);
            backend.Copy(whole, copy, Length);
            Assert.Equal(values, Download(backend, type, copy, Length));
        }

        Assert.True(ran > 0, "The backend holds no integer type.");
    }

    [SkippableFact]
    public void A_kernel_writes_only_the_requested_prefix_and_refuses_bad_operands()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        var elementwise = Elementwise(backend);
        const GpuElementType Type = GpuElementType.Int32;
        var values = Values(Type, 1000, 5);
        var sentinel = new long[1000];
        for (int i = 0; i < sentinel.Length; i++) sentinel[i] = -7;
        using var a = Upload(backend, Type, values);
        using var output = Upload(backend, Type, sentinel);
        using var floats = backend.AllocateBuffer(1000);
        elementwise.BinaryInteger(GpuIntegerBinaryOperation.Add, a, a, output, 300);
        var actual = Download(backend, Type, output, 1000);
        for (int i = 0; i < 1000; i++)
            Assert.Equal(i < 300 ? GpuIntegerLayout.Wrap(values[i] + values[i], Type) : -7, actual[i]);

        Assert.Throws<ArgumentException>(() => elementwise.BinaryInteger(GpuIntegerBinaryOperation.Add, a, floats, output, 10));
        Assert.Throws<ArgumentException>(() => elementwise.BinaryInteger(GpuIntegerBinaryOperation.Add, a, a, floats, 10));
        Assert.Throws<ArgumentException>(() => elementwise.BinaryInteger(GpuIntegerBinaryOperation.Add, a, output, output, 10));
        Assert.Throws<ArgumentException>(() => elementwise.UnaryInteger(GpuIntegerUnaryOperation.Negate, a, a, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => elementwise.BinaryInteger(GpuIntegerBinaryOperation.Add, a, a, output, 1001));
        Assert.Throws<ArgumentException>(() => Reductions(backend).SumInteger(floats, 10));
        if (elementwise.SupportsIntegerElementwise(GpuElementType.Bool))
        {
            using var flags = backend.AllocateBuffer(16, GpuElementType.Bool);
            using var flagsOut = backend.AllocateBuffer(16, GpuElementType.Bool);
            Assert.Throws<ArgumentException>(() => elementwise.BinaryInteger(GpuIntegerBinaryOperation.Add, flags, flags, flagsOut, 16));
            Assert.Throws<ArgumentException>(() => elementwise.UnaryInteger(GpuIntegerUnaryOperation.Negate, flags, flagsOut, 16));
        }
    }
}

/// <summary>The backend the engine picked on this host (OpenCL, CUDA or HIP where one is present).</summary>
[Collection("DirectGpuSerial")]
public sealed class EngineBackendIntegerTests : GpuIntegerBackendContractTests
{
    // The backend whatever it implements: one without the integer interface fails the tests, it does not skip them.
    protected override IDirectGpuBackend? Device()
        => AiDotNetEngine.Current is DirectGpuTensorEngine gpu ? gpu.GetBackend() : null;

    protected override string SkipReason => "No GPU backend on this host.";
}

/// <summary>Vulkan: GLSL Int32 kernels behind the runtime GLSL compiler.</summary>
[Collection("VulkanGlobalState")]
public sealed class VulkanIntegerTests : GpuIntegerBackendContractTests
{
    private readonly IDirectGpuBackend? _backend;

    public VulkanIntegerTests()
    {
        try
        {
            var backend = AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanBackend.Instance;
            // Gated on what the kernels need (a device and the GLSL compiler), not on the capability under test: a
            // device that has both and reports no Int32 support fails these tests instead of skipping them.
            if (backend.Initialize() && backend.IsGlslCompilerAvailable) _backend = backend;
        }
        catch (Exception unavailable) when (unavailable is DllNotFoundException or InvalidOperationException or NotSupportedException)
        {
            _backend = null;
        }
    }

    protected override IDirectGpuBackend? Device() => _backend;

    protected override string SkipReason => "No Vulkan device with libshaderc.";
}

/// <summary>Metal: Int32 kernels in Metal Shading Language.</summary>
[Collection("DirectGpuSerial")]
public sealed class MetalIntegerTests : GpuIntegerBackendContractTests, IDisposable
{
    private readonly AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalBackend? _backend;

    public MetalIntegerTests()
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

    public void Dispose() => _backend?.Dispose();
}

#if NET7_0_OR_GREATER
/// <summary>WebGPU on Dawn: WGSL i32 kernels.</summary>
[Collection("DirectGpuSerial")]
public sealed class WebGpuIntegerTests : GpuIntegerBackendContractTests
{
    protected override IDirectGpuBackend? Device() => DawnHostedWebGpu.BackendOrNull();

    protected override string SkipReason => "Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).";
}
#endif

#endif
