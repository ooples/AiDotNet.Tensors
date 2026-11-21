using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class GpuElementTypeTests
{
    [Theory]
    [InlineData(GpuElementType.Float64, 8)]
    [InlineData(GpuElementType.Int64, 8)]
    [InlineData(GpuElementType.Float32, 4)]
    [InlineData(GpuElementType.Int32, 4)]
    [InlineData(GpuElementType.Float16, 2)]
    [InlineData(GpuElementType.BFloat16, 2)]
    [InlineData(GpuElementType.Int16, 2)]
    [InlineData(GpuElementType.Int8, 1)]
    [InlineData(GpuElementType.UInt8, 1)]
    [InlineData(GpuElementType.Bool, 1)]
    public void SizeInBytes_IsTheDeviceWidthOfTheType(GpuElementType type, int bytes)
        => Assert.Equal(bytes, GpuElementTypes.SizeInBytes(type));

    [Fact]
    public void SizeInBytes_OfEveryDefinedTypeButRawBytes_IsDefined()
    {
        // A type added to the enum without a size would only fail at its first allocation.
        foreach (GpuElementType type in Enum.GetValues(typeof(GpuElementType)))
        {
            if (type == GpuElementType.Unspecified)
                Assert.Throws<ArgumentOutOfRangeException>(() => GpuElementTypes.SizeInBytes(type));
            else
                Assert.InRange(GpuElementTypes.SizeInBytes(type), 1, 8);
        }
    }

    [Fact]
    public void TryFromClrType_MapsEachDeviceRepresentableType()
    {
        AssertMaps<float>(GpuElementType.Float32);
        AssertMaps<double>(GpuElementType.Float64);
        AssertMaps<Half>(GpuElementType.Float16);
        AssertMaps<BFloat16>(GpuElementType.BFloat16);
        AssertMaps<sbyte>(GpuElementType.Int8);
        AssertMaps<byte>(GpuElementType.UInt8);
        AssertMaps<short>(GpuElementType.Int16);
        AssertMaps<int>(GpuElementType.Int32);
        AssertMaps<long>(GpuElementType.Int64);
        AssertMaps<bool>(GpuElementType.Bool);
    }

    [Fact]
    public void TryFromClrType_RefusesATypeWithNoDeviceRepresentation()
    {
        Assert.False(GpuElementTypes.TryFromClrType<decimal>(out var type));
        Assert.Equal(GpuElementType.Unspecified, type);
        Assert.False(GpuElementTypes.TryFromClrType(typeof(string), out _));
    }

    [Fact]
    public void Classification_SeparatesFloatingPointIntegerAndBool()
    {
        foreach (GpuElementType type in Enum.GetValues(typeof(GpuElementType)))
        {
            bool floating = GpuElementTypes.IsFloatingPoint(type), integer = GpuElementTypes.IsInteger(type);
            Assert.False(floating && integer);
            if (type is GpuElementType.Bool or GpuElementType.Unspecified) Assert.False(floating || integer);
            else Assert.True(floating || integer);
        }
    }

    [SkippableFact]
    public void DeviceBuffers_ReportTheirElementType()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");
        using var floats = backend.AllocateBuffer(8);
        using var bytes = backend.AllocateByteBuffer(8);
        Assert.Equal(GpuElementType.Float32, floats.ElementType);
        Assert.Equal(GpuElementType.Unspecified, bytes.ElementType);
    }

    [SkippableFact]
    public void IntBuffers_ReportInt32_AndARecycledAllocationIsAFloatBufferAgain()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");
        // An odd size nothing else in the suite pools, so the float rents below can only be served by these buffers.
        const int Size = 7919;
        using (var sized = backend.AllocateIntBuffer(Size))
            Assert.Equal(GpuElementType.Int32, sized.ElementType);
        using (var uploaded = backend.AllocateIntBuffer(new int[Size]))
            Assert.Equal(GpuElementType.Int32, uploaded.ElementType);
        // Int and float buffers share one pool: whichever allocation comes back, it must not still say Int32.
        for (int i = 0; i < 4; i++)
        {
            using var floats = backend.AllocateBuffer(Size);
            Assert.Equal(GpuElementType.Float32, floats.ElementType);
        }
    }

    [Fact]
    public void Tag_RefusesAnElementTypeOfAnotherWidth()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        if (gpu?.GetBackend() is not { } backend) return;
        using var floats = backend.AllocateBuffer(16);
        Assert.Throws<InvalidOperationException>(() => GpuBufferTyping.Tag(floats, GpuElementType.Float64));
        Assert.Equal(GpuElementType.Float32, floats.ElementType);
    }

    [SkippableFact]
    public void TypedBuffers_RoundTripEveryElementTypeTheBackendHolds()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");

        // Every type has storage on CUDA, HIP and OpenCL (double where the device has it); a backend that reports a type
        // must round-trip it bit for bit, and one that does not must refuse the allocation.
        RoundTrip(backend, GpuElementType.Float32, new[] { 1.5f, -2.25f, float.MaxValue, float.Epsilon, 0f });
        RoundTrip(backend, GpuElementType.Float64, new[] { 1.5, -2.25, double.MaxValue, double.Epsilon, 1e-300 });
        RoundTrip(backend, GpuElementType.Int32, new[] { 1, -2, int.MaxValue, int.MinValue, 0 });
        RoundTrip(backend, GpuElementType.Int64, new[] { 1L, -2L, long.MaxValue, long.MinValue, 1L << 40 });
        RoundTrip(backend, GpuElementType.Int16, new short[] { 1, -2, short.MaxValue, short.MinValue, 0 });
        RoundTrip(backend, GpuElementType.Int8, new sbyte[] { 1, -2, sbyte.MaxValue, sbyte.MinValue, 0 });
        RoundTrip(backend, GpuElementType.UInt8, new byte[] { 1, 2, byte.MaxValue, 0, 128 });
        RoundTrip(backend, GpuElementType.Bool, new[] { true, false, true, true, false });
        RoundTrip(backend, GpuElementType.Float16, new[] { (Half)1.5f, (Half)(-2.25f), Half.MaxValue, Half.Epsilon, (Half)0f });
        RoundTrip(backend, GpuElementType.BFloat16, new[] { (BFloat16)1.5f, (BFloat16)(-2.25f), (BFloat16)3f, (BFloat16)0f, (BFloat16)100f });
    }

    [SkippableFact]
    public void TypedTransfer_RefusesAHostTypeThatIsNotTheBuffersElementType()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");
        using var floats = backend.AllocateBuffer(4, GpuElementType.Float32);
        Assert.Throws<ArgumentException>(() => backend.UploadBuffer<int>(floats, new int[4]));
        Assert.Throws<ArgumentException>(() => backend.DownloadBuffer<double>(floats, new double[4]));
        Assert.Throws<ArgumentException>(() => backend.UploadBuffer<float>(floats, new float[5]));
        Assert.Throws<NotSupportedException>(() => backend.AllocateBuffer(4, GpuElementType.Unspecified));
        Assert.False(backend.SupportsElementType(GpuElementType.Unspecified));
        Assert.True(backend.SupportsElementType(GpuElementType.Float32));
    }

    private static void RoundTrip<T>(IDirectGpuBackend backend, GpuElementType type, T[] values) where T : unmanaged
    {
        if (!backend.SupportsElementType(type))
        {
            Assert.Throws<NotSupportedException>(() => backend.AllocateBuffer(values.Length, type));
            return;
        }
        using var buffer = backend.AllocateBuffer(values.Length, type);
        Assert.Equal(type, buffer.ElementType);
        Assert.Equal(values.Length, buffer.Size);
        Assert.True(buffer.SizeInBytes >= (long)values.Length * GpuElementTypes.SizeInBytes(type));
        backend.UploadBuffer<T>(buffer, values);
        var back = new T[values.Length];
        backend.DownloadBuffer<T>(buffer, back);
        // Bitwise: a transfer must not round, reorder or reinterpret.
        Assert.True(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan())
            .SequenceEqual(System.Runtime.InteropServices.MemoryMarshal.AsBytes(back.AsSpan())), $"{type} did not round-trip.");
    }

    private static void AssertMaps<T>(GpuElementType expected)
    {
        Assert.True(GpuElementTypes.TryFromClrType<T>(out var type));
        Assert.Equal(expected, type);
    }
}
