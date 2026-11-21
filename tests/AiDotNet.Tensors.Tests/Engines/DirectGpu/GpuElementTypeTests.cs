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

    private static void AssertMaps<T>(GpuElementType expected)
    {
        Assert.True(GpuElementTypes.TryFromClrType<T>(out var type));
        Assert.Equal(expected, type);
    }
}
