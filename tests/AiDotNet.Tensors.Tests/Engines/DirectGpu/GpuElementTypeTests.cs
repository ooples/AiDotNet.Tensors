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

    private static void AssertMaps<T>(GpuElementType expected)
    {
        Assert.True(GpuElementTypes.TryFromClrType<T>(out var type));
        Assert.Equal(expected, type);
    }
}
