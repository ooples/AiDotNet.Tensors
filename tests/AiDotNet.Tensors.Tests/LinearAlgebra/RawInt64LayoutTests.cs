using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// #1135: the host side of the raw int64 device layout -- each integer element type widens to int64 (ulong keeping its
/// bits) and narrows back unchecked, so a round trip is the identity at every extreme value.
/// </summary>
public class RawInt64LayoutTests
{
    private static void RoundTrip<T>(T[] values, long[] widened)
    {
        Assert.True(Tensor<T>.IsRawInt64Element());
        Assert.Equal(widened, Tensor<T>.ToRawInt64(values));
        Assert.Equal(values, Tensor<T>.FromRawInt64(widened));
    }

    [Fact]
    public void Long_RoundTrips() =>
        RoundTrip(new[] { long.MinValue, -1L, 0L, 1L, long.MaxValue }, new[] { long.MinValue, -1L, 0L, 1L, long.MaxValue });

    [Fact]
    public void ULong_KeepsItsBits() =>
        RoundTrip(new[] { 0UL, 1UL, (ulong)long.MaxValue, 1UL << 63, ulong.MaxValue },
            new[] { 0L, 1L, long.MaxValue, long.MinValue, -1L });

    [Fact]
    public void Short_SignExtends() =>
        RoundTrip(new short[] { short.MinValue, -1, 0, short.MaxValue }, new long[] { short.MinValue, -1, 0, short.MaxValue });

    [Fact]
    public void UShort_ZeroExtends() =>
        RoundTrip(new ushort[] { 0, 1, ushort.MaxValue }, new long[] { 0, 1, ushort.MaxValue });

    [Fact]
    public void SByte_SignExtends() =>
        RoundTrip(new sbyte[] { sbyte.MinValue, -1, 0, sbyte.MaxValue }, new long[] { sbyte.MinValue, -1, 0, sbyte.MaxValue });

    [Fact]
    public void Byte_ZeroExtends() =>
        RoundTrip(new byte[] { 0, 1, byte.MaxValue }, new long[] { 0, 1, byte.MaxValue });

    [Fact]
    public void UInt_ZeroExtends() =>
        RoundTrip(new uint[] { 0, 1, uint.MaxValue }, new long[] { 0, 1, uint.MaxValue });

    [Fact]
    public void Narrowing_WrapsLikeAnUncheckedCast()
    {
        // A device result already wrapped to the type narrows exactly; a wider value wraps as C#'s unchecked cast does.
        Assert.Equal(new short[] { -32768, 1 }, Tensor<short>.FromRawInt64(new long[] { 32768, 65537 }));
        Assert.Equal(new byte[] { 0, 255 }, Tensor<byte>.FromRawInt64(new long[] { 256, -1 }));
    }

    [Fact]
    public void OtherElementTypes_HaveNoRawInt64Layout()
    {
        Assert.False(Tensor<int>.IsRawInt64Element());
        Assert.False(Tensor<float>.IsRawInt64Element());
        Assert.Throws<NotSupportedException>(() => Tensor<float>.ToRawInt64(new[] { 1f }));
        Assert.Throws<NotSupportedException>(() => Tensor<double>.FromRawInt64(new[] { 1L }));
        Assert.Throws<ArgumentException>(() =>
            Tensor<float>.FromGpuBuffer(null!, null!, new[] { 1 }, bufferContainsRawInt64: true));
    }
}
