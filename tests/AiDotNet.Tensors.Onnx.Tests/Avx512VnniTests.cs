using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// B5 contract: Avx512Vnni.DotInt8 computes the signed int32 dot product
/// of a uint8 activation span and an int8 weight span. Locking the
/// reference semantics in here means the future VNNI SIMD kernel has a
/// bit-exact oracle to match.
/// </summary>
public class Avx512VnniTests
{
    [Fact]
    public void DotInt8_MatchesManualComputation()
    {
        byte[] a = { 1, 2, 3, 4, 255, 128, 0, 10 };
        sbyte[] b = { -1, 2, -3, 4, -128, 127, 100, 0 };
        int expected = 0;
        for (int i = 0; i < a.Length; i++) expected += a[i] * b[i];
        Assert.Equal(expected, Avx512Vnni.DotInt8(a, b, a.Length));
    }

    [Fact]
    public void DotInt8_ZeroLength_ReturnsZero()
    {
        byte[] a = { 1, 2, 3 };
        sbyte[] b = { -1, -2, -3 };
        Assert.Equal(0, Avx512Vnni.DotInt8(a, b, 0));
    }

    [Fact]
    public void DotInt8_LargeVector_MatchesReference()
    {
        var rng = new Random(0xFADE);
        const int N = 1024;
        var a = new byte[N];
        var b = new sbyte[N];
        for (int i = 0; i < N; i++)
        {
            a[i] = (byte)rng.Next(0, 256);
            b[i] = (sbyte)rng.Next(-128, 128);
        }
        int expected = 0;
        for (int i = 0; i < N; i++) expected += a[i] * b[i];
        Assert.Equal(expected, Avx512Vnni.DotInt8(a, b, N));
    }

    [Fact]
    public void DotInt8_LengthExceedsSpans_Throws()
    {
        byte[] a = { 1 };
        sbyte[] b = { 1 };
        Assert.Throws<ArgumentException>(() => Avx512Vnni.DotInt8(a, b, 10));
    }
}
