// Copyright (c) AiDotNet. All rights reserved.
// Tests for the image codec surface (Issue #217 tail). PNG is pure
// managed so tested here; JPEG / WebP are native-bound and their
// decode/encode paths throw PlatformNotSupportedException when
// libjpeg-turbo / libwebp aren't present — we only verify that the
// error path surfaces clearly.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Image;

public class ImageCodecTests
{
    private readonly CpuEngine _cpu = new();

    private static Tensor<byte> SyntheticImage(int h, int w, int c)
    {
        var data = new byte[h * w * c];
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        for (int k = 0; k < c; k++)
            data[(y * w + x) * c + k] = (byte)((y * 17 + x * 31 + k * 97) & 0xFF);
        return new Tensor<byte>(data, new[] { h, w, c });
    }

    [Theory]
    [InlineData(1)]  // grey
    [InlineData(3)]  // RGB
    [InlineData(4)]  // RGBA
    public void Png_RoundtripMatchesOriginal(int channels)
    {
        var original = SyntheticImage(17, 23, channels);
        var encoded = _cpu.ImageEncode(original, ImageFormat.Png);
        // Encoded stream must start with the PNG 8-byte signature.
        byte[] sig = { 137, 80, 78, 71, 13, 10, 26, 10 };
        for (int i = 0; i < sig.Length; i++) Assert.Equal(sig[i], encoded[i]);

        var decoded = _cpu.ImageDecode(encoded);
        Assert.Equal(original.Shape.ToArray(), decoded.Shape.ToArray());
        var o = original.AsSpan();
        var d = decoded.AsSpan();
        Assert.Equal(o.Length, d.Length);
        for (int i = 0; i < o.Length; i++) Assert.Equal(o[i], d[i]);
    }

    [Fact]
    public void ImageDecode_AutoDetectsPng()
    {
        var img = SyntheticImage(5, 7, 3);
        var encoded = _cpu.ImageEncode(img, ImageFormat.Png);
        // No format argument — should auto-detect from magic bytes.
        var decoded = _cpu.ImageDecode(encoded);
        Assert.Equal(new[] { 5, 7, 3 }, decoded.Shape.ToArray());
    }

    [Fact]
    public void ImageDecode_Jpeg_NativeMissingThrowsInformative()
    {
        // Pass enough bytes past the short-input guard so we reach the
        // actual native decoder (either PlatformNotSupported when the
        // library isn't installed, or InvalidDataException from a bogus
        // stream when it is).
        var bytes = new byte[32];
        bytes[0] = 0xFF; bytes[1] = 0xD8; bytes[2] = 0xFF; bytes[3] = 0xE0;
        var ex = Assert.ThrowsAny<Exception>(() => _cpu.ImageDecode(bytes));
        // Either PlatformNotSupportedException (lib missing) or
        // InvalidDataException (lib present, bogus bytes) is acceptable —
        // we just need a clear, reached error.
        Assert.True(ex is PlatformNotSupportedException || ex is System.IO.InvalidDataException
                    || ex is InvalidOperationException,
            $"unexpected exception type: {ex.GetType().Name}: {ex.Message}");
    }

    [Fact]
    public void ImageEncode_RejectsUnsupportedChannelCount()
    {
        var weird = new Tensor<byte>(new byte[2 * 2 * 5], new[] { 2, 2, 5 });
        Assert.Throws<ArgumentException>(() => _cpu.ImageEncode(weird, ImageFormat.Png));
    }

    [Fact]
    public void ImageDecode_RejectsTooShortInput()
    {
        Assert.Throws<ArgumentException>(() => _cpu.ImageDecode(new byte[4]));
    }
}
