// Copyright (c) AiDotNet. All rights reserved.
// Image codec CPU implementations — Issue #217 tail.
// - PNG: pure managed decoder + encoder (no native deps). Supports
//   non-interlaced 8-bit colour types 0/2/4/6 (grey, rgb, grey+alpha,
//   rgba). 16-bit + interlaced fall back to an explicit error — rare in
//   practice.
// - JPEG: native libjpeg-turbo bindings (dynamic, loaded on first use).
// - WebP: native libwebp bindings (dynamic).
// Native loaders throw <see cref="PlatformNotSupportedException"/> with a
// clear message if the shared library can't be loaded so callers can
// ship users an actionable install instruction.

using System;
using System.IO;
using System.IO.Compression;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<byte> ImageDecode(byte[] encoded, ImageFormat? format = null)
    {
        if (encoded is null) throw new ArgumentNullException(nameof(encoded));
        if (encoded.Length < 8) throw new ArgumentException("encoded image is too short.");

        ImageFormat detected = format ?? DetectFormat(encoded);
        return detected switch
        {
            ImageFormat.Png => PngCodec.Decode(encoded),
            ImageFormat.Jpeg => JpegBinding.Decode(encoded),
            ImageFormat.WebP => WebPBinding.Decode(encoded),
            _ => throw new ArgumentOutOfRangeException(nameof(format)),
        };
    }

    /// <inheritdoc/>
    public virtual byte[] ImageEncode(Tensor<byte> image, ImageFormat format, int quality = 90)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (image.Rank != 3)
            throw new ArgumentException("image must be rank-3 [H, W, C].");
        if ((format == ImageFormat.Jpeg || format == ImageFormat.WebP) && (quality < 0 || quality > 100))
            throw new ArgumentOutOfRangeException(nameof(quality),
                $"quality must be in [0, 100] for lossy formats; got {quality}.");
        int C = image._shape[2];
        // PNG colour types 0/2/3/4/6 map to C in {1, 3, 1, 2, 4}. Accept
        // all five (2 = grey+alpha) to match what PngCodec.Encode actually
        // supports. Only JPEG is channel-restricted below.
        if (C != 1 && C != 2 && C != 3 && C != 4)
            throw new ArgumentException("channels must be 1 (grey), 2 (grey+alpha), 3 (RGB), or 4 (RGBA).");
        if (format == ImageFormat.Jpeg && C == 2)
            throw new ArgumentException("JPEG does not support 2-channel grey+alpha.");
        return format switch
        {
            ImageFormat.Png => PngCodec.Encode(image),
            ImageFormat.Jpeg => JpegBinding.Encode(image, quality),
            ImageFormat.WebP => WebPBinding.Encode(image, quality),
            _ => throw new ArgumentOutOfRangeException(nameof(format)),
        };
    }

    /// <summary>Sniff the format from magic bytes.</summary>
    private static ImageFormat DetectFormat(byte[] bytes)
    {
        if (bytes.Length >= 8
            && bytes[0] == 137 && bytes[1] == 80 && bytes[2] == 78 && bytes[3] == 71
            && bytes[4] == 13 && bytes[5] == 10 && bytes[6] == 26 && bytes[7] == 10)
            return ImageFormat.Png;
        if (bytes.Length >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF)
            return ImageFormat.Jpeg;
        if (bytes.Length >= 12 && bytes[0] == 'R' && bytes[1] == 'I' && bytes[2] == 'F' && bytes[3] == 'F'
            && bytes[8] == 'W' && bytes[9] == 'E' && bytes[10] == 'B' && bytes[11] == 'P')
            return ImageFormat.WebP;
        throw new ArgumentException("unrecognised image format (PNG / JPEG / WebP supported).");
    }
}

// ============================================================================
// PNG — pure managed encoder + decoder.
// ============================================================================
internal static class PngCodec
{
    private static readonly byte[] Signature = { 137, 80, 78, 71, 13, 10, 26, 10 };

    public static Tensor<byte> Decode(byte[] data)
    {
        for (int i = 0; i < 8; i++)
            if (data[i] != Signature[i]) throw new InvalidDataException("not a PNG file.");

        int pos = 8;
        int width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
        byte[]? palette = null;
        var idat = new MemoryStream();

        while (pos < data.Length)
        {
            if (pos + 8 > data.Length) throw new InvalidDataException("PNG: truncated chunk header.");
            int chunkLen = ReadBE(data, pos); pos += 4;
            if (chunkLen < 0 || chunkLen > data.Length - pos - 4)
                throw new InvalidDataException("PNG: chunk length exceeds buffer.");
            string type = System.Text.Encoding.ASCII.GetString(data, pos, 4); pos += 4;
            if (type == "IHDR")
            {
                if (chunkLen < 13) throw new InvalidDataException("PNG: truncated IHDR.");
                width = ReadBE(data, pos); height = ReadBE(data, pos + 4);
                bitDepth = data[pos + 8]; colorType = data[pos + 9];
                interlace = data[pos + 12];
            }
            else if (type == "IDAT")
            {
                idat.Write(data, pos, chunkLen);
            }
            else if (type == "PLTE")
            {
                palette = new byte[chunkLen];
                Array.Copy(data, pos, palette, 0, chunkLen);
            }
            pos += chunkLen + 4;  // skip data + CRC
            if (type == "IEND") break;
        }

        if (width <= 0 || height <= 0 || (long)width * height > int.MaxValue / 8)
            throw new InvalidDataException($"PNG: invalid or oversized dimensions {width}×{height}.");
        if (bitDepth != 8) throw new NotSupportedException("PNG: only 8-bit depth supported.");
        if (interlace != 0) throw new NotSupportedException("PNG: interlaced images not supported.");
        int channels = colorType switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, 6 => 4, _ => throw new InvalidDataException("PNG: unknown colour type.") };
        if (colorType == 3 && palette is null) throw new InvalidDataException("PNG: indexed image without PLTE.");

        // zlib-wrapped DEFLATE — skip 2-byte zlib header.
        var zlib = idat.ToArray();
        int skip = 2;
        byte[] raw;
        using (var inflated = new MemoryStream())
        using (var infl = new DeflateStream(new MemoryStream(zlib, skip, zlib.Length - skip), CompressionMode.Decompress))
        {
            infl.CopyTo(inflated);
            raw = inflated.ToArray();
        }

        int bpp = channels;  // bytes per pixel at 8-bit
        int stride = width * bpp;
        var outBytes = new byte[width * height * (colorType == 3 ? 3 : channels)];
        var prevRow = new byte[stride];
        int src = 0;
        int dstChannels = colorType == 3 ? 3 : channels;

        for (int y = 0; y < height; y++)
        {
            if (src >= raw.Length) throw new InvalidDataException("PNG: truncated raster.");
            byte filter = raw[src++];
            var row = new byte[stride];
            Array.Copy(raw, src, row, 0, stride);
            src += stride;
            ApplyFilter(filter, row, prevRow, bpp);

            if (colorType == 3)  // indexed — expand to RGB via palette
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = row[x] * 3;
                    if (palette is null || idx + 2 >= palette.Length)
                        throw new InvalidDataException("PNG: palette index out of range.");
                    int off = (y * width + x) * 3;
                    outBytes[off] = palette[idx];
                    outBytes[off + 1] = palette[idx + 1];
                    outBytes[off + 2] = palette[idx + 2];
                }
            }
            else
            {
                Array.Copy(row, 0, outBytes, y * stride, stride);
            }
            Array.Copy(row, prevRow, stride);
        }

        return new Tensor<byte>(outBytes, new[] { height, width, dstChannels });
    }

    private static void ApplyFilter(byte filter, byte[] row, byte[] prev, int bpp)
    {
        switch (filter)
        {
            case 0: break;  // None
            case 1:  // Sub
                for (int i = bpp; i < row.Length; i++) row[i] = (byte)(row[i] + row[i - bpp]);
                break;
            case 2:  // Up
                for (int i = 0; i < row.Length; i++) row[i] = (byte)(row[i] + prev[i]);
                break;
            case 3:  // Average
                for (int i = 0; i < row.Length; i++)
                {
                    byte left = i < bpp ? (byte)0 : row[i - bpp];
                    row[i] = (byte)(row[i] + (left + prev[i]) / 2);
                }
                break;
            case 4:  // Paeth
                for (int i = 0; i < row.Length; i++)
                {
                    byte a = i < bpp ? (byte)0 : row[i - bpp];
                    byte b = prev[i];
                    byte c = i < bpp ? (byte)0 : prev[i - bpp];
                    int p = a + b - c;
                    int pa = Math.Abs(p - a);
                    int pb = Math.Abs(p - b);
                    int pc = Math.Abs(p - c);
                    byte pr = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
                    row[i] = (byte)(row[i] + pr);
                }
                break;
            default: throw new InvalidDataException($"PNG: unknown filter {filter}.");
        }
    }

    public static byte[] Encode(Tensor<byte> image)
    {
        int H = image._shape[0], W = image._shape[1], C = image._shape[2];
        byte colorType = C switch { 1 => 0, 2 => 4, 3 => 2, 4 => 6, _ => throw new ArgumentException("PNG: unsupported channel count") };

        var src = image.AsSpan();
        var raw = new byte[H * (1 + W * C)];
        int dst = 0;
        for (int y = 0; y < H; y++)
        {
            raw[dst++] = 0;  // filter type None
            src.Slice(y * W * C, W * C).CopyTo(raw.AsSpan(dst));
            dst += W * C;
        }

        // DEFLATE-compress, then wrap in zlib (2-byte header + adler32 footer).
        byte[] compressed;
        using (var ms = new MemoryStream())
        {
            using (var defl = new DeflateStream(ms, CompressionLevel.Optimal, leaveOpen: true))
                defl.Write(raw, 0, raw.Length);
            compressed = ms.ToArray();
        }
        var zlib = new byte[compressed.Length + 6];
        zlib[0] = 0x78; zlib[1] = 0x9C;  // zlib header, default level
        Array.Copy(compressed, 0, zlib, 2, compressed.Length);
        uint adler = Adler32(raw);
        int apos = compressed.Length + 2;
        zlib[apos] = (byte)(adler >> 24);
        zlib[apos + 1] = (byte)(adler >> 16);
        zlib[apos + 2] = (byte)(adler >> 8);
        zlib[apos + 3] = (byte)adler;

        // Assemble PNG stream.
        using var outMs = new MemoryStream();
        outMs.Write(Signature, 0, 8);

        // IHDR chunk: 13 bytes: width(4), height(4), bitDepth(1), colorType(1), compression(1), filter(1), interlace(1).
        var ihdr = new byte[13];
        WriteBE(ihdr, 0, W); WriteBE(ihdr, 4, H);
        ihdr[8] = 8; ihdr[9] = colorType; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
        WriteChunk(outMs, "IHDR", ihdr);
        WriteChunk(outMs, "IDAT", zlib);
        WriteChunk(outMs, "IEND", Array.Empty<byte>());
        return outMs.ToArray();
    }

    private static void WriteChunk(Stream s, string type, byte[] data)
    {
        var header = new byte[4];
        WriteBE(header, 0, data.Length);
        s.Write(header, 0, 4);
        var typeBytes = System.Text.Encoding.ASCII.GetBytes(type);
        s.Write(typeBytes, 0, 4);
        s.Write(data, 0, data.Length);
        uint crc = Crc32(typeBytes, 0, 4, 0);
        crc = Crc32(data, 0, data.Length, crc);
        var crcBytes = new byte[4];
        WriteBE(crcBytes, 0, (int)crc);
        s.Write(crcBytes, 0, 4);
    }

    private static int ReadBE(byte[] d, int o) => (d[o] << 24) | (d[o + 1] << 16) | (d[o + 2] << 8) | d[o + 3];
    private static void WriteBE(byte[] d, int o, int v)
    { d[o] = (byte)(v >> 24); d[o + 1] = (byte)(v >> 16); d[o + 2] = (byte)(v >> 8); d[o + 3] = (byte)v; }

    private static readonly uint[] CrcTable = BuildCrcTable();
    private static uint[] BuildCrcTable()
    {
        var t = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }
    private static uint Crc32(byte[] data, int offset, int length, uint seed)
    {
        uint c = seed ^ 0xFFFFFFFF;
        for (int i = 0; i < length; i++) c = CrcTable[(c ^ data[offset + i]) & 0xFF] ^ (c >> 8);
        return c ^ 0xFFFFFFFF;
    }

    private static uint Adler32(byte[] data)
    {
        uint a = 1, b = 0;
        const uint mod = 65521;
        for (int i = 0; i < data.Length; i++)
        {
            a = (a + data[i]) % mod;
            b = (b + a) % mod;
        }
        return (b << 16) | a;
    }
}

// ============================================================================
// JPEG — P/Invoke to libjpeg-turbo (the TurboJPEG API). The library is
// loaded on first call; a missing library raises a clear error telling
// users how to install it. TurboJPEG's DLL name differs per platform
// ("turbojpeg" on Win/Linux, "libturbojpeg.0" on macOS) so we try the
// common ones in order.
// ============================================================================
internal static class JpegBinding
{
    // TurboJPEG pixel-format codes (from turbojpeg.h).
    private const int TJPF_RGB = 0;
    private const int TJPF_RGBA = 7;
    private const int TJPF_GRAY = 3;
    // TJ_SAMP_* subsampling codes.
    private const int TJSAMP_444 = 0;

    // Library tried in order on first call. Native P/Invoke will throw
    // DllNotFoundException if the entry point can't be resolved — that's
    // the canonical way to detect "native lib missing" in .NET.
    [System.Runtime.InteropServices.DllImport("turbojpeg", EntryPoint = "tjInitDecompress", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern IntPtr tjInitDecompress();
    [System.Runtime.InteropServices.DllImport("turbojpeg", EntryPoint = "tjDecompressHeader3", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern int tjDecompressHeader3(IntPtr handle, byte[] jpegBuf, ulong jpegSize, out int width, out int height, out int subsamp, out int colorspace);
    [System.Runtime.InteropServices.DllImport("turbojpeg", EntryPoint = "tjDecompress2", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern int tjDecompress2(IntPtr handle, byte[] jpegBuf, ulong jpegSize, byte[] dstBuf, int width, int pitch, int height, int pixelFormat, int flags);
    [System.Runtime.InteropServices.DllImport("turbojpeg", EntryPoint = "tjInitCompress", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern IntPtr tjInitCompress();
    [System.Runtime.InteropServices.DllImport("turbojpeg", EntryPoint = "tjCompress2", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern int tjCompress2(IntPtr handle, byte[] srcBuf, int width, int pitch, int height, int pixelFormat, ref IntPtr jpegBuf, ref ulong jpegSize, int jpegSubsamp, int jpegQual, int flags);
    [System.Runtime.InteropServices.DllImport("turbojpeg", EntryPoint = "tjDestroy", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern int tjDestroy(IntPtr handle);
    [System.Runtime.InteropServices.DllImport("turbojpeg", EntryPoint = "tjFree", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern void tjFree(IntPtr buf);

    public static Tensor<byte> Decode(byte[] data)
    {
        IntPtr h;
        try { h = tjInitDecompress(); }
        catch (DllNotFoundException ex)
        {
            throw new PlatformNotSupportedException(
                "JPEG decode requires libjpeg-turbo. Install it system-wide " +
                "(apt install libturbojpeg / brew install jpeg-turbo / Windows " +
                "MSI from libjpeg-turbo.org) so 'turbojpeg' can be loaded.", ex);
        }
        if (h == IntPtr.Zero) throw new InvalidOperationException("tjInitDecompress returned null.");
        try
        {
            int r = tjDecompressHeader3(h, data, (ulong)data.Length, out int w, out int height, out _, out _);
            if (r != 0) throw new InvalidDataException("JPEG header read failed.");
            if (w <= 0 || height <= 0 || (long)w * height * 3 > int.MaxValue)
                throw new InvalidDataException($"JPEG: invalid or oversized dimensions {w}×{height}.");
            var px = new byte[w * height * 3];
            r = tjDecompress2(h, data, (ulong)data.Length, px, w, 0, height, TJPF_RGB, 0);
            if (r != 0) throw new InvalidDataException("JPEG decode failed.");
            return new Tensor<byte>(px, new[] { height, w, 3 });
        }
        finally { tjDestroy(h); }
    }

    public static byte[] Encode(Tensor<byte> image, int quality)
    {
        int H = image._shape[0], W = image._shape[1], C = image._shape[2];
        int pixFmt = C switch { 1 => TJPF_GRAY, 3 => TJPF_RGB, 4 => TJPF_RGBA, _ => -1 };
        if (pixFmt < 0) throw new ArgumentException("JPEG: unsupported channel count.");
        IntPtr h;
        try { h = tjInitCompress(); }
        catch (DllNotFoundException ex)
        {
            throw new PlatformNotSupportedException(
                "JPEG encode requires libjpeg-turbo. Install it system-wide.", ex);
        }
        if (h == IntPtr.Zero) throw new InvalidOperationException("tjInitCompress returned null.");
        IntPtr dst = IntPtr.Zero;
        ulong dstLen = 0;
        try
        {
            var srcBytes = image.AsSpan().ToArray();
            int r = tjCompress2(h, srcBytes, W, 0, H, pixFmt, ref dst, ref dstLen, TJSAMP_444, quality, 0);
            if (r != 0) throw new InvalidDataException("JPEG encode failed.");
            var result = new byte[(int)dstLen];
            System.Runtime.InteropServices.Marshal.Copy(dst, result, 0, (int)dstLen);
            return result;
        }
        finally
        {
            if (dst != IntPtr.Zero) tjFree(dst);
            tjDestroy(h);
        }
    }
}

// ============================================================================
// WebP — P/Invoke to libwebp. Simple API: WebPDecodeRGBA / WebPEncodeRGBA.
// ============================================================================
internal static class WebPBinding
{
    [System.Runtime.InteropServices.DllImport("webp", EntryPoint = "WebPDecodeRGBA", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern IntPtr WebPDecodeRGBA(byte[] data, nuint dataSize, out int width, out int height);
    [System.Runtime.InteropServices.DllImport("webp", EntryPoint = "WebPEncodeRGBA", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern nuint WebPEncodeRGBA(byte[] rgba, int width, int height, int stride, float quality, out IntPtr output);
    [System.Runtime.InteropServices.DllImport("webp", EntryPoint = "WebPFree", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern void WebPFree(IntPtr ptr);

    public static Tensor<byte> Decode(byte[] data)
    {
        IntPtr ptr;
        int w, h;
        try { ptr = WebPDecodeRGBA(data, (nuint)data.Length, out w, out h); }
        catch (DllNotFoundException ex)
        {
            throw new PlatformNotSupportedException(
                "WebP decode requires libwebp. Install it system-wide (apt " +
                "install libwebp-dev / brew install webp / Windows NuGet) so " +
                "'webp' can be loaded.", ex);
        }
        if (ptr == IntPtr.Zero) throw new InvalidDataException("WebP decode failed.");
        if (w <= 0 || h <= 0 || (long)w * h * 4 > int.MaxValue)
        {
            WebPFree(ptr);
            throw new InvalidDataException($"WebP: invalid or oversized dimensions {w}×{h}.");
        }
        try
        {
            var bytes = new byte[w * h * 4];
            System.Runtime.InteropServices.Marshal.Copy(ptr, bytes, 0, bytes.Length);
            return new Tensor<byte>(bytes, new[] { h, w, 4 });
        }
        finally { WebPFree(ptr); }
    }

    public static byte[] Encode(Tensor<byte> image, int quality)
    {
        int H = image._shape[0], W = image._shape[1], C = image._shape[2];
        if (C != 4) throw new ArgumentException("WebP encode needs 4-channel RGBA; pad if needed.");
        IntPtr outPtr = IntPtr.Zero;
        nuint outLen;
        try { outLen = WebPEncodeRGBA(image.AsSpan().ToArray(), W, H, W * 4, quality, out outPtr); }
        catch (DllNotFoundException ex)
        {
            throw new PlatformNotSupportedException(
                "WebP encode requires libwebp. Install it system-wide.", ex);
        }
        if (outLen == 0 || outPtr == IntPtr.Zero) throw new InvalidDataException("WebP encode failed.");
        try
        {
            var result = new byte[(int)outLen];
            System.Runtime.InteropServices.Marshal.Copy(outPtr, result, 0, (int)outLen);
            return result;
        }
        finally { WebPFree(outPtr); }
    }
}
