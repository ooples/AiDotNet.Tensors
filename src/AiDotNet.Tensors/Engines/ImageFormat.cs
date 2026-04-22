// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Image codec types supported by <see cref="IEngine.ImageDecode"/> and
/// <see cref="IEngine.ImageEncode"/>. PNG is implemented in pure managed
/// C# (no native deps). JPEG and WebP delegate to native bindings —
/// libjpeg-turbo / libwebp — and throw a descriptive
/// <c>PlatformNotSupportedException</c> if the native shared library
/// isn't loadable.
/// </summary>
public enum ImageFormat
{
    Png,
    Jpeg,
    WebP,
}
