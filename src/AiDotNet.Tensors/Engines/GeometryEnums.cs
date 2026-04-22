// Copyright (c) AiDotNet. All rights reserved.
// Enumerations for the torchvision-style geometry / sampling ops added by
// Issue #217: Interpolate, Pad, GridSample, AffineGrid.
namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Interpolation algorithm for <see cref="IEngine.Interpolate{T}"/> and
/// <see cref="IEngine.InterpolateByScale{T}"/>. Maps one-to-one onto
/// <c>torch.nn.functional.interpolate</c>'s <c>mode=</c> parameter.
/// </summary>
public enum InterpolateMode
{
    /// <summary>Nearest-neighbour (valid for 1D/2D/3D).</summary>
    Nearest,
    /// <summary>Linear (1D only — for 2D use <see cref="Bilinear"/>).</summary>
    Linear,
    /// <summary>Bilinear (2D only).</summary>
    Bilinear,
    /// <summary>Bicubic (2D only). Cubic Catmull-Rom kernel.</summary>
    Bicubic,
    /// <summary>Trilinear (3D only).</summary>
    Trilinear,
    /// <summary>Adaptive area averaging — equivalent to adaptive_avg_pool2d
    /// when the scale is an integer ratio. Valid for 1D/2D/3D.</summary>
    Area,
}

/// <summary>
/// Padding mode for <see cref="IEngine.PadNd{T}"/>. Matches
/// <c>torch.nn.functional.pad</c>'s <c>mode=</c> parameter.
/// </summary>
public enum PadMode
{
    /// <summary>Fill the added region with a caller-provided constant
    /// value (torchvision's default).</summary>
    Constant,
    /// <summary>Reflect across the boundary, excluding the boundary pixel
    /// itself (torch default reflect).</summary>
    Reflect,
    /// <summary>Repeat the boundary pixel (aka "edge" / "border" /
    /// "replicate").</summary>
    Replicate,
    /// <summary>Wrap around toroidally (circular padding).</summary>
    Circular,
}

/// <summary>
/// Sampling algorithm for <see cref="IEngine.GridSample{T}"/>. Extends the
/// original bilinear-only API with the full torchvision surface.
/// </summary>
public enum GridSampleMode
{
    /// <summary>Bilinear interpolation (default in torchvision).</summary>
    Bilinear,
    /// <summary>Nearest-neighbour.</summary>
    Nearest,
    /// <summary>Bicubic interpolation (2D only; matches torchvision's
    /// convention of using a Catmull-Rom cubic kernel).</summary>
    Bicubic,
}

/// <summary>
/// Out-of-bounds handling for <see cref="IEngine.GridSample{T}"/>. Matches
/// torchvision's <c>padding_mode=</c> parameter.
/// </summary>
public enum GridSamplePadding
{
    /// <summary>Out-of-bounds samples read 0.</summary>
    Zeros,
    /// <summary>Out-of-bounds samples clamp to the nearest boundary pixel
    /// (aka edge / replicate).</summary>
    Border,
    /// <summary>Out-of-bounds samples reflect across the boundary.</summary>
    Reflection,
}
