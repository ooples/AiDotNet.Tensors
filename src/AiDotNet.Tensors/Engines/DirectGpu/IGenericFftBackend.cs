// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Generic-precision, arbitrary-length FFT (#959, #962): batched radix-2 for powers of two and device-resident
/// Bluestein otherwise, with float32 arithmetic and the requested storage width at load/store boundaries. Forward is
/// unscaled; inverse is scaled by 1/n.
/// </summary>
internal interface IGenericFftBackend
{
    /// <summary>True only once every kernel the element type needs is compiled on this device.</summary>
    bool SupportsFftElementType(FftElementType type);

    void ConvertFloatToFftStorage(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type);

    void ConvertFftStorageToFloat(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type);

    /// <summary>In-place transform of <paramref name="batch"/> split real/imaginary signals of length <paramref name="n"/>.</summary>
    void LaunchFftGeneric(IGpuBuffer real, IGpuBuffer imaginary, int batch, int n, bool inverse, FftElementType type);
}
