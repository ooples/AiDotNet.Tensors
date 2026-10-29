// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The backend contract behind IEngine.FftGeneric.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Backends with a device-resident, generic-precision, arbitrary-length FFT: batched radix-2 for power-of-two lengths
/// and Bluestein for every other length, on split real / imaginary buffers stored as <see cref="FftElementType"/>.
/// Arithmetic is float32 whatever the storage type; the transform is unscaled forward and scaled by 1/n inverse.
/// </summary>
/// <remarks>
/// Implemented by the CUDA backend (#959) and the Vulkan backend (#964). The remaining backends are tracked per backend:
/// HIP #961, OpenCL #962, Metal #963 and WebGPU #965. A backend without it does not get a host fallback: an explicit
/// <c>IEngine.FftGeneric</c> request throws <see cref="System.NotSupportedException"/>.
/// </remarks>
internal interface IGenericFftBackend
{
    /// <summary>
    /// True when this backend routes power-of-two lengths through the generic FFT too; false when it keeps a dedicated
    /// power-of-two kernel and uses the generic path only for the other lengths.
    /// </summary>
    bool PrefersGenericFftForPowerOfTwo { get; }

    /// <summary>
    /// True when every kernel the transform needs for <paramref name="type"/> is compiled and the device can run it.
    /// The result is authoritative: a false answer is never worked around with a host copy.
    /// </summary>
    bool SupportsFftElementType(FftElementType type);

    /// <summary>
    /// Narrows <paramref name="count"/> float32 values into <paramref name="type"/> storage. The destination is a byte
    /// buffer of at least <c>count * type.ByteSize()</c> bytes (<c>IDirectGpuBackend.AllocateByteBuffer</c>).
    /// </summary>
    void ConvertFloatToFftStorage(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type);

    /// <summary>Widens <paramref name="count"/> values of <paramref name="type"/> storage to float32.</summary>
    void ConvertFftStorageToFloat(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type);

    /// <summary>
    /// Transforms <paramref name="batch"/> rows of <paramref name="n"/> complex values in place. Both buffers hold
    /// <c>batch * n</c> values of <paramref name="type"/>.
    /// </summary>
    void LaunchFftGeneric(IGpuBuffer real, IGpuBuffer imaginary, int batch, int n, bool inverse, FftElementType type);
}
