// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IChannelsLastMixedConvBackend"/> conversions on HIP, Metal, OpenCL, Vulkan and WebGPU: the layout change
/// in FP32 (<see cref="IChannelsLastBackend.TryChangeLayout"/>), then the backend's own storage conversion
/// (<see cref="IGenericFftBackend"/>), in two passes rather than CUDA's one. These backends' convolutions are NCHW, so
/// they do not stage autocast convolutions channels-last (SupportsChannelsLastMixedConv is false); the conversions are
/// real so a channels-last 16-bit operand is never a dead end on them.
/// </summary>
internal static class PortableChannelsLastMixedConv
{
    private static FftElementType Storage(GpuMixedPrecisionDataType type) => type switch
    {
        GpuMixedPrecisionDataType.Half => FftElementType.Float16,
        GpuMixedPrecisionDataType.BFloat16 => FftElementType.BFloat16,
        _ => throw new System.ArgumentOutOfRangeException(nameof(type), type, "Not a 16-bit storage type."),
    };

    internal static void Narrow(IDirectGpuBackend backend, IGpuBuffer source, IGpuBuffer destination, int batch,
        int channels, int spatial, GpuMixedPrecisionDataType type)
    {
        int count = checked(batch * channels * spatial);
        using var staged = backend.AllocateBuffer(count);
        if (!((IChannelsLastBackend)backend).TryChangeLayout(source, staged, batch, channels, spatial, toChannelsLast: true))
            throw new System.InvalidOperationException("The channels-last layout change did not run.");
        ((IGenericFftBackend)backend).ConvertFloatToFftStorage(staged, destination, count, Storage(type));
    }

    internal static void Widen(IDirectGpuBackend backend, IGpuBuffer source, IGpuBuffer destination, int batch,
        int channels, int spatial, GpuMixedPrecisionDataType type)
    {
        int count = checked(batch * channels * spatial);
        using var staged = backend.AllocateBuffer(count);
        ((IGenericFftBackend)backend).ConvertFftStorageToFloat(source, staged, count, Storage(type));
        if (!((IChannelsLastBackend)backend).TryChangeLayout(staged, destination, batch, channels, spatial, toChannelsLast: false))
            throw new System.InvalidOperationException("The channels-last layout change did not run.");
    }
}
