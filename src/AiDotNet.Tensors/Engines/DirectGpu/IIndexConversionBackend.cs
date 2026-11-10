// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Internal capability interface for the integer index conversions the any-rank gather/scatter paths run on the
    /// device. These were members of <see cref="IDirectGpuBackend"/>; adding members to a shipped public interface
    /// breaks every external implementer (package validation CP0006), so the backends opt in here instead.
    /// </summary>
    internal interface IIndexConversionBackend
    {
        /// <summary>
        /// Raw int32 values (the storage of device-resident index tensors such as argmax indices) to float:
        /// output[i] = (float)raw[i]. The numeric float form the float-indexed kernels (take-along-dim) read.
        /// </summary>
        void Int32ToFloat(IGpuBuffer raw, IGpuBuffer output, int n);

        /// <summary>
        /// Each write's position along the scatter axis for [outer, srcDim, inner] writes, as a float:
        /// output[g] = (g / inner) % srcDim, computed in integers on the device (no host upload).
        /// </summary>
        void ScatterPositions(IGpuBuffer output, int srcDim, int inner, int n);
    }

    internal static class IndexConversion
    {
        /// <summary>
        /// The backend's index conversions; a backend without them reports a capability gap, which the engine answers
        /// with its CPU path.
        /// </summary>
        public static IIndexConversionBackend Of(IDirectGpuBackend backend)
            => backend as IIndexConversionBackend
               ?? throw new GpuCapabilityUnavailableException(
                   $"{backend.GetType().Name} does not implement the device index conversions (Int32ToFloat, ScatterPositions).");
    }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA
{
    public sealed partial class CudaBackend : IIndexConversionBackend { }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP
{
    public sealed partial class HipBackend : IIndexConversionBackend { }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IIndexConversionBackend { }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    public sealed partial class MetalBackend : IIndexConversionBackend { }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan
{
    public sealed partial class VulkanBackend : IIndexConversionBackend { }
}

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu
{
    public sealed partial class WebGpuBackend : IIndexConversionBackend { }
}
#endif
