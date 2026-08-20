using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA
{
    /// <summary>Precision capabilities exposed by the CUDA backend.</summary>
    public sealed partial class CudaBackend : IGpuPrecisionBackend, IGpuFp16ElementwiseBackend
    {
        /// <inheritdoc/>
        public IReadOnlyList<GpuPrecisionCapability> GetPrecisionCapabilities(GpuPrecisionOperation operation)
            => GpuPrecisionCapabilityCatalog.Create(
                SupportsFp16(operation),
                Fp16Implementation(operation),
                fp16ReducesStorageBytes: true,
                fp16OutputStorage: IsFp16Elementwise(operation) ? GpuScalarType.Float16 : GpuScalarType.Float32,
                fp16MultiplyType: IsFp16Elementwise(operation) ? GpuScalarType.Float32 : GpuScalarType.Float16,
                fp32Implementation: IsGemm(operation)
                    ? GpuPrecisionImplementation.VendorLibrary
                    : GpuPrecisionImplementation.Native,
                fp32MultiplyType: UsesTensorFloat32(operation)
                    ? GpuScalarType.TensorFloat32
                    : GpuScalarType.Float32,
                supportsTensorFloat32: UsesTensorFloat32(operation));

        private bool SupportsFp16(GpuPrecisionOperation operation) => operation switch
        {
            GpuPrecisionOperation.MatMul => SupportsHgemm,
            GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu
                => SupportsFp16NativeOps,
            GpuPrecisionOperation.Convolution => Fp16Im2colAvailable,
            _ => false,
        };

        private static bool IsFp16Elementwise(GpuPrecisionOperation operation)
            => operation is GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu;

        private static bool IsGemm(GpuPrecisionOperation operation)
            => operation is GpuPrecisionOperation.MatMul
                or GpuPrecisionOperation.MatMulTransposed
                or GpuPrecisionOperation.BatchMatMul;

        private bool UsesTensorFloat32(GpuPrecisionOperation operation)
            => IsGemm(operation)
                && _ccMajor >= 8
                && _fp32GemmComputeType == CuBlasNative.CUBLAS_COMPUTE_32F;

        private static GpuPrecisionImplementation Fp16Implementation(GpuPrecisionOperation operation)
            => operation switch
            {
                GpuPrecisionOperation.MatMul => GpuPrecisionImplementation.VendorLibrary,
                GpuPrecisionOperation.Convolution => GpuPrecisionImplementation.Composite,
                _ => GpuPrecisionImplementation.Native,
            };
    }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP
{
    /// <summary>Precision capabilities exposed by the HIP backend.</summary>
    public sealed partial class HipBackend : IGpuPrecisionBackend, IGpuFp16ElementwiseBackend
    {
        /// <inheritdoc/>
        public IReadOnlyList<GpuPrecisionCapability> GetPrecisionCapabilities(GpuPrecisionOperation operation)
            => GpuPrecisionCapabilityCatalog.Create(
                SupportsFp16(operation),
                Fp16Implementation(operation),
                fp16ReducesStorageBytes: true,
                fp16OutputStorage: IsFp16Elementwise(operation) ? GpuScalarType.Float16 : GpuScalarType.Float32,
                fp16MultiplyType: IsFp16Elementwise(operation) ? GpuScalarType.Float32 : GpuScalarType.Float16,
                fp32Implementation: IsGemm(operation)
                    ? GpuPrecisionImplementation.VendorLibrary
                    : GpuPrecisionImplementation.Native);

        private bool SupportsFp16(GpuPrecisionOperation operation) => operation switch
        {
            GpuPrecisionOperation.MatMul => SupportsHgemm,
            GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu
                => SupportsFp16NativeOps,
            GpuPrecisionOperation.Convolution => Fp16Im2colAvailable,
            _ => false,
        };

        private static bool IsFp16Elementwise(GpuPrecisionOperation operation)
            => operation is GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu;

        private static bool IsGemm(GpuPrecisionOperation operation)
            => operation is GpuPrecisionOperation.MatMul
                or GpuPrecisionOperation.MatMulTransposed
                or GpuPrecisionOperation.BatchMatMul;

        private static GpuPrecisionImplementation Fp16Implementation(GpuPrecisionOperation operation)
            => operation switch
            {
                GpuPrecisionOperation.MatMul => GpuPrecisionImplementation.VendorLibrary,
                GpuPrecisionOperation.Convolution => GpuPrecisionImplementation.Composite,
                _ => GpuPrecisionImplementation.Native,
            };
    }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    /// <summary>Precision capabilities exposed by the Metal backend.</summary>
    public sealed partial class MetalBackend : IGpuPrecisionBackend, IGpuFp16ElementwiseBackend
    {
        /// <inheritdoc/>
        public IReadOnlyList<GpuPrecisionCapability> GetPrecisionCapabilities(GpuPrecisionOperation operation)
            => GpuPrecisionCapabilityCatalog.Create(
                SupportsFp16(operation),
                GpuPrecisionImplementation.Native,
                fp16ReducesStorageBytes: true,
                fp16OutputStorage: IsFp16Elementwise(operation) ? GpuScalarType.Float16 : GpuScalarType.Float32,
                fp16MultiplyType: GpuScalarType.Float32);

        private bool SupportsFp16(GpuPrecisionOperation operation) => operation switch
        {
            GpuPrecisionOperation.MatMul => SupportsHgemm,
            GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu
                => SupportsFp16NativeOps,
            GpuPrecisionOperation.Convolution => Fp16Im2colAvailable,
            _ => false,
        };

        private static bool IsFp16Elementwise(GpuPrecisionOperation operation)
            => operation is GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu;
    }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>Precision capabilities exposed by the OpenCL backend.</summary>
    public sealed partial class OpenClBackend : IGpuPrecisionBackend, IGpuFp16ElementwiseBackend
    {
        /// <inheritdoc/>
        public IReadOnlyList<GpuPrecisionCapability> GetPrecisionCapabilities(GpuPrecisionOperation operation)
            => GpuPrecisionCapabilityCatalog.Create(
                SupportsFp16(operation),
                GpuPrecisionImplementation.Packed,
                fp16ReducesStorageBytes: true,
                fp16OutputStorage: IsFp16Elementwise(operation) ? GpuScalarType.Float16 : GpuScalarType.Float32,
                fp16MultiplyType: GpuScalarType.Float32);

        private bool SupportsFp16(GpuPrecisionOperation operation) => operation switch
        {
            GpuPrecisionOperation.MatMul => SupportsHgemm,
            GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu
                => SupportsFp16NativeOps,
            GpuPrecisionOperation.Convolution => Fp16Im2colAvailable,
            _ => false,
        };

        private static bool IsFp16Elementwise(GpuPrecisionOperation operation)
            => operation is GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu;
    }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan
{
    /// <summary>Precision capabilities exposed by the Vulkan backend.</summary>
    public sealed partial class VulkanBackend : IGpuPrecisionBackend, IGpuFp16ElementwiseBackend
    {
        /// <inheritdoc/>
        public IReadOnlyList<GpuPrecisionCapability> GetPrecisionCapabilities(GpuPrecisionOperation operation)
            => GpuPrecisionCapabilityCatalog.Create(
                SupportsFp16(operation),
                GpuPrecisionImplementation.Packed,
                fp16ReducesStorageBytes: true,
                fp16OutputStorage: IsFp16Elementwise(operation) ? GpuScalarType.Float16 : GpuScalarType.Float32,
                fp16MultiplyType: GpuScalarType.Float32);

        private bool SupportsFp16(GpuPrecisionOperation operation) => operation switch
        {
            GpuPrecisionOperation.MatMul => SupportsHgemm,
            GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu
                => SupportsFp16NativeOps,
            GpuPrecisionOperation.Convolution => Fp16Im2colAvailable,
            _ => false,
        };

        private static bool IsFp16Elementwise(GpuPrecisionOperation operation)
            => operation is GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu;
    }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu
{
    /// <summary>Precision capabilities exposed by the WebGPU backend.</summary>
#if NET7_0_OR_GREATER
    public sealed partial class WebGpuBackend : IGpuPrecisionBackend, IGpuFp16ElementwiseBackend
#else
    public sealed partial class WebGpuBackend : IGpuPrecisionBackend
#endif
    {
        /// <inheritdoc/>
        public IReadOnlyList<GpuPrecisionCapability> GetPrecisionCapabilities(GpuPrecisionOperation operation)
#if NET7_0_OR_GREATER
            => GpuPrecisionCapabilityCatalog.Create(
                SupportsFp16(operation),
                GpuPrecisionImplementation.Emulated,
                // The current WebGPU route truncates values to FP16 precision but stores them in FP32 buffers.
                // It must not claim the memory or transfer benefit of native shader-f16.
                fp16ReducesStorageBytes: false,
                fp16OutputStorage: IsFp16Elementwise(operation) ? GpuScalarType.Float16 : GpuScalarType.Float32,
                fp16MultiplyType: GpuScalarType.Float32);
#else
            => GpuPrecisionCapabilityCatalog.Float32Only;
#endif

#if NET7_0_OR_GREATER
        private bool SupportsFp16(GpuPrecisionOperation operation) => operation switch
        {
            GpuPrecisionOperation.MatMul => SupportsHgemm,
            GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu
                => SupportsFp16NativeOps,
            GpuPrecisionOperation.Convolution => Fp16Im2colAvailable,
            _ => false,
        };

        private static bool IsFp16Elementwise(GpuPrecisionOperation operation)
            => operation is GpuPrecisionOperation.Add or GpuPrecisionOperation.Relu or GpuPrecisionOperation.Gelu;
#endif
    }
}
