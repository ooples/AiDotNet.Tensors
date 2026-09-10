// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen;

/// <summary>The backend-native artifact produced before a generated kernel can execute.</summary>
public enum CodegenNativeArtifactKind
{
    /// <summary>Machine code emitted by the .NET JIT.</summary>
    DotNetMachineCode = 0,
    /// <summary>A CUDA driver module JIT-compiled from Triton output or direct PTX.</summary>
    CudaDriverModule = 1,
    /// <summary>An AMD GPU code object produced by HIPRTC.</summary>
    HipCodeObject = 2,
    /// <summary>A device-and-driver-specific binary produced by the OpenCL compiler.</summary>
    OpenClDeviceBinary = 3,
    /// <summary>A compiled Metal library and compute pipeline.</summary>
    MetalComputePipeline = 4,
    /// <summary>A SPIR-V shader module and Vulkan compute pipeline.</summary>
    VulkanComputePipeline = 5,
    /// <summary>An implementation-compiled WebGPU compute pipeline.</summary>
    WebGpuComputePipeline = 6
}

/// <summary>The concrete compiler path used to reach the executable artifact.</summary>
public enum CodegenNativeCompilationKind
{
    /// <summary>The .NET JIT lowers managed IL to host machine code.</summary>
    DotNetJit = 0,
    /// <summary>Triton lowers its program through LLVM into a CUDA driver module.</summary>
    TritonCudaJit = 1,
    /// <summary>HIPRTC compiles HIP source into an AMD GPU code object.</summary>
    HipRtc = 2,
    /// <summary>The selected OpenCL implementation compiles OpenCL C for its physical device.</summary>
    OpenClDriverCompiler = 3,
    /// <summary>Metal compiles MSL into a library and compute-pipeline state.</summary>
    MetalLibraryCompiler = 4,
    /// <summary>The WebGPU implementation compiles WGSL into its device pipeline.</summary>
    WebGpuImplementationCompiler = 5,
    /// <summary>Shaderc lowers GLSL to SPIR-V and Vulkan creates the device compute pipeline.</summary>
    ShadercToVulkanPipeline = 6,
    /// <summary>Direct PTX is restored as a cubin or JIT-loaded into a CUDA device module.</summary>
    CudaPtxModuleLoader = 7
}

/// <summary>
/// Type-safe lowering contract from a generated kernel target to the runtime compiler backend and
/// executable device artifact that must be produced before execution. This is a requirement, not
/// proof that a source emitter has completed native compilation. It prevents CUDA/PTX from being
/// treated as the generic name for hardware-native GPU compilation.
/// </summary>
public readonly record struct CodegenTargetRuntime(
    KernelTuningBackend Backend,
    CodegenNativeCompilationKind Compilation,
    CodegenNativeArtifactKind NativeArtifact,
    bool IsGpu)
{
    /// <summary>Gets the native runtime contract for every declared code-generation target.</summary>
    public static CodegenTargetRuntime For(CodegenTarget target) => target switch
    {
        CodegenTarget.CpuDotNetJit =>
            new(KernelTuningBackend.DotNetJit, CodegenNativeCompilationKind.DotNetJit,
                CodegenNativeArtifactKind.DotNetMachineCode, false),
        CodegenTarget.CpuAvx512 =>
            new(KernelTuningBackend.DotNetJit, CodegenNativeCompilationKind.DotNetJit,
                CodegenNativeArtifactKind.DotNetMachineCode, false),
        CodegenTarget.Triton =>
            new(KernelTuningBackend.Cuda, CodegenNativeCompilationKind.TritonCudaJit,
                CodegenNativeArtifactKind.CudaDriverModule, true),
        CodegenTarget.Hip =>
            new(KernelTuningBackend.Hip, CodegenNativeCompilationKind.HipRtc,
                CodegenNativeArtifactKind.HipCodeObject, true),
        CodegenTarget.OpenCl =>
            new(KernelTuningBackend.OpenCl, CodegenNativeCompilationKind.OpenClDriverCompiler,
                CodegenNativeArtifactKind.OpenClDeviceBinary, true),
        CodegenTarget.Msl =>
            new(KernelTuningBackend.Metal, CodegenNativeCompilationKind.MetalLibraryCompiler,
                CodegenNativeArtifactKind.MetalComputePipeline, true),
        CodegenTarget.Wgsl =>
            new(KernelTuningBackend.WebGpu, CodegenNativeCompilationKind.WebGpuImplementationCompiler,
                CodegenNativeArtifactKind.WebGpuComputePipeline, true),
        CodegenTarget.Glsl =>
            new(KernelTuningBackend.Vulkan, CodegenNativeCompilationKind.ShadercToVulkanPipeline,
                CodegenNativeArtifactKind.VulkanComputePipeline, true),
        CodegenTarget.DirectPtx =>
            new(KernelTuningBackend.Cuda, CodegenNativeCompilationKind.CudaPtxModuleLoader,
                CodegenNativeArtifactKind.CudaDriverModule, true),
        _ => throw new ArgumentOutOfRangeException(nameof(target))
    };
}
