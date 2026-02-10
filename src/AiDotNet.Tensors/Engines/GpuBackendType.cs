namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Identifies the DirectGpu backend type for hardware capability reporting.
/// </summary>
public enum GpuBackendType
{
    /// <summary>
    /// No GPU backend available.
    /// </summary>
    None,

    /// <summary>
    /// NVIDIA CUDA backend for GeForce, RTX, Quadro, and Tesla GPUs.
    /// </summary>
    Cuda,

    /// <summary>
    /// OpenCL backend for cross-platform GPU support (AMD, Intel, NVIDIA fallback).
    /// </summary>
    OpenCl,

    /// <summary>
    /// AMD HIP/ROCm backend for Radeon and Instinct GPUs.
    /// </summary>
    Hip,

    /// <summary>
    /// Apple Metal backend for M1, M2, M3, M4 chips and iOS/macOS GPUs.
    /// Uses Metal Performance Shaders (MPS) for optimized neural network operations.
    /// </summary>
    Metal,

    /// <summary>
    /// WebGPU backend for browser-based GPU acceleration via WASM.
    /// Supports Chrome 113+, Edge, Firefox, and Safari (experimental).
    /// </summary>
    WebGpu,

    /// <summary>
    /// Vulkan backend for cross-platform GPU support.
    /// Excellent for Android, Linux, and as a fallback on Windows.
    /// </summary>
    Vulkan,

    /// <summary>
    /// Unknown or undetected backend type.
    /// </summary>
    Unknown
}
