// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// GPU vendor identification.
/// </summary>
public enum GpuVendor
{
    /// <summary>
    /// Unknown or unsupported vendor.
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// AMD (Radeon, Instinct).
    /// </summary>
    AMD = 1,

    /// <summary>
    /// NVIDIA (GeForce, RTX, Quadro, Tesla).
    /// </summary>
    NVIDIA = 2,

    /// <summary>
    /// Intel (Arc, UHD, Iris).
    /// </summary>
    Intel = 3,

    /// <summary>
    /// Apple (M1, M2, M3 GPUs via Metal - future support).
    /// </summary>
    Apple = 4
}

/// <summary>
/// GPU backend type preference.
/// </summary>
public enum GpuBackendPreference
{
    /// <summary>
    /// Auto-detect best backend for the GPU vendor.
    /// NVIDIA: CUDA (primary), OpenCL (fallback)
    /// AMD: HIP/ROCm (if available), OpenCL (fallback)
    /// Intel: OpenCL or Vulkan
    /// Apple: Metal
    /// Browser: WebGPU
    /// </summary>
    Auto = 0,

    /// <summary>
    /// Force CUDA backend (NVIDIA only).
    /// </summary>
    CUDA = 1,

    /// <summary>
    /// Force OpenCL backend (all vendors except Apple).
    /// </summary>
    OpenCL = 2,

    /// <summary>
    /// Force HIP/ROCm backend (AMD only).
    /// </summary>
    HIP = 3,

    /// <summary>
    /// Force Metal backend (Apple Silicon only).
    /// Uses Metal Performance Shaders (MPS) for optimized operations.
    /// </summary>
    Metal = 4,

    /// <summary>
    /// Force WebGPU backend (browser environment only).
    /// Requires WebGPU-enabled browser (Chrome 113+, Edge, Firefox).
    /// </summary>
    WebGPU = 5,

    /// <summary>
    /// Force Vulkan backend (cross-platform).
    /// Best for Android, also works on Linux, Windows, and macOS via MoltenVK.
    /// </summary>
    Vulkan = 6
}
