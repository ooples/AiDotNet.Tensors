namespace AiDotNet.Tensors.Exceptions;

/// <summary>
/// Reasons why CUDA might not be available on a system.
/// Used by <see cref="CudaNotFoundException"/> to provide specific troubleshooting guidance.
/// </summary>
public enum CudaUnavailableReason
{
    /// <summary>
    /// Unknown reason - general troubleshooting steps will be provided.
    /// </summary>
    Unknown,

    /// <summary>
    /// Application is running as a 32-bit process. CUDA requires 64-bit.
    /// </summary>
    Not64BitProcess,

    /// <summary>
    /// No NVIDIA GPU detected on this system.
    /// AMD and Intel GPUs are not supported by CUDA.
    /// </summary>
    NoNvidiaGpu,

    /// <summary>
    /// NVIDIA display driver not installed (nvcuda.dll / libcuda.so missing).
    /// The CUDA driver is included with NVIDIA display drivers.
    /// </summary>
    NoCudaDriver,

    /// <summary>
    /// cuBLAS library not installed (cublas64_12.dll / libcublas.so.12 missing).
    /// Install via: dotnet add package AiDotNet.Native.CUDA
    /// Or install NVIDIA CUDA Toolkit.
    /// </summary>
    NoCuBlas,

    /// <summary>
    /// CUDA driver is installed but no CUDA-capable devices were found.
    /// This can happen if the GPU doesn't support the required CUDA version.
    /// </summary>
    NoDevices
}
