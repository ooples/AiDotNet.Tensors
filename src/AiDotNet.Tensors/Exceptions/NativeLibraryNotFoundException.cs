using System;

namespace AiDotNet.Tensors.Exceptions;

/// <summary>
/// Exception thrown when a required native library is not found.
/// Provides helpful guidance on how to install the missing dependency.
/// </summary>
public class NativeLibraryNotFoundException : InvalidOperationException
{
    /// <summary>
    /// Gets the name of the native library that was not found.
    /// </summary>
    public string LibraryName { get; }

    /// <summary>
    /// Gets the NuGet package name that provides this library.
    /// </summary>
    public string NuGetPackage { get; }

    /// <summary>
    /// Gets the feature that requires this library.
    /// </summary>
    public string Feature { get; }

    /// <summary>
    /// Creates a new NativeLibraryNotFoundException.
    /// </summary>
    /// <param name="libraryName">The name of the missing library.</param>
    /// <param name="nugetPackage">The NuGet package that provides this library.</param>
    /// <param name="feature">The feature that requires this library.</param>
    public NativeLibraryNotFoundException(string libraryName, string nugetPackage, string feature)
        : base(FormatMessage(libraryName, nugetPackage, feature))
    {
        LibraryName = libraryName;
        NuGetPackage = nugetPackage;
        Feature = feature;
    }

    /// <summary>
    /// Creates a new NativeLibraryNotFoundException with an inner exception.
    /// </summary>
    public NativeLibraryNotFoundException(string libraryName, string nugetPackage, string feature, Exception innerException)
        : base(FormatMessage(libraryName, nugetPackage, feature), innerException)
    {
        LibraryName = libraryName;
        NuGetPackage = nugetPackage;
        Feature = feature;
    }

    private static string FormatMessage(string libraryName, string nugetPackage, string feature)
    {
        return $"""
            Native library '{libraryName}' is required for {feature} but was not found.

            To enable this feature, install the NuGet package:
              dotnet add package {nugetPackage}

            For more information, see: https://github.com/ooples/AiDotNet.Tensors#optional-acceleration-packages
            """;
    }
}

/// <summary>
/// Exception thrown when OpenBLAS is required but not available.
/// </summary>
public class OpenBlasNotFoundException : NativeLibraryNotFoundException
{
    private const string LibName = "OpenBLAS";
    private const string Package = "AiDotNet.Native.OpenBLAS";

    /// <summary>
    /// Creates a new OpenBlasNotFoundException for the specified feature.
    /// </summary>
    /// <param name="feature">The feature that requires OpenBLAS.</param>
    public OpenBlasNotFoundException(string feature = "CPU-accelerated BLAS operations")
        : base(LibName, Package, feature)
    {
    }
}

/// <summary>
/// Exception thrown when CLBlast is required but not available.
/// </summary>
public class ClBlastNotFoundException : NativeLibraryNotFoundException
{
    private const string LibName = "CLBlast";
    private const string Package = "AiDotNet.Native.CLBlast";

    /// <summary>
    /// Creates a new ClBlastNotFoundException for the specified feature.
    /// </summary>
    /// <param name="feature">The feature that requires CLBlast.</param>
    public ClBlastNotFoundException(string feature = "GPU-accelerated OpenCL BLAS operations")
        : base(LibName, Package, feature)
    {
    }
}

/// <summary>
/// Exception thrown when CUDA is required but not available.
/// Provides detailed troubleshooting steps for beginners.
/// </summary>
public class CudaNotFoundException : InvalidOperationException
{
    /// <summary>
    /// Gets the reason why CUDA is not available.
    /// </summary>
    public CudaUnavailableReason Reason { get; }

    /// <summary>
    /// Gets the feature that requires CUDA.
    /// </summary>
    public string Feature { get; }

    /// <summary>
    /// Creates a new CudaNotFoundException with automatic reason detection.
    /// </summary>
    /// <param name="feature">The feature that requires CUDA.</param>
    public CudaNotFoundException(string feature = "NVIDIA GPU acceleration")
        : this(DetectReason(), feature)
    {
    }

    /// <summary>
    /// Creates a new CudaNotFoundException with a specific reason.
    /// </summary>
    /// <param name="reason">The reason CUDA is unavailable.</param>
    /// <param name="feature">The feature that requires CUDA.</param>
    public CudaNotFoundException(CudaUnavailableReason reason, string feature = "NVIDIA GPU acceleration")
        : base(FormatMessage(reason, feature))
    {
        Reason = reason;
        Feature = feature;
    }

    private static CudaUnavailableReason DetectReason()
    {
        // Check if we're on a supported platform
        if (!Environment.Is64BitProcess)
            return CudaUnavailableReason.Not64BitProcess;

        // Try to detect why CUDA isn't working
        try
        {
            // Check if CUDA driver is installed (nvcuda.dll / libcuda.so)
            if (!IsCudaDriverInstalled())
                return CudaUnavailableReason.NoCudaDriver;

            // Check if cuBLAS is available
            if (!IsCuBlasInstalled())
                return CudaUnavailableReason.NoCuBlas;

            // Check if there's an NVIDIA GPU
            if (!HasNvidiaGpu())
                return CudaUnavailableReason.NoNvidiaGpu;

            return CudaUnavailableReason.Unknown;
        }
        catch
        {
            return CudaUnavailableReason.Unknown;
        }
    }

    private static bool IsCudaDriverInstalled()
    {
#if NET5_0_OR_GREATER
        try
        {
            var libName = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Windows) ? "nvcuda" : "libcuda.so.1";
            return System.Runtime.InteropServices.NativeLibrary.TryLoad(libName, out _);
        }
        catch
        {
            return false;
        }
#else
        return false;
#endif
    }

    private static bool IsCuBlasInstalled()
    {
#if NET5_0_OR_GREATER
        try
        {
            var libName = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Windows) ? "cublas64_12" : "libcublas.so.12";
            return System.Runtime.InteropServices.NativeLibrary.TryLoad(libName, out _);
        }
        catch
        {
            return false;
        }
#else
        return false;
#endif
    }

    private static bool HasNvidiaGpu()
    {
        // This is a simplified check - the full check requires calling CUDA APIs
        return IsCudaDriverInstalled();
    }

    private static string FormatMessage(CudaUnavailableReason reason, string feature)
    {
        var baseMessage = $"CUDA is required for {feature} but is not available.";

        return reason switch
        {
            CudaUnavailableReason.Not64BitProcess => $"""
                {baseMessage}

                PROBLEM: You are running a 32-bit process. CUDA requires 64-bit.

                SOLUTION:
                  1. Ensure your project targets x64 or AnyCPU (prefer 64-bit)
                  2. In Visual Studio: Project Properties > Build > Platform target > x64
                  3. Or add to your .csproj: <PlatformTarget>x64</PlatformTarget>
                """,

            CudaUnavailableReason.NoNvidiaGpu => $"""
                {baseMessage}

                PROBLEM: No NVIDIA GPU detected on this system.

                CUDA acceleration requires an NVIDIA GPU (GeForce, Quadro, or Tesla).
                AMD and Intel GPUs are not supported by CUDA.

                ALTERNATIVES:
                  - Use OpenCL for AMD/Intel GPUs: dotnet add package AiDotNet.Native.CLBlast
                  - Use CPU acceleration: dotnet add package AiDotNet.Native.OpenBLAS
                """,

            CudaUnavailableReason.NoCudaDriver => $"""
                {baseMessage}

                PROBLEM: NVIDIA CUDA driver (nvcuda.dll) not found.

                SOLUTIONS (try in order):

                1. Install/Update NVIDIA GPU drivers:
                   - Download from: https://www.nvidia.com/drivers
                   - The CUDA driver is included with NVIDIA display drivers

                2. If drivers are installed but CUDA still not found:
                   - Restart your computer after driver installation
                   - Ensure the driver version supports CUDA 12.x (driver 525.60+)

                3. Verify installation:
                   - Open Command Prompt and run: nvidia-smi
                   - If this fails, the driver is not properly installed
                """,

            CudaUnavailableReason.NoCuBlas => $"""
                {baseMessage}

                PROBLEM: cuBLAS library (cublas64_12.dll) not found.

                The NVIDIA driver is installed, but the cuBLAS library is missing.

                SOLUTIONS:

                1. Install the AiDotNet.Native.CUDA NuGet package:
                   dotnet add package AiDotNet.Native.CUDA

                2. Or install NVIDIA CUDA Toolkit:
                   - Download from: https://developer.nvidia.com/cuda-downloads
                   - Select CUDA 12.x for your operating system
                   - After installation, restart your application

                3. Or copy cuBLAS DLLs to your application folder:
                   - cublas64_12.dll
                   - cublasLt64_12.dll
                   - cudart64_12.dll
                   These can be found in CUDA Toolkit: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\
                """,

            _ => $"""
                {baseMessage}

                TROUBLESHOOTING STEPS:

                1. Check if you have an NVIDIA GPU:
                   - Open Device Manager > Display adapters
                   - Look for "NVIDIA GeForce", "NVIDIA Quadro", or "NVIDIA Tesla"

                2. Install/Update NVIDIA drivers:
                   - Download from: https://www.nvidia.com/drivers

                3. Install cuBLAS libraries:
                   dotnet add package AiDotNet.Native.CUDA

                4. Verify CUDA is working:
                   - Open Command Prompt and run: nvidia-smi
                   - This should show your GPU and driver version

                ALTERNATIVES if you don't have an NVIDIA GPU:
                  - OpenCL (AMD/Intel): dotnet add package AiDotNet.Native.CLBlast
                  - CPU only: dotnet add package AiDotNet.Native.OpenBLAS

                For more help: https://github.com/ooples/AiDotNet.Tensors/issues
                """
        };
    }
}

/// <summary>
/// Exception thrown when OpenCL is required but not available.
/// </summary>
public class OpenClNotFoundException : NativeLibraryNotFoundException
{
    private const string LibName = "OpenCL";
    private const string Package = "OpenCL runtime (system installation)";

    /// <summary>
    /// Creates a new OpenClNotFoundException for the specified feature.
    /// </summary>
    /// <param name="feature">The feature that requires OpenCL.</param>
    public OpenClNotFoundException(string feature = "GPU-accelerated operations via OpenCL")
        : base(LibName, Package, feature)
    {
    }
}
