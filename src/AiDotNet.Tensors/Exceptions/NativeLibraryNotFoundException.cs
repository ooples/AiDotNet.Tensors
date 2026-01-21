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
/// </summary>
public class CudaNotFoundException : NativeLibraryNotFoundException
{
    private const string LibName = "CUDA";
    private const string Package = "AiDotNet.Native.CUDA";

    /// <summary>
    /// Creates a new CudaNotFoundException for the specified feature.
    /// </summary>
    /// <param name="feature">The feature that requires CUDA.</param>
    public CudaNotFoundException(string feature = "NVIDIA GPU acceleration")
        : base(LibName, Package, feature)
    {
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
