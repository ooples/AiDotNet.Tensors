using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Exceptions;
#if NET5_0_OR_GREATER
using System.Collections.Generic;
#endif

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Provides centralized detection for native acceleration libraries.
/// This class detects whether optional performance packages are installed
/// and provides helpful guidance when they are not.
/// </summary>
public static class NativeLibraryDetector
{
    private static readonly Lazy<NativeLibraryStatus> _status = new(DetectAllLibraries);

    /// <summary>
    /// Gets the current native library detection status.
    /// </summary>
    public static NativeLibraryStatus Status => _status.Value;

    /// <summary>
    /// Gets whether CLBlast is available for GPU-accelerated OpenCL BLAS operations.
    /// </summary>
    public static bool HasClBlast => Status.HasClBlast;

    /// <summary>
    /// Gets whether OpenCL runtime is available.
    /// </summary>
    public static bool HasOpenCl => Status.HasOpenCl;

    /// <summary>
    /// Gets whether CUDA runtime is available for NVIDIA GPU acceleration.
    /// </summary>
    public static bool HasCuda => Status.HasCuda;

    /// <summary>
    /// Gets whether HIP/ROCm runtime is available for AMD GPU acceleration.
    /// </summary>
    public static bool HasHip => Status.HasHip;

    /// <summary>
    /// Ensures CLBlast is available, throwing a helpful exception if not.
    /// </summary>
    /// <param name="feature">Optional: The feature that requires CLBlast.</param>
    /// <exception cref="ClBlastNotFoundException">Thrown when CLBlast is not available.</exception>
    public static void RequireClBlast(string feature = "GPU-accelerated OpenCL BLAS operations")
    {
        if (!HasClBlast)
        {
            throw new ClBlastNotFoundException(feature);
        }
    }

    /// <summary>
    /// Ensures OpenCL runtime is available, throwing a helpful exception if not.
    /// </summary>
    /// <param name="feature">Optional: The feature that requires OpenCL.</param>
    /// <exception cref="OpenClNotFoundException">Thrown when OpenCL is not available.</exception>
    public static void RequireOpenCl(string feature = "GPU-accelerated operations via OpenCL")
    {
        if (!HasOpenCl)
        {
            throw new OpenClNotFoundException(feature);
        }
    }

    /// <summary>
    /// Ensures CUDA is available, throwing a helpful exception if not.
    /// </summary>
    /// <param name="feature">Optional: The feature that requires CUDA.</param>
    /// <exception cref="CudaNotFoundException">Thrown when CUDA is not available.</exception>
    public static void RequireCuda(string feature = "NVIDIA GPU acceleration")
    {
        if (!HasCuda)
        {
            throw new CudaNotFoundException(feature);
        }
    }

    /// <summary>
    /// Gets a human-readable summary of available native libraries.
    /// </summary>
    public static string GetStatusSummary()
    {
        var status = Status;
        return $"""
            Native Library Status:
              GPU Acceleration:
                OpenCL Runtime: {(status.HasOpenCl ? "Available" : "Not found")}
                CLBlast: {(status.HasClBlast ? "Available" : "Not found")}
                CUDA: {(status.HasCuda ? "Available" : "Not found")}
                HIP/ROCm: {(status.HasHip ? "Available" : "Not found")}

            To install optional packages:
              dotnet add package AiDotNet.Native.CLBlast   # 10x+ faster GPU matrix ops
            """;
    }

    private static NativeLibraryStatus DetectAllLibraries()
    {
        return new NativeLibraryStatus
        {
            HasClBlast = DetectClBlast(),
            HasOpenCl = DetectOpenCl(),
            HasCuda = DetectCuda(),
            HasHip = DetectHip()
        };
    }

    private static bool DetectClBlast()
    {
#if NET5_0_OR_GREATER
        return TryLoadLibrary("clblast");
#else
        return false;
#endif
    }

    private static bool DetectOpenCl()
    {
#if NET5_0_OR_GREATER
        return TryLoadLibrary("OpenCL");
#else
        return false;
#endif
    }

    private static bool DetectCuda()
    {
#if NET5_0_OR_GREATER
        if (!Environment.Is64BitProcess)
            return false;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return TryLoadLibrary("nvcuda.dll");
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return TryLoadLibrary("libcuda.so.1") || TryLoadLibrary("libcuda.so");
        }
#endif
        return false;
    }

    private static bool DetectHip()
    {
#if NET5_0_OR_GREATER
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return TryLoadLibrary("amdhip64.dll");
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return TryLoadLibrary("libamdhip64.so") || TryLoadLibrary("libamdhip64.so.5");
        }
#endif
        return false;
    }

#if NET5_0_OR_GREATER
    private static bool TryLoadLibrary(string name)
    {
        try
        {
            if (NativeLibrary.TryLoad(name, out var handle))
            {
                NativeLibrary.Free(handle);
                return true;
            }
        }
        catch
        {
            // Ignore load failures
        }
        return false;
    }
#endif
}

/// <summary>
/// Contains the detection status for all native acceleration libraries.
/// </summary>
public class NativeLibraryStatus
{
    /// <summary>Gets or sets whether CLBlast is available.</summary>
    public bool HasClBlast { get; set; }

    /// <summary>Gets or sets whether OpenCL runtime is available.</summary>
    public bool HasOpenCl { get; set; }

    /// <summary>Gets or sets whether CUDA runtime is available.</summary>
    public bool HasCuda { get; set; }

    /// <summary>Gets or sets whether HIP/ROCm runtime is available.</summary>
    public bool HasHip { get; set; }

    /// <summary>Gets whether any GPU acceleration is available.</summary>
    public bool HasGpuAcceleration => HasOpenCl || HasCuda || HasHip;
}
