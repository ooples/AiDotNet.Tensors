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
    /// Gets whether OpenBLAS is available for CPU-accelerated BLAS operations.
    /// </summary>
    public static bool HasOpenBlas => Status.HasOpenBlas;

    /// <summary>
    /// Gets whether Intel MKL is available for CPU-accelerated BLAS operations.
    /// </summary>
    public static bool HasMkl => Status.HasMkl;

    /// <summary>
    /// Gets whether any CPU BLAS library is available (OpenBLAS or MKL).
    /// </summary>
    public static bool HasCpuBlas => Status.HasCpuBlas;

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
    /// Ensures OpenBLAS is available, throwing a helpful exception if not.
    /// </summary>
    /// <param name="feature">Optional: The feature that requires OpenBLAS.</param>
    /// <exception cref="OpenBlasNotFoundException">Thrown when OpenBLAS is not available.</exception>
    public static void RequireOpenBlas(string feature = "CPU-accelerated BLAS operations")
    {
        if (!HasOpenBlas)
        {
            throw new OpenBlasNotFoundException(feature);
        }
    }

    /// <summary>
    /// Ensures any CPU BLAS library is available, throwing a helpful exception if not.
    /// </summary>
    /// <param name="feature">Optional: The feature that requires CPU BLAS.</param>
    /// <exception cref="OpenBlasNotFoundException">Thrown when no CPU BLAS is available.</exception>
    public static void RequireCpuBlas(string feature = "CPU-accelerated matrix operations")
    {
        if (!HasCpuBlas)
        {
            throw new OpenBlasNotFoundException(feature);
        }
    }

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
              CPU BLAS:
                OpenBLAS: {(status.HasOpenBlas ? "Available" : "Not found")}
                Intel MKL: {(status.HasMkl ? "Available" : "Not found")}

              GPU Acceleration:
                OpenCL Runtime: {(status.HasOpenCl ? "Available" : "Not found")}
                CLBlast: {(status.HasClBlast ? "Available" : "Not found")}
                CUDA: {(status.HasCuda ? "Available" : "Not found")}
                HIP/ROCm: {(status.HasHip ? "Available" : "Not found")}

            To install optional packages:
              dotnet add package AiDotNet.Native.OpenBLAS  # 2x faster CPU matrix ops
              dotnet add package AiDotNet.Native.CLBlast   # 10x+ faster GPU matrix ops
            """;
    }

    private static NativeLibraryStatus DetectAllLibraries()
    {
        return new NativeLibraryStatus
        {
            HasOpenBlas = DetectOpenBlas(),
            HasMkl = DetectMkl(),
            HasClBlast = DetectClBlast(),
            HasOpenCl = DetectOpenCl(),
            HasCuda = DetectCuda(),
            HasHip = DetectHip()
        };
    }

    private static bool DetectOpenBlas()
    {
#if NET5_0_OR_GREATER
        foreach (var name in GetOpenBlasLibraryNames())
        {
            if (TryLoadLibrary(name))
            {
                return true;
            }
        }
#endif
        return false;
    }

    private static bool DetectMkl()
    {
#if NET5_0_OR_GREATER
        foreach (var name in GetMklLibraryNames())
        {
            if (TryLoadLibrary(name))
            {
                return true;
            }
        }
#endif
        return false;
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

    private static IEnumerable<string> GetOpenBlasLibraryNames()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            yield return "openblas.dll";
            yield return "libopenblas.dll";
            yield return "libopenblas64_.dll";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            yield return "libopenblas.so";
            yield return "libopenblas.so.0";
            yield return "libopenblas64_.so";
            yield return "libopenblas64_.so.0";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            yield return "libopenblas.dylib";
        }
    }

    private static IEnumerable<string> GetMklLibraryNames()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            yield return "mkl_rt.dll";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            yield return "libmkl_rt.so";
            yield return "libmkl_rt.so.2";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            yield return "libmkl_rt.dylib";
        }
    }
#endif
}

/// <summary>
/// Contains the detection status for all native acceleration libraries.
/// </summary>
public class NativeLibraryStatus
{
    /// <summary>Gets or sets whether OpenBLAS is available.</summary>
    public bool HasOpenBlas { get; set; }

    /// <summary>Gets or sets whether Intel MKL is available.</summary>
    public bool HasMkl { get; set; }

    /// <summary>Gets whether any CPU BLAS library is available.</summary>
    public bool HasCpuBlas => HasOpenBlas || HasMkl;

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
