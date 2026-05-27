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
    /// Gets a human-readable explanation of why the native CPU BLAS is unavailable,
    /// or <c>null</c> when it loaded successfully (<see cref="HasCpuBlas"/> is <c>true</c>).
    /// <para>
    /// Issue #444: when <c>libopenblas</c> is deployed but still reports as not found, this
    /// distinguishes the common causes — package not installed, a missing transitive
    /// dependency causing the OS loader to fail (<c>ERROR_MOD_NOT_FOUND</c>), an
    /// architecture mismatch, or an explicit <c>AIDOTNET_USE_BLAS</c> opt-out.
    /// </para>
    /// </summary>
    public static string? OpenBlasLoadDiagnostic => Helpers.BlasProvider.LoadError;

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
        // Issue #444: when OpenBLAS is not active, append the captured load
        // reason so a deployed-but-not-loaded library is diagnosable in place.
        var diagnostic = OpenBlasLoadDiagnostic;
        string openBlasLine = status.HasOpenBlas
            ? "Available"
            : string.IsNullOrEmpty(diagnostic) ? "Not found" : $"Not found ({diagnostic})";
        return $"""
            Native Library Status:
              CPU BLAS:
                OpenBLAS: {openBlasLine}
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
        // Issue #444: report the TRUE state of the native CPU BLAS path. These
        // detectors previously hardcoded `false` (a stale assumption from when
        // matmul ran exclusively through SimdGemm), but the engine's GEMM
        // dispatch now routes through Helpers.BlasProvider, which loads
        // libopenblas by default (industry-standard BLAS-on; opt out with
        // AIDOTNET_USE_BLAS=0). Hardcoding `false` made AccelerationDiagnostics
        // report "OpenBLAS: Not found" even when libopenblas was deployed,
        // loaded, and actively servicing matmul — a diagnostic that
        // contradicted reality. Delegating to BlasProvider keeps the diagnostic
        // surface consistent with the path real matmul actually takes.
        return Helpers.BlasProvider.IsOpenBlasActive;
    }

    private static bool DetectMkl()
    {
        // Issue #444: see DetectOpenBlas. True only when the consumer opted into
        // MKL routing (AIDOTNET_BLAS_PROVIDER=mkl*) AND MklImports loaded.
        return Helpers.BlasProvider.IsMklActive;
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

    // GetOpenBlasLibraryNames / GetMklLibraryNames removed in
    // feat/finish-mkl-replacement: DetectOpenBlas and DetectMkl short-circuit
    // to return false, so the library-name enumerators they previously fed
    // were unreachable. See git history for the original lists if a user ever
    // wants to re-enable CPU BLAS detection.
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
