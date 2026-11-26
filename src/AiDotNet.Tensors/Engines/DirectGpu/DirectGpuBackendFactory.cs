// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Factory for creating GPU backends with automatic vendor detection.
/// </summary>
/// <remarks>
/// <para><b>Backend Selection Strategy:</b></para>
/// <list type="bullet">
/// <item><b>NVIDIA</b>: CUDA (primary) → OpenCL (fallback)</item>
/// <item><b>AMD</b>: HIP/ROCm (if available) → OpenCL (fallback)</item>
/// <item><b>Intel</b>: OpenCL</item>
/// </list>
/// <para>
/// The factory automatically detects the GPU vendor and selects the most performant
/// backend. For NVIDIA GPUs, CUDA typically provides 10-20% better performance than
/// OpenCL. For AMD GPUs, HIP/ROCm can provide better performance when available.
/// </para>
/// </remarks>
public static class DirectGpuBackendFactory
{
    private static readonly object _lock = new();
    private static IDirectGpuBackend? _cachedBackend;
    private static GpuVendor _detectedVendor = GpuVendor.Unknown;

    /// <summary>
    /// Gets the detected GPU vendor from the last Create() call.
    /// </summary>
    public static GpuVendor DetectedVendor => _detectedVendor;

    /// <summary>
    /// Creates the best available GPU backend for this system.
    /// </summary>
    /// <param name="preference">Backend preference (default: Auto).</param>
    /// <param name="deviceIndex">GPU device index for multi-GPU systems.</param>
    /// <param name="logger">Optional logger for diagnostics.</param>
    /// <returns>The GPU backend, or null if no GPU is available.</returns>
    public static IDirectGpuBackend? Create(
        GpuBackendPreference preference = GpuBackendPreference.Auto,
        int deviceIndex = 0,
        ILogger? logger = null)
    {
        lock (_lock)
        {
            // Return cached backend if already created
            if (_cachedBackend is not null && _cachedBackend.IsAvailable)
            {
                var required = GpuBackendOptions.RequiredBackend;
                if (required != GpuBackendType.None && !IsBackend(_cachedBackend, required))
                    throw new InvalidOperationException(
                        $"{nameof(GpuBackendOptions)}.{nameof(GpuBackendOptions.RequiredBackend)} is {required}, but this " +
                        $"process's shared GPU backend is {_cachedBackend.BackendName}, created before the requirement was set.");
                return _cachedBackend;
            }

            _cachedBackend = CreateInternal(preference, deviceIndex, logger);
            return _cachedBackend;
        }
    }

    /// <summary>
    /// Creates a new GPU backend without caching (for testing/benchmarking).
    /// </summary>
    public static IDirectGpuBackend? CreateNew(
        GpuBackendPreference preference = GpuBackendPreference.Auto,
        int deviceIndex = 0,
        ILogger? logger = null)
    {
        return CreateInternal(preference, deviceIndex, logger);
    }

    /// <summary>
    /// Detects the GPU vendor from the system.
    /// </summary>
    public static GpuVendor DetectVendor()
    {
        // Prefer CUDA detection FIRST. The OpenCL probe below returns the vendor of OpenCL device 0,
        // which on a mixed-vendor box (an NVIDIA discrete GPU alongside an AMD/Intel iGPU that exposes
        // its own OpenCL platform) can be the integrated AMD/Intel part — silently routing an
        // NVIDIA+CUDA machine down the AMD/OpenCL fallback path (CreateAmdBackend → OpenCL) even though
        // cuBLAS/NVRTC load fine. If the CUDA runtime is present this is an NVIDIA box; trust that and
        // let CreateNvidiaBackend pick CUDA (its own OpenCL fallback still covers a CUDA-init failure).
        if (CudaBackend.IsCudaAvailable)
        {
            return GpuVendor.NVIDIA;
        }

        // Try OpenCL next (works on all vendors)
        if (DirectOpenClContext.IsAvailable)
        {
            try
            {
                using var context = new DirectOpenClContext();
                return ParseVendor(context.DeviceVendor);
            }
            catch
            {
                // Fall through to other detection methods
            }
        }

        // Try HIP (AMD only)
        if (HipBackend.IsHipAvailable)
        {
            return GpuVendor.AMD;
        }

        return GpuVendor.Unknown;
    }

    /// <summary>
    /// Gets information about all available GPUs.
    /// </summary>
    public static GpuInfo[] GetAvailableGpus()
    {
        var gpus = new System.Collections.Generic.List<GpuInfo>();

        // Check OpenCL devices - enumerate all GPU devices across all platforms
        if (DirectOpenClContext.IsAvailable)
        {
            int deviceCount = DirectOpenClContext.GetDeviceCount();
            for (int deviceIndex = 0; deviceIndex < deviceCount; deviceIndex++)
            {
                try
                {
                    using var context = new DirectOpenClContext(deviceIndex);
                    gpus.Add(new GpuInfo
                    {
                        DeviceIndex = deviceIndex,
                        DeviceName = context.DeviceName,
                        Vendor = ParseVendor(context.DeviceVendor),
                        VendorName = context.DeviceVendor,
                        AvailableBackends = GetAvailableBackends(ParseVendor(context.DeviceVendor)),
                        ComputeUnits = (int)context.MaxComputeUnits,
                        GlobalMemoryBytes = (long)context.GlobalMemSize
                    });
                }
                catch
                {
                    // Ignore detection errors for individual devices
                }
            }
        }

        return gpus.ToArray();
    }

    private static bool IsBackend(IDirectGpuBackend backend, GpuBackendType type)
        => backend.BackendName.StartsWith(GpuBackendOptions.NameOf(type), StringComparison.OrdinalIgnoreCase);

    private static IDirectGpuBackend? CreateInternal(
        GpuBackendPreference preference,
        int deviceIndex,
        ILogger? logger)
    {
        var required = GpuBackendOptions.RequiredBackend;
        if (required != GpuBackendType.None) return CreateRequired(required, preference, deviceIndex, logger);

        // Detect vendor first
        _detectedVendor = DetectVendor();

        if (_detectedVendor == GpuVendor.Unknown)
        {
            logger?.LogWarning("No GPU detected. GPU acceleration will be disabled.");
            return null;
        }

        // Handle explicit preference
        if (preference != GpuBackendPreference.Auto)
        {
            return CreateWithPreference(preference, deviceIndex, logger);
        }

        // Auto-select based on vendor
        return _detectedVendor switch
        {
            GpuVendor.NVIDIA => CreateNvidiaBackend(deviceIndex, logger),
            GpuVendor.AMD => CreateAmdBackend(deviceIndex, logger),
            GpuVendor.Intel => CreateIntelBackend(deviceIndex, logger),
            _ => CreateFallbackBackend(deviceIndex, logger)
        };
    }

    // RequiredBackend set: only that backend is tried, whatever the vendor or preference, and its failure throws rather
    // than leaving the caller on another backend or on no GPU.
    private static IDirectGpuBackend CreateRequired(
        GpuBackendType required, GpuBackendPreference preference, int deviceIndex, ILogger? logger)
    {
        string prefix = $"{nameof(GpuBackendOptions)}.{nameof(GpuBackendOptions.RequiredBackend)} is {required}";
        GpuBackendPreference matching = required switch
        {
            GpuBackendType.Cuda => GpuBackendPreference.CUDA,
            GpuBackendType.Hip => GpuBackendPreference.HIP,
            GpuBackendType.OpenCl => GpuBackendPreference.OpenCL,
            _ => throw new NotSupportedException(
                $"{prefix}, which this factory cannot create; create a {nameof(DirectGpuEngine)} instead."),
        };
        if (preference != GpuBackendPreference.Auto && preference != matching)
            throw new ArgumentException($"{prefix}, which contradicts the requested {preference} backend.", nameof(preference));
        string? failure = null;
        var backend = required switch
        {
            GpuBackendType.Cuda => TryCreateCuda(deviceIndex, logger, out failure),
            GpuBackendType.Hip => TryCreateHip(deviceIndex, logger),
            _ => TryCreateOpenCl(deviceIndex, logger),
        };
        return backend ?? throw new InvalidOperationException(
            $"{prefix}, and the {GpuBackendOptions.NameOf(required)} backend for device {deviceIndex} could not be " +
            $"created: {failure ?? "no reason was reported"}.");
    }

    private static IDirectGpuBackend? CreateWithPreference(
        GpuBackendPreference preference,
        int deviceIndex,
        ILogger? logger)
    {
        return preference switch
        {
            GpuBackendPreference.CUDA => TryCreateCuda(deviceIndex, logger, out _),
            GpuBackendPreference.HIP => TryCreateHip(deviceIndex, logger),
            GpuBackendPreference.OpenCL => TryCreateOpenCl(deviceIndex, logger),
            _ => null
        };
    }

    internal static IDirectGpuBackend? CreateNvidiaBackend(int deviceIndex, ILogger? logger)
    {
        // Try CUDA first (better performance on NVIDIA)
        var cuda = TryCreateCuda(deviceIndex, logger, out var cudaFailure);
        if (cuda is not null)
        {
            logger?.LogInformation("Using CUDA backend for NVIDIA GPU (optimal performance).");
            return cuda;
        }

        // An NVIDIA GPU whose CUDA backend failed: never a quiet switch to OpenCL.
        string reason = cudaFailure ?? "the CUDA backend could not be created";
        if (GpuBackendOptions.RequiredBackend == GpuBackendType.Cuda)
            throw new InvalidOperationException(
                $"{nameof(GpuBackendOptions)}.{nameof(GpuBackendOptions.RequiredBackend)} is Cuda, and the CUDA backend " +
                $"for NVIDIA device {deviceIndex} could not be created: {reason}.");
        logger?.LogWarning("CUDA backend for NVIDIA device {DeviceIndex} unavailable ({Reason}); falling back to OpenCL.",
            deviceIndex, reason);
        var openCl = TryCreateOpenCl(deviceIndex, logger);
        GpuBackendOptions.WarnCudaFallback(reason, openCl is null ? "no GPU backend" : "OpenCL");
        return openCl;
    }

    private static IDirectGpuBackend? CreateAmdBackend(int deviceIndex, ILogger? logger)
    {
        // Try HIP first (better performance on AMD with ROCm)
        var hip = TryCreateHip(deviceIndex, logger);
        if (hip is not null)
        {
            logger?.LogInformation("Using HIP/ROCm backend for AMD GPU (optimal performance).");
            return hip;
        }

        // Fall back to OpenCL (works on all AMD GPUs)
        logger?.LogInformation("HIP/ROCm not available, using OpenCL for AMD GPU.");
        return TryCreateOpenCl(deviceIndex, logger);
    }

    private static IDirectGpuBackend? CreateIntelBackend(int deviceIndex, ILogger? logger)
    {
        // Intel only supports OpenCL (or oneAPI/SYCL in future)
        logger?.LogInformation("Using OpenCL backend for Intel GPU.");
        return TryCreateOpenCl(deviceIndex, logger);
    }

    private static IDirectGpuBackend? CreateFallbackBackend(int deviceIndex, ILogger? logger)
    {
        // Try OpenCL as universal fallback
        return TryCreateOpenCl(deviceIndex, logger);
    }

    /// <summary>Replaces the CUDA backend construction (by device index), so a test can make CUDA initialization fail.</summary>
    internal static Func<int, IDirectGpuBackend>? CudaBackendFactoryForTests { get; set; }

    private static IDirectGpuBackend? TryCreateCuda(int deviceIndex, ILogger? logger, out string? failure)
    {
        failure = null;
        var factory = CudaBackendFactoryForTests;
        if (factory is null && !CudaBackend.IsCudaAvailable)
        {
            failure = "the CUDA driver or NVRTC library could not be loaded";
            return null;
        }

        try
        {
            var backend = factory is not null ? factory(deviceIndex) : new CudaBackend(deviceIndex);
            if (backend.IsAvailable)
            {
                return backend;
            }
            backend.Dispose();
            failure = $"the CUDA driver loaded but device {deviceIndex} is not usable";
        }
        catch (Exception ex)
        {
            failure = $"{ex.GetType().Name}: {ex.Message}";
            logger?.LogWarning(ex, "Failed to initialize CUDA backend on device {DeviceIndex}.", deviceIndex);
        }

        return null;
    }

    private static IDirectGpuBackend? TryCreateHip(int deviceIndex, ILogger? logger)
    {
        if (!HipBackend.IsHipAvailable)
        {
            return null;
        }

        try
        {
            var backend = new HipBackend(deviceIndex);
            if (backend.IsAvailable)
            {
                return backend;
            }
            if (backend.InitializationException is Exception initializationException)
            {
                logger?.LogWarning(initializationException,
                    "HIP backend initialization failed on device {DeviceIndex}.", deviceIndex);
            }
            backend.Dispose();
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "Failed to initialize HIP backend on device {DeviceIndex}.", deviceIndex);
        }

        return null;
    }

    private static IDirectGpuBackend? TryCreateOpenCl(int deviceIndex, ILogger? logger)
    {
        if (!DirectOpenClContext.IsAvailable)
        {
            return null;
        }

        try
        {
            var backend = new OpenClBackend(deviceIndex, logger);
            if (backend.IsAvailable)
            {
                return backend;
            }
            if (backend.InitializationException is Exception initializationException)
            {
                logger?.LogWarning(initializationException,
                    "OpenCL backend initialization failed on device {DeviceIndex}.", deviceIndex);
            }
            backend.Dispose();
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "Failed to initialize OpenCL backend on device {DeviceIndex}.", deviceIndex);
        }

        return null;
    }

    private static GpuVendor ParseVendor(string vendorString)
    {
        if (string.IsNullOrEmpty(vendorString))
        {
            return GpuVendor.Unknown;
        }

        var upper = vendorString.ToUpperInvariant();

        if (upper.Contains("AMD") || upper.Contains("ADVANCED MICRO") || upper.Contains("ATI"))
        {
            return GpuVendor.AMD;
        }

        if (upper.Contains("NVIDIA") || upper.Contains("GEFORCE") || upper.Contains("QUADRO") || upper.Contains("TESLA"))
        {
            return GpuVendor.NVIDIA;
        }

        if (upper.Contains("INTEL"))
        {
            return GpuVendor.Intel;
        }

        if (upper.Contains("APPLE"))
        {
            return GpuVendor.Apple;
        }

        return GpuVendor.Unknown;
    }

    private static GpuBackendPreference[] GetAvailableBackends(GpuVendor vendor)
    {
        return vendor switch
        {
            GpuVendor.NVIDIA => new[] { GpuBackendPreference.CUDA, GpuBackendPreference.OpenCL },
            GpuVendor.AMD => new[] { GpuBackendPreference.HIP, GpuBackendPreference.OpenCL },
            GpuVendor.Intel => new[] { GpuBackendPreference.OpenCL },
            _ => new[] { GpuBackendPreference.OpenCL }
        };
    }

    /// <summary>
    /// Clears the cached backend (for testing).
    /// </summary>
    public static void ClearCache()
    {
        lock (_lock)
        {
            _cachedBackend?.Dispose();
            _cachedBackend = null;
            _detectedVendor = GpuVendor.Unknown;
        }
    }
}

/// <summary>
/// Information about an available GPU.
/// </summary>
public readonly struct GpuInfo
{
    /// <summary>
    /// Device index for multi-GPU systems.
    /// </summary>
    public int DeviceIndex { get; init; }

    /// <summary>
    /// GPU device name (e.g., "AMD Radeon RX 5500 XT").
    /// </summary>
    public string DeviceName { get; init; }

    /// <summary>
    /// GPU vendor.
    /// </summary>
    public GpuVendor Vendor { get; init; }

    /// <summary>
    /// Vendor name string from driver.
    /// </summary>
    public string VendorName { get; init; }

    /// <summary>
    /// Available backend types for this GPU.
    /// </summary>
    public GpuBackendPreference[] AvailableBackends { get; init; }

    /// <summary>
    /// Number of compute units.
    /// </summary>
    public int ComputeUnits { get; init; }

    /// <summary>
    /// Total GPU memory in bytes.
    /// </summary>
    public long GlobalMemoryBytes { get; init; }

    /// <summary>
    /// GPU memory in human-readable format.
    /// </summary>
    public string MemoryFormatted => GlobalMemoryBytes switch
    {
        >= 1024L * 1024 * 1024 => $"{GlobalMemoryBytes / (1024.0 * 1024 * 1024):F1} GB",
        >= 1024L * 1024 => $"{GlobalMemoryBytes / (1024.0 * 1024):F1} MB",
        _ => $"{GlobalMemoryBytes} bytes"
    };

    public override string ToString() =>
        $"{DeviceName} ({Vendor}, {ComputeUnits} CUs, {MemoryFormatted})";
}
