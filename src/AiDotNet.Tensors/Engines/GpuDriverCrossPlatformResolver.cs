// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using System.Reflection;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Maps GPU driver/runtime library names across operating systems at RUNTIME.
///
/// <para>WHY: several native bindings historically baked the library name in with
/// <c>#if WINDOWS</c> compile-time switches (e.g. <c>"nvcuda"</c> vs <c>"libcuda"</c>).
/// The NuGet package ships ONE <c>lib/netX/AiDotNet.Tensors.dll</c> for all platforms,
/// so whichever OS the package was BUILT on became the only OS whose GPU libraries
/// could load — a package built on a Linux CI runner silently lost CUDA/HIP on every
/// Windows machine (DllImport("libcuda") can never resolve there; the engine fell back
/// to OpenCL with no visible error). This resolver makes the baked-in name irrelevant:
/// any known alias of a GPU driver/runtime library resolves to the current platform's
/// real library.</para>
/// </summary>
internal static class GpuDriverCrossPlatformResolver
{
    private static bool _registered;
    private static readonly object _lock = new();

    /// <summary>Idempotent registration into the shared assembly-level resolver chain.</summary>
    public static void EnsureRegistered()
    {
        lock (_lock)
        {
            if (_registered) return;
            _registered = true;
            NativeLibraryResolverRegistry.Register(Resolve);
        }
    }

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        string[]? candidates = MapToPlatformCandidates(libraryName);
        if (candidates is null)
            return IntPtr.Zero;

        foreach (var candidate in candidates)
        {
            // Skip the name the loader is already trying — returning Zero lets the
            // default search handle it; we only add the cross-OS aliases.
            if (string.Equals(candidate, libraryName, StringComparison.Ordinal))
                continue;
            if (NativeLibrary.TryLoad(candidate, assembly, searchPath, out var handle))
                return handle;
        }
        return IntPtr.Zero;
    }

    /// <summary>
    /// Returns the current platform's candidate names for a known GPU library alias,
    /// or null when the name is not one of ours (lets the chain continue).
    /// </summary>
    private static string[]? MapToPlatformCandidates(string libraryName)
    {
        bool windows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

        // NVIDIA driver (CUDA driver API)
        if (libraryName is "nvcuda" or "libcuda" or "libcuda.so.1" or "libcuda.so")
            return windows ? ["nvcuda"] : ["libcuda.so.1", "libcuda.so"];

        // NVRTC (runtime compiler) — newest first; exact-version probing still happens
        // upstream in NvrtcNativeBindings.ResolveApi, this is the cross-OS fallback.
        if (libraryName.StartsWith("nvrtc64_", StringComparison.Ordinal) ||
            libraryName.StartsWith("libnvrtc", StringComparison.Ordinal))
        {
            return windows
                ? ["nvrtc64_130_0", "nvrtc64_120_0", "nvrtc64_12", "nvrtc64_118_0", "nvrtc64_112_0"]
                : ["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.8", "libnvrtc.so.11", "libnvrtc.so"];
        }

        // CUDA runtime
        if (libraryName.StartsWith("cudart64_", StringComparison.Ordinal) ||
            libraryName.StartsWith("libcudart", StringComparison.Ordinal))
        {
            return windows
                ? ["cudart64_13", "cudart64_12", "cudart64_110"]
                : ["libcudart.so.13", "libcudart.so.12", "libcudart.so"];
        }

        // cuBLAS (CuBlasNative has its own version-fallback resolver; this covers the
        // cross-OS direction it cannot: a Linux-built assembly asking for libcublas on Windows)
        if (libraryName.StartsWith("cublas64_", StringComparison.Ordinal) ||
            libraryName.StartsWith("libcublas", StringComparison.Ordinal))
        {
            return windows
                ? ["cublas64_13", "cublas64_12", "cublas64_11"]
                : ["libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so"];
        }

        // AMD HIP driver/runtime + hiprtc
        if (libraryName.StartsWith("amdhip64", StringComparison.Ordinal) ||
            libraryName.StartsWith("libamdhip64", StringComparison.Ordinal))
            return windows ? ["amdhip64_6", "amdhip64"] : ["libamdhip64.so.6", "libamdhip64.so", "libamdhip64"];
        if (libraryName.StartsWith("hiprtc", StringComparison.Ordinal) ||
            libraryName.StartsWith("libhiprtc", StringComparison.Ordinal))
            return windows ? ["hiprtc0604", "hiprtc"] : ["libhiprtc.so.6", "libhiprtc.so", "libhiprtc"];

        return null;
    }
}
#endif
