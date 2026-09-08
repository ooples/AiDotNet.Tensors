using System.Reflection;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

internal enum RocmNativeLibraryKind
{
    Runtime = 0,
    RuntimeCompiler = 1,
    HipBlas = 2
}

internal enum RocmOperatingSystem
{
    Unsupported = 0,
    Windows = 1,
    Linux = 2
}

/// <summary>
/// Provides runtime-OS ROCm library resolution for the platform-neutral NuGet assembly.
/// </summary>
internal static class RocmNativeLibraryResolver
{
    internal const string RuntimeImportName = "aidotnet_rocm_runtime";
    internal const string RuntimeCompilerImportName = "aidotnet_rocm_hiprtc";
    internal const string HipBlasImportName = "hipblas";

    internal static RocmOperatingSystem CurrentOperatingSystem
    {
        get
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return RocmOperatingSystem.Windows;
            }

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return RocmOperatingSystem.Linux;
            }

            return RocmOperatingSystem.Unsupported;
        }
    }

    internal static IReadOnlyList<string> GetCandidates(
        RocmNativeLibraryKind kind,
        RocmOperatingSystem operatingSystem,
        string? rocmBinPath)
    {
        string[] libraryNames = (kind, operatingSystem) switch
        {
            (RocmNativeLibraryKind.Runtime, RocmOperatingSystem.Windows) =>
                ["amdhip64_7.dll", "amdhip64_6.dll"],
            (RocmNativeLibraryKind.RuntimeCompiler, RocmOperatingSystem.Windows) =>
                ["hiprtc.dll", "hiprtc0700.dll", "hiprtc0604.dll", "hiprtc0603.dll",
                    "hiprtc0602.dll", "hiprtc0601.dll", "hiprtc0600.dll"],
            (RocmNativeLibraryKind.HipBlas, RocmOperatingSystem.Windows) =>
                ["hipblas.dll"],
            (RocmNativeLibraryKind.Runtime, RocmOperatingSystem.Linux) =>
                ["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"],
            (RocmNativeLibraryKind.RuntimeCompiler, RocmOperatingSystem.Linux) =>
                ["libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so"],
            (RocmNativeLibraryKind.HipBlas, RocmOperatingSystem.Linux) =>
                ["libhipblas.so.3", "libhipblas.so.2", "libhipblas.so"],
            _ => Array.Empty<string>()
        };

        var candidates = new List<string>(libraryNames.Length * 3);
        AddDirectoryCandidates(candidates, rocmBinPath, libraryNames);
        AddDirectoryCandidates(candidates, AppContext.BaseDirectory, libraryNames);
        candidates.AddRange(libraryNames);

        if (kind == RocmNativeLibraryKind.RuntimeCompiler &&
            operatingSystem == RocmOperatingSystem.Windows &&
            !string.IsNullOrWhiteSpace(rocmBinPath) &&
            Directory.Exists(rocmBinPath))
        {
            // Windows hipRTC encodes the installed ROCm minor version in its filename.
            // Discover future 6.x/7.x versions without hard-coding every release, while
            // excluding the separate builtins support DLL.
            try
            {
                string[] discoveredCompilers = Directory.GetFiles(rocmBinPath, "hiprtc*.dll")
                    .Where(path => !Path.GetFileName(path).Contains("builtins", StringComparison.OrdinalIgnoreCase))
                    .OrderByDescending(path => path, StringComparer.OrdinalIgnoreCase)
                    .Where(path => !candidates.Contains(path, StringComparer.OrdinalIgnoreCase))
                    .ToArray();
                candidates.InsertRange(0, discoveredCompilers);
            }
            catch (IOException)
            {
                // A concurrent install/uninstall can invalidate the directory between the
                // existence check and enumeration. The stable candidates below still apply.
            }
            catch (UnauthorizedAccessException)
            {
                // Standard loader probing remains available when the install directory cannot
                // be enumerated by the current process.
            }
        }

        return candidates;
    }

#if NET5_0_OR_GREATER
    internal static IntPtr Resolve(
        string libraryName,
        Assembly assembly,
        DllImportSearchPath? searchPath,
        string? rocmBinPath)
    {
        RocmNativeLibraryKind? kind = libraryName switch
        {
            RuntimeImportName => RocmNativeLibraryKind.Runtime,
            RuntimeCompilerImportName => RocmNativeLibraryKind.RuntimeCompiler,
            HipBlasImportName => RocmNativeLibraryKind.HipBlas,
            _ => null
        };

        if (!kind.HasValue)
        {
            return IntPtr.Zero;
        }

        foreach (string candidate in GetCandidates(kind.Value, CurrentOperatingSystem, rocmBinPath))
        {
            if (NativeLibrary.TryLoad(candidate, assembly, searchPath, out IntPtr handle))
            {
                return handle;
            }
        }

        return IntPtr.Zero;
    }
#endif

    private static void AddDirectoryCandidates(
        ICollection<string> candidates,
        string? directory,
        IEnumerable<string> libraryNames)
    {
        if (string.IsNullOrWhiteSpace(directory))
        {
            return;
        }

        foreach (string libraryName in libraryNames)
        {
            candidates.Add(Path.Combine(directory, libraryName));
        }
    }
}
