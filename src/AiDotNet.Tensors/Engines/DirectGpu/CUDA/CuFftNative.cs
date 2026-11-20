// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Runtime.InteropServices;
#if NET5_0_OR_GREATER
using System.Reflection;
#endif

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

internal static class CuFftNative
{
#if WINDOWS
    private const string Library = "cufft64_11";
#else
    private const string Library = "libcufft.so.11";
#endif
    internal static readonly bool Available = Probe();
    internal static int Version { get; private set; }

    private static bool Probe()
    {
#if NET5_0_OR_GREATER
        NativeLibraryResolverRegistry.Register(Resolve);
#endif
        try
        {
            int status = GetVersion(out int version);
            if (status != 0) return false;
            Version = version;
            return true;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
    }

#if NET5_0_OR_GREATER
    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (name != Library) return IntPtr.Zero;
        string[] candidates = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? new[] { "cufft64_12.dll", "cufft64_11.dll", "cufft64_10.dll" }   // cuFFT 12 ships with CUDA 13
            : new[] { "libcufft.so.12", "libcufft.so.11", "libcufft.so.10", "libcufft.so" };
        foreach (string candidate in candidates)
            if (NativeLibrary.TryLoad(candidate, assembly, searchPath, out var handle)) return handle;
        string? cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (!string.IsNullOrEmpty(cudaPath))
        {
            string[] directories = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? new[] { Path.Combine(cudaPath, "bin", "x64"), Path.Combine(cudaPath, "bin") }
                : new[] { Path.Combine(cudaPath, "lib64"), Path.Combine(cudaPath, "lib") };
            foreach (string directory in directories)
                foreach (string candidate in candidates)
                    if (NativeLibrary.TryLoad(Path.Combine(directory, candidate), out var handle)) return handle;
        }
        return IntPtr.Zero;
    }
#endif

    internal static void Check(int status, string operation)
    {
        if (status != 0) throw new InvalidOperationException($"{operation} failed with cuFFT status {status}.");
    }

    [DllImport(Library, EntryPoint = "cufftGetVersion", CallingConvention = CallingConvention.Cdecl)]
    private static extern int GetVersion(out int version);
    [DllImport(Library, EntryPoint = "cufftCreate", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Create(out int plan);
    [DllImport(Library, EntryPoint = "cufftSetAutoAllocation", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int SetAutoAllocation(int plan, int allocate);
    [DllImport(Library, EntryPoint = "cufftMakePlan1d", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int MakePlan1d(int plan, int n, int type, int batch, out UIntPtr workspace);
    [DllImport(Library, EntryPoint = "cufftSetStream", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int SetStream(int plan, IntPtr stream);
    [DllImport(Library, EntryPoint = "cufftGetSize", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int GetSize(int plan, out UIntPtr workspace);
    [DllImport(Library, EntryPoint = "cufftSetWorkArea", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int SetWorkArea(int plan, IntPtr workspace);
    [DllImport(Library, EntryPoint = "cufftExecR2C", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int ExecR2C(int plan, IntPtr input, IntPtr output);
    [DllImport(Library, EntryPoint = "cufftExecC2C", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int ExecC2C(int plan, IntPtr input, IntPtr output, int direction);
    [DllImport(Library, EntryPoint = "cufftDestroy", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Destroy(int plan);
}
