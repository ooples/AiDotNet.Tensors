// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// Cached probe for an OpenCL ICD on this host. Uses
/// <c>clGetPlatformIDs</c> with a zero-sized buffer to discover whether
/// any OpenCL platform is registered. The probe is cached at module
/// init so the hot path is a single static-bool read.
/// </summary>
internal static class OpenClPlatformProbe
{
    // Windows ICD: "OpenCL.dll"; Linux ICD: "libOpenCL.so.1" / "libOpenCL.so"; macOS: framework path.
    private const string Lib = "OpenCL";

#if NET5_0_OR_GREATER
    static OpenClPlatformProbe()
    {
        NativeLibrary.SetDllImportResolver(typeof(OpenClPlatformProbe).Assembly, ResolveLib);
    }

    private static IntPtr ResolveLib(string name, System.Reflection.Assembly asm, DllImportSearchPath? path)
    {
        if (name != Lib) return IntPtr.Zero;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            if (NativeLibrary.TryLoad("libOpenCL.so.1", asm, path, out var h)) return h;
            if (NativeLibrary.TryLoad("libOpenCL.so", asm, path, out h)) return h;
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            if (NativeLibrary.TryLoad("/System/Library/Frameworks/OpenCL.framework/OpenCL", asm, path, out var h)) return h;
        }
        return IntPtr.Zero;
    }
#endif

    [DllImport(Lib)]
    private static extern int clGetPlatformIDs(uint numEntries, IntPtr platforms, out uint numPlatforms);

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            int err = clGetPlatformIDs(0, IntPtr.Zero, out uint num);
            return err == 0 && num > 0;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
