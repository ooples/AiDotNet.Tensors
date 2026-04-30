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
        AiDotNet.Tensors.Engines.NativeLibraryResolverRegistry.Register(ResolveLib);
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
    internal static extern int clGetPlatformIDs(uint numEntries, IntPtr[]? platforms, out uint numPlatforms);

    // Issue #276 sub-feature 4: real OpenCL host-pointer + SVM bindings.
    [DllImport(Lib)]
    internal static extern int clGetDeviceIDs(IntPtr platform, ulong deviceType, uint num, IntPtr[]? devices, out uint numDevicesOut);

    [DllImport(Lib)]
    internal static extern IntPtr clCreateContext(IntPtr properties, uint numDevices, IntPtr[] devices, IntPtr pfnNotify, IntPtr userData, out int errcodeRet);

    [DllImport(Lib)]
    internal static extern int clReleaseContext(IntPtr context);

    [DllImport(Lib)]
    internal static extern IntPtr clCreateBuffer(IntPtr context, ulong flags, UIntPtr size, IntPtr hostPtr, out int errcodeRet);

    [DllImport(Lib)]
    internal static extern int clReleaseMemObject(IntPtr memObject);

    [DllImport(Lib)]
    internal static extern IntPtr clSVMAlloc(IntPtr context, ulong flags, UIntPtr size, uint alignment);

    [DllImport(Lib)]
    internal static extern void clSVMFree(IntPtr context, IntPtr svmPtr);

    internal const ulong CL_MEM_READ_WRITE = 1UL << 0;
    internal const ulong CL_MEM_ALLOC_HOST_PTR = 1UL << 4;
    internal const ulong CL_MEM_SVM_FINE_GRAIN_BUFFER = 1UL << 10;
    internal const ulong CL_DEVICE_TYPE_GPU = 1UL << 2;

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            int err = clGetPlatformIDs(0, null, out uint num);
            return err == 0 && num > 0;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
