// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Bindings against the Metal Performance Shaders RNG kernels
/// (<c>MPSMatrixRandomPhilox</c> / <c>MPSMatrixRandomMTGP32</c>). MPS
/// is the Apple-side counterpart to cuRAND / rocRAND. The kernels are
/// dispatched on the system's Metal device via the Objective-C runtime;
/// availability is gated on running on macOS, since libobjc and the
/// MPS framework only exist there.
///
/// <para>The IsAvailable probe is platform-gated rather than DllImport-
/// gated: on Windows / Linux we never even attempt to call libobjc.
/// On macOS we look up the MPS class names via <c>objc_getClass</c>.</para>
/// </summary>
internal static class MpsRngNative
{
    private const string LibObjc = "/usr/lib/libobjc.A.dylib";
    private const string LibMps = "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders";

    [DllImport(LibObjc, EntryPoint = "objc_getClass")]
    private static extern IntPtr objc_getClass([MarshalAs(UnmanagedType.LPStr)] string name);

    /// <summary>True when running on macOS and the MPS framework is loadable.</summary>
    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return false;
        try
        {
            // On macOS, force-load the MPS framework via LoadLibrary; the
            // class lookup confirms the framework registered the symbols.
#if NET5_0_OR_GREATER
            if (!NativeLibrary.TryLoad(LibMps, out _)) return false;
#endif
            var philoxClass = objc_getClass("MPSMatrixRandomPhilox");
            return philoxClass != IntPtr.Zero;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
