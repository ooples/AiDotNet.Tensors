// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Bindings against MPS's LSTM kernel
/// (<c>MPSGraph.LSTM</c> on Apple silicon, <c>MPSCNNLSTM</c> on older
/// Macs). Same pattern as <see cref="MpsLinalgNative"/>: probe via
/// <c>objc_getClass</c> after force-loading the MPS framework. The
/// <see cref="MpsLstm"/> wrapper falls back to <see cref="CpuRnn"/>
/// when the probe returns false, which keeps non-macOS hosts running.
/// </summary>
internal static class MpsLstmNative
{
    private const string LibMps = "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders";
    private const string LibObjc = "/usr/lib/libobjc.A.dylib";

    [DllImport(LibObjc, EntryPoint = "objc_getClass")]
    private static extern IntPtr objc_getClass([MarshalAs(UnmanagedType.LPStr)] string name);

    /// <summary>True when the MPS LSTM kernel class is loadable.</summary>
    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return false;
        try
        {
#if NET5_0_OR_GREATER
            if (!NativeLibrary.TryLoad(LibMps, out _)) return false;
#endif
            // MPSGraph is the modern path; MPSCNNLSTM is the legacy fallback.
            return objc_getClass("MPSGraph") != IntPtr.Zero
                || objc_getClass("MPSCNNLSTM") != IntPtr.Zero;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
