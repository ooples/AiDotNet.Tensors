// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.Profiling.Native;

/// <summary>
/// Lightweight bridge to NVIDIA NVTX (NVIDIA Tools Extension) so a profiler
/// session can emit ranges that show up in Nsight Systems / Nsight Compute
/// without the user having to annotate anything.
///
/// <para>NVTX lives in <c>nvToolsExt64_1.dll</c> (Windows) /
/// <c>libnvToolsExt.so.1</c> (Linux). When the library is missing — typical
/// on CPU-only boxes and CI runners — every method here is a silent no-op,
/// so the engine call sites can call them unconditionally without paying any
/// cost on the missing-lib path.</para>
///
/// <para>The full NVTX surface is large (categories, payloads, domain
/// scoping, custom event types). We bind only the two functions that
/// matter for a profiler timeline:
/// <c>nvtxRangePushA</c> and <c>nvtxRangePopA</c>. They form a thread-local
/// LIFO stack — push at scope start, pop on dispose. Nsight Systems renders
/// each level of the stack as a band on the timeline.</para>
///
/// <para>The engine calls in via <see cref="Profiler.OpScope(string)"/>
/// when <see cref="ProfilerActivities.Gpu"/> is in the active session's
/// <see cref="ProfilerOptions.Activities"/>. CPU-only sessions do not pay
/// the marshalling cost.</para>
/// </summary>
public static class NvtxBridge
{
    // Resolve once + cache the available state. Probing the native library
    // each call would burn ~100 ns/op even when missing — too much for a
    // hot-path that ops hit billions of times per training run.
    private static readonly bool _available = TryProbe();

    /// <summary>True iff NVTX dlopen succeeded on this process.</summary>
    public static bool IsAvailable => _available;

    /// <summary>Pushes a named range onto the calling thread's NVTX stack.
    /// No-op when <see cref="IsAvailable"/> is false.</summary>
    public static void PushRange(string name)
    {
        if (!_available) return;
        try { NativeMethods.nvtxRangePushA(name); }
        catch { /* swallow — never crash the host on a profiler hook */ }
    }

    /// <summary>Pops the topmost range. No-op when unavailable.</summary>
    public static void PopRange()
    {
        if (!_available) return;
        try { NativeMethods.nvtxRangePopA(); }
        catch { /* swallow */ }
    }

    private static bool TryProbe()
    {
        try
        {
            NativeMethods.nvtxRangePushA("__aidotnet_probe");
            NativeMethods.nvtxRangePopA();
            return true;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }

    private static class NativeMethods
    {
        // Windows ships the lib as nvToolsExt64_1.dll; Linux as libnvToolsExt.so.1.
        // .NET resolves the right name per-platform via DllImportSearchPath rules.
        private const string Lib = "nvToolsExt64_1";

        // NVTX uses CDECL on every platform NVIDIA ships it for; pin the calling
        // convention so the ABI is unambiguous (the platform default of Winapi /
        // stdcall on x86 Windows would corrupt the call frame on this entry-point set).
        [DllImport(Lib, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true,
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern int nvtxRangePushA(string message);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern int nvtxRangePopA();
    }
}
