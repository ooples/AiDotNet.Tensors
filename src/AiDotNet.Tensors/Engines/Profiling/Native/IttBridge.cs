// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.Profiling.Native;

/// <summary>
/// Bridge to Intel ITT (Instrumentation and Tracing Technology) so the
/// profiler can emit ranges that VTune / Intel Trace Analyzer pick up.
///
/// <para>Same shape as <see cref="NvtxBridge"/> — silent no-op when the
/// native lib is missing. The Intel surface needs a one-time
/// <c>__itt_domain_create</c> / <c>__itt_string_handle_create</c> dance per
/// label, which we cache below for the same hot-path-free reason.</para>
/// </summary>
public static class IttBridge
{
    private static readonly bool _available = TryProbe();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, IntPtr> _stringHandles = new();
    private static IntPtr _domain;

    /// <summary>True iff <c>libittnotify</c> is loadable on this process.</summary>
    public static bool IsAvailable => _available;

    /// <summary>Pushes a named task on the calling thread's ITT task stack.</summary>
    public static void PushTask(string name)
    {
        if (!_available) return;
        try
        {
            var handle = _stringHandles.GetOrAdd(name, n => NativeMethods.__itt_string_handle_create(n));
            NativeMethods.__itt_task_begin(_domain, NativeMethods.__itt_null, NativeMethods.__itt_null, handle);
        }
        catch { /* swallow — never crash the host */ }
    }

    /// <summary>Ends the topmost task on the calling thread.</summary>
    public static void PopTask()
    {
        if (!_available) return;
        try { NativeMethods.__itt_task_end(_domain); }
        catch { /* swallow */ }
    }

    private static bool TryProbe()
    {
        try
        {
            _domain = NativeMethods.__itt_domain_create("AiDotNet.Tensors");
            return _domain != IntPtr.Zero;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }

    private static class NativeMethods
    {
        // libittnotify ships in Intel oneAPI / VTune installations.
        private const string Lib = "libittnotify";

        public static readonly IntPtr __itt_null = IntPtr.Zero;

        [DllImport(Lib, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern IntPtr __itt_domain_create(string name);

        [DllImport(Lib, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern IntPtr __itt_string_handle_create(string name);

        [DllImport(Lib)]
        internal static extern void __itt_task_begin(IntPtr domain, IntPtr taskId, IntPtr parentId, IntPtr name);

        [DllImport(Lib)]
        internal static extern void __itt_task_end(IntPtr domain);
    }
}
