// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Shared <see cref="NativeLibrary.SetDllImportResolver"/> registry.
/// .NET's resolver API allows exactly one resolver per assembly, but
/// AiDotNet.Tensors has many native bindings (cuBLAS / cuRAND / cuSparse /
/// cuSolver / cuDNN / rocBLAS / rocRAND / rocSolver / MIOpen / OpenCL /
/// Vulkan / WebGPU / NCCL / MPI / UCC) that each want platform-specific
/// SONAME aliasing. This registry installs ONE resolver per process and
/// dispatches to every registered handler in order until one returns a
/// non-zero handle.
///
/// <para>Per-binding cctors call <see cref="Register"/> with a delegate
/// that returns either a loaded <see cref="IntPtr"/> handle or
/// <see cref="IntPtr.Zero"/> ("not my library, try the next handler").
/// The shared dispatcher walks the chain until something resolves or
/// the loader falls through to standard search.</para>
/// </summary>
public static class NativeLibraryResolverRegistry
{
    private static readonly object _lock = new();
    private static readonly List<DllImportResolver> _handlers = new();
    private static bool _installed;

    /// <summary>Registers a per-library resolver. Idempotent: the
    /// dispatcher is installed exactly once per assembly load.</summary>
    public static void Register(DllImportResolver handler)
    {
        if (handler is null) throw new ArgumentNullException(nameof(handler));
        lock (_lock)
        {
            _handlers.Add(handler);
            if (_installed) return;
            try
            {
                NativeLibrary.SetDllImportResolver(typeof(NativeLibraryResolverRegistry).Assembly, Dispatch);
                _installed = true;
            }
            catch (InvalidOperationException)
            {
                // Some other code path already registered a resolver on
                // this assembly. We can't override; chain registration
                // becomes a no-op for late-arriving handlers. Each
                // handler's per-lib short-circuit returns IntPtr.Zero so
                // the loader falls through to standard search — runtime
                // still resolves canonical library names correctly.
                _installed = true;
            }
        }
    }

    private static IntPtr Dispatch(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        DllImportResolver[] snapshot;
        lock (_lock) snapshot = _handlers.ToArray();
        foreach (var h in snapshot)
        {
            try
            {
                IntPtr handle = h(libraryName, assembly, searchPath);
                if (handle != IntPtr.Zero) return handle;
            }
            catch
            {
                // Defensive: a faulty handler must not break the chain.
            }
        }
        return IntPtr.Zero;
    }
}
#endif
