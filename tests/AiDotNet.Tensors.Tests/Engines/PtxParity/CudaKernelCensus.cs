// Copyright (c) AiDotNet. All rights reserved.
// PTX transfer tracking. The authoritative census of every CUDA kernel — the
// full set that must be replaced by a promoted PTX kernel before the CUDA
// kernels can be deleted. Reflected from the same GetKernelNames() registries
// the backend compiles, so it cannot drift from what actually ships.
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace AiDotNet.Tensors.Tests.Engines.PtxParity;

/// <summary>
/// The union of every <c>public static string[] GetKernelNames()</c> registry
/// declared under the product assembly's <c>.DirectGpu.CUDA</c> namespace — i.e.
/// every CUDA kernel the backend can launch. This is the denominator for the
/// "when can we delete the CUDA kernels?" question.
/// </summary>
public static class CudaKernelCensus
{
    public static IReadOnlyCollection<string> KernelNames { get; } = Discover();

    private static SortedSet<string> Discover()
    {
        var set = new SortedSet<string>(StringComparer.Ordinal);
        Assembly asm = typeof(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend).Assembly;
        const string marker = ".DirectGpu.CUDA";
        Type[] types;
        try { types = asm.GetTypes(); }
        catch (ReflectionTypeLoadException ex) { types = ex.Types.Where(t => t is not null).Cast<Type>().ToArray(); }

        foreach (Type t in types)
        {
            if (t.Namespace is null || t.Namespace.IndexOf(marker, StringComparison.Ordinal) < 0) continue;
            MethodInfo? m = t.GetMethod("GetKernelNames", BindingFlags.Public | BindingFlags.Static,
                binder: null, types: Type.EmptyTypes, modifiers: null);
            if (m is null || m.ReturnType != typeof(string[])) continue;
            try
            {
                if (m.Invoke(null, null) is string[] names)
                    foreach (string n in names)
                        if (!string.IsNullOrWhiteSpace(n)) set.Add(n);
            }
            catch { /* a registry that needs a device/context — best effort, skip */ }
        }
        return set;
    }
}
#endif
