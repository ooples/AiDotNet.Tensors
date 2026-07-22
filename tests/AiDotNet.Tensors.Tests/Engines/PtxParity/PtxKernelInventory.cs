// Copyright (c) AiDotNet. All rights reserved.
// PTX-vs-CUDA-vs-CPU parity scaffold. Reflection inventory of the hand-emitted
// direct-PTX kernel family, so parity coverage auto-syncs: add a Ptx*Kernel and
// the coverage audit immediately requires a parity decision for it.
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace AiDotNet.Tensors.Tests.Engines.PtxParity;

/// <summary>
/// Discovers every hand-emitted direct-PTX kernel type in the AiDotNet.Tensors
/// assembly by convention (internal sealed <c>Ptx*Kernel</c> in the CUDA.Ptx
/// namespace). The inventory is the single source of truth the parity coverage
/// audit checks the registry against — nothing is hard-coded, so a newly added
/// kernel cannot slip through uncovered.
/// </summary>
public static class PtxKernelInventory
{
    private const string PtxNamespace = "AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx";

    /// <summary>Simple type names of all direct-PTX kernels, e.g. "PtxFusedResidualRmsNormD64Kernel".</summary>
    public static IReadOnlyList<string> KernelTypeNames { get; } = Discover();

    private static IReadOnlyList<string> Discover()
    {
        // Anchor on a known public type to reach the assembly; internal kernel
        // types are still returned by GetTypes().
        Assembly assembly = typeof(AiDotNet.Tensors.Engines.CpuEngine).Assembly;
        Type[] types;
        try
        {
            types = assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException ex)
        {
            types = ex.Types.Where(t => t is not null).Cast<Type>().ToArray();
        }

        return types
            .Where(t => t.Namespace == PtxNamespace
                        && t.Name.StartsWith("Ptx", StringComparison.Ordinal)
                        && t.Name.EndsWith("Kernel", StringComparison.Ordinal)
                        && t.IsClass)
            .Select(t => t.Name)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToArray();
    }
}
#endif
