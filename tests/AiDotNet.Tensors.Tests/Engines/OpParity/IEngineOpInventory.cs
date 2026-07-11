// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775). Full-surface op enumeration.
// Reflects the IEngine surface at runtime so the parity scaffold auto-syncs with it — every
// tensor-returning op is enumerated, and coverage is measured against this ground truth. (A
// Roslyn source generator could emit the same inventory at compile time; reflection gives the
// identical auto-sync with far less machinery and is the recommended mechanism here.)
#if !NETFRAMEWORK

using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

public static class IEngineOpInventory
{
    /// <summary>Distinct names of every public <see cref="IEngine"/> method that returns a
    /// <c>Tensor&lt;T&gt;</c> — the op surface the parity scaffold must eventually cover.</summary>
    public static IReadOnlyList<string> TensorReturningOps() =>
        typeof(IEngine)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(IsTensorReturning)
            .Select(m => m.Name)
            .Distinct(System.StringComparer.Ordinal)
            .OrderBy(n => n, System.StringComparer.Ordinal)
            .ToList();

    /// <summary>Total public IEngine method overloads (incl. non-tensor-returning) — the broad
    /// "~789 op" surface figure; kept for the coverage report's denominator context.</summary>
    public static int TotalPublicMethodOverloads() =>
        typeof(IEngine).GetMethods(BindingFlags.Public | BindingFlags.Instance).Length;

    private static bool IsTensorReturning(MethodInfo m)
    {
        var rt = m.ReturnType;
        // Tensor<T> reflects as the open/closed generic named "Tensor`1".
        return rt.IsGenericType && rt.Name == "Tensor`1";
    }
}
#endif
