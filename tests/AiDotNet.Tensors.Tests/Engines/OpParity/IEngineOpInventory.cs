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

    private static bool IsTensorReturning(MethodInfo m) => ProducesTensor(m.ReturnType);

    /// <summary>A return type "produces a tensor" if it is a bare <c>Tensor&lt;T&gt;</c>, an ARRAY of
    /// them (<c>Tensor&lt;T&gt;[]</c>), or a tuple carrying at least one — so multi-output ops like
    /// split / unstack / meshgrid enter the coverage denominator instead of silently hiding parity gaps.</summary>
    private static bool ProducesTensor(System.Type rt)
    {
        // Tensor<T> reflects as the open/closed generic named "Tensor`1".
        if (rt.IsGenericType && rt.Name == "Tensor`1") return true;
        // Tensor<T>[] (or jagged arrays of tensors).
        if (rt.IsArray && rt.GetElementType() is { } el && ProducesTensor(el)) return true;
        // (Value)Tuple<...> with a tensor in any slot — split/unstack/meshgrid-style multi-returns.
        if (rt.IsGenericType && rt.FullName is { } fn && fn.StartsWith("System.ValueTuple`", System.StringComparison.Ordinal))
            return rt.GetGenericArguments().Any(ProducesTensor);
        return false;
    }
}
#endif
