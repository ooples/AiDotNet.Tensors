// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Registry of fusion patterns consumed by <see cref="CpuFusionPass"/>.
/// Patterns are tried in declaration order (the dictionary preserves
/// insertion order on a forward iteration via the snapshot list, so
/// pattern priority is implicit in registration order).
///
/// <para>OCP-compliant extension: introduce a new <see cref="IFusionPattern"/>
/// implementation and call <see cref="Register"/> at module init. The
/// pass itself never changes.</para>
/// </summary>
internal static class FusionPatternRegistry
{
    private static readonly ConcurrentDictionary<string, IFusionPattern> _patterns = new();
    // Insertion-order list snapshot. We rebuild this on Register/Unregister
    // so the hot Run() path can iterate without rebuilding a list per call.
    private static volatile IReadOnlyList<IFusionPattern> _ordered = Array.Empty<IFusionPattern>();
    private static readonly object _orderLock = new();

    static FusionPatternRegistry()
    {
        // Bootstrap with the built-in patterns. Order matters when more
        // than one pattern could match the same starting node — narrower
        // (more specific) patterns first. LoRA's chain has 4 ops and is
        // strictly more specific than the 1-op MatMul-start of Linear,
        // so LoRA goes first.
        Register(new LoRAFusionPattern());
        Register(new SparseLinearFusionPattern());
        Register(new DDIMStepFusionPattern());
        Register(new LinearFusionPattern());
    }

    /// <summary>
    /// Registers (or replaces) a fusion pattern. Pattern order in the
    /// matcher is the order of first registration.
    /// </summary>
    public static void Register(IFusionPattern pattern)
    {
        if (pattern is null) throw new ArgumentNullException(nameof(pattern));
        bool isNew = _patterns.TryAdd(pattern.Name, pattern);
        if (!isNew)
        {
            // Replace in place — preserves position in the ordered list.
            _patterns[pattern.Name] = pattern;
            // No need to rebuild order list; same name keeps the slot.
            return;
        }
        lock (_orderLock)
        {
            var prev = _ordered;
            var next = new List<IFusionPattern>(prev.Count + 1);
            next.AddRange(prev);
            next.Add(pattern);
            _ordered = next;
        }
    }

    /// <summary>Removes a pattern by name. Returns false if no such pattern was registered.</summary>
    public static bool Unregister(string name)
    {
        if (!_patterns.TryRemove(name, out _)) return false;
        lock (_orderLock)
        {
            var prev = _ordered;
            var next = new List<IFusionPattern>(prev.Count);
            foreach (var p in prev)
                if (p.Name != name) next.Add(p);
            _ordered = next;
        }
        return true;
    }

    /// <summary>Snapshot of the currently registered patterns, in priority order.</summary>
    public static IReadOnlyList<IFusionPattern> Patterns => _ordered;
}
