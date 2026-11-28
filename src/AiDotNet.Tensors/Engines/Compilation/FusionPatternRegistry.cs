// Copyright (c) CrowdTrainer LLC. All rights reserved.

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
        Register(new ConvBiasBatchNormFusionPattern());   // before the conv + bias and BatchNorm fusions it subsumes
        Register(new BatchNormAddReluFusionPattern());
        Register(new BatchNormReluFusionPattern());
        Register(new BatchNormRelu6FusionPattern());
        Register(new BatchNormLeakyReluFusionPattern());
        Register(new ConvChannelEpilogueFusionPattern());   // before ConvBias: a conv whose bias feeds more steps
        Register(new ConvBiasFusionPattern());
        Register(new ConvTransposeBiasFusionPattern());
        Register(new DepthwiseConvBiasFusionPattern());
        Register(new MultiHeadAttentionPackedFusionPattern());
        Register(new LinearMseLossFusionPattern());
        Register(new MseLossFusionPattern());
        Register(new RotaryPositionEmbeddingFusionPattern());
        Register(new LogSoftmaxCrossEntropyFusionPattern());
        Register(new NormResidualRopeFusionPattern());
        Register(new RmsNormResidualFusionPattern());
        Register(new LayerNormResidualFusionPattern());
        Register(new SwiGluFusionPattern());
        Register(new BroadcastAddActivationFusionPattern());
        Register(new ElementwiseChainFusionPattern());
    }

    /// <summary>
    /// Registers (or replaces) a fusion pattern. Pattern order in the
    /// matcher is the order of first registration.
    /// </summary>
    public static void Register(IFusionPattern pattern)
    {
        if (pattern is null) throw new ArgumentNullException(nameof(pattern));
        // Always rebuild under the order lock so dictionary update and
        // ordered-snapshot rebuild are atomic. Previously a re-registration
        // updated _patterns but left _ordered pointing at the stale instance —
        // CpuFusionPass kept calling the old pattern.
        lock (_orderLock)
        {
            bool isNew = _patterns.TryAdd(pattern.Name, pattern);
            if (!isNew)
            {
                _patterns[pattern.Name] = pattern;
                // Rebuild the ordered snapshot in place, swapping the new
                // instance into the slot the old one occupied so registration
                // priority is preserved.
                var prevR = _ordered;
                var nextR = new List<IFusionPattern>(prevR.Count);
                foreach (var p in prevR)
                    nextR.Add(p.Name == pattern.Name ? pattern : p);
                _ordered = nextR;
                return;
            }

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
        // Take the lock BEFORE the dictionary remove so we can't race with
        // a concurrent Register(name) — that race would otherwise leave
        // _patterns containing a new instance while _ordered drops it,
        // silently disabling the pattern in the fusion pass.
        lock (_orderLock)
        {
            if (!_patterns.TryRemove(name, out _)) return false;
            var prev = _ordered;
            var next = new List<IFusionPattern>(prev.Count);
            foreach (var p in prev)
                if (p.Name != name) next.Add(p);
            _ordered = next;
            return true;
        }
    }

    // Names suspended on the current logical flow (thread or async context). Suspension used to remove the pattern
    // process-wide, so a test suspending a fusion changed what every test running in parallel compiled.
    private static readonly System.Threading.AsyncLocal<string[]?> s_suspended = new();

    /// <summary>
    /// Hides a pattern from the fusion pass on the calling logical flow until the returned scope is disposed. Other
    /// threads keep seeing it, and its priority slot is untouched (<see cref="Unregister"/> followed by
    /// <see cref="Register"/> would move it to the end of the order). Null when no such pattern is registered.
    /// </summary>
    internal static IDisposable? Suspend(string name)
    {
        if (!_patterns.ContainsKey(name)) return null;
        var previous = s_suspended.Value;
        var next = previous is null ? new[] { name } : AppendName(previous, name);
        s_suspended.Value = next;
        return new Restore(previous);
    }

    private static string[] AppendName(string[] names, string name)
    {
        var next = new string[names.Length + 1];
        Array.Copy(names, next, names.Length);
        next[names.Length] = name;
        return next;
    }

    private sealed class Restore : IDisposable
    {
        private readonly string[]? _previous;
        private int _disposed;

        public Restore(string[]? previous) { _previous = previous; }

        public void Dispose()
        {
            if (System.Threading.Interlocked.Exchange(ref _disposed, 1) != 0) return;
            s_suspended.Value = _previous;
        }
    }

    /// <summary>Snapshot of the currently registered patterns, in priority order.</summary>
    public static IReadOnlyList<IFusionPattern> Patterns
    {
        get
        {
            var ordered = _ordered;
            var suspended = s_suspended.Value;
            if (suspended is null) return ordered;
            var visible = new List<IFusionPattern>(ordered.Count);
            foreach (var pattern in ordered)
                if (Array.IndexOf(suspended, pattern.Name) < 0) visible.Add(pattern);
            return visible;
        }
    }
}
