using System.Diagnostics;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Per-op backward timing aggregator. Enabled when env var
/// AIDOTNET_BWD_TIMING=1. Aggregates wall ticks by backward delegate
/// method name across a backward pass and prints a summary on demand.
/// Used to diagnose which backward function dominates the
/// chain.Execute wall time (issue #327 investigation).
/// </summary>
internal static class BackwardTiming
{
    private static readonly bool _enabled =
        Environment.GetEnvironmentVariable("AIDOTNET_BWD_TIMING") == "1";
    [ThreadStatic]
    private static Dictionary<string, (long ticks, int calls)>? _aggregator;

    internal static bool Enabled => _enabled;

    internal static void Record(string op, long ticks)
    {
        if (!_enabled) return;
        _aggregator ??= new Dictionary<string, (long, int)>();
        var key = op ?? "(null)";
        if (_aggregator.TryGetValue(key, out var prior))
            _aggregator[key] = (prior.ticks + ticks, prior.calls + 1);
        else
            _aggregator[key] = (ticks, 1);
    }

    internal static void DumpAndReset()
    {
        DumpAndReset(Console.WriteLine);
    }

    /// <summary>
    /// Phase A profiling (#338): emits the per-op breakdown via the
    /// caller-supplied writer in table format. Default Console writer
    /// preserves the legacy behaviour; tests can pass an xUnit
    /// <c>ITestOutputHelper.WriteLine</c>-style delegate.
    /// </summary>
    internal static void DumpAndReset(Action<string> writer)
    {
        if (!_enabled || _aggregator is null) return;
        writer(string.Empty);
        writer($"{"Backward op",-40}{"calls",10}{"total ms",14}{"avg µs",14}");
        long totalTicks = 0;
        int totalCalls = 0;
        foreach (var kv in _aggregator) { totalTicks += kv.Value.ticks; totalCalls += kv.Value.calls; }
        var sorted = new List<(string op, long ticks, int calls)>();
        foreach (var kv in _aggregator) sorted.Add((kv.Key, kv.Value.ticks, kv.Value.calls));
        sorted.Sort((a, b) => b.ticks.CompareTo(a.ticks));
        foreach (var (op, ticks, calls) in sorted)
        {
            double ms = ticks * 1000.0 / Stopwatch.Frequency;
            double us = ticks * 1_000_000.0 / Stopwatch.Frequency / calls;
            writer($"{op,-40}{calls,10}{ms,14:F3}{us,14:F1}");
        }
        writer($"{"TOTAL",-40}{totalCalls,10}{totalTicks * 1000.0 / Stopwatch.Frequency,14:F3}");
        _aggregator.Clear();
    }

    /// <summary>
    /// Phase A profiling (#338): emits the per-op breakdown as CSV
    /// rows. Format: <c>function,calls,total_ticks,total_ms,pct_of_total</c>.
    /// Sorted by total_ticks descending. The aggregator is RESET after
    /// emission (consistent with <see cref="DumpAndReset()"/>).
    /// </summary>
    internal static void DumpAndResetCsv(Action<string> writer)
    {
        if (!_enabled || _aggregator is null) return;
        long totalTicks = 0;
        foreach (var kv in _aggregator) totalTicks += kv.Value.ticks;
        writer("function,calls,total_ticks,total_ms,pct_of_total");
        var sorted = new List<(string op, long ticks, int calls)>();
        foreach (var kv in _aggregator) sorted.Add((kv.Key, kv.Value.ticks, kv.Value.calls));
        sorted.Sort((a, b) => b.ticks.CompareTo(a.ticks));
        double freqMs = 1000.0 / Stopwatch.Frequency;
        foreach (var (op, ticks, calls) in sorted)
        {
            double ms = ticks * freqMs;
            double pct = totalTicks > 0 ? 100.0 * ticks / totalTicks : 0.0;
            writer($"{op},{calls},{ticks},{ms:F3},{pct:F2}");
        }
        _aggregator.Clear();
    }
}

/// <summary>
/// Pre-compiled delegate chain for backward execution. Captures the exact
/// sequence of backward function calls from the first backward pass and
/// replays them on subsequent passes — no tape walking, no graph traversal,
/// no dictionary lookups. Just straight-line delegate invocations.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class CompiledDelegateChain<T>
{
    private readonly BackwardStep<T>[] _steps;
    // Logical step count. The backing array may be larger when rented
    // from BackwardScratch<T> — only indices [0, _count) hold live data.
    private readonly int _count;

    internal CompiledDelegateChain(BackwardStep<T>[] steps)
        : this(steps, steps.Length)
    {
    }

    internal CompiledDelegateChain(BackwardStep<T>[] steps, int count)
    {
        _steps = steps;
        _count = count;
    }

    /// <summary>
    /// Exposes the underlying step array so callers (the persistent-tape cleanup
    /// in <c>ComputeGradientsViaGraphCore</c>) can walk it to release per-step
    /// gradient retentions without re-doing the topological traversal that
    /// produced the chain.
    /// </summary>
    internal BackwardStep<T>[] Steps => _steps;

    /// <summary>
    /// Releases the saved-for-backward references held by every step in this
    /// chain — drops `Output`, `Inputs`, `SavedState`, and `Backward` to
    /// `null`/`default`, so the chain no longer pins forward-pass
    /// intermediates after disposal. AiDotNet#1340 / AiDotNet.Tensors#1340
    /// follow-up: under <c>GradientTapeOptions.Persistent = true</c> (the
    /// default), each <c>Train()</c> call cached its `CompiledDelegateChain`
    /// onto the tape. When the tape was disposed, `_cachedDelegateChain` was
    /// nulled — but if any reference path retained the chain instance (which
    /// can happen via finalizer ordering, async continuations, or pooled
    /// references in <see cref="BackwardScratch{T}"/>), the chain's
    /// `_steps[]` array still held 30+ tensor references per step group,
    /// preventing the runtime from collecting the forward intermediates.
    /// Measured: ~79 KB/call retention on a 164k-param Transformer (L=2,
    /// dModel=128) at 500 calls, projecting to 3.8 GB at 50k calls.
    /// Explicitly clearing the step references at tape Dispose drops the
    /// retention rate to near zero. Each cleared step becomes
    /// `default(BackwardStep&lt;T&gt;)` so a subsequent `Execute()` call
    /// would no-op safely — but persistent tape reuse requires the chain
    /// for replay, so this MUST be called only on the path that's about
    /// to discard the tape (i.e. <c>GradientTape&lt;T&gt;.Dispose()</c>).
    /// </summary>
    internal void Clear()
    {
        for (int i = 0; i < _count; i++)
        {
            // `default(BackwardStep<T>)` zeroes all four fields:
            // Output, Inputs, Backward, SavedState.
            _steps[i] = default;
        }
    }

    /// <summary>
    /// Executes the pre-compiled backward chain. Each step is a captured closure
    /// that calls the backward function with the right inputs and accumulates gradients.
    /// </summary>
    /// <param name="useScratch">
    /// True when the caller has already acquired
    /// <see cref="BackwardScratch{T}"/> and grants permission to rent
    /// pooled grads/seed buffers. False forces fresh allocation
    /// (used by nested backward calls and standalone callers).
    /// </param>
    internal Dictionary<Tensor<T>, Tensor<T>> Execute(
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>>? sources,
        IEngine engine,
        bool useScratch = false)
    {
        Dictionary<Tensor<T>, Tensor<T>> grads = useScratch
            ? BackwardScratch<T>.RentGrads(_count + 1)
            : new Dictionary<Tensor<T>, Tensor<T>>(
                _count + 1,
                ReferenceEqualityComparer<Tensor<T>>.Instance);

        // Seed gradient — cached per-thread by length so the ones-tensor
        // allocation doesn't repeat for fixed-shape losses (the typical
        // training-loop pattern).
        Tensor<T> seedGrad;
        if (loss.Length == 1)
        {
            // Tiny: just allocate. Caching a 1-element tensor doesn't pay.
            var numOps = Helpers.MathHelper.GetNumericOperations<T>();
            seedGrad = new Tensor<T>(new[] { numOps.One }, new[] { 1 });
        }
        else if (useScratch)
        {
            seedGrad = BackwardScratch<T>.RentSeedGradient(loss._shape);
        }
        else
        {
            var numOps = Helpers.MathHelper.GetNumericOperations<T>();
            var onesData = new T[loss.Length];
            var one = numOps.One;
            for (int j = 0; j < onesData.Length; j++) onesData[j] = one;
            seedGrad = new Tensor<T>(onesData, loss._shape);
        }
        grads[loss] = seedGrad;

        // Replay chain — straight-line delegate calls, no traversal.
        // Iterate only over the live range [0, _count).
        bool timing = BackwardTiming.Enabled;

        // OOM fix: free each forward activation's GPU buffer on its LAST use — right after its step's backward; in
        // reverse-topological order a step's output is fully consumed once its step runs (every consumer was an
        // earlier step). Previously all forward activations stayed GPU-pinned for the whole backward, so the working
        // set pegged at the card memory limit → OOM at scale (a tiny d256/L2 model OOM'd a 12 GB card past ~200K
        // tokens). Skip the loss + sources (leaf params). Gated to the GPU engine; materialize-then-free is safe.
        var gpuEngine = engine as AiDotNet.Tensors.Engines.DirectGpuTensorEngine;
        HashSet<Tensor<T>>? freeSkip = null;
        if (gpuEngine is not null)
        {
            freeSkip = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance) { loss };
            if (sources is not null) foreach (var s in sources) freeSkip.Add(s);
        }

        for (int i = 0; i < _count; i++)
        {
            ref var step = ref _steps[i];
            if (!grads.TryGetValue(step.Output, out var gradOutput))
                continue;

            long start = timing ? Stopwatch.GetTimestamp() : 0;
            step.Backward(gradOutput, step.Inputs, step.Output,
                step.SavedState ?? Array.Empty<object>(), engine, grads);
            if (timing)
            {
                long ticks = Stopwatch.GetTimestamp() - start;
                BackwardTiming.Record(step.Backward.Method.Name, ticks);
            }

            if (gpuEngine is not null && !freeSkip!.Contains(step.Output))
                gpuEngine.InvalidateGpuCacheForTensor(step.Output);
        }

        if (sources is not null)
        {
            var filtered = new Dictionary<Tensor<T>, Tensor<T>>(
                sources.Count, ReferenceEqualityComparer<Tensor<T>>.Instance);
            foreach (var source in sources)
                if (grads.TryGetValue(source, out var grad))
                    filtered[source] = grad;
            return filtered;
        }

        if (useScratch)
        {
            // Caller wants the full grads dict — return a copy so the
            // pooled dictionary can be reused on the next backward
            // without surprising the caller with mutating state.
            return new Dictionary<Tensor<T>, Tensor<T>>(
                grads, ReferenceEqualityComparer<Tensor<T>>.Instance);
        }

        return grads;
    }
}

/// <summary>A single step in the compiled backward chain.</summary>
internal struct BackwardStep<T>
{
    public Tensor<T> Output;
    public Tensor<T>[] Inputs;
    public BackwardFunction<T> Backward;
    public object[]? SavedState;
}
