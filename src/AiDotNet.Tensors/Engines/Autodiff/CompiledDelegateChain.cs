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
        if (!_enabled || _aggregator is null) return;
        Console.WriteLine();
        Console.WriteLine($"{"Backward op",-40}{"calls",10}{"total ms",14}{"avg µs",14}");
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
            Console.WriteLine($"{op,-40}{calls,10}{ms,14:F3}{us,14:F1}");
        }
        Console.WriteLine($"{"TOTAL",-40}{totalCalls,10}{totalTicks * 1000.0 / Stopwatch.Frequency,14:F3}");
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
