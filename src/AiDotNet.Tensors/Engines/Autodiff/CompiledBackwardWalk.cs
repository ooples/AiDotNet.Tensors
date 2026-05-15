// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Issue #338 Item 3 — compiled-IL backward.
/// <para>
/// Replaces <see cref="RebindablePlanCache{T}.TryExecute"/>'s dispatch loop
/// with a specialised walker per cached forward pattern. The eventual
/// implementation emits a <see cref="System.Reflection.Emit.DynamicMethod"/>
/// that bakes in the index sequence + per-entry backward delegate calls,
/// eliminating the loop counter, bounds-check, and per-iteration entry
/// lookup overhead. Equivalent to PyTorch <c>torch.compile</c>'s backward
/// codegen pass.
/// </para>
/// <para>
/// Status — this commit lands the integration scaffolding:
/// <list type="bullet">
///   <item>Static API surface (<see cref="TryGetWalker"/> / <see cref="Register"/>) keyed by pattern hash.</item>
///   <item>Cached walker shape (<see cref="CompiledWalker{T}"/>) that future commits replace with a JIT-emitted DynamicMethod delegate.</item>
///   <item>Today the walker is a passthrough — it indirects to the same per-entry dispatch loop <see cref="RebindablePlanCache{T}"/> uses. The structural extraction here is the prerequisite for the IL specialisation in subsequent commits.</item>
/// </list>
/// </para>
/// <para>
/// Enable via <c>AIDOTNET_COMPILED_BACKWARD=1</c>. Off by default while
/// the IL emission is incomplete; flipping the env var routes
/// <see cref="GradientTape{T}.ComputeGradientsViaGraphCore"/> through this
/// path instead of the inline replay loop.
/// </para>
/// </summary>
internal static class CompiledBackwardWalk<T>
{
    /// <summary>
    /// Feature flag: <c>AIDOTNET_COMPILED_BACKWARD</c> set to any non-empty
    /// non-zero value enables the compiled-walk path. Read once at first
    /// access — flipping the env var mid-process doesn't toggle the flag.
    /// </summary>
    private static readonly Lazy<bool> s_enabled = new(() =>
    {
        var raw = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_BACKWARD");
        if (string.IsNullOrEmpty(raw)) return false;
        if (raw == "0" || string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase))
            return false;
        return true;
    });

    /// <summary>True when the compiled-walk path is enabled for this process.</summary>
    internal static bool Enabled => s_enabled.Value;

    /// <summary>
    /// Walker delegate shape. Takes the same arguments as the inline
    /// <see cref="RebindablePlanCache{T}.TryExecute"/> replay loop. Returns
    /// the gradient dictionary (caller filters by sources). Returning
    /// <c>null</c> signals a stale cache — fall back to the fresh-walk
    /// dispatcher.
    /// </summary>
    internal delegate Dictionary<Tensor<T>, Tensor<T>>? CompiledWalker(
        TapeEntryArena<T> entries,
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>>? sources,
        IEngine engine);

    /// <summary>
    /// Per-thread cache of pattern-hash → compiled walker. Mirrors
    /// <see cref="RebindablePlanCache{T}"/>'s thread-local structure so the
    /// two caches stay in lock-step.
    /// </summary>
    [ThreadStatic]
    private static Dictionary<long, CompiledWalker>? s_walkers;

    /// <summary>
    /// Looks up the walker for a given pattern hash. Returns <c>null</c>
    /// when no walker has been compiled for this pattern yet.
    /// </summary>
    internal static CompiledWalker? TryGetWalker(long patternHash)
    {
        var walkers = s_walkers;
        if (walkers is null) return null;
        return walkers.TryGetValue(patternHash, out var walker) ? walker : null;
    }

    /// <summary>
    /// Registers a compiled walker for a given pattern hash. Called by
    /// the JIT layer the first time a fresh-tape replay's cache key fires;
    /// subsequent fires of the same pattern hit the cache.
    /// </summary>
    internal static void Register(long patternHash, CompiledWalker walker)
    {
        if (walker is null) throw new ArgumentNullException(nameof(walker));
        (s_walkers ??= new Dictionary<long, CompiledWalker>(8))[patternHash] = walker;
    }

    /// <summary>
    /// Clears the thread-local walker cache. Used by test isolation; the
    /// production path never invalidates walkers — once a pattern is
    /// compiled it's reused for the thread's lifetime.
    /// </summary>
    internal static void ResetForTests()
    {
        s_walkers?.Clear();
        s_walkers = null;
    }

    /// <summary>
    /// Builds a compiled walker for the given plan. The current implementation
    /// is a passthrough — it forwards to the same per-entry dispatch loop
    /// <see cref="RebindablePlanCache{T}.TryExecute"/> uses. Subsequent
    /// commits replace this body with a <see cref="System.Reflection.Emit.DynamicMethod"/>
    /// that bakes in the index sequence + per-entry backward delegate
    /// signatures, eliminating per-iteration loop / bounds-check overhead.
    /// </summary>
    /// <param name="reverseTopoIndices">Reverse-topo entry-index sequence
    /// from <see cref="RebindablePlanCache{T}"/>. Captured by reference;
    /// caller must not mutate after passing in.</param>
    internal static CompiledWalker Compile(int[] reverseTopoIndices)
    {
        if (reverseTopoIndices is null) throw new ArgumentNullException(nameof(reverseTopoIndices));

        // Passthrough specialisation. Captures the indices array directly
        // into the closure; future IL-emitted version replaces this with a
        // DynamicMethod whose IL hard-codes the entire index walk.
        var indices = reverseTopoIndices;

        return (entries, loss, sources, engine) =>
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            var grads = new Dictionary<Tensor<T>, Tensor<T>>(
                indices.Length + 1,
                ReferenceEqualityComparer<Tensor<T>>.Instance);

            // Seed gradient — identical to RebindablePlanCache's seeding.
            Tensor<T> seedGrad;
            if (loss.Length == 1)
            {
                seedGrad = new Tensor<T>(new[] { numOps.One }, (int[])loss._shape.Clone());
            }
            else
            {
                var onesData = new T[loss.Length];
                var one = numOps.One;
                for (int j = 0; j < onesData.Length; j++) onesData[j] = one;
                seedGrad = new Tensor<T>(onesData, loss._shape);
            }
            grads[loss] = seedGrad;

            var inputsBuffer1 = new Tensor<T>[1];
            var inputsBuffer2 = new Tensor<T>[2];
            var inputsBuffer3 = new Tensor<T>[3];
            try
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    int idx = indices[i];
                    if (idx < 0 || idx >= entries.Count) return null;
                    ref var entry = ref entries[idx];

                    if (!grads.TryGetValue(entry.Output, out var gradOutput))
                        continue;

                    entry.Backward(
                        gradOutput,
                        entry.GetInputsArrayInto(inputsBuffer1, inputsBuffer2, inputsBuffer3),
                        entry.Output,
                        entry.SavedState ?? Array.Empty<object>(),
                        engine,
                        grads);
                }
            }
            finally
            {
                inputsBuffer1[0] = null!;
                inputsBuffer2[0] = null!; inputsBuffer2[1] = null!;
                inputsBuffer3[0] = null!; inputsBuffer3[1] = null!; inputsBuffer3[2] = null!;
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

            return grads;
        };
    }
}
