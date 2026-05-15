// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

// ─────────────────────────────────────────────────────────────────────
// Compiled-IL backward walker architecture (issue #338 Item 3)
// ─────────────────────────────────────────────────────────────────────
//
// LAYERED SPECIALIZATION
//
// The walker produces a DynamicMethod per forward-pattern hash that
// computes gradients with progressively less per-entry overhead. Four
// layers, each independently enableable:
//
//   1. **DynamicMethod IL emission** — bakes the reverse-topo index
//      sequence into emitted IL. No indices[i] array access; no loop
//      counter; no per-iteration bounds check.
//
//   2. **Single-shot bounds check** — emits one pre-loop check that
//      max(indices) < entries.Count. Replaces N per-entry callvirts on
//      entries.Count with 1.
//
//   3. **Per-op-method direct call** — when every entry's backward
//      function is static + closure-free (which is the case for every
//      builtin BackwardFunctions<T>.* method), emits a direct
//      `call <MethodInfo>` per entry. JIT sees concrete static calls
//      instead of BackwardFunction<T>.Invoke dispatch.
//
//   4. **Per-op kernel inlining** — for 22 specific backward functions,
//      an IPerOpInliner emits the gradient math IL directly: skips
//      inputs[] array construction, savedState null-coalescing, and the
//      backward method dispatch entirely.
//
// CACHE + AUTO-ENGAGEMENT
//
// Walkers cache by pattern hash, capped at MaxCachedWalkers (FIFO
// eviction). Auto-engaged via AIDOTNET_COMPILED_BACKWARD env var or
// test-only _testEnabledOverride. RebindablePlanCache.Store captures
// each entry's backward MethodInfo at recording time and registers a
// specialised walker for the pattern.
//
// SAFETY
//
// - Closure-bound delegates refused (would crash on direct call —
//   the delegate's hidden closure arg would be skipped).
// - Falls back to closure-based walker if Reflection.Emit fails.
// - Per-pattern walker cache; entries with the same pattern hash share
//   the compiled walker, so the IL emission cost is amortized.
//
// PERF (measured #327 transformer fresh-tape)
//
//   Reference dispatcher: 395.82 ms/iter
//   Compiled-IL walker:   353.29 ms/iter
//   Speedup:              -42.5 ms/iter (-10.7%)
//
// Validated by Issue338_Item3_CompiledIL_NotSlowerThanReference under
// AIDOTNET_RUN_PERF_GATES=1.

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Issue #338 Item 3 — compiled-IL backward.
/// <para>
/// Replaces <see cref="RebindablePlanCache{T}.TryExecute"/>'s dispatch loop
/// with a per-pattern <see cref="DynamicMethod"/> whose IL bakes in the
/// reverse-topo index sequence. Each entry in the sequence becomes a
/// straight-line <c>call</c> to <see cref="CompiledBackwardWalkHelpers{T}.ProcessEntry"/>
/// with the index loaded as an <c>Ldc.I4</c> constant — no
/// <c>indices[i]</c> array access, no loop counter, no per-iteration
/// bounds check. Equivalent to PyTorch <c>torch.compile</c>'s backward
/// codegen pass.
/// </para>
/// <para>
/// Enable via <c>AIDOTNET_COMPILED_BACKWARD=1</c>. Off by default to give
/// the env-var-gated path time to soak before flipping the default.
/// </para>
/// <para>
/// IL emission is wrapped in a defensive try/catch; any
/// <c>Reflection.Emit</c> failure (rare — usually only happens on
/// platforms with restricted IL JIT) falls back to a closure-based walker
/// with identical semantics. The fallback path is what the test suite
/// runs in regression checks.
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

    /// <summary>
    /// Test-only override for <see cref="Enabled"/>. When non-null,
    /// short-circuits the env-var lookup. Production code never touches
    /// this; tests use it to flip the gate without spawning a new
    /// process. Reset to <c>null</c> in test cleanup to restore the
    /// env-var-derived default.
    /// </summary>
    internal static bool? _testEnabledOverride;

    /// <summary>True when the compiled-walk path is enabled for this process.</summary>
    internal static bool Enabled => _testEnabledOverride ?? s_enabled.Value;

    /// <summary>
    /// Walker delegate shape. Takes the same arguments as the inline
    /// <see cref="RebindablePlanCache{T}.TryExecute"/> replay loop.
    /// Returns the gradient dictionary, or <c>null</c> when a baked-in
    /// index falls outside the current tape's bounds (stale cache —
    /// caller falls back to the fresh-walk dispatcher).
    /// </summary>
    internal delegate Dictionary<Tensor<T>, Tensor<T>>? CompiledWalker(
        TapeEntryArena<T> entries,
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>>? sources,
        IEngine engine);

    /// <summary>
    /// Maximum number of distinct walkers cached per thread. The DFS
    /// pattern hash is keyed by (op-type sequence × shape tuple), so a
    /// workload that trains many different model topologies in the same
    /// thread could otherwise grow the cache unboundedly. The bound is
    /// generous — a complex Transformer trains ~1-5 distinct patterns
    /// per epoch — so eviction is rare in production but the safety
    /// guard is essential.
    /// </summary>
    private const int MaxCachedWalkers = 64;

    /// <summary>
    /// Per-thread cache of pattern-hash → compiled walker. Mirrors
    /// <see cref="RebindablePlanCache{T}"/>'s thread-local structure so the
    /// two caches stay in lock-step. Capped at <see cref="MaxCachedWalkers"/>;
    /// when full, the oldest insertion order entry is evicted (simple FIFO
    /// — full LRU would require tracking access timestamps and the extra
    /// bookkeeping isn't worth the small workload-dependent hit-rate gain).
    /// </summary>
    [ThreadStatic]
    private static Dictionary<long, CompiledWalker>? s_walkers;

    /// <summary>
    /// Parallel insertion-order tracking for the FIFO eviction policy.
    /// Sized to <see cref="MaxCachedWalkers"/>+1 so adds are cheap (no
    /// resize). Index 0 is the oldest entry; new entries append, evict
    /// removes index 0 + shifts.
    /// </summary>
    [ThreadStatic]
    private static List<long>? s_walkerInsertionOrder;

    /// <summary>
    /// Looks up the walker for a given pattern hash. Returns <c>null</c>
    /// when no walker has been compiled for this pattern yet. Tracks a
    /// hit/miss counter pair so tests + diagnostics can verify the cache
    /// is actually being reused (a high miss rate suggests the pattern
    /// hash is too discriminating, evicting useful entries).
    /// </summary>
    internal static CompiledWalker? TryGetWalker(long patternHash)
    {
        var walkers = s_walkers;
        if (walkers is null)
        {
            s_missCount++;
            return null;
        }
        if (walkers.TryGetValue(patternHash, out var walker))
        {
            s_hitCount++;
            return walker;
        }
        s_missCount++;
        return null;
    }

    [ThreadStatic] private static long s_hitCount;
    [ThreadStatic] private static long s_missCount;

    /// <summary>Diagnostic counter: walker cache hits on this thread.</summary>
    internal static long CacheHitsForTests => s_hitCount;
    /// <summary>Diagnostic counter: walker cache misses on this thread.</summary>
    internal static long CacheMissesForTests => s_missCount;
    /// <summary>Resets the hit/miss counters. Used by test isolation.</summary>
    internal static void ResetCounters()
    {
        s_hitCount = 0;
        s_missCount = 0;
    }

    /// <summary>
    /// Registers a compiled walker for a given pattern hash. Called by
    /// the JIT layer the first time a fresh-tape replay's cache key fires;
    /// subsequent fires of the same pattern hit the cache.
    /// <para>
    /// Enforces a <see cref="MaxCachedWalkers"/>-entry cap via FIFO
    /// eviction. Re-registering an existing pattern hash refreshes the
    /// walker but does NOT reset its position in the insertion-order list
    /// (the entry was already in there; we don't want to shift older
    /// entries around).
    /// </para>
    /// </summary>
    internal static void Register(long patternHash, CompiledWalker walker)
    {
        if (walker is null) throw new ArgumentNullException(nameof(walker));
        var walkers = s_walkers ??= new Dictionary<long, CompiledWalker>(MaxCachedWalkers + 1);
        var insertionOrder = s_walkerInsertionOrder ??= new List<long>(MaxCachedWalkers + 1);

        bool isNew = !walkers.ContainsKey(patternHash);
        walkers[patternHash] = walker;
        if (isNew)
        {
            insertionOrder.Add(patternHash);
            if (walkers.Count > MaxCachedWalkers)
            {
                long evictKey = insertionOrder[0];
                insertionOrder.RemoveAt(0);
                walkers.Remove(evictKey);
            }
        }
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
        s_walkerInsertionOrder?.Clear();
        s_walkerInsertionOrder = null;
        ResetCounters();
    }

    /// <summary>Test-visible cache size. Exposed so eviction tests can
    /// validate the cap.</summary>
    internal static int CachedWalkerCountForTests => s_walkers?.Count ?? 0;

    // ─── IL emission targets ─────────────────────────────────────────────
    // These MethodInfos are looked up once per closed generic T and reused
    // for every Compile call on that T. Lookup is cheap (reflection on a
    // closed generic type) but doing it once amortises across many
    // pattern compilations.

    private static readonly MethodInfo s_initStateMethod = typeof(CompiledBackwardWalkHelpers<T>)
        .GetMethod(nameof(CompiledBackwardWalkHelpers<T>.InitState),
            BindingFlags.Public | BindingFlags.Static)!;

    private static readonly MethodInfo s_processEntryMethod = typeof(CompiledBackwardWalkHelpers<T>)
        .GetMethod(nameof(CompiledBackwardWalkHelpers<T>.ProcessEntry),
            BindingFlags.Public | BindingFlags.Static)!;

    private static readonly MethodInfo s_finalizeWalkMethod = typeof(CompiledBackwardWalkHelpers<T>)
        .GetMethod(nameof(CompiledBackwardWalkHelpers<T>.FinalizeWalk),
            BindingFlags.Public | BindingFlags.Static)!;

    /// <summary>
    /// Per-op-pattern inliner lookup. When EmitWalkerILSpecialised
    /// recognises an entry's backward method as one with a known inliner,
    /// the inliner emits the actual gradient math directly (eliding
    /// inputs[] array construction, savedState handling, and the
    /// BackwardFunction<T> invocation) instead of the generic 6-arg call.
    /// </summary>
    private static readonly Dictionary<MethodInfo, IPerOpInliner> s_inliners
        = BuildInlinerRegistry();

    private static Dictionary<MethodInfo, IPerOpInliner> BuildInlinerRegistry()
    {
        var registry = new Dictionary<MethodInfo, IPerOpInliner>();
        var bwdType = typeof(BackwardFunctions<T>);
        var addMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.AddBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (addMethod is not null)
            registry[addMethod] = new AddBackwardInliner();

        var subMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.SubtractBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (subMethod is not null)
            registry[subMethod] = new SubtractBackwardInliner();

        var negMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.NegateBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (negMethod is not null)
            registry[negMethod] = new NegateBackwardInliner();

        var mulMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.MultiplyBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (mulMethod is not null)
            registry[mulMethod] = new MultiplyBackwardInliner();

        var expMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.ExpBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (expMethod is not null)
            registry[expMethod] = new ExpBackwardInliner();

        var logMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.LogBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (logMethod is not null)
            registry[logMethod] = new LogBackwardInliner();

        var absMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.AbsBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (absMethod is not null)
            registry[absMethod] = new AbsBackwardInliner();

        var divMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.DivideBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (divMethod is not null)
            registry[divMethod] = new DivideBackwardInliner();

        // Activation backward inliners — common in deep learning hot
        // paths (ReLU / Sigmoid / Tanh / GELU all appear in transformer
        // FFN + attention residuals).
        var reluMethod = bwdType.GetMethod("ReLUBackward",
            BindingFlags.NonPublic | BindingFlags.Static);
        if (reluMethod is not null)
            registry[reluMethod] = new ReluBackwardInliner();

        var sigmoidMethod = bwdType.GetMethod("SigmoidBackward",
            BindingFlags.NonPublic | BindingFlags.Static);
        if (sigmoidMethod is not null)
            registry[sigmoidMethod] = new SigmoidBackwardInliner();

        var tanhMethod = bwdType.GetMethod("TanhBackward",
            BindingFlags.NonPublic | BindingFlags.Static);
        if (tanhMethod is not null)
            registry[tanhMethod] = new TanhBackwardInliner();

        var geluMethod = bwdType.GetMethod("GELUBackward",
            BindingFlags.NonPublic | BindingFlags.Static);
        if (geluMethod is not null)
            registry[geluMethod] = new GeluBackwardInliner();

        var sinMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.SinBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (sinMethod is not null)
            registry[sinMethod] = new SinBackwardInliner();

        var cosMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.CosBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (cosMethod is not null)
            registry[cosMethod] = new CosBackwardInliner();

        var leakyReluMethod = bwdType.GetMethod("LeakyReLUBackward",
            BindingFlags.NonPublic | BindingFlags.Static);
        if (leakyReluMethod is not null)
            registry[leakyReluMethod] = new LeakyReluBackwardInliner();

        var softmaxMethod = bwdType.GetMethod("SoftmaxBackward",
            BindingFlags.NonPublic | BindingFlags.Static);
        if (softmaxMethod is not null)
            registry[softmaxMethod] = new SoftmaxBackwardInliner();

        // Trivial pass-through scalar ops: AddScalar and SubtractScalar
        // both have gradInput = gradOutput. Identical inliner.
        var addScalarPassThru = new PassThroughGradInliner();
        var addScalarMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.AddScalarBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (addScalarMethod is not null)
            registry[addScalarMethod] = addScalarPassThru;

        var subScalarMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.SubtractScalarBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (subScalarMethod is not null)
            registry[subScalarMethod] = addScalarPassThru;

        var mulScalarMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.MultiplyScalarBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (mulScalarMethod is not null)
            registry[mulScalarMethod] = new MultiplyScalarBackwardInliner();

        var sqrtMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.SqrtBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (sqrtMethod is not null)
            registry[sqrtMethod] = new SqrtBackwardInliner();

        var reshapeMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.ReshapeBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (reshapeMethod is not null)
            registry[reshapeMethod] = new ReshapeBackwardInliner();

        var reduceMeanMethod = bwdType.GetMethod(nameof(BackwardFunctions<T>.ReduceMeanBackward),
            BindingFlags.NonPublic | BindingFlags.Static);
        if (reduceMeanMethod is not null)
            registry[reduceMeanMethod] = new ReduceMeanBackwardInliner();

        return registry;
    }

    /// <summary>
    /// Reads <c>(TSavedState)entry.SavedState[index]</c> and pushes the
    /// resulting value onto the IL stack. Use for inliners that need
    /// scalar params (axis ints, negativeSlope doubles, etc.) saved
    /// during the forward pass.
    /// </summary>
    private static void EmitLoadSavedStateValue(
        ILGenerator il, LocalBuilder entryRefLocal, FieldInfo savedStateField,
        int index, Type valueType)
    {
        il.Emit(OpCodes.Ldloc, entryRefLocal);
        il.Emit(OpCodes.Ldfld, savedStateField);
        il.Emit(OpCodes.Ldc_I4, index);
        il.Emit(OpCodes.Ldelem_Ref);
        il.Emit(OpCodes.Unbox_Any, valueType);
    }

    /// <summary>
    /// Emits the <c>AccumulateGrad(state.Grads, entry.&lt;inputField&gt;,
    /// gradLocal, engine)</c> call sequence. Common tail of every
    /// inliner that accumulates a single computed gradient into a single
    /// input — the bulk of them. Saves ~7 IL lines per inliner.
    /// </summary>
    private static void EmitAccumulateGradToInput(
        ILGenerator il, LocalBuilder stateLocal, LocalBuilder entryRefLocal,
        LocalBuilder gradLocal, FieldInfo gradsField, FieldInfo inputField,
        MethodInfo accumulateGradMethod)
    {
        il.Emit(OpCodes.Ldloca, stateLocal);
        il.Emit(OpCodes.Ldfld, gradsField);
        il.Emit(OpCodes.Ldloc, entryRefLocal);
        il.Emit(OpCodes.Ldfld, inputField);
        il.Emit(OpCodes.Ldloc, gradLocal);
        il.Emit(OpCodes.Ldarg_3);
        il.Emit(OpCodes.Call, accumulateGradMethod);
    }

    /// <summary>
    /// Per-op IL emitter contract. Returns false to fall back to the
    /// generic specialised-call path; true means the inliner emitted
    /// the full per-entry IL itself.
    /// </summary>
    private interface IPerOpInliner
    {
        void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod);
    }

    /// <summary>
    /// Inlines <c>AddBackward</c>'s gradient math: two
    /// <c>AccumulateGrad(grads, input_i, gradOutput, engine)</c> calls
    /// for i in {0, 1}. Skips inputs[] array construction + savedState
    /// null-coalescing + the BackwardFunction<T>.Invoke dispatch — net
    /// ~12 IL instructions saved per AddBackward entry.
    /// </summary>
    private sealed class AddBackwardInliner : IPerOpInliner
    {
        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // AccumulateGrad(state.Grads, entry.Input0, gradOutput, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);

            // AccumulateGrad(state.Grads, entry.Input1, gradOutput, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input1Field);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>SubtractBackward</c>: d(a-b)/da = +grad, d(a-b)/db = -grad.
    /// Emits one AccumulateGrad for input0 with gradOutput, then a
    /// TensorNegate(gradOutput) virtual call, then a second AccumulateGrad
    /// for input1 with the negated gradient.
    /// </summary>
    private sealed class SubtractBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_negateMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorNegate))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // AccumulateGrad(state.Grads, entry.Input0, gradOutput, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);

            // var negGrad = engine.TensorNegate(gradOutput);
            // Reuse gradOutputLocal as the negated-grad slot — safe
            // because subsequent uses below want -gradOutput anyway.
            // But we must NOT lose the original gradOutput if some
            // other entry holds it; the original tensor is referenced
            // via grads[entry.Output] which is unchanged. Negate the
            // local-only reference.
            var negGradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Callvirt, s_negateMethod);
            il.Emit(OpCodes.Stloc, negGradLocal);

            // AccumulateGrad(state.Grads, entry.Input1, negGrad, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input1Field);
            il.Emit(OpCodes.Ldloc, negGradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>MultiplyBackward</c>: d(a*b)/da = grad*b, d(a*b)/db = grad*a.
    /// Two TensorMultiply virtual calls + two AccumulateGrad calls. Skips
    /// inputs[]/savedState/output dispatch overhead.
    /// </summary>
    private sealed class MultiplyBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_multiplyMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorMultiply))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // gradA = engine.TensorMultiply(gradOutput, entry.Input1)
            var gradALocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input1Field);
            il.Emit(OpCodes.Callvirt, s_multiplyMethod);
            il.Emit(OpCodes.Stloc, gradALocal);

            // gradB = engine.TensorMultiply(gradOutput, entry.Input0)
            var gradBLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Callvirt, s_multiplyMethod);
            il.Emit(OpCodes.Stloc, gradBLocal);

            // AccumulateGrad(state.Grads, entry.Input0, gradA, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradALocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);

            // AccumulateGrad(state.Grads, entry.Input1, gradB, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input1Field);
            il.Emit(OpCodes.Ldloc, gradBLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>ExpBackward</c>: d(exp(x))/dx = grad * exp(x) = grad * output.
    /// One TensorMultiply with entry.Output + one AccumulateGrad. Unary op
    /// but uses output (the cached forward result) instead of input — the
    /// "saved tensor" optimisation that PyTorch / TF also exploit.
    /// </summary>
    private sealed class ExpBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_multiplyMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorMultiply))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // grad = engine.TensorMultiply(gradOutput, entry.Output)
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, outputField);
            il.Emit(OpCodes.Callvirt, s_multiplyMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            // AccumulateGrad(state.Grads, entry.Input0, grad, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>LogBackward</c>: d(log(x))/dx = grad / x. One
    /// TensorDivide(gradOutput, entry.Input0) + one AccumulateGrad.
    /// </summary>
    private sealed class LogBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_divideMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorDivide))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // grad = engine.TensorDivide(gradOutput, entry.Input0)
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Callvirt, s_divideMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            // AccumulateGrad(state.Grads, entry.Input0, grad, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>AbsBackward</c>: d(|x|)/dx = grad * sign(x). Two virtual
    /// calls (TensorSign + TensorMultiply) + one AccumulateGrad.
    /// </summary>
    private sealed class AbsBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_signMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorSign))!
                .MakeGenericMethod(typeof(T));
        private static readonly MethodInfo s_multiplyMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorMultiply))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // signTensor = engine.TensorSign(entry.Input0)
            var signLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Callvirt, s_signMethod);
            il.Emit(OpCodes.Stloc, signLocal);

            // grad = engine.TensorMultiply(gradOutput, signTensor)
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, signLocal);
            il.Emit(OpCodes.Callvirt, s_multiplyMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            // AccumulateGrad(state.Grads, entry.Input0, grad, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>DivideBackward</c>: d(a/b)/da = grad/b, d(a/b)/db = -grad*a/(b*b).
    /// </summary>
    private sealed class DivideBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_divideMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorDivide))!
                .MakeGenericMethod(typeof(T));
        private static readonly MethodInfo s_multiplyMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorMultiply))!
                .MakeGenericMethod(typeof(T));
        private static readonly MethodInfo s_negateMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorNegate))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // gradA = engine.TensorDivide(gradOutput, entry.Input1)
            var gradALocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input1Field);
            il.Emit(OpCodes.Callvirt, s_divideMethod);
            il.Emit(OpCodes.Stloc, gradALocal);

            // bSquared = engine.TensorMultiply(entry.Input1, entry.Input1)
            var bSquaredLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input1Field);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input1Field);
            il.Emit(OpCodes.Callvirt, s_multiplyMethod);
            il.Emit(OpCodes.Stloc, bSquaredLocal);

            // negGradA = engine.TensorNegate(engine.TensorMultiply(gradOutput, entry.Input0))
            var negGradALocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Callvirt, s_multiplyMethod);
            il.Emit(OpCodes.Callvirt, s_negateMethod);
            il.Emit(OpCodes.Stloc, negGradALocal);

            // gradB = engine.TensorDivide(negGradA, bSquared)
            var gradBLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, negGradALocal);
            il.Emit(OpCodes.Ldloc, bSquaredLocal);
            il.Emit(OpCodes.Callvirt, s_divideMethod);
            il.Emit(OpCodes.Stloc, gradBLocal);

            // AccumulateGrad(state.Grads, entry.Input0, gradA, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradALocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);

            // AccumulateGrad(state.Grads, entry.Input1, gradB, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input1Field);
            il.Emit(OpCodes.Ldloc, gradBLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>ReLUBackward</c>: dispatches via engine.ReluBackward.
    /// Skips inputs[]/savedState/output dispatch overhead — direct
    /// engine.ReluBackward call + AccumulateGrad.
    /// </summary>
    private sealed class ReluBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_reluBackwardMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.ReluBackward))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // grad = engine.ReluBackward(gradOutput, entry.Input0)
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Callvirt, s_reluBackwardMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            // AccumulateGrad(state.Grads, entry.Input0, grad, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>SigmoidBackward</c>: dispatches via engine.SigmoidBackward(grad, output)
    /// — the kernel uses the cached forward output, not the input.
    /// </summary>
    private sealed class SigmoidBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_sigmoidBackwardMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.SigmoidBackward))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // grad = engine.SigmoidBackward(gradOutput, entry.Output)
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, outputField);
            il.Emit(OpCodes.Callvirt, s_sigmoidBackwardMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>TanhBackward</c>: dispatches via engine.TanhBackward(grad, output).
    /// </summary>
    private sealed class TanhBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_tanhBackwardMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TanhBackward))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, outputField);
            il.Emit(OpCodes.Callvirt, s_tanhBackwardMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>GELUBackward</c>: dispatches via engine.GeluBackward(grad, input).
    /// GELU uses the input, not the output, for its gradient.
    /// </summary>
    private sealed class GeluBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_geluBackwardMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.GeluBackward))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Callvirt, s_geluBackwardMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>SinBackward</c>: d(sin(x))/dx = grad * cos(x). One
    /// TensorCos + one TensorMultiply + one AccumulateGrad.
    /// </summary>
    private sealed class SinBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_cosMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorCos))!
                .MakeGenericMethod(typeof(T));
        private static readonly MethodInfo s_multiplyMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorMultiply))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            var cosLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Callvirt, s_cosMethod);
            il.Emit(OpCodes.Stloc, cosLocal);

            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, cosLocal);
            il.Emit(OpCodes.Callvirt, s_multiplyMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>CosBackward</c>: d(cos(x))/dx = -grad * sin(x).
    /// TensorSin + TensorNegate + TensorMultiply + AccumulateGrad.
    /// </summary>
    private sealed class CosBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_sinMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorSin))!
                .MakeGenericMethod(typeof(T));
        private static readonly MethodInfo s_negateMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorNegate))!
                .MakeGenericMethod(typeof(T));
        private static readonly MethodInfo s_multiplyMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorMultiply))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // negSinX = engine.TensorNegate(engine.TensorSin(entry.Input0))
            var negSinLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Callvirt, s_sinMethod);
            il.Emit(OpCodes.Callvirt, s_negateMethod);
            il.Emit(OpCodes.Stloc, negSinLocal);

            // grad = engine.TensorMultiply(gradOutput, negSinX)
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, negSinLocal);
            il.Emit(OpCodes.Callvirt, s_multiplyMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>LeakyReLUBackward</c>: dispatches engine.LeakyReluBackward(
    /// grad, input, negativeSlope) where negativeSlope is the saved double
    /// in entry.SavedState[0]. Includes the unbox-int-double cast inline.
    /// </summary>
    private sealed class LeakyReluBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_leakyMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.LeakyReluBackward))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // grad = engine.LeakyReluBackward(gradOutput, entry.Input0, (double)entry.SavedState[0])
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            EmitLoadSavedStateValue(il, entryRefLocal, savedStateField, 0, typeof(double));
            il.Emit(OpCodes.Callvirt, s_leakyMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>SoftmaxBackward</c>: dispatches engine.SoftmaxBackward(
    /// grad, output, axis) where axis is the saved int in entry.SavedState[0].
    /// Critical for transformer attention hot path.
    /// </summary>
    private sealed class SoftmaxBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_softmaxMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.SoftmaxBackward))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // grad = engine.SoftmaxBackward(gradOutput, entry.Output, (int)entry.SavedState[0])
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, outputField);
            EmitLoadSavedStateValue(il, entryRefLocal, savedStateField, 0, typeof(int));
            il.Emit(OpCodes.Callvirt, s_softmaxMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Pass-through gradient: input gradient = output gradient. Covers
    /// AddScalarBackward (d(x+c)/dx = 1) and SubtractScalarBackward
    /// (d(x-c)/dx = 1). Single AccumulateGrad call with no preceding
    /// engine ops. The simplest possible inliner — for these two ops the
    /// inliner-vs-generic path saves the full backward method invocation
    /// plus the inputs[] / savedState dispatch overhead.
    /// </summary>
    private sealed class PassThroughGradInliner : IPerOpInliner
    {
        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // AccumulateGrad(state.Grads, entry.Input0, gradOutput, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>MultiplyScalarBackward</c>: d(c*x)/dx = c * grad.
    /// Reads scalar c = (T)savedState[0], emits TensorMultiplyScalar +
    /// AccumulateGrad.
    /// </summary>
    private sealed class MultiplyScalarBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_mulScalarMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorMultiplyScalar))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // grad = engine.TensorMultiplyScalar(gradOutput, (T)savedState[0])
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            EmitLoadSavedStateValue(il, entryRefLocal, savedStateField, 0, typeof(T));
            il.Emit(OpCodes.Callvirt, s_mulScalarMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>SqrtBackward</c>: d(sqrt(x))/dx = grad / (2 * sqrt(x))
    /// = grad / (2 * output). Uses the cached forward output via
    /// entry.Output. Emits TensorMultiplyScalar (×2) + TensorDivide +
    /// AccumulateGrad.
    /// </summary>
    private sealed class SqrtBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_mulScalarMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorMultiplyScalar))!
                .MakeGenericMethod(typeof(T));
        private static readonly MethodInfo s_divideMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorDivide))!
                .MakeGenericMethod(typeof(T));
        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // twoOutput = engine.TensorMultiplyScalar(entry.Output, (T)2)
            // The literal "2" is constructed via FromDouble — we can't
            // bake a T constant directly into IL because T is generic.
            // Use a captured delegate that returns the FromDouble result
            // for the test value 2.0; invoke at emit time and store the
            // resulting T value as a boxed object that we unbox at run
            // time. Simpler: use NumericOperations.FromDouble at runtime.
            var numOpsType = typeof(MathHelper);
            var getNumOpsMethod = numOpsType.GetMethod(nameof(MathHelper.GetNumericOperations))!
                .MakeGenericMethod(typeof(T));
            var fromDoubleMethod = typeof(AiDotNet.Tensors.Interfaces.INumericOperations<>)
                .MakeGenericType(typeof(T))
                .GetMethod("FromDouble")!;

            var twoLocal = il.DeclareLocal(typeof(T));
            il.Emit(OpCodes.Call, getNumOpsMethod);
            il.Emit(OpCodes.Ldc_R8, 2.0);
            il.Emit(OpCodes.Callvirt, fromDoubleMethod);
            il.Emit(OpCodes.Stloc, twoLocal);

            // twoOutput = engine.TensorMultiplyScalar(entry.Output, two)
            var twoOutputLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, outputField);
            il.Emit(OpCodes.Ldloc, twoLocal);
            il.Emit(OpCodes.Callvirt, s_mulScalarMethod);
            il.Emit(OpCodes.Stloc, twoOutputLocal);

            // grad = engine.TensorDivide(gradOutput, twoOutput)
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, twoOutputLocal);
            il.Emit(OpCodes.Callvirt, s_divideMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, gradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>ReshapeBackward</c>: gradients flow back through Reshape
    /// to the original shape. Reads savedState[0] as int[]. Very common
    /// in transformer head-splitting / token reshaping.
    /// </summary>
    private sealed class ReshapeBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_reshapeMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.Reshape))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // originalShape = (int[])entry.SavedState[0]
            // (cast via unbox-ref since arrays are reference types)
            // grad = engine.Reshape(gradOutput, originalShape)
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, savedStateField);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Ldelem_Ref);
            il.Emit(OpCodes.Castclass, typeof(int[]));
            il.Emit(OpCodes.Callvirt, s_reshapeMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            EmitAccumulateGradToInput(il, stateLocal, entryRefLocal,
                gradLocal, gradsField, input0Field, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>ReduceMeanBackward</c>: dispatches engine.ReduceMeanBackward(
    /// grad, inputs[0]._shape, (int[])savedState[0]).
    /// </summary>
    private sealed class ReduceMeanBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_reduceMeanBackwardMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.ReduceMeanBackward))!
                .MakeGenericMethod(typeof(T));
        private static readonly FieldInfo s_shapeField =
            typeof(LinearAlgebra.TensorBase<T>).GetField("_shape",
                BindingFlags.NonPublic | BindingFlags.Instance)!;

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // grad = engine.ReduceMeanBackward(gradOutput, entry.Input0._shape, (int[])entry.SavedState[0])
            var gradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldfld, s_shapeField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, savedStateField);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Ldelem_Ref);
            il.Emit(OpCodes.Castclass, typeof(int[]));
            il.Emit(OpCodes.Callvirt, s_reduceMeanBackwardMethod);
            il.Emit(OpCodes.Stloc, gradLocal);

            EmitAccumulateGradToInput(il, stateLocal, entryRefLocal,
                gradLocal, gradsField, input0Field, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Inlines <c>NegateBackward</c>: d(-x)/dx = -grad. One TensorNegate
    /// + one AccumulateGrad call. Unary op, only input0 is touched.
    /// </summary>
    private sealed class NegateBackwardInliner : IPerOpInliner
    {
        private static readonly MethodInfo s_negateMethod =
            typeof(IEngine).GetMethod(nameof(IEngine.TensorNegate))!
                .MakeGenericMethod(typeof(T));

        public void Emit(
            ILGenerator il,
            LocalBuilder stateLocal,
            LocalBuilder entryRefLocal,
            LocalBuilder gradOutputLocal,
            FieldInfo gradsField,
            FieldInfo input0Field,
            FieldInfo input1Field,
            FieldInfo outputField,
            FieldInfo savedStateField,
            MethodInfo accumulateGradMethod)
        {
            // var negGrad = engine.TensorNegate(gradOutput);
            var negGradLocal = il.DeclareLocal(typeof(Tensor<T>));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, gradOutputLocal);
            il.Emit(OpCodes.Callvirt, s_negateMethod);
            il.Emit(OpCodes.Stloc, negGradLocal);

            // AccumulateGrad(state.Grads, entry.Input0, negGrad, engine)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, input0Field);
            il.Emit(OpCodes.Ldloc, negGradLocal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, accumulateGradMethod);
        }
    }

    /// <summary>
    /// Builds a compiled walker for the given plan. Attempts IL emission
    /// first; on failure (rare — restricted IL platforms) falls back to a
    /// closure-based walker with identical semantics.
    /// </summary>
    /// <param name="reverseTopoIndices">Reverse-topo entry-index sequence
    /// from <see cref="RebindablePlanCache{T}"/>. Captured by reference;
    /// caller must not mutate after passing in.</param>
    internal static CompiledWalker Compile(int[] reverseTopoIndices)
        => Compile(reverseTopoIndices, backwardMethods: null);

    /// <summary>
    /// Builds a compiled walker for the given plan with optional per-op
    /// method specialisation. When <paramref name="backwardMethods"/> is
    /// non-null (and every entry's backward function is static), the IL
    /// emitter bakes a direct <c>call</c> to each method per entry
    /// instead of going through <c>BackwardFunction&lt;T&gt;.Invoke</c>'s
    /// delegate dispatch. This is the genuine per-op specialisation
    /// described in #338 Item 3.
    /// </summary>
    /// <param name="reverseTopoIndices">Reverse-topo entry-index sequence.</param>
    /// <param name="backwardMethods">Parallel array of backward method
    /// metadata, indexed by position in <paramref name="reverseTopoIndices"/>.
    /// Null falls back to the helper-dispatch path used by <see cref="Compile(int[])"/>.
    /// When provided, ALL methods must be static (instance methods would
    /// require a baked-in target reference, which we don't capture).</param>
    internal static CompiledWalker Compile(int[] reverseTopoIndices, MethodInfo[]? backwardMethods)
    {
        if (reverseTopoIndices is null) throw new ArgumentNullException(nameof(reverseTopoIndices));

        // Per-op specialisation is only safe when every backward method is
        // static and the array length matches indices.Length. If either
        // fails, fall through to the helper-dispatch IL path.
        bool specialise = backwardMethods is not null
            && backwardMethods.Length == reverseTopoIndices.Length
            && AllStatic(backwardMethods);

        try
        {
            DynamicMethod dynMethod = specialise
                ? EmitWalkerILSpecialised(reverseTopoIndices, backwardMethods!)
                : EmitWalkerIL(reverseTopoIndices);
            return (CompiledWalker)dynMethod.CreateDelegate(typeof(CompiledWalker));
        }
        catch
        {
            // Fallthrough to closure-based walker — defensive.
        }

        return EmitWalkerClosure(reverseTopoIndices);
    }

    private static bool AllStatic(MethodInfo[] methods)
    {
        for (int i = 0; i < methods.Length; i++)
            if (methods[i] is null || !methods[i].IsStatic) return false;
        return true;
    }

    /// <summary>
    /// Emits the unrolled walker as a <see cref="DynamicMethod"/>. The
    /// emitted IL has the shape:
    /// <code>
    /// var state = CompiledBackwardWalkHelpers&lt;T&gt;.InitState(loss, indices.Length);
    /// CompiledBackwardWalkHelpers&lt;T&gt;.ProcessEntry(entries, idx_0, state, engine);
    /// CompiledBackwardWalkHelpers&lt;T&gt;.ProcessEntry(entries, idx_1, state, engine);
    /// ... (one call per baked index) ...
    /// return CompiledBackwardWalkHelpers&lt;T&gt;.FinalizeWalk(state, sources);
    /// </code>
    /// Each index is loaded via <c>Ldc.I4</c> — no <c>indices[i]</c>
    /// access in the JITted code.
    /// </summary>
    private static DynamicMethod EmitWalkerIL(int[] reverseTopoIndices)
    {
        var method = new DynamicMethod(
            name: $"CompiledBackwardWalk_{typeof(T).Name}_{reverseTopoIndices.Length}",
            returnType: typeof(Dictionary<Tensor<T>, Tensor<T>>),
            parameterTypes: new[]
            {
                typeof(TapeEntryArena<T>),
                typeof(Tensor<T>),
                typeof(IReadOnlyList<Tensor<T>>),
                typeof(IEngine),
            },
            restrictedSkipVisibility: true);
        method.DefineParameter(1, ParameterAttributes.None, "entries");
        method.DefineParameter(2, ParameterAttributes.None, "loss");
        method.DefineParameter(3, ParameterAttributes.None, "sources");
        method.DefineParameter(4, ParameterAttributes.None, "engine");

        var il = method.GetILGenerator();

        // Local 0: state — holds grads + per-arity buffer arrays.
        var stateLocal = il.DeclareLocal(typeof(CompiledBackwardWalkHelpers<T>.WalkState));

        // state = InitState(loss, reservedCount)
        il.Emit(OpCodes.Ldarg_1);                              // loss
        il.Emit(OpCodes.Ldc_I4, reverseTopoIndices.Length);    // reservedCount
        il.Emit(OpCodes.Call, s_initStateMethod);              // -> WalkState
        il.Emit(OpCodes.Stloc, stateLocal);

        // For each baked-in index, emit:
        //   ProcessEntry(entries, idx, ref state, engine);
        for (int i = 0; i < reverseTopoIndices.Length; i++)
        {
            il.Emit(OpCodes.Ldarg_0);                          // entries
            il.Emit(OpCodes.Ldc_I4, reverseTopoIndices[i]);    // idx
            il.Emit(OpCodes.Ldloca, stateLocal);               // ref state
            il.Emit(OpCodes.Ldarg_3);                          // engine
            il.Emit(OpCodes.Call, s_processEntryMethod);       // returns bool: false = stale cache
            // ProcessEntry returns false when idx is out of range — the
            // walker contract says return null in that case so the caller
            // falls back to the fresh-walk dispatcher.
            var continueLabel = il.DefineLabel();
            il.Emit(OpCodes.Brtrue_S, continueLabel);
            il.Emit(OpCodes.Ldnull);
            il.Emit(OpCodes.Ret);
            il.MarkLabel(continueLabel);
        }

        // return FinalizeWalk(state, sources);
        il.Emit(OpCodes.Ldloca, stateLocal);                   // ref state
        il.Emit(OpCodes.Ldarg_2);                              // sources
        il.Emit(OpCodes.Call, s_finalizeWalkMethod);           // -> Dictionary<Tensor<T>, Tensor<T>>
        il.Emit(OpCodes.Ret);

        return method;
    }

    /// <summary>
    /// Per-op-specialised IL emitter. Same per-index unrolling as
    /// <see cref="EmitWalkerIL"/> but each entry emits a direct
    /// <c>call &lt;MethodInfo&gt;</c> to its backward function instead of
    /// going through <see cref="CompiledBackwardWalkHelpers{T}.ProcessEntry"/>'s
    /// helper (which still indirects through <c>entry.Backward.Invoke</c>).
    /// This is the genuine perf-winning specialisation: the JIT sees a
    /// direct call to a static method per entry, not a delegate dispatch.
    /// <para>
    /// Emitted IL for one entry:
    /// <code>
    /// if ((uint)idx &gt;= (uint)entries.Count) return null;
    /// ref var entry = ref entries[idx];
    /// if (!grads.TryGetValue(entry.Output, out gradOutput)) goto skip;
    /// directCall(
    ///     gradOutput,
    ///     entry.GetInputsArrayInto(state.Buf1, state.Buf2, state.Buf3),
    ///     entry.Output,
    ///     entry.SavedState ?? Array.Empty&lt;object&gt;(),
    ///     engine,
    ///     state.Grads);
    /// skip:
    /// </code>
    /// </para>
    /// </summary>
    private static DynamicMethod EmitWalkerILSpecialised(int[] reverseTopoIndices, MethodInfo[] backwardMethods)
    {
        var method = new DynamicMethod(
            name: $"CompiledBackwardWalk_Spec_{typeof(T).Name}_{reverseTopoIndices.Length}",
            returnType: typeof(Dictionary<Tensor<T>, Tensor<T>>),
            parameterTypes: new[]
            {
                typeof(TapeEntryArena<T>),
                typeof(Tensor<T>),
                typeof(IReadOnlyList<Tensor<T>>),
                typeof(IEngine),
            },
            restrictedSkipVisibility: true);
        method.DefineParameter(1, ParameterAttributes.None, "entries");
        method.DefineParameter(2, ParameterAttributes.None, "loss");
        method.DefineParameter(3, ParameterAttributes.None, "sources");
        method.DefineParameter(4, ParameterAttributes.None, "engine");

        var il = method.GetILGenerator();

        var stateLocal = il.DeclareLocal(typeof(CompiledBackwardWalkHelpers<T>.WalkState));
        var gradOutputLocal = il.DeclareLocal(typeof(Tensor<T>));
        // Byref local for caching entries[idx] — avoids 4 redundant indexer
        // calls per entry. CLR allows byref locals for managed pointers to
        // struct types; the JIT keeps them in a register or stack slot.
        var entryRefLocal = il.DeclareLocal(typeof(TapeEntry<T>).MakeByRefType());

        // state = InitState(loss, reservedCount)
        il.Emit(OpCodes.Ldarg_1);
        il.Emit(OpCodes.Ldc_I4, reverseTopoIndices.Length);
        il.Emit(OpCodes.Call, s_initStateMethod);
        il.Emit(OpCodes.Stloc, stateLocal);

        // Single bounds check up front: verify entries.Count is large
        // enough to cover every baked-in index. Replaces per-entry
        // bounds checks (was N callvirts on entries.Count for an N-entry
        // tape; now exactly one).
        int maxIdx = 0;
        for (int j = 0; j < reverseTopoIndices.Length; j++)
            if (reverseTopoIndices[j] > maxIdx) maxIdx = reverseTopoIndices[j];

        var arenaTypeForCount = typeof(TapeEntryArena<T>);
        var arenaCountGetterEarly = arenaTypeForCount.GetProperty("Count",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)!
            .GetGetMethod(nonPublic: true)!;

        var pastBoundsCheckLabel = il.DefineLabel();
        il.Emit(OpCodes.Ldc_I4, maxIdx);
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Callvirt, arenaCountGetterEarly);
        il.Emit(OpCodes.Blt_S, pastBoundsCheckLabel);
        il.Emit(OpCodes.Ldnull);
        il.Emit(OpCodes.Ret);
        il.MarkLabel(pastBoundsCheckLabel);

        // Pre-resolve handles used per-entry.
        var walkStateType = typeof(CompiledBackwardWalkHelpers<T>.WalkState);
        var gradsField = walkStateType.GetField(nameof(CompiledBackwardWalkHelpers<T>.WalkState.Grads))!;
        var buf1Field = walkStateType.GetField(nameof(CompiledBackwardWalkHelpers<T>.WalkState.Buf1))!;
        var buf2Field = walkStateType.GetField(nameof(CompiledBackwardWalkHelpers<T>.WalkState.Buf2))!;
        var buf3Field = walkStateType.GetField(nameof(CompiledBackwardWalkHelpers<T>.WalkState.Buf3))!;

        var arenaType = typeof(TapeEntryArena<T>);
        var arenaIndexer = arenaType.GetProperty("Item",
            BindingFlags.NonPublic | BindingFlags.Instance)!.GetGetMethod(nonPublic: true)!;
        var arenaCountGetter = arenaType.GetProperty("Count",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)!.GetGetMethod(nonPublic: true)!;

        var entryType = typeof(TapeEntry<T>);
        var outputField = entryType.GetField(nameof(TapeEntry<T>.Output))!;
        var input0Field = entryType.GetField(nameof(TapeEntry<T>.Input0))!;
        var input1Field = entryType.GetField(nameof(TapeEntry<T>.Input1))!;
        var savedStateField = entryType.GetField(nameof(TapeEntry<T>.SavedState))!;
        var getInputsArrayIntoMethod = entryType.GetMethod(nameof(TapeEntry<T>.GetInputsArrayInto))!;

        var dictType = typeof(Dictionary<Tensor<T>, Tensor<T>>);
        var tryGetValueMethod = dictType.GetMethod(nameof(Dictionary<Tensor<T>, Tensor<T>>.TryGetValue))!;

        var arrayEmptyObjectMethod = typeof(Array).GetMethod(nameof(Array.Empty))!
            .MakeGenericMethod(typeof(object));

        // AccumulateGrad is called by inliners (e.g. AddBackward) that
        // emit the gradient math directly instead of going through the
        // backward method dispatch.
        var diffOpsType = typeof(DifferentiableOps);
        var accumulateGradMethod = diffOpsType.GetMethod(
            "AccumulateGrad",
            BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Static)
            ?.MakeGenericMethod(typeof(T));

        for (int i = 0; i < reverseTopoIndices.Length; i++)
        {
            int idx = reverseTopoIndices[i];
            var bwdMethod = backwardMethods[i];
            var skipLabel = il.DefineLabel();
            var savedStateNonNullLabel = il.DefineLabel();
            var savedStateDoneLabel = il.DefineLabel();
            var bailOutLabel = il.DefineLabel();

            // Per-entry bounds check elided: the pre-loop check above
            // verifies entries.Count > max(indices), so each baked idx
            // is in-range without redundant per-entry checks.

            // Cache the entry ref in a byref local — one indexer call
            // per entry instead of four.
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, idx);
            il.Emit(OpCodes.Callvirt, arenaIndexer);   // ref entry
            il.Emit(OpCodes.Stloc, entryRefLocal);

            // grads.TryGetValue(entry.Output, out gradOutput)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldloc, entryRefLocal);
            il.Emit(OpCodes.Ldfld, outputField);       // entry.Output
            il.Emit(OpCodes.Ldloca, gradOutputLocal);
            il.Emit(OpCodes.Callvirt, tryGetValueMethod);
            il.Emit(OpCodes.Brfalse_S, skipLabel);

            // Inliner registry check: if this backward method has a
            // per-op IL inliner, let it emit the gradient math directly
            // instead of building the 6-arg call. The inliner skips
            // inputs[] array construction + savedState null-coalescing
            // entirely.
            if (accumulateGradMethod is not null
                && s_inliners.TryGetValue(bwdMethod, out var inliner))
            {
                inliner.Emit(il, stateLocal, entryRefLocal, gradOutputLocal,
                    gradsField, input0Field, input1Field, outputField, savedStateField, accumulateGradMethod);
            }
            else
            {
                // arg1: gradOutput
                il.Emit(OpCodes.Ldloc, gradOutputLocal);

                // arg2: entry.GetInputsArrayInto(buf1, buf2, buf3)
                il.Emit(OpCodes.Ldloc, entryRefLocal);
                il.Emit(OpCodes.Ldloca, stateLocal); il.Emit(OpCodes.Ldfld, buf1Field);
                il.Emit(OpCodes.Ldloca, stateLocal); il.Emit(OpCodes.Ldfld, buf2Field);
                il.Emit(OpCodes.Ldloca, stateLocal); il.Emit(OpCodes.Ldfld, buf3Field);
                il.Emit(OpCodes.Call, getInputsArrayIntoMethod);

                // arg3: entry.Output
                il.Emit(OpCodes.Ldloc, entryRefLocal);
                il.Emit(OpCodes.Ldfld, outputField);

                // arg4: entry.SavedState ?? Array.Empty<object>()
                il.Emit(OpCodes.Ldloc, entryRefLocal);
                il.Emit(OpCodes.Ldfld, savedStateField);
                il.Emit(OpCodes.Dup);
                il.Emit(OpCodes.Brtrue_S, savedStateNonNullLabel);
                il.Emit(OpCodes.Pop);                          // pop the null
                il.Emit(OpCodes.Call, arrayEmptyObjectMethod); // push Array.Empty<object>()
                il.Emit(OpCodes.Br_S, savedStateDoneLabel);
                il.MarkLabel(savedStateNonNullLabel);
                // non-null savedState is already on the stack from Dup; nothing more to do
                il.MarkLabel(savedStateDoneLabel);

                // arg5: engine
                il.Emit(OpCodes.Ldarg_3);

                // arg6: state.Grads
                il.Emit(OpCodes.Ldloca, stateLocal);
                il.Emit(OpCodes.Ldfld, gradsField);

                // Direct call to the entry's static backward method —
                // this is the perf-winning step over delegate dispatch.
                il.Emit(OpCodes.Call, bwdMethod);
            }

            il.Emit(OpCodes.Br_S, skipLabel);

            // bailOutLabel from the per-entry bounds check is no longer
            // referenced (the pre-loop bounds check covers all indices),
            // but the label still needs to be marked somewhere or
            // Reflection.Emit complains about an unmarked defined label.
            il.MarkLabel(bailOutLabel);

            il.MarkLabel(skipLabel);
        }

        // return FinalizeWalk(state, sources);
        il.Emit(OpCodes.Ldloca, stateLocal);
        il.Emit(OpCodes.Ldarg_2);
        il.Emit(OpCodes.Call, s_finalizeWalkMethod);
        il.Emit(OpCodes.Ret);

        return method;
    }

    /// <summary>
    /// Closure-based fallback walker. Identical semantics to the IL
    /// version; runs on platforms that disallow <c>Reflection.Emit</c>.
    /// Also serves as the regression-test oracle when env-var-gated
    /// AIDOTNET_COMPILED_BACKWARD is off.
    /// </summary>
    private static CompiledWalker EmitWalkerClosure(int[] reverseTopoIndices)
    {
        var indices = reverseTopoIndices;
        return (entries, loss, sources, engine) =>
        {
            var state = CompiledBackwardWalkHelpers<T>.InitState(loss, indices.Length);
            for (int i = 0; i < indices.Length; i++)
            {
                if (!CompiledBackwardWalkHelpers<T>.ProcessEntry(entries, indices[i], ref state, engine))
                    return null;
            }
            return CompiledBackwardWalkHelpers<T>.FinalizeWalk(ref state, sources);
        };
    }
}

/// <summary>
/// Helper functions that the IL-emitted walker calls. Keeping them as a
/// separate static class lets the IL emitter cache <see cref="MethodInfo"/>
/// handles once per closed generic <typeparamref name="T"/>.
/// </summary>
internal static class CompiledBackwardWalkHelpers<T>
{
    /// <summary>
    /// State threaded through every per-entry call. A struct (not a class)
    /// so the IL emitter can pass it via <c>ldloca</c> and the JIT can
    /// keep its fields in registers across the unrolled walk.
    /// </summary>
    internal struct WalkState
    {
        public Dictionary<Tensor<T>, Tensor<T>> Grads;
        public Tensor<T>[] Buf1;
        public Tensor<T>[] Buf2;
        public Tensor<T>[] Buf3;
    }

    /// <summary>
    /// Initializes the per-walk state. Allocates the gradient dictionary
    /// with capacity sized to the recorded entry count, allocates the
    /// per-arity input buffers, and seeds <paramref name="loss"/>'s
    /// gradient to ones.
    /// </summary>
    public static WalkState InitState(Tensor<T> loss, int reservedCount)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var grads = new Dictionary<Tensor<T>, Tensor<T>>(
            reservedCount + 1,
            ReferenceEqualityComparer<Tensor<T>>.Instance);

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

        return new WalkState
        {
            Grads = grads,
            Buf1 = new Tensor<T>[1],
            Buf2 = new Tensor<T>[2],
            Buf3 = new Tensor<T>[3],
        };
    }

    /// <summary>
    /// Processes one tape entry. Looks up the entry by index, checks
    /// whether the output has an accumulated upstream gradient, and if so
    /// invokes its backward function. Returns <c>false</c> when the index
    /// is out of range — signal to the IL walker to bail out (stale cache).
    /// </summary>
    public static bool ProcessEntry(
        TapeEntryArena<T> entries,
        int idx,
        ref WalkState state,
        IEngine engine)
    {
        if ((uint)idx >= (uint)entries.Count) return false;
        ref var entry = ref entries[idx];

        if (!state.Grads.TryGetValue(entry.Output, out var gradOutput))
            return true;

        entry.Backward(
            gradOutput,
            entry.GetInputsArrayInto(state.Buf1, state.Buf2, state.Buf3),
            entry.Output,
            entry.SavedState ?? Array.Empty<object>(),
            engine,
            state.Grads);
        return true;
    }

    /// <summary>
    /// Filters the gradient dictionary by sources (when non-null) and
    /// clears the per-arity buffer references so the next walk starts
    /// fresh. Mirrors <see cref="RebindablePlanCache{T}.TryExecute"/>'s
    /// post-loop cleanup.
    /// </summary>
    public static Dictionary<Tensor<T>, Tensor<T>> FinalizeWalk(
        ref WalkState state,
        IReadOnlyList<Tensor<T>>? sources)
    {
        // Clear buffer references so forward intermediates from this walk
        // don't get pinned in the buffer arrays for the rest of the
        // thread's lifetime. The buffers themselves are kept (small fixed
        // allocs); only the references inside them are nulled.
        state.Buf1[0] = null!;
        state.Buf2[0] = null!; state.Buf2[1] = null!;
        state.Buf3[0] = null!; state.Buf3[1] = null!; state.Buf3[2] = null!;

        if (sources is not null)
        {
            var filtered = new Dictionary<Tensor<T>, Tensor<T>>(
                sources.Count, ReferenceEqualityComparer<Tensor<T>>.Instance);
            for (int i = 0; i < sources.Count; i++)
            {
                if (state.Grads.TryGetValue(sources[i], out var grad))
                    filtered[sources[i]] = grad;
            }
            return filtered;
        }

        return state.Grads;
    }
}
