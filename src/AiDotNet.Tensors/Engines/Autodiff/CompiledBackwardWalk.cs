// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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

        // state = InitState(loss, reservedCount)
        il.Emit(OpCodes.Ldarg_1);
        il.Emit(OpCodes.Ldc_I4, reverseTopoIndices.Length);
        il.Emit(OpCodes.Call, s_initStateMethod);
        il.Emit(OpCodes.Stloc, stateLocal);

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
        var savedStateField = entryType.GetField(nameof(TapeEntry<T>.SavedState))!;
        var getInputsArrayIntoMethod = entryType.GetMethod(nameof(TapeEntry<T>.GetInputsArrayInto))!;

        var dictType = typeof(Dictionary<Tensor<T>, Tensor<T>>);
        var tryGetValueMethod = dictType.GetMethod(nameof(Dictionary<Tensor<T>, Tensor<T>>.TryGetValue))!;

        var arrayEmptyObjectMethod = typeof(Array).GetMethod(nameof(Array.Empty))!
            .MakeGenericMethod(typeof(object));

        for (int i = 0; i < reverseTopoIndices.Length; i++)
        {
            int idx = reverseTopoIndices[i];
            var bwdMethod = backwardMethods[i];
            var skipLabel = il.DefineLabel();
            var savedStateNonNullLabel = il.DefineLabel();
            var savedStateDoneLabel = il.DefineLabel();
            var bailOutLabel = il.DefineLabel();

            // Bounds check: idx < entries.Count else return null.
            il.Emit(OpCodes.Ldc_I4, idx);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Callvirt, arenaCountGetter);
            il.Emit(OpCodes.Bge_S, bailOutLabel);  // idx >= Count

            // grads.TryGetValue(entries[idx].Output, out gradOutput)
            il.Emit(OpCodes.Ldloca, stateLocal);
            il.Emit(OpCodes.Ldfld, gradsField);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, idx);
            il.Emit(OpCodes.Callvirt, arenaIndexer);   // ref entry
            il.Emit(OpCodes.Ldfld, outputField);       // entry.Output
            il.Emit(OpCodes.Ldloca, gradOutputLocal);
            il.Emit(OpCodes.Callvirt, tryGetValueMethod);
            il.Emit(OpCodes.Brfalse_S, skipLabel);

            // arg1: gradOutput
            il.Emit(OpCodes.Ldloc, gradOutputLocal);

            // arg2: entries[idx].GetInputsArrayInto(buf1, buf2, buf3)
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, idx);
            il.Emit(OpCodes.Callvirt, arenaIndexer);
            il.Emit(OpCodes.Ldloca, stateLocal); il.Emit(OpCodes.Ldfld, buf1Field);
            il.Emit(OpCodes.Ldloca, stateLocal); il.Emit(OpCodes.Ldfld, buf2Field);
            il.Emit(OpCodes.Ldloca, stateLocal); il.Emit(OpCodes.Ldfld, buf3Field);
            il.Emit(OpCodes.Call, getInputsArrayIntoMethod);

            // arg3: entries[idx].Output
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, idx);
            il.Emit(OpCodes.Callvirt, arenaIndexer);
            il.Emit(OpCodes.Ldfld, outputField);

            // arg4: entries[idx].SavedState ?? Array.Empty<object>()
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, idx);
            il.Emit(OpCodes.Callvirt, arenaIndexer);
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

            il.Emit(OpCodes.Br_S, skipLabel);

            il.MarkLabel(bailOutLabel);
            il.Emit(OpCodes.Ldnull);
            il.Emit(OpCodes.Ret);

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
