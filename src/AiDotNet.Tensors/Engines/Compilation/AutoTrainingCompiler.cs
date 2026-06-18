using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Auto-compiles repeated training step patterns for zero-overhead replay.
/// When GradientTape detects the same forward op sequence on consecutive steps,
/// it compiles the backward graph for optimized gradient computation.
///
/// This is Layer 3 of the TensorCodec training system:
///   Layer 1: AutoTensorCache — zero-alloc tensor pool
///   Layer 2: AutoTracer — auto-compile hot forward ops
///   Layer 3: AutoTrainingCompiler — auto-compile backward pass (this)
///
/// Transparent to user code: just write normal training loops.
/// </summary>
internal static class AutoTrainingCompiler
{
    internal static bool Enabled { get; set; } = true;

    /// <summary>Resets all thread-static state. Used by tests to ensure isolation.</summary>
    internal static void ResetState()
    {
        _state = null;
        ReplayMode = false;
    }

    /// <summary>
    /// When true, DifferentiableOps.RecordIfActive skips tape recording entirely.
    /// Set after auto-compilation succeeds — the compiled backward graph replaces
    /// the tape, so recording new entries would be wasted work (~200ns per op saved).
    /// Cleared when the forward pattern changes (new model, different input shape).
    /// </summary>
    [ThreadStatic]
    internal static bool ReplayMode;

    [ThreadStatic]
    private static AutoTrainingState? _state;

    private static AutoTrainingState State => _state ??= new AutoTrainingState();

    /// <summary>
    /// Called by GradientTape.GradientAndUpdate after backward pass completes.
    /// Records the training step pattern for future compilation.
    /// </summary>
    internal static void RecordStep<T>(TapeEntryArena<T> entries, int entryCount, Tensor<T>? loss = null)
    {
        if (!Enabled) return;
        // Repeat-detection hash: STRUCTURE ONLY (op sequence + shapes + element
        // type). Every forward pass creates fresh activation / loss tensors,
        // so folding their identities in here would make every step hash
        // differently and the compile threshold would never fire on real
        // training loops (which is exactly what happened on PR #333's first
        // attempt at a unified-hash design — the AutoTrainingCompilerTests
        // PersistentTape_* assertions all failed). Cache LOOKUP uses a stricter
        // key (structure + parameter-tensor identities, via
        // <see cref="TryGetCompiledBackward{T}"/>) so cloned models still miss
        // the cache and trigger a fresh compile.
        long hash = ComputeStructureHash(entries, entryCount);
        State.RecordStepHash(hash);
    }

    /// <summary>
    /// Called by GradientTape.ComputeGradients to check if a compiled backward
    /// graph exists for the current training pattern.
    /// </summary>
    internal static CompiledBackwardGraph<T>? TryGetCompiledBackward<T>(
        GradientTape<T> tape,
        Tensor<T> loss,
        Tensor<T>[]? sources)
    {
        if (!Enabled) return null;
        if (!State.HasCompiledPlan) return null;

        // Only use compiled backward if tape is persistent (entries survive between calls)
        if (!tape.Options.Persistent) return null;

        // Compiled-plan retrieval hash: structure + parameter-tensor identities.
        // The compiled backward graph holds direct Tensor<T> references to the
        // specific parameter tensors it was built against; replaying it on a
        // cloned model (same architecture, fresh parameter instances) would
        // silently produce wrong gradients. So the cache key must include
        // SOURCE identities. Activation / loss identities are deliberately NOT
        // included — only sources, which are the long-lived parameter tensors
        // the caller actually wants gradients for.
        long currentHash = ComputeStructureHash(tape.Entries, tape.EntryCount);
        currentHash = IncludeTargetIdentity(currentHash, loss, sources);
        // Match on BOTH the full hash AND exact source reference-equality. The
        // hash alone is collision-prone (XOR of 32-bit GetHashCode values), and
        // the state is thread-static — a different model on the same thread could
        // collide and replay the wrong backward against the wrong parameters
        // (the flaky, batch-only LossStrictly failures). The reference check makes
        // a cross-model hit impossible.
        if (!State.MatchesCompiledHash(currentHash) || !State.SourcesMatch(sources))
        {
            // Pattern / sources changed — disable replay mode so recording resumes
            ReplayMode = false;
            return null;
        }

        var compiled = State.TryGetCompiledBackward<T>();
        // Only enable replay mode if we actually got a compiled backward graph.
        // If TryGetCompiledBackward returns null (type mismatch, etc.), recording
        // must continue so the tape-based backward path can function.
        ReplayMode = compiled is not null;
        return compiled;
    }

    /// <summary>
    /// Called after ComputeGradients to store a compiled backward graph
    /// when a repeating pattern is detected.
    /// </summary>
    internal static void TryCompileBackward<T>(
        GradientTape<T> tape,
        Tensor<T> loss,
        Tensor<T>[]? sources)
    {
        if (!Enabled) return;
        if (!tape.Options.Persistent) return;
        if (!State.ShouldCompile) return;

        // Size gate: only compile steps whose forward activation volume is large enough to benefit
        // from GPU-resident execution. Tiny models (small RL policies, shallow tabular nets) gain
        // nothing from compilation — kernel-launch/transfer overhead dominates — and on the GPU
        // buffer-cache path they can trip the deferred-materializer-vs-eviction race (#226), so we
        // keep them on the eager path. Large models (transformer-scale training) still compile.
        if (TotalForwardElements(tape) < CompiledTrainingMinForwardElements)
        {
            // Mark this pattern as not-to-compile so we don't re-evaluate every step; it runs eager.
            State.MarkCompilationFailed();
            return;
        }

        // Clone-then-train safety: when `sources` is null the cache key
        // collapses to structure-only, which would let a cloned model with
        // identical architecture hit the original's compiled plan and replay
        // gradients against the wrong parameter tensors. Refuse to compile
        // until the caller passes an explicit sources set — they almost
        // always do (every consumer that actually uses
        // <c>tape.ComputeGradients(loss, parameters)</c> passes parameters),
        // and the rare null-sources callers simply pay the recording-path
        // cost rather than risk a wrong-gradient replay.
        if (sources is null) return;

        try
        {
            var compiled = tape.CompileBackward(loss, sources);
            // Store the cache key (structure + sources). Matches the lookup
            // computation in TryGetCompiledBackward — both must agree.
            long hash = ComputeStructureHash(tape.Entries, tape.EntryCount);
            hash = IncludeTargetIdentity(hash, loss, sources);
            State.StoreCompiledBackwardWithHash(compiled, hash, sources);
            // Drop the plan's strong reference to the compilation-time loss
            // tensor. The compile cache is thread-static and persists for
            // the lifetime of the worker thread; without this release, the
            // loss tensor (and its entire GradFn chain back through every
            // forward intermediate) would stay reachable for the whole
            // process, manifesting as a per-iteration leak under
            // GradientTapeLeakTests at issue #283 scale (tensor-arena pool
            // reuse makes early-iteration intermediates the same Tensor
            // instances as later iterations'). Subsequent ComputeGradients
            // calls always pass the current step's loss to Execute(loss),
            // so the parameterless fallback isn't needed on this path.
            compiled.ReleaseCompilationTimeLoss();
            ReplayMode = true; // Enable replay mode now that backward is compiled
        }
        catch
        {
            // Compilation failed — mark as attempted to prevent infinite retry
            State.MarkCompilationFailed();
        }
    }

    /// <summary>
    /// Incorporates source tensor identities into the pattern hash so that different
    /// parameter sets don't reuse the wrong backward plan.
    /// Note: loss tensor identity is NOT included because loss tensors are recreated
    /// each forward pass. The op structure (captured in tape entries hash) already
    /// identifies the computation pattern including the loss function type.
    /// </summary>
    /// <remarks>
    /// When <paramref name="sources"/> is null the function is a no-op. That
    /// path is now closed inside <see cref="TryCompileBackward"/> via a
    /// short-circuit return, so the cache cannot store a plan keyed on
    /// structure-only — clone-then-train remains safe.
    /// </remarks>
    private static long IncludeTargetIdentity<T>(long hash, Tensor<T> loss, Tensor<T>[]? sources)
    {
        if (sources is not null)
        {
            foreach (var src in sources)
            {
                hash ^= RuntimeHelpers.GetHashCode(src);
                hash *= unchecked((long)0x100000001b3L);
            }
        }
        return hash;
    }

    /// <summary>
    /// Structure-only hash: op name sequence + output shapes + element type.
    /// Two training steps with the same model architecture and same input
    /// shapes produce the same hash even though their activation / loss
    /// tensors are different object instances (which they always are —
    /// each forward pass creates fresh intermediates).
    /// </summary>
    /// <remarks>
    /// <para>This is the right hash for <see cref="RecordStep{T}"/>'s repeat
    /// detection: we want consecutive train steps of the same model to be
    /// recognized as a repeat so the compile threshold actually fires.
    /// Including activation / intermediate / loss identities would make
    /// every step hash differently and pattern repeats would never trigger.</para>
    /// <para>Per-Train cost: O(entryCount) — one hash op per tape entry. On
    /// VGG16-BN that's ~100 hash ops out of a 38 s iteration — negligible.</para>
    /// <para><b>Visibility:</b> internal (not private) because PR #331's
    /// RebindablePlanCache fresh-tape path and the cross-tape detection
    /// walk in GradientTape both consult this hash directly to keep the
    /// non-persistent fast path in sync with the persistent compile cache.
    /// Keep both consumers on the SAME hash function — splitting them
    /// would let the cache lookups disagree silently.</para>
    /// </remarks>
    internal static long ComputeStructureHash<T>(TapeEntryArena<T> entries, int entryCount)
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        // Include element type so float and double plans don't collide.
        hash ^= typeof(T).GetHashCode();
        hash *= unchecked((long)0x100000001b3L);
        for (int i = 0; i < entryCount; i++)
        {
            ref var entry = ref entries[i];
            hash ^= entry.OperationName.GetHashCode();
            hash *= unchecked((long)0x100000001b3L);
            if (entry.Output is not null)
            {
                // Shape only — NOT identity. Two steps with the same model
                // shape produce the same structure hash.
                foreach (int dim in entry.Output._shape)
                {
                    hash ^= dim;
                    hash *= unchecked((long)0x100000001b3L);
                }
            }
        }
        return hash;
    }

    /// <summary>
    /// Minimum total forward-activation element count for a training step to be worth compiling.
    /// Below this, eager execution is used: compilation overhead exceeds any GPU benefit at this
    /// size and the GPU buffer-cache deferred-materializer path (#226) is avoided. Override with the
    /// AIDOTNET_COMPILE_MIN_ELEMENTS environment variable (read once at type load).
    /// </summary>
    private static readonly long CompiledTrainingMinForwardElementsDefault =
        long.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_COMPILE_MIN_ELEMENTS"), out var configured) && configured > 0
            ? configured
            : 1_000_000L;

    /// <summary>
    /// Test-only override of the compile size gate. The env-backed default is a
    /// <c>static readonly</c> read once at type load, so tests that exercise the
    /// replay mechanism on deliberately tiny tapes cannot lower it via the env var
    /// after the type is initialized. Set this (and reset to <c>null</c> in teardown)
    /// to force compilation of small steps without a real million-element model.
    /// Null = use the env-or-1M default. Mirrors the established test-hook pattern
    /// (e.g. MixedPrecisionEmit.TestOverrideEnabled).
    /// </summary>
    internal static long? TestMinForwardElementsOverride { get; set; }

    /// <summary>Effective size-gate threshold: the test override when set, else the env-or-1M default.</summary>
    private static long CompiledTrainingMinForwardElements
        => TestMinForwardElementsOverride ?? CompiledTrainingMinForwardElementsDefault;
    /// <summary>Sum of forward-activation element counts on the tape (early-exits once the threshold is reached).</summary>
    private static long TotalForwardElements<T>(GradientTape<T> tape)
    {
        long total = 0;
        int count = tape.EntryCount;
        for (int i = 0; i < count; i++)
        {
            ref var entry = ref tape.Entries[i];
            var output = entry.Output;
            if (output is null) continue;
            long elements = 1;
            foreach (int dim in output._shape)
            {
                elements *= dim;
            }

            total += elements;
            if (total >= CompiledTrainingMinForwardElements)
            {
                return total;
            }
        }

        return total;
    }
}

/// <summary>
/// Per-thread state for auto-training compilation. Tracks step patterns
/// and stores compiled backward graphs.
/// </summary>
internal sealed class AutoTrainingState
{
    private long _lastStepHash;    // Pattern hash (ops + shapes only) for repeat detection
    private long _compiledHash;    // Full hash (pattern + sources) for compiled backward lookup
    private int _repeatCount;
    private object? _compiledBackward; // CompiledBackwardGraph<T> — type-erased
    private bool _compiledStored;
    // Weak references to the EXACT source tensors the compiled plan was built
    // against. The full hash (_compiledHash) folds in RuntimeHelpers.GetHashCode
    // of these sources, but hash codes are 32-bit and XOR-combined, so a
    // DIFFERENT model on the same (thread-static) state could collide and replay
    // the wrong backward — silently producing wrong gradients (root cause of the
    // flaky, batch-only LossStrictly failures). Verifying reference-equality of
    // the sources on lookup makes a cross-model hit impossible. Weak so the plan
    // never pins another model's parameter tensors alive.
    private System.WeakReference<object>[]? _compiledSourceRefs;

    private const int CompileThreshold = 2;

    private bool _compilationFailed;

    internal bool HasCompiledPlan => _compiledStored;
    internal bool ShouldCompile => _repeatCount >= CompileThreshold && !_compiledStored && !_compilationFailed;
    internal bool MatchesCompiledHash(long hash) => _compiledStored && hash == _compiledHash;
    internal void MarkCompilationFailed() => _compilationFailed = true;
    internal int RepeatCount => _repeatCount;
    internal bool CompilationFailed => _compilationFailed;

    internal void RecordStepHash(long hash)
    {
        if (hash == _lastStepHash && _lastStepHash != 0)
        {
            _repeatCount++;
        }
        else
        {
            // New STRUCTURE pattern (different model / input shape). The
            // thread-static state persists across models and tests on the same
            // thread, so a previously-compiled plan here belongs to the OLD
            // structure and its source tensors. Drop it completely: keeping it
            // (the previous behaviour, which only cleared when !_compiledStored)
            // let a structurally-different model leave a stale plan in the single
            // slot, which then blocked its own compilation and — on a hash
            // collision — could be replayed against the wrong parameters. A model
            // whose structure genuinely repeats will recompile on its next steps.
            _lastStepHash = hash;
            _repeatCount = 1;
            _compilationFailed = false;
            _compiledBackward = null;
            _compiledStored = false;
            _compiledHash = 0;
            _compiledSourceRefs = null;
        }
    }

    internal void StoreCompiledBackward<T>(CompiledBackwardGraph<T> compiled)
    {
        _compiledBackward = compiled;
        _compiledStored = true;
    }

    internal void StoreCompiledBackwardWithHash<T>(CompiledBackwardGraph<T> compiled, long hash, Tensor<T>[]? sources)
    {
        _compiledBackward = compiled;
        _compiledStored = true;
        _compiledHash = hash; // Store full hash (pattern + sources) for lookup matching
        // Record the EXACT source tensors (weakly) so lookup can confirm the
        // plan belongs to THIS model and not a hash-colliding different one.
        if (sources is null || sources.Length == 0)
        {
            _compiledSourceRefs = null;
        }
        else
        {
            _compiledSourceRefs = new System.WeakReference<object>[sources.Length];
            for (int i = 0; i < sources.Length; i++)
                _compiledSourceRefs[i] = new System.WeakReference<object>(sources[i]);
        }
    }

    /// <summary>
    /// True only if the stored plan was compiled against EXACTLY these source
    /// tensor instances (same count, each reference-equal). Guards against a
    /// hash collision serving one model's plan to another. A null/empty stored
    /// source set only matches a null/empty query (structure-only plans are
    /// refused at compile time, so this is the defensive case).
    /// </summary>
    internal bool SourcesMatch<T>(Tensor<T>[]? sources)
    {
        var stored = _compiledSourceRefs;
        int queryLen = sources?.Length ?? 0;
        int storedLen = stored?.Length ?? 0;
        if (queryLen != storedLen) return false;
        if (storedLen == 0) return true;
        for (int i = 0; i < storedLen; i++)
        {
            if (!stored![i].TryGetTarget(out var obj) || !ReferenceEquals(obj, sources![i]))
                return false;
        }
        return true;
    }

    internal CompiledBackwardGraph<T>? TryGetCompiledBackward<T>()
    {
        return _compiledBackward as CompiledBackwardGraph<T>;
    }
}
