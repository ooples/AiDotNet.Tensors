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
        // Single full-identity hash. We considered splitting into a structure-only
        // hash for repeat detection + identity-augmented hash for cache lookup
        // (see PR #333 review thread), but actually splitting them exposes latent
        // bugs in the compiled-replay path: tests in BatchNorm3DIntegrationTests,
        // GradientTapeReplayModeTests, TensorCopyToTests, and TorchFuncPhase1Tests
        // start failing because auto-compile fires in scenarios it didn't before,
        // and the compiled backward graph stores tensor references that aren't
        // valid in those scenarios (KeyNotFoundException on gradients dictionary).
        // The TODO is to fix the compiled-replay path so it handles createGraph,
        // gradient-dict consistency, and cross-tape replay correctly — only then
        // can we safely split the hashes. Until then keep the full hash so compile
        // only fires when activation / loss / parameter identities all match,
        // which is rare in practice (every forward creates fresh intermediates)
        // and limits the scope of replay-path bugs that can be triggered.
        long hash = ComputePatternHash(entries, entryCount);
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

        // Validate current tape pattern AND loss/sources identity match the
        // compiled plan. Full-hash matching keeps the compile cache safe
        // against clone-then-train (different parameter tensors) and against
        // tape-aliasing (same op pattern, different activation/loss objects).
        long currentHash = ComputePatternHash(tape.Entries, tape.EntryCount);
        currentHash = IncludeTargetIdentity(currentHash, loss, sources);
        if (!State.MatchesCompiledHash(currentHash))
        {
            // Pattern changed — disable replay mode so recording resumes
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

        try
        {
            var compiled = tape.CompileBackward(loss, sources);
            // Store the same hash TryGetCompiledBackward will compute for
            // lookup — full identity-augmented pattern hash so cache lookup
            // and cache store agree.
            long hash = ComputePatternHash(tape.Entries, tape.EntryCount);
            hash = IncludeTargetIdentity(hash, loss, sources);
            State.StoreCompiledBackwardWithHash(compiled, hash);
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
    /// Computes a hash of the training step pattern: op names + output identities
    /// + output shapes + input identities + element type. Used by both
    /// <see cref="RecordStep{T}"/> for repeat detection and by
    /// <see cref="TryGetCompiledBackward{T}"/> / <see cref="TryCompileBackward{T}"/>
    /// (after composition with <see cref="IncludeTargetIdentity{T}"/>) for the
    /// compiled-plan cache key.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Output tensor identities (<c>RuntimeHelpers.GetHashCode</c>) are included
    /// so a cloned model — same architecture, fresh parameter tensors — produces
    /// a different hash from its source and gets a fresh compile rather than
    /// replaying the original's backward plan against the wrong tensor refs.
    /// </para>
    /// <para>
    /// <b>Repeat-detection trade-off:</b> activation / loss tensors are typically
    /// recreated each forward pass, so the hash differs across steps and the
    /// compile threshold rarely fires unless the consumer reuses tensor instances
    /// (e.g. via <c>TensorArena</c>'s pooled reuse). A future refactor could split
    /// this into a structure-only hash for repeat detection + an identity-augmented
    /// hash for cache lookup so compile fires more aggressively, but that requires
    /// the compiled-replay path to first correctly handle createGraph, gradient-dict
    /// consistency, and cross-tape replay — see BatchNorm3DIntegrationTests and
    /// GradientTapeReplayModeTests for the surface that would need to be fixed
    /// alongside the split.
    /// </para>
    /// </remarks>
    private static long ComputePatternHash<T>(TapeEntryArena<T> entries, int entryCount)
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        // Include element type so float and double plans don't collide
        hash ^= typeof(T).GetHashCode();
        hash *= unchecked((long)0x100000001b3L);
        for (int i = 0; i < entryCount; i++)
        {
            ref var entry = ref entries[i];
            hash ^= entry.OperationName.GetHashCode();
            hash *= unchecked((long)0x100000001b3L);
            if (entry.Output is not null)
            {
                // Identity disambiguates same-shape outputs across models
                // (e.g., cloned network's intermediates vs original's).
                hash ^= RuntimeHelpers.GetHashCode(entry.Output);
                hash *= unchecked((long)0x100000001b3L);
                foreach (int dim in entry.Output._shape)
                {
                    hash ^= dim;
                    hash *= unchecked((long)0x100000001b3L);
                }
            }
            // Hash input tensor identities directly from the inline
            // Input0/Input1/Input2/InputsOverflow fields. GetInputsArray()
            // allocates a Tensor<T>[1..3] on every call and this hash runs
            // per RecordStep / TryGetCompiledBackward / TryCompileBackward,
            // so on large graphs (BERT ~150 entries) that's 150 small-array
            // allocs per Train call we avoid here.
            if (entry.InputsOverflow is not null)
            {
                foreach (var inp in entry.InputsOverflow)
                {
                    if (inp is null) continue;
                    hash ^= RuntimeHelpers.GetHashCode(inp);
                    hash *= unchecked((long)0x100000001b3L);
                }
            }
            else
            {
                if (entry.InputCount >= 1 && entry.Input0 is not null)
                {
                    hash ^= RuntimeHelpers.GetHashCode(entry.Input0);
                    hash *= unchecked((long)0x100000001b3L);
                }
                if (entry.InputCount >= 2 && entry.Input1 is not null)
                {
                    hash ^= RuntimeHelpers.GetHashCode(entry.Input1);
                    hash *= unchecked((long)0x100000001b3L);
                }
                if (entry.InputCount >= 3 && entry.Input2 is not null)
                {
                    hash ^= RuntimeHelpers.GetHashCode(entry.Input2);
                    hash *= unchecked((long)0x100000001b3L);
                }
            }
        }
        return hash;
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
        else if (!_compiledStored || hash != _lastStepHash)
        {
            // New pattern — reset state. But don't drop compiled state if
            // the pattern hash matches (only the full hash differs due to sources).
            _lastStepHash = hash;
            _repeatCount = 1;
            if (!_compiledStored)
            {
                _compilationFailed = false;
                _compiledBackward = null;
            }
        }
    }

    internal void StoreCompiledBackward<T>(CompiledBackwardGraph<T> compiled)
    {
        _compiledBackward = compiled;
        _compiledStored = true;
    }

    internal void StoreCompiledBackwardWithHash<T>(CompiledBackwardGraph<T> compiled, long hash)
    {
        _compiledBackward = compiled;
        _compiledStored = true;
        _compiledHash = hash; // Store full hash (pattern + sources) for lookup matching
    }

    internal CompiledBackwardGraph<T>? TryGetCompiledBackward<T>()
    {
        return _compiledBackward as CompiledBackwardGraph<T>;
    }
}
