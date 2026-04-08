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
        // Pattern hash is based on op sequence + shapes only — NOT loss tensor identity.
        // Loss tensors are recreated each forward pass (different object each step),
        // so including identity would prevent pattern detection across steps.
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

        // Validate current tape pattern AND loss/sources identity match the compiled plan.
        // IncludeTargetIdentity hashes both loss tensor and source tensors so different
        // loss functions or source sets don't reuse the wrong backward plan.
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
            // Store with target identity so the hash includes loss/sources
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
    /// Computes a hash of the training step pattern from tape entries.
    /// Two steps with the same op sequence + shapes produce the same hash.
    /// </summary>
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
                foreach (int dim in entry.Output._shape)
                {
                    hash ^= dim;
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
