using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Auto-Tracing JIT: transparently compiles repeated tensor operation patterns
/// into optimized plans for zero-overhead replay. Fully automatic — no user
/// code changes needed.
///
/// How it works:
/// 1. First call: operation executes eagerly, trace records the op + delegate
/// 2. Second call with same pattern: pattern recognized, graph compiled from trace
/// 3. Third+ calls: compiled plan replayed — zero allocation, pinned SIMD direct
///
/// This is Layer 2 of the TensorCodec system:
///   Layer 1: AutoTensorCache — zero-alloc eager ops (always active)
///   Layer 2: AutoTracer — auto-compile hot paths (this, always active)
///   Layer 3: TensorCodec optimizations — spectral/dataflow/algebraic
///
/// Enabled by default. Disable via AutoTracer.Enabled = false or
/// TensorCodecOptions.Current.EnableCompilation = false.
/// </summary>
internal static class AutoTracer
{
    /// <summary>Whether auto-tracing is enabled (default: true).</summary>
    internal static bool Enabled { get; set; } = true;

    [ThreadStatic]
    private static AutoTracerState? _state;

    internal static AutoTracerState State => _state ??= new AutoTracerState();

    /// <summary>
    /// Called by CpuEngine before each operation. Returns a compiled plan
    /// if one exists for the current operation sequence.
    /// </summary>
    /// <param name="opName">Operation name (e.g., "TensorAdd", "ReLU")</param>
    /// <param name="outputShape">Expected output shape for hash matching.</param>
    /// <param name="paramHash">Extra hash for op-specific parameters (axis, alpha, min/max)
    /// that affect the result. Two calls with the same opName+shape but different paramHash
    /// are treated as different operations — prevents silent wrong-result bugs.</param>
    internal static CompiledInferencePlan<T>? TryGetCompiledPlan<T>(string opName, int[] outputShape, long paramHash = 0)
    {
        if (!Enabled || GraphMode.IsActive || !TensorCodecOptions.Current.EnableCompilation) return null;
        // Don't return compiled plans during this thread's GradientTape lifecycle
        // (forward AND backward). The narrower IsRecording<T>() check returns
        // false during backward (Current is suspended for the backward walk),
        // which would let backward ops replay compiled plans — bypassing
        // autograd. Issue #283: the matching RecordOp side captures forward
        // intermediates in closures during backward, pinning them past tape
        // Dispose. Use _threadTapeDepth (ThreadStatic) so a training thread
        // doesn't suppress AutoTracer on unrelated inference threads in the
        // same process.
        if (Autodiff.DifferentiableOps.ThreadTapeActive()) return null;
        return State.TryGetPlan<T>(opName, outputShape, paramHash);
    }

    /// <summary>
    /// Called by CpuEngine after each operation. Records the op with its
    /// execute delegate for future compilation.
    /// </summary>
    /// <param name="opName">Operation name (e.g., "TensorAdd", "ReLU")</param>
    /// <param name="result">The eagerly computed result tensor</param>
    /// <param name="replayDelegate">A delegate that can re-execute this op inside GraphMode.
    /// Signature: (engine) => engine.OpName(inputs...) — captures the input tensors.</param>
    /// <param name="paramHash">Extra hash for op-specific parameters (must match TryGetCompiledPlan).</param>
    internal static void RecordOp<T>(string opName, Tensor<T> result, Func<IEngine, Tensor<T>> replayDelegate, long paramHash = 0)
    {
        if (!Enabled || GraphMode.IsActive || !TensorCodecOptions.Current.EnableCompilation) return;
        // Skip recording for the ENTIRE GradientTape lifecycle on this thread,
        // not just while the tape is "currently recording" (Current != null).
        // During the backward walk the tape suspends Current to null so backward
        // ops don't append to it; with only the IsRecording<T>() check,
        // AutoTracer would happily record those backward ops here, capturing
        // the forward intermediates they consume in `replayDelegate`'s closure.
        // Those closures sit in _currentSequence until they hit the 128-op cap
        // or a pattern-match clears them — exactly the residual ~400 KB/call
        // leak signature reported in #283. _threadTapeDepth (ThreadStatic) is
        // > 0 for this thread's whole tape lifetime (constructor → Dispose);
        // it does NOT trigger on unrelated inference threads that share the
        // process with a separate training thread.
        if (Autodiff.DifferentiableOps.ThreadTapeActive()) return;
        State.RecordOp(opName, result._shape, replayDelegate, paramHash);
    }
}

/// <summary>
/// A recorded operation in the trace — captures the execute delegate for replay.
/// </summary>
internal sealed class TracedOp
{
    internal readonly string OpName;
    internal readonly int[] OutputShape;
    internal readonly object ReplayDelegate; // Func<IEngine, Tensor<T>> — type-erased
    internal readonly long ParamHash;

    internal TracedOp(string opName, int[] outputShape, object replayDelegate, long paramHash = 0)
    {
        OpName = opName;
        OutputShape = outputShape;
        ReplayDelegate = replayDelegate;
        ParamHash = paramHash;
    }
}

/// <summary>
/// Per-thread auto-tracer state. Tracks operation sequences and auto-compiles
/// when a pattern repeats.
/// </summary>
internal sealed class AutoTracerState
{
    // Sentinel object to mark patterns that failed compilation (prevents infinite retries)
    private static readonly object FailedCompilationSentinel = new();

    private readonly List<TracedOp> _currentSequence = new();
    private readonly Dictionary<long, object> _compiledPlans = new();
    private readonly LinkedList<long> _evictionOrder = new();
    private int _patternRepeatCount;
    private long _lastPatternHash;

    private const int CompileThreshold = 2;
    private const int MaxSequenceLength = 128;

    /// <summary>
    /// Maximum number of compiled plans cached per thread.
    /// When exceeded, the oldest plan is evicted (LRU).
    /// Prevents unbounded memory growth in long-running processes.
    /// </summary>
    private const int MaxCompiledPlans = 64;

    internal CompiledInferencePlan<T>? TryGetPlan<T>(string opName, int[] outputShape, long paramHash = 0)
    {
        long hash = ComputeLookupHash(opName, outputShape, paramHash, typeof(T).GetHashCode());
        if (_compiledPlans.TryGetValue(hash, out var plan))
        {
            // Don't clear sequence or return for FailedCompilationSentinel —
            // that would cause the tracer to repeatedly re-record and re-attempt
            // compilation for patterns that already failed.
            if (ReferenceEquals(plan, FailedCompilationSentinel))
                return null;

            // Reset sequence since we're using the compiled plan
            _currentSequence.Clear();

            // Update LRU order — move this plan to end (most recently used)
            _evictionOrder.Remove(hash);
            _evictionOrder.AddLast(hash);

            return plan as CompiledInferencePlan<T>;
        }
        return null;
    }

    internal void RecordOp<T>(string opName, int[] outputShape, Func<IEngine, Tensor<T>> replayDelegate, long paramHash = 0)
    {
        if (_currentSequence.Count >= MaxSequenceLength)
        {
            _currentSequence.Clear();
            _patternRepeatCount = 0;
            return;
        }

        _currentSequence.Add(new TracedOp(opName, (int[])outputShape.Clone(), replayDelegate, paramHash));

        long hash = ComputeCurrentHash();
        if (hash == _lastPatternHash && _lastPatternHash != 0)
        {
            _patternRepeatCount++;
            // Use the same key computation as TryCompile (includes type hash)
            // to correctly detect both compiled plans and FailedCompilationSentinel entries.
            long storageKey = hash;
            storageKey ^= typeof(T).GetHashCode();
            storageKey *= unchecked((long)0x100000001b3L);
            if (_patternRepeatCount >= CompileThreshold && !_compiledPlans.ContainsKey(storageKey))
            {
                TryCompile<T>(hash);
            }
        }
        else
        {
            _lastPatternHash = hash;
            _patternRepeatCount = 1;
        }
    }

    /// <summary>
    /// Compiles the traced sequence into a CompiledInferencePlan.
    /// Replays all recorded ops inside a GraphMode scope, then calls CompileInference.
    /// </summary>
    private void TryCompile<T>(long patternHash)
    {
        // Include type hash in storage key so it matches the lookup hash used in TryGetPlan.
        // ComputeCurrentHash() doesn't include type, but TryGetPlan uses ComputeLookupHash
        // which does — so we must store under the same key that lookup will use.
        long storageKey = patternHash;
        storageKey ^= typeof(T).GetHashCode();
        storageKey *= unchecked((long)0x100000001b3L);

        try
        {
            using var scope = GraphMode.Enable();
            var engine = AiDotNetEngine.Current;
            if (engine is null) return;

            // Replay each traced op inside GraphMode — this builds the lazy graph
            foreach (var tracedOp in _currentSequence)
            {
                if (tracedOp.ReplayDelegate is Func<IEngine, Tensor<T>> replay)
                {
                    replay(engine);
                }
            }

            // Compile the captured graph into an optimized plan
            var plan = scope.CompileInference<T>();
            if (plan is not null)
            {
                // Evict oldest plan if at capacity (LRU eviction)
                while (_compiledPlans.Count >= MaxCompiledPlans && _evictionOrder.Count > 0)
                {
                    long oldest = _evictionOrder.First!.Value;
                    _evictionOrder.RemoveFirst();
                    // Dispose evicted plan if it implements IDisposable (GCHandle cleanup)
                    if (_compiledPlans.TryGetValue(oldest, out var evicted) && evicted is IDisposable disposable)
                        disposable.Dispose();
                    _compiledPlans.Remove(oldest);
                }

                _compiledPlans[storageKey] = plan;
                _evictionOrder.AddLast(storageKey);
            }

            _currentSequence.Clear();
        }
        catch
        {
            // Compilation failed — mark pattern as attempted so we don't retry indefinitely.
            _compiledPlans[storageKey] = FailedCompilationSentinel;
            _evictionOrder.AddLast(storageKey);
        }
    }

    private long ComputeLookupHash(string nextOp, int[] shape, long paramHash = 0, int typeHash = 0)
    {
        long hash = ComputeCurrentHash();
        hash ^= nextOp.GetHashCode();
        hash *= unchecked((long)0x100000001b3L);
        for (int i = 0; i < shape.Length; i++)
        {
            hash ^= shape[i];
            hash *= unchecked((long)0x100000001b3L);
        }
        // Include parameter hash so ops with different params (axis, alpha, etc.)
        // are treated as distinct operations — prevents silent wrong-result bugs
        if (paramHash != 0)
        {
            hash ^= paramHash;
            hash *= unchecked((long)0x100000001b3L);
        }
        // Include element type so plans compiled for float don't match double
        if (typeHash != 0)
        {
            hash ^= typeHash;
            hash *= unchecked((long)0x100000001b3L);
        }
        return hash;
    }

    private long ComputeCurrentHash()
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        foreach (var op in _currentSequence)
        {
            hash ^= op.OpName.GetHashCode();
            hash *= unchecked((long)0x100000001b3L);
            foreach (int dim in op.OutputShape)
            {
                hash ^= dim;
                hash *= unchecked((long)0x100000001b3L);
            }
            if (op.ParamHash != 0)
            {
                hash ^= op.ParamHash;
                hash *= unchecked((long)0x100000001b3L);
            }
        }
        return hash;
    }
}
