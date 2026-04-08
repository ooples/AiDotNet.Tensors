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
/// Enabled by default. Disable via TensorCacheSettings or AutoTracer.Enabled = false.
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
        if (!Enabled || GraphMode.IsActive) return null;
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
        if (!Enabled || GraphMode.IsActive) return;
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
        long hash = ComputeLookupHash(opName, outputShape, paramHash);
        if (_compiledPlans.TryGetValue(hash, out var plan))
        {
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
            if (_patternRepeatCount >= CompileThreshold && !_compiledPlans.ContainsKey(hash))
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
                    _compiledPlans.Remove(oldest);
                }

                _compiledPlans[patternHash] = plan;
                _evictionOrder.AddLast(patternHash);
            }

            _currentSequence.Clear();
        }
        catch
        {
            // Compilation failed — mark pattern as attempted so we don't retry indefinitely.
            _compiledPlans[patternHash] = FailedCompilationSentinel;
        }
    }

    private long ComputeLookupHash(string nextOp, int[] shape, long paramHash = 0)
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
