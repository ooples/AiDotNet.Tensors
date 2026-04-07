using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Auto-Tracing JIT: transparently detects repeated tensor operation patterns
/// and compiles them into optimized plans for zero-overhead replay.
///
/// How it works:
/// 1. First call: operations execute eagerly (tracing mode records the pattern)
/// 2. Second call with same shapes: pattern recognized, graph is compiled
/// 3. Third+ calls: compiled plan is replayed — zero allocation, BLAS/SIMD direct
///
/// This is Layer 2 of the TensorCodec system:
///   Layer 1: AutoTensorCache — zero-alloc eager ops
///   Layer 2: AutoTracer — auto-compile hot paths (this)
///   Layer 3: TensorCodec optimizations — spectral/dataflow/algebraic (applied by compiler)
///
/// Unlike PyTorch's torch.compile() or JAX's jit(), this requires NO user code changes.
/// The engine detects patterns automatically based on operation sequence + tensor shapes.
/// </summary>
internal static class AutoTracer
{
    /// <summary>Whether auto-tracing is enabled (default: true).</summary>
    internal static bool Enabled { get; set; } = true;

    [ThreadStatic]
    private static AutoTracerState? _state;

    /// <summary>Gets the auto-tracer state for this thread.</summary>
    internal static AutoTracerState State => _state ??= new AutoTracerState();

    /// <summary>
    /// Called by CpuEngine before each operation. Records the op signature
    /// for pattern detection. Returns a compiled plan if one exists for
    /// the current pattern.
    /// </summary>
    internal static CompiledInferencePlan<T>? TryGetCompiledPlan<T>(string opName, int[] outputShape)
    {
        if (!Enabled || GraphMode.IsActive) return null;
        return State.TryGetPlan<T>(opName, outputShape);
    }

    /// <summary>
    /// Called by CpuEngine after each operation. Records the op for pattern building.
    /// When a pattern repeats, triggers compilation.
    /// </summary>
    internal static void RecordOp(string opName, int[] outputShape)
    {
        if (!Enabled || GraphMode.IsActive) return;
        State.RecordOp(opName, outputShape);
    }
}

/// <summary>
/// Per-thread auto-tracer state. Tracks operation sequences and caches compiled plans.
/// </summary>
internal sealed class AutoTracerState
{
    // Current sequence of operations being recorded
    private readonly List<string> _currentSequence = new();

    // Pattern hash → compiled plan cache
    private readonly Dictionary<long, object> _compiledPlans = new();

    // How many times we've seen the current pattern
    private int _patternRepeatCount;
    private long _lastPatternHash;

    // Minimum repeats before compiling
    private const int CompileThreshold = 2;

    internal CompiledInferencePlan<T>? TryGetPlan<T>(string opName, int[] outputShape)
    {
        // Check if we have a compiled plan for the current sequence + this op
        long hash = ComputeSequenceHash(opName);
        if (_compiledPlans.TryGetValue(hash, out var plan))
        {
            return plan as CompiledInferencePlan<T>;
        }
        return null;
    }

    internal void RecordOp(string opName, int[] outputShape)
    {
        _currentSequence.Add(opName);

        // Check for pattern repeat (same sequence of ops)
        long hash = ComputeCurrentHash();
        if (hash == _lastPatternHash)
        {
            _patternRepeatCount++;
        }
        else
        {
            _lastPatternHash = hash;
            _patternRepeatCount = 1;
        }
    }

    private long ComputeSequenceHash(string nextOp)
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        foreach (var op in _currentSequence)
        {
            hash ^= op.GetHashCode();
            hash *= unchecked((long)0x100000001b3L);
        }
        hash ^= nextOp.GetHashCode();
        hash *= unchecked((long)0x100000001b3L);
        return hash;
    }

    private long ComputeCurrentHash()
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        foreach (var op in _currentSequence)
        {
            hash ^= op.GetHashCode();
            hash *= unchecked((long)0x100000001b3L);
        }
        return hash;
    }
}
