using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Caches compiled inference and training plans keyed by input shape.
/// Automatically recompiles when input dimensions change.
///
/// Thread-safe: each thread gets its own compiled plans via the underlying
/// GraphMode/AutoTracer thread-local infrastructure.
///
/// Usage:
///   var cache = new CompiledModelCache&lt;float&gt;();
///   var output = cache.GetOrCompileInference(input._shape, () => model.Forward(input));
/// </summary>
public sealed class CompiledModelCache<T> : IDisposable
{
    private readonly Dictionary<long, ICompiledPlan<T>> _inferencePlans = new();
    private readonly Dictionary<long, ICompiledTrainingPlan<T>> _trainingPlans = new();
    private bool _disposed;

    /// <summary>
    /// Gets a cached inference plan for the given input shape, or compiles a new one.
    /// The forward action is traced once on cache miss and compiled into an optimized plan.
    /// </summary>
    /// <param name="inputShape">Shape of the input tensor (used as cache key).</param>
    /// <param name="forward">The forward pass to trace. Called once during compilation.</param>
    /// <returns>The compiled inference plan.</returns>
    public ICompiledPlan<T> GetOrCompileInference(int[] inputShape, Action forward)
    {
        long key = ComputeShapeKey(inputShape);
        if (_inferencePlans.TryGetValue(key, out var cached) && cached.IsValid(inputShape))
            return cached;

        // Trace the forward pass under GraphMode and compile
        using var scope = GraphMode.Enable();
        forward();
        var plan = scope.CompileInference<T>();

        // Dispose old plan if shape changed
        if (_inferencePlans.TryGetValue(key, out var old))
            old.Dispose();

        _inferencePlans[key] = plan;
        return plan;
    }

    /// <summary>
    /// Gets a cached training plan for the given input shape, or compiles a new one.
    /// The forward + loss computation is traced once and compiled with backward pass.
    /// </summary>
    /// <param name="inputShape">Shape of the input tensor (used as cache key).</param>
    /// <param name="forwardAndLoss">The forward + loss computation to trace.</param>
    /// <param name="parameters">Trainable parameters for gradient computation.</param>
    /// <returns>The compiled training plan.</returns>
    public ICompiledTrainingPlan<T> GetOrCompileTraining(
        int[] inputShape, Action forwardAndLoss, Tensor<T>[] parameters)
    {
        long key = ComputeShapeKey(inputShape);
        if (_trainingPlans.TryGetValue(key, out var cached))
            return cached;

        // Trace forward + loss under GraphMode and compile with backward
        using var scope = GraphMode.Enable();
        forwardAndLoss();
        var plan = scope.CompileTraining(parameters);

        // Dispose old plan if exists
        if (_trainingPlans.TryGetValue(key, out var old))
            old.Dispose();

        _trainingPlans[key] = plan;
        return plan;
    }

    /// <summary>
    /// Invalidates all cached plans. Call when model structure changes
    /// (e.g., adding/removing layers, changing activation functions).
    /// </summary>
    public void Invalidate()
    {
        foreach (var plan in _inferencePlans.Values) plan.Dispose();
        foreach (var plan in _trainingPlans.Values) plan.Dispose();
        _inferencePlans.Clear();
        _trainingPlans.Clear();
    }

    /// <summary>Number of cached inference plans.</summary>
    public int InferencePlanCount => _inferencePlans.Count;

    /// <summary>Number of cached training plans.</summary>
    public int TrainingPlanCount => _trainingPlans.Count;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Invalidate();
    }

    private static long ComputeShapeKey(int[] shape)
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        hash ^= typeof(T).GetHashCode();
        hash *= unchecked((long)0x100000001b3L);
        for (int i = 0; i < shape.Length; i++)
        {
            hash ^= shape[i];
            hash *= unchecked((long)0x100000001b3L);
        }
        return hash;
    }
}
